"""JSON configuration loading and validation.

Only the Python standard library is required.  JSON was chosen for the first
version so configurations are portable and parsing semantics are unambiguous;
YAML support can be added later without changing the normalized dataclasses.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .mechanisms import build_mixer
from .schema import (
    AlwaysActiveParameterGroup,
    DeploymentConfig,
    EngineConfig,
    LayerGroupConfig,
    ModelConfig,
    RoutedExpertGroup,
    WeightConfig,
)


class ConfigurationError(ValueError):
    """Raised when a model/deployment configuration is incomplete or invalid."""


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{path} must be an object")
    return value


def _sequence(value: Any, path: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ConfigurationError(f"{path} must be an array")
    return value


def _string(config: Mapping[str, Any], key: str, path: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{path}.{key} must be a non-empty string")
    return value


def _number(
    config: Mapping[str, Any],
    key: str,
    path: str,
    *,
    default: float | None = None,
) -> float:
    if key not in config:
        if default is not None:
            return default
        raise ConfigurationError(f"{path}.{key} is required")
    value = config[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{path}.{key} must be a number")
    return float(value)


def _nonnegative(
    config: Mapping[str, Any],
    key: str,
    path: str,
    *,
    default: float | None = None,
) -> float:
    value = _number(config, key, path, default=default)
    if value < 0:
        raise ConfigurationError(f"{path}.{key} must be >= 0")
    return value


def _positive(
    config: Mapping[str, Any],
    key: str,
    path: str,
    *,
    default: float | None = None,
) -> float:
    value = _number(config, key, path, default=default)
    if value <= 0:
        raise ConfigurationError(f"{path}.{key} must be > 0")
    return value


def _positive_int(config: Mapping[str, Any], key: str, path: str) -> int:
    value = _positive(config, key, path)
    if not value.is_integer():
        raise ConfigurationError(f"{path}.{key} must be an integer")
    return int(value)


def _optional_bits(
    config: Mapping[str, Any], key: str, path: str
) -> float | None:
    return _positive(config, key, path) if key in config else None


def _boolean(
    config: Mapping[str, Any], key: str, path: str, default: bool
) -> bool:
    value = config.get(key, default)
    if not isinstance(value, bool):
        raise ConfigurationError(f"{path}.{key} must be true or false")
    return value


def _fraction(
    config: Mapping[str, Any], key: str, path: str, default: float
) -> float:
    value = _number(config, key, path, default=default)
    if not 0.0 <= value <= 1.0:
        raise ConfigurationError(f"{path}.{key} must be between 0 and 1")
    return value


def _parse_deployment(config: Mapping[str, Any]) -> DeploymentConfig:
    path = "deployment"
    weight_bits = _positive(config, "weight_bits", path)
    kv_bits = _positive(config, "kv_bits", path)
    return DeploymentConfig(
        weight_bits=weight_bits,
        expert_weight_bits=_positive(
            config, "expert_weight_bits", path, default=weight_bits
        ),
        kv_bits=kv_bits,
        index_bits=_positive(config, "index_bits", path, default=kv_bits),
        state_bits=_positive(config, "state_bits", path, default=16.0),
        mac_flops=_positive(config, "mac_flops", path, default=2.0),
        include_kv_write=_boolean(config, "include_kv_write", path, True),
        include_index_write=_boolean(
            config, "include_index_write", path, True
        ),
        include_state_write=_boolean(
            config, "include_state_write", path, True
        ),
        weight_hbm_fraction=_fraction(
            config, "weight_hbm_fraction", path, 1.0
        ),
        kv_hbm_fraction=_fraction(config, "kv_hbm_fraction", path, 1.0),
        index_hbm_fraction=_fraction(
            config, "index_hbm_fraction", path, 1.0
        ),
        state_hbm_fraction=_fraction(
            config, "state_hbm_fraction", path, 1.0
        ),
        weight_read_multiplier=_positive(
            config, "weight_read_multiplier", path, default=1.0
        ),
        activation_bytes_per_output_token=_nonnegative(
            config,
            "activation_bytes_per_output_token",
            path,
            default=0.0,
        ),
        extra_flops_per_output_token=_nonnegative(
            config, "extra_flops_per_output_token", path, default=0.0
        ),
        activation_bytes_per_input_token=_nonnegative(
            config,
            "activation_bytes_per_input_token",
            path,
            default=0.0,
        ),
        extra_flops_per_input_token=_nonnegative(
            config, "extra_flops_per_input_token", path, default=0.0
        ),
    )


def _parse_explicit_unique(
    config: Mapping[str, Any],
    path: str,
    expert_count: int,
    selected_per_token: int,
    key: str,
) -> dict[int, float]:
    raw = config.get(key, {})
    raw_mapping = _mapping(raw, f"{path}.{key}")
    result: dict[int, float] = {}
    for raw_batch, raw_value in raw_mapping.items():
        try:
            batch = int(raw_batch)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(
                f"{path}.{key} keys must be positive integer counts"
            ) from exc
        if batch <= 0 or isinstance(raw_value, bool) or not isinstance(
            raw_value, (int, float)
        ):
            raise ConfigurationError(
                f"{path}.{key} has invalid entry"
            )
        value = float(raw_value)
        lower = float(selected_per_token)
        upper = float(min(expert_count, batch * selected_per_token))
        if not lower <= value <= upper:
            raise ConfigurationError(
                f"{path}.{key}[{batch}] must be between selected_per_token="
                f"{selected_per_token} and min(expert_count, count * "
                f"selected_per_token)={upper:g}"
            )
        result[batch] = value
    return result


def _parse_expert_group(
    config: Mapping[str, Any], index: int
) -> RoutedExpertGroup:
    path = f"model.weights.routed_expert_groups[{index}]"
    expert_count = _positive_int(config, "expert_count", path)
    selected = _positive_int(config, "selected_per_token", path)
    if selected > expert_count:
        raise ConfigurationError(
            f"{path}.selected_per_token cannot exceed expert_count"
        )

    mode = config.get("routing_mode", "uniform_independent")
    supported = {
        "uniform_independent",
        "same_experts",
        "explicit_unique",
        "no_batch_reuse",
    }
    if mode not in supported:
        raise ConfigurationError(
            f"{path}.routing_mode must be one of {sorted(supported)}"
        )
    explicit_batches = _parse_explicit_unique(
        config,
        path,
        expert_count,
        selected,
        "expected_unique_experts_by_batch",
    )
    explicit_tokens = _parse_explicit_unique(
        config,
        path,
        expert_count,
        selected,
        "expected_unique_experts_by_active_tokens",
    )
    if mode == "explicit_unique" and not (explicit_batches or explicit_tokens):
        raise ConfigurationError(
            f"{path} requires expected_unique_experts_by_batch and/or "
            "expected_unique_experts_by_active_tokens"
        )

    return RoutedExpertGroup(
        name=_string(config, "name", path),
        layers=_positive_int(config, "layers", path),
        expert_count=expert_count,
        selected_per_token=selected,
        parameters_per_expert=_positive(
            config, "parameters_per_expert", path
        ),
        routing_mode=str(mode),
        expected_unique_experts_by_batch=explicit_batches,
        expected_unique_experts_by_active_tokens=explicit_tokens,
        weight_bits=_optional_bits(config, "weight_bits", path),
    )


def _parse_weights(config: Mapping[str, Any]) -> WeightConfig:
    path = "model.weights"
    raw_groups = _sequence(config.get("routed_expert_groups", []), path)
    groups = tuple(
        _parse_expert_group(_mapping(group, f"{path}[{index}]"), index)
        for index, group in enumerate(raw_groups)
    )
    names = [group.name for group in groups]
    if len(names) != len(set(names)):
        raise ConfigurationError(f"{path} routed expert group names must be unique")

    raw_always_groups = config.get("always_active_parameter_groups")
    if raw_always_groups is not None and "always_active_parameters" in config:
        raise ConfigurationError(
            f"{path} must use either always_active_parameters or "
            "always_active_parameter_groups, not both"
        )
    if raw_always_groups is None:
        always_groups = (
            AlwaysActiveParameterGroup(
                name="always_active",
                parameters=_nonnegative(
                    config, "always_active_parameters", path
                ),
            ),
        )
    else:
        parsed_always: list[AlwaysActiveParameterGroup] = []
        for index, raw_group in enumerate(
            _sequence(raw_always_groups, f"{path}.always_active_parameter_groups")
        ):
            group_path = f"{path}.always_active_parameter_groups[{index}]"
            group = _mapping(raw_group, group_path)
            parsed_always.append(
                AlwaysActiveParameterGroup(
                    name=_string(group, "name", group_path),
                    parameters=_nonnegative(group, "parameters", group_path),
                    weight_bits=_optional_bits(group, "weight_bits", group_path),
                )
            )
        if not parsed_always:
            raise ConfigurationError(
                f"{path}.always_active_parameter_groups cannot be empty"
            )
        always_names = [group.name for group in parsed_always]
        if len(always_names) != len(set(always_names)):
            raise ConfigurationError(
                f"{path} always-active group names must be unique"
            )
        always_groups = tuple(parsed_always)

    result = WeightConfig(
        always_active_parameter_groups=always_groups,
        routed_expert_groups=groups,
        weight_bits=_optional_bits(config, "weight_bits", path),
        output_head_parameters=_nonnegative(
            config, "output_head_parameters", path, default=0.0
        ),
        output_head_parameters_configured=(
            "output_head_parameters" in config
        ),
        output_head_weight_bits=_optional_bits(
            config, "output_head_weight_bits", path
        ),
    )
    if result.output_head_parameters > result.always_active_parameters:
        raise ConfigurationError(
            f"{path}.output_head_parameters cannot exceed the "
            "always-active parameter total"
        )
    return result


def _parse_layer_group(
    config: Mapping[str, Any], index: int
) -> LayerGroupConfig:
    path = f"model.layer_groups[{index}]"
    raw_mixers = _sequence(config.get("mixers"), f"{path}.mixers")
    if not raw_mixers:
        raise ConfigurationError(f"{path}.mixers cannot be empty")
    mixers: list[Mapping[str, Any]] = []
    for mixer_index, raw_mixer in enumerate(raw_mixers):
        mixer = dict(
            _mapping(raw_mixer, f"{path}.mixers[{mixer_index}]")
        )
        try:
            # Build once while parsing so unsupported or incomplete mechanisms
            # fail before a long grid calculation starts.
            build_mixer(mixer, f"{path}.mixers[{mixer_index}]")
        except ValueError as exc:
            raise ConfigurationError(str(exc)) from exc
        mixers.append(mixer)
    return LayerGroupConfig(
        name=_string(config, "name", path),
        layers=_positive_int(config, "layers", path),
        mixers=tuple(mixers),
    )


def _parse_model(config: Mapping[str, Any]) -> ModelConfig:
    path = "model"
    max_context: int | None
    if config.get("max_context_tokens") is None:
        max_context = None
    else:
        max_context = _positive_int(config, "max_context_tokens", path)

    raw_groups = _sequence(config.get("layer_groups"), f"{path}.layer_groups")
    if not raw_groups:
        raise ConfigurationError(f"{path}.layer_groups cannot be empty")
    groups = tuple(
        _parse_layer_group(
            _mapping(group, f"{path}.layer_groups[{index}]"), index
        )
        for index, group in enumerate(raw_groups)
    )

    metadata = config.get("metadata", {})
    return ModelConfig(
        name=_string(config, "name", path),
        max_context_tokens=max_context,
        weights=_parse_weights(_mapping(config.get("weights"), f"{path}.weights")),
        layer_groups=groups,
        metadata=dict(_mapping(metadata, f"{path}.metadata")),
    )


def _int_list(value: Any, path: str, *, positive: bool) -> tuple[int, ...]:
    result: list[int] = []
    for index, item in enumerate(_sequence(value, path)):
        if isinstance(item, bool) or not isinstance(item, int):
            raise ConfigurationError(f"{path}[{index}] must be an integer")
        if (positive and item <= 0) or (not positive and item < 0):
            relation = "> 0" if positive else ">= 0"
            raise ConfigurationError(f"{path}[{index}] must be {relation}")
        result.append(item)
    return tuple(result)


def _nested_positive_int_lists(
    value: Any, path: str
) -> tuple[tuple[int, ...], ...]:
    result: list[tuple[int, ...]] = []
    for index, item in enumerate(_sequence(value, path)):
        batch = _int_list(item, f"{path}[{index}]", positive=True)
        if not batch:
            raise ConfigurationError(f"{path}[{index}] cannot be empty")
        result.append(batch)
    return tuple(result)


def parse_engine_config(raw: Mapping[str, Any]) -> EngineConfig:
    """Validate a decoded JSON object and return normalized typed config."""

    config = _mapping(raw, "root")
    version = config.get("schema_version")
    if version != 1:
        raise ConfigurationError("schema_version must be 1")

    analysis = _mapping(config.get("analysis", {}), "analysis")
    prefill = _mapping(analysis.get("prefill", {}), "analysis.prefill")
    return EngineConfig(
        schema_version=1,
        model=_parse_model(_mapping(config.get("model"), "model")),
        deployment=_parse_deployment(
            _mapping(config.get("deployment"), "deployment")
        ),
        default_contexts=_int_list(
            analysis.get("contexts", []), "analysis.contexts", positive=False
        ),
        default_batches=_int_list(
            analysis.get("batches", []), "analysis.batches", positive=True
        ),
        default_prefill_lengths=_int_list(
            prefill.get("prompt_lengths", []),
            "analysis.prefill.prompt_lengths",
            positive=True,
        ),
        default_prefill_batches=_int_list(
            prefill.get("batches", []),
            "analysis.prefill.batches",
            positive=True,
        ),
        default_prefill_token_budgets=_int_list(
            prefill.get("token_budgets", []),
            "analysis.prefill.token_budgets",
            positive=True,
        ),
        default_ragged_prefill_batches=_nested_positive_int_lists(
            prefill.get("ragged_batches", []),
            "analysis.prefill.ragged_batches",
        ),
    )


def load_engine_config(path: str | Path) -> EngineConfig:
    """Load and validate one UTF-8 JSON engine configuration file."""

    config_path = Path(path)
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ConfigurationError(
            f"invalid JSON in {config_path}: line {exc.lineno}, column {exc.colno}"
        ) from exc
    except OSError as exc:
        raise ConfigurationError(f"cannot read {config_path}: {exc}") from exc
    return parse_engine_config(_mapping(raw, "root"))
