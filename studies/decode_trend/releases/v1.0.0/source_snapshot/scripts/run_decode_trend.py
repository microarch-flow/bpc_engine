#!/usr/bin/env python3
"""Run the audited Decode trend grid and write research-grade artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from decode_engine.config import ConfigurationError, load_engine_config  # noqa: E402
from decode_engine.engine import calculate_decode  # noqa: E402
from decode_engine.schema import EngineConfig, WorkCost  # noqa: E402


DEFAULT_MANIFEST = (
    PROJECT_ROOT / "studies" / "decode_trend" / "models.json"
)
DEFAULT_OUTPUT_DIR = Path("/tmp/bpc_engine_decode_trend")
ENGINE_VERSION = "0.1.0"
RESULT_SCHEMA_VERSION = 1
SAFE_INTEGER_MAX = 2**53
AUDIT_DIMENSIONS = (
    "flops",
    "logical_hbm_traffic",
    "cache_capacity",
    "weight_capacity",
)
AUDIT_STATUSES = {
    "supported",
    "partially_supported",
    "unsupported",
    "not_applicable",
}

FLOP_FIELDS = (
    "parameter_flops",
    "attention_flops",
    "index_flops",
    "state_flops",
    "extra_flops",
)
TRAFFIC_FIELDS = (
    "weight_read_bytes",
    "kv_read_bytes",
    "kv_write_bytes",
    "index_read_bytes",
    "index_write_bytes",
    "state_read_bytes",
    "state_write_bytes",
    "activation_bytes",
    "other_read_bytes",
)
REQUIRED_MODEL_FIELDS = (
    "model_release_id",
    "deployment_profile_id",
    "year",
    "organization",
    "release_date",
    "config_path",
    "checkpoint",
    "checkpoint_revision",
    "sample_roles",
    "parameters",
    "capacity",
    "context",
    "context_anchors",
    "calculation_status",
    "source_refs",
    "known_gaps",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate the Decode trend C x B grid and enrich it with "
            "audited model facts."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"Machine-readable model manifest (default: {DEFAULT_MANIFEST})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Artifact directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help="Run one model_release_id. Repeat to select multiple models.",
    )
    parser.add_argument(
        "--contexts",
        nargs="+",
        type=int,
        help=(
            "Override context points. Without this option, common points and "
            "model-specific anchors are combined."
        ),
    )
    parser.add_argument(
        "--batches",
        nargs="+",
        type=int,
        help="Override Batch points. Defaults to the manifest main grid.",
    )
    parser.add_argument(
        "--include-stress-batches",
        action="store_true",
        help="Append the manifest's stress Batch points.",
    )
    return parser


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"invalid JSON in {path}: line {exc.lineno}, column {exc.colno}"
        ) from exc
    except OSError as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} root must be an object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _positive_int_list(value: Any, path: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{path} must be a non-empty array")
    result: list[int] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            raise ValueError(f"{path}[{index}] must be a positive integer")
        result.append(item)
    if len(result) != len(set(result)):
        raise ValueError(f"{path} must not contain duplicates")
    return tuple(result)


def _required_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a non-empty string")
    return value


def _required_positive_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{path} must be a positive integer")
    return value


def _optional_positive_int(value: Any, path: str) -> int | None:
    if value is None:
        return None
    return _required_positive_int(value, path)


def _audit_status_map(value: Any, path: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be an object")
    if set(value) != set(AUDIT_DIMENSIONS):
        raise ValueError(
            f"{path} must contain exactly {list(AUDIT_DIMENSIONS)}"
        )
    result: dict[str, str] = {}
    for dimension in AUDIT_DIMENSIONS:
        status = value[dimension]
        if status not in AUDIT_STATUSES:
            raise ValueError(
                f"{path}.{dimension} has unsupported status {status!r}"
            )
        result[dimension] = status
    return result


def _active_matrix_parameters(config: EngineConfig) -> int:
    weights = config.model.weights
    active = weights.always_active_parameters + sum(
        group.layers
        * group.selected_per_token
        * group.parameters_per_expert
        for group in weights.routed_expert_groups
    )
    if not float(active).is_integer():
        raise ValueError("active matrix parameter total must be an integer")
    return int(active)


def _mechanism_profile(config: EngineConfig) -> dict[str, Any]:
    layer_counts: dict[str, int] = {}
    for group in config.model.layer_groups:
        for mixer in group.mixers:
            kind = str(mixer.get("kind"))
            if kind == "fixed_cost":
                continue
            if kind == "softmax_attention":
                layout = str(mixer.get("kv_layout", {}).get("kind"))
                access = str(mixer.get("access", {}).get("kind"))
                label = f"softmax_attention:{layout}:{access}"
            else:
                label = kind
            layer_counts[label] = layer_counts.get(label, 0) + group.layers

    weights = config.model.weights
    default_weight_bits = weights.weight_bits or config.deployment.weight_bits
    return {
        "mechanism_layer_counts": layer_counts,
        "always_active_weight_groups": [
            {
                "name": group.name,
                "parameters": group.parameters,
                "weight_bits": group.weight_bits or default_weight_bits,
            }
            for group in weights.always_active_parameter_groups
        ],
        "routed_expert_groups": [
            {
                "name": group.name,
                "layers": group.layers,
                "expert_count": group.expert_count,
                "selected_per_token": group.selected_per_token,
                "parameters_per_expert": group.parameters_per_expert,
                "routing_mode": group.routing_mode,
                "weight_bits": (
                    group.weight_bits or config.deployment.expert_weight_bits
                ),
            }
            for group in weights.routed_expert_groups
        ],
        "deployment_defaults": {
            "weight_bits": config.deployment.weight_bits,
            "expert_weight_bits": config.deployment.expert_weight_bits,
            "kv_bits": config.deployment.kv_bits,
            "index_bits": config.deployment.index_bits,
            "state_bits": config.deployment.state_bits,
            "mac_flops": config.deployment.mac_flops,
        },
    }


def _validate_model_entry(
    entry: Mapping[str, Any],
    *,
    manifest_path: Path,
) -> tuple[Path, EngineConfig, dict[str, Any]]:
    missing = [name for name in REQUIRED_MODEL_FIELDS if name not in entry]
    if missing:
        raise ValueError(f"model entry is missing fields: {missing}")

    model_id = _required_string(entry["model_release_id"], "model_release_id")
    _required_string(
        entry["deployment_profile_id"], f"{model_id}.deployment_profile_id"
    )
    _required_positive_int(entry["year"], f"{model_id}.year")
    _required_string(entry["organization"], f"{model_id}.organization")
    _required_string(entry["release_date"], f"{model_id}.release_date")
    _required_string(entry["checkpoint"], f"{model_id}.checkpoint")
    if entry["checkpoint_revision"] is not None:
        _required_string(
            entry["checkpoint_revision"], f"{model_id}.checkpoint_revision"
        )

    config_path = (
        manifest_path.parent
        / _required_string(entry["config_path"], f"{model_id}.config_path")
    ).resolve()
    try:
        config_path.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise ValueError(f"{model_id}.config_path must stay inside the project") from exc
    if not config_path.is_file():
        raise ValueError(f"{model_id}.config_path does not exist: {config_path}")
    config = load_engine_config(config_path)

    parameters = entry["parameters"]
    if not isinstance(parameters, Mapping):
        raise ValueError(f"{model_id}.parameters must be an object")
    resident_parameters = _required_positive_int(
        parameters.get("decode_resident_parameter_elements"),
        f"{model_id}.parameters.decode_resident_parameter_elements",
    )
    audited_active = _required_positive_int(
        parameters.get("active_matrix_parameter_elements_per_token"),
        f"{model_id}.parameters.active_matrix_parameter_elements_per_token",
    )
    derived_active = _active_matrix_parameters(config)
    if audited_active != derived_active:
        raise ValueError(
            f"{model_id} active matrix parameters disagree: "
            f"manifest={audited_active}, config={derived_active}"
        )
    if audited_active > resident_parameters:
        raise ValueError(
            f"{model_id} active parameters cannot exceed resident parameters"
        )

    capacity = entry["capacity"]
    if not isinstance(capacity, Mapping):
        raise ValueError(f"{model_id}.capacity must be an object")
    decode_capacity = _required_positive_int(
        capacity.get("decode_profile_weight_capacity_bytes"),
        f"{model_id}.capacity.decode_profile_weight_capacity_bytes",
    )
    full_capacity = _optional_positive_int(
        capacity.get("full_checkpoint_capacity_bytes"),
        f"{model_id}.capacity.full_checkpoint_capacity_bytes",
    )
    if full_capacity is not None and full_capacity < decode_capacity:
        raise ValueError(
            f"{model_id} full checkpoint capacity cannot be smaller than "
            "Decode profile capacity"
        )
    _required_string(capacity.get("status"), f"{model_id}.capacity.status")

    context = entry["context"]
    if not isinstance(context, Mapping):
        raise ValueError(f"{model_id}.context must be an object")
    advertised = _required_positive_int(
        context.get("advertised_max_context_tokens_at_release"),
        f"{model_id}.context.advertised_max_context_tokens_at_release",
    )
    for name in (
        "trained_max_context_tokens",
        "evaluated_max_context_tokens",
        "deployed_max_context_tokens",
    ):
        _optional_positive_int(context.get(name), f"{model_id}.context.{name}")
    if not isinstance(context.get("effective_context_observations"), list):
        raise ValueError(
            f"{model_id}.context.effective_context_observations must be an array"
        )

    anchors = entry["context_anchors"]
    if not isinstance(anchors, list):
        raise ValueError(f"{model_id}.context_anchors must be an array")
    for index, anchor in enumerate(anchors):
        if not isinstance(anchor, Mapping):
            raise ValueError(f"{model_id}.context_anchors[{index}] must be an object")
        _required_positive_int(
            anchor.get("tokens"),
            f"{model_id}.context_anchors[{index}].tokens",
        )
        tags = anchor.get("tags")
        if (
            not isinstance(tags, list)
            or not tags
            or any(not isinstance(tag, str) or not tag for tag in tags)
        ):
            raise ValueError(
                f"{model_id}.context_anchors[{index}].tags must contain strings"
            )

    source_refs = entry["source_refs"]
    sources = config.model.metadata.get("sources", {})
    if not isinstance(source_refs, Mapping) or not isinstance(sources, Mapping):
        raise ValueError(f"{model_id} source maps must be objects")
    unknown_refs = sorted(
        {
            source_id
            for refs in source_refs.values()
            for source_id in refs
            if source_id not in sources
        }
    )
    if unknown_refs:
        raise ValueError(f"{model_id} has unknown source_refs: {unknown_refs}")

    if config.model.max_context_tokens != advertised:
        raise ValueError(
            f"{model_id} advertised context {advertised} must equal "
            f"model.max_context_tokens={config.model.max_context_tokens}"
        )

    profile = {
        **dict(entry),
        "config_path": str(config_path.relative_to(PROJECT_ROOT)),
        "config_sha256": _sha256(config_path),
        "model_name": config.model.name,
        "active_parameter_ratio": audited_active / resident_parameters,
        **_mechanism_profile(config),
    }
    return config_path, config, profile


def _validate_manifest(
    manifest: Mapping[str, Any],
    manifest_path: Path,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], list[dict[str, Any]]]:
    if manifest.get("schema_version") != 1:
        raise ValueError("manifest.schema_version must be 1")
    _required_string(manifest.get("study_id"), "manifest.study_id")
    common_contexts = _positive_int_list(
        manifest.get("common_contexts"), "manifest.common_contexts"
    )
    main_batches = _positive_int_list(
        manifest.get("main_batches"), "manifest.main_batches"
    )
    stress_batches = _positive_int_list(
        manifest.get("stress_batches"), "manifest.stress_batches"
    )
    raw_models = manifest.get("models")
    if not isinstance(raw_models, list) or not raw_models:
        raise ValueError("manifest.models must be a non-empty array")
    models: list[dict[str, Any]] = []
    ids: set[str] = set()
    profiles: set[tuple[str, str]] = set()
    for index, raw_entry in enumerate(raw_models):
        if not isinstance(raw_entry, Mapping):
            raise ValueError(f"manifest.models[{index}] must be an object")
        model_id = raw_entry.get("model_release_id")
        profile_id = raw_entry.get("deployment_profile_id")
        if model_id in ids:
            raise ValueError(f"duplicate model_release_id: {model_id}")
        key = (str(model_id), str(profile_id))
        if key in profiles:
            raise ValueError(f"duplicate model/profile pair: {key}")
        ids.add(str(model_id))
        profiles.add(key)
        config_path, config, profile = _validate_model_entry(
            raw_entry,
            manifest_path=manifest_path,
        )
        models.append(
            {
                "entry": dict(raw_entry),
                "config_path": config_path,
                "config": config,
                "profile": profile,
            }
        )
    return common_contexts, main_batches, stress_batches, models


def _validate_mechanism_audit(
    audit: Mapping[str, Any],
    models: Sequence[dict[str, Any]],
) -> tuple[str, dict[str, dict[str, Any]]]:
    if audit.get("schema_version") != 1:
        raise ValueError("mechanism audit schema_version must be 1")
    audit_id = _required_string(audit.get("audit_id"), "audit.audit_id")
    if tuple(audit.get("dimensions", ())) != AUDIT_DIMENSIONS:
        raise ValueError(
            "mechanism audit dimensions must match the runner contract"
        )
    definitions = audit.get("status_definitions")
    if (
        not isinstance(definitions, Mapping)
        or set(definitions) != AUDIT_STATUSES
    ):
        raise ValueError(
            "mechanism audit status_definitions must define every status"
        )

    raw_profiles = audit.get("profiles")
    if not isinstance(raw_profiles, Mapping) or not raw_profiles:
        raise ValueError("mechanism audit profiles must be a non-empty object")
    profiles: dict[str, dict[str, Any]] = {}
    for profile_id, raw_profile in raw_profiles.items():
        if not isinstance(raw_profile, Mapping):
            raise ValueError(f"audit.profiles.{profile_id} must be an object")
        profiles[profile_id] = {
            **dict(raw_profile),
            "dimensions": _audit_status_map(
                raw_profile.get("dimensions"),
                f"audit.profiles.{profile_id}.dimensions",
            ),
        }

    raw_models = audit.get("models")
    if not isinstance(raw_models, list):
        raise ValueError("mechanism audit models must be an array")
    audit_by_id: dict[str, Mapping[str, Any]] = {}
    for index, raw_model in enumerate(raw_models):
        if not isinstance(raw_model, Mapping):
            raise ValueError(f"audit.models[{index}] must be an object")
        model_id = _required_string(
            raw_model.get("model_release_id"),
            f"audit.models[{index}].model_release_id",
        )
        if model_id in audit_by_id:
            raise ValueError(f"duplicate mechanism audit model: {model_id}")
        audit_by_id[model_id] = raw_model

    expected_ids = {
        model["entry"]["model_release_id"] for model in models
    }
    if set(audit_by_id) != expected_ids:
        raise ValueError(
            "mechanism audit model IDs must exactly match the model manifest"
        )

    expanded_by_id: dict[str, dict[str, Any]] = {}
    status_rank = {
        "supported": 0,
        "partially_supported": 1,
        "unsupported": 2,
    }
    for model in models:
        model_id = model["entry"]["model_release_id"]
        raw_model = audit_by_id[model_id]
        overall = _audit_status_map(
            raw_model.get("overall"), f"audit.{model_id}.overall"
        )
        raw_mechanisms = raw_model.get("mechanisms")
        if not isinstance(raw_mechanisms, list) or not raw_mechanisms:
            raise ValueError(f"audit.{model_id}.mechanisms must be non-empty")

        config_labels = model["profile"]["mechanism_layer_counts"]
        routed_groups = {
            group["name"]: group["layers"]
            for group in model["profile"]["routed_expert_groups"]
        }
        seen_labels: set[str] = set()
        seen_groups: set[str] = set()
        expanded_mechanisms: list[dict[str, Any]] = []
        for index, raw_mechanism in enumerate(raw_mechanisms):
            if not isinstance(raw_mechanism, Mapping):
                raise ValueError(
                    f"audit.{model_id}.mechanisms[{index}] must be an object"
                )
            has_label = "config_label" in raw_mechanism
            has_group = "routed_group" in raw_mechanism
            if has_label == has_group:
                raise ValueError(
                    f"audit.{model_id}.mechanisms[{index}] must identify "
                    "exactly one config_label or routed_group"
                )
            key_name = "config_label" if has_label else "routed_group"
            mechanism_key = _required_string(
                raw_mechanism[key_name],
                f"audit.{model_id}.mechanisms[{index}].{key_name}",
            )
            expected_layers = (
                config_labels.get(mechanism_key)
                if has_label
                else routed_groups.get(mechanism_key)
            )
            layers = _required_positive_int(
                raw_mechanism.get("layers"),
                f"audit.{model_id}.mechanisms[{index}].layers",
            )
            if expected_layers != layers:
                raise ValueError(
                    f"audit {model_id} {mechanism_key} layers disagree: "
                    f"audit={layers}, config={expected_layers}"
                )
            (seen_labels if has_label else seen_groups).add(mechanism_key)

            profile_id = _required_string(
                raw_mechanism.get("profile"),
                f"audit.{model_id}.mechanisms[{index}].profile",
            )
            if profile_id not in profiles:
                raise ValueError(
                    f"audit {model_id} references unknown profile {profile_id}"
                )
            dimensions = dict(profiles[profile_id]["dimensions"])
            overrides = raw_mechanism.get("overrides", {})
            if not isinstance(overrides, Mapping):
                raise ValueError(
                    f"audit.{model_id}.mechanisms[{index}].overrides "
                    "must be an object"
                )
            unknown_overrides = set(overrides) - set(AUDIT_DIMENSIONS)
            if unknown_overrides:
                raise ValueError(
                    f"audit {model_id} has unknown dimension overrides: "
                    f"{sorted(unknown_overrides)}"
                )
            for dimension, status in overrides.items():
                if status not in AUDIT_STATUSES:
                    raise ValueError(
                        f"audit {model_id} has unsupported status {status!r}"
                    )
                dimensions[dimension] = status

            for dimension, status in dimensions.items():
                if status == "not_applicable":
                    continue
                overall_status = overall[dimension]
                if overall_status == "not_applicable":
                    raise ValueError(
                        f"audit {model_id} overall {dimension} cannot be "
                        "not_applicable"
                    )
                if status_rank[overall_status] < status_rank[status]:
                    raise ValueError(
                        f"audit {model_id} overall {dimension} is more "
                        "confident than mechanism evidence"
                    )
            expanded_mechanisms.append(
                {
                    key_name: mechanism_key,
                    "layers": layers,
                    "profile": profile_id,
                    "dimensions": dimensions,
                    "boundary": profiles[profile_id]["boundary"],
                    "anchors": profiles[profile_id]["anchors"],
                }
            )

        if seen_labels != set(config_labels):
            raise ValueError(
                f"audit {model_id} sequence mechanism coverage mismatch"
            )
        if seen_groups != set(routed_groups):
            raise ValueError(
                f"audit {model_id} routed-group coverage mismatch"
            )
        known_gaps = raw_model.get("known_gaps")
        if not isinstance(known_gaps, list) or any(
            not isinstance(gap, str) or not gap for gap in known_gaps
        ):
            raise ValueError(f"audit.{model_id}.known_gaps must be strings")
        expanded_by_id[model_id] = {
            "audit_id": audit_id,
            "overall": overall,
            "mechanisms": expanded_mechanisms,
            "known_gaps": known_gaps,
        }
    return audit_id, expanded_by_id


def _git_value(args: Sequence[str]) -> str | None:
    completed = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    value = completed.stdout.strip()
    return value if completed.returncode == 0 and value else None


def _context_tags(
    entry: Mapping[str, Any],
    context: int,
    common_contexts: set[int],
) -> list[str]:
    tags: list[str] = []
    if context in common_contexts:
        tags.append("common_power_of_two")
    if context > 1_048_576:
        tags.append("future_context_stress")
    for anchor in entry["context_anchors"]:
        if anchor["tokens"] == context:
            tags.extend(anchor["tags"])
    return sorted(set(tags))


def _within(context: int, limit: int | None) -> bool | None:
    return None if limit is None else context <= limit


def _work_csv(prefix: str, work: WorkCost) -> dict[str, float]:
    result = {
        f"{prefix}{name}": getattr(work, name)
        for name in (*FLOP_FIELDS, *TRAFFIC_FIELDS)
    }
    result[f"{prefix}total_flops"] = work.total_flops
    result[f"{prefix}engine_total_bytes"] = work.total_bytes
    result[f"{prefix}logical_hbm_bytes"] = work.logical_hbm_bytes
    return result


def _build_records(
    *,
    run_id: str,
    study_id: str,
    model: Mapping[str, Any],
    contexts: Sequence[int],
    batches: Sequence[int],
    common_contexts: set[int],
    main_batches: set[int],
    stress_batches: set[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    entry = model["entry"]
    config: EngineConfig = model["config"]
    profile = model["profile"]
    p3_audit = profile["p3_audit"]
    context_facts = entry["context"]
    advertised = context_facts["advertised_max_context_tokens_at_release"]
    decode_capacity = entry["capacity"][
        "decode_profile_weight_capacity_bytes"
    ]
    full_capacity = entry["capacity"]["full_checkpoint_capacity_bytes"]
    routing_assumptions = {
        group.name: group.routing_mode
        for group in config.model.weights.routed_expert_groups
    }

    json_records: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for batch in batches:
        batch_class = (
            "main"
            if batch in main_batches
            else "stress"
            if batch in stress_batches
            else "custom"
        )
        for context in contexts:
            is_extrapolated = context > advertised
            result = calculate_decode(
                config.model,
                config.deployment,
                [context] * batch,
                allow_extrapolation=is_extrapolated,
            )
            if result.per_output_work.activation_bytes != 0:
                raise ValueError(
                    f"{entry['model_release_id']} produced non-zero Decode "
                    "activation traffic; the trend contract excludes it"
                )

            cache_per_request = result.cache_capacity_per_request_average
            cache_total = result.cache_capacity_total
            persistent_decode = decode_capacity + cache_total.total_bytes
            persistent_full = (
                full_capacity + cache_total.total_bytes
                if full_capacity is not None
                else None
            )
            logical_bytes = result.per_output_work.logical_hbm_bytes
            total_flops = result.per_output_work.total_flops
            tags = _context_tags(entry, context, common_contexts)
            row_warnings: list[str] = []
            if is_extrapolated:
                row_warnings.append("context_exceeds_advertised_release_limit")
            for dimension, status in p3_audit["overall"].items():
                if status in {"partially_supported", "unsupported"}:
                    row_warnings.append(f"p3_{dimension}_{status}")
            for label, value in (
                ("batch_cache_bytes", cache_total.total_bytes),
                ("persistent_decode_profile_bytes", persistent_decode),
                ("persistent_full_checkpoint_bytes", persistent_full),
            ):
                if value is None:
                    continue
                if value > SAFE_INTEGER_MAX:
                    warning = (
                        f"{entry['model_release_id']} C={context} B={batch} "
                        f"{label} exceeds exact IEEE-754 integer range"
                    )
                    row_warnings.append("value_above_float_exact_integer_range")
                    warnings.append(warning)
                    break

            scope = {
                "is_extrapolated": is_extrapolated,
                "within_advertised_context": context <= advertised,
                "within_trained_context": _within(
                    context, context_facts["trained_max_context_tokens"]
                ),
                "within_evaluated_context": _within(
                    context, context_facts["evaluated_max_context_tokens"]
                ),
                "within_deployed_context": _within(
                    context, context_facts["deployed_max_context_tokens"]
                ),
                "context_anchor_tags": tags,
            }
            capacity = {
                "decode_profile_weight_capacity_bytes": decode_capacity,
                "full_checkpoint_capacity_bytes": full_capacity,
                "kv_cache_bytes_per_request": cache_per_request.kv_bytes,
                "index_cache_bytes_per_request": cache_per_request.index_bytes,
                "state_cache_bytes_per_request": cache_per_request.state_bytes,
                "cache_bytes_per_request": cache_per_request.total_bytes,
                "kv_cache_bytes_total": cache_total.kv_bytes,
                "index_cache_bytes_total": cache_total.index_bytes,
                "state_cache_bytes_total": cache_total.state_bytes,
                "batch_cache_bytes": cache_total.total_bytes,
                "persistent_decode_profile_bytes": persistent_decode,
                "persistent_full_checkpoint_bytes": persistent_full,
            }
            derived = {
                "logical_hbm_bytes_per_flop": (
                    logical_bytes / total_flops
                    if total_flops
                    else None
                ),
                "flops_per_logical_hbm_byte": (
                    total_flops / logical_bytes
                    if logical_bytes
                    else None
                ),
                "tbps_per_pflops": (
                    1000.0 * logical_bytes / total_flops
                    if total_flops
                    else None
                ),
                "cache_to_decode_weight_capacity_ratio": (
                    cache_total.total_bytes / decode_capacity
                ),
            }
            audit = {
                **entry["calculation_status"],
                "routing_assumptions": routing_assumptions,
                "p3_overall": p3_audit["overall"],
                "p3_mechanisms": p3_audit["mechanisms"],
                "p3_known_gaps": p3_audit["known_gaps"],
                "warnings": sorted(set(row_warnings)),
            }
            json_record = {
                "result_schema_version": RESULT_SCHEMA_VERSION,
                "run_id": run_id,
                "study_id": study_id,
                "model_release_id": entry["model_release_id"],
                "deployment_profile_id": entry["deployment_profile_id"],
                "config_sha256": profile["config_sha256"],
                "context_C": context,
                "concurrency_B": batch,
                "batch_class": batch_class,
                "scope": scope,
                "step_work": result.step_work.to_dict(),
                "per_output_work": result.per_output_work.to_dict(),
                "capacity": capacity,
                "derived": derived,
                "expert_weight_sets_read": dict(
                    result.expert_weight_sets_read
                ),
                "audit": audit,
            }
            json_records.append(json_record)

            csv_row: dict[str, Any] = {
                "result_schema_version": RESULT_SCHEMA_VERSION,
                "run_id": run_id,
                "study_id": study_id,
                "year": entry["year"],
                "organization": entry["organization"],
                "model_release_id": entry["model_release_id"],
                "deployment_profile_id": entry["deployment_profile_id"],
                "config_path": profile["config_path"],
                "config_sha256": profile["config_sha256"],
                "release_date": entry["release_date"],
                "checkpoint": entry["checkpoint"],
                "checkpoint_revision": entry["checkpoint_revision"] or "",
                "context_C": context,
                "concurrency_B": batch,
                "batch_class": batch_class,
                "is_extrapolated": is_extrapolated,
                "within_advertised_context": scope[
                    "within_advertised_context"
                ],
                "within_trained_context": scope["within_trained_context"],
                "within_evaluated_context": scope[
                    "within_evaluated_context"
                ],
                "within_deployed_context": scope["within_deployed_context"],
                "context_anchor_tags": json.dumps(tags),
                "decode_resident_parameter_elements": entry["parameters"][
                    "decode_resident_parameter_elements"
                ],
                "active_matrix_parameter_elements_per_token": entry[
                    "parameters"
                ]["active_matrix_parameter_elements_per_token"],
                "active_parameter_ratio": profile[
                    "active_parameter_ratio"
                ],
                **_work_csv("step_", result.step_work),
                **_work_csv("per_token_", result.per_output_work),
                **capacity,
                **derived,
                "expert_weight_sets_read": json.dumps(
                    dict(result.expert_weight_sets_read),
                    sort_keys=True,
                ),
                "routing_assumptions": json.dumps(
                    routing_assumptions,
                    sort_keys=True,
                ),
                "flops_status": audit["flops"],
                "traffic_status": audit["traffic"],
                "capacity_status": audit["capacity"],
                "p3_flops_support": p3_audit["overall"]["flops"],
                "p3_logical_hbm_traffic_support": p3_audit["overall"][
                    "logical_hbm_traffic"
                ],
                "p3_cache_capacity_support": p3_audit["overall"][
                    "cache_capacity"
                ],
                "p3_weight_capacity_support": p3_audit["overall"][
                    "weight_capacity"
                ],
                "p3_known_gaps": json.dumps(
                    p3_audit["known_gaps"],
                    ensure_ascii=False,
                ),
                "calculation_assumptions": json.dumps(
                    audit["assumptions"],
                    ensure_ascii=False,
                ),
                "calculation_warnings": json.dumps(audit["warnings"]),
            }
            csv_rows.append(csv_row)
    return json_records, csv_rows, warnings


def _validate_result_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    checked = 0
    extrapolated = 0
    for row in rows:
        checked += 1
        batch = row["concurrency_B"]
        if row["is_extrapolated"]:
            extrapolated += 1
        for field in FLOP_FIELDS:
            if not math.isclose(
                row[f"step_{field}"],
                row[f"per_token_{field}"] * batch,
                rel_tol=1e-12,
                abs_tol=1e-6,
            ):
                raise ValueError(f"step/token normalization failed for {field}")
        for field in TRAFFIC_FIELDS:
            if not math.isclose(
                row[f"step_{field}"],
                row[f"per_token_{field}"] * batch,
                rel_tol=1e-12,
                abs_tol=1e-6,
            ):
                raise ValueError(f"step/token normalization failed for {field}")
        if not math.isclose(
            row["batch_cache_bytes"],
            row["cache_bytes_per_request"] * batch,
            rel_tol=1e-12,
            abs_tol=1e-6,
        ):
            raise ValueError("Batch/request Cache normalization failed")
        if row["per_token_activation_bytes"] != 0:
            raise ValueError("Decode trend activation bytes must be zero")
        if not math.isclose(
            row["persistent_decode_profile_bytes"],
            row["decode_profile_weight_capacity_bytes"]
            + row["batch_cache_bytes"],
            rel_tol=1e-12,
            abs_tol=1e-6,
        ):
            raise ValueError("Decode persistent capacity identity failed")
        if row["full_checkpoint_capacity_bytes"] is None:
            if row["persistent_full_checkpoint_bytes"] is not None:
                raise ValueError(
                    "Unknown full-checkpoint capacity must remain unknown"
                )
        elif not math.isclose(
            row["persistent_full_checkpoint_bytes"],
            row["full_checkpoint_capacity_bytes"]
            + row["batch_cache_bytes"],
            rel_tol=1e-12,
            abs_tol=1e-6,
        ):
            raise ValueError("Full-checkpoint persistent capacity identity failed")
    return {"rows_checked": checked, "extrapolated_rows": extrapolated}


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, values: Iterable[Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(
                json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n"
            )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty result CSV")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest_path = args.manifest.resolve()
        manifest = _load_json(manifest_path)
        (
            common_contexts,
            main_batches,
            stress_batches,
            models,
        ) = _validate_manifest(manifest, manifest_path)
        audit_path = (
            manifest_path.parent
            / _required_string(
                manifest.get("mechanism_audit_path"),
                "manifest.mechanism_audit_path",
            )
        ).resolve()
        audit_document = _load_json(audit_path)
        audit_id, audit_by_model = _validate_mechanism_audit(
            audit_document,
            models,
        )
        for model in models:
            model_id = model["entry"]["model_release_id"]
            model["profile"]["p3_audit"] = audit_by_model[model_id]

        if args.models:
            selected_ids = set(args.models)
            known_ids = {
                model["entry"]["model_release_id"] for model in models
            }
            unknown = sorted(selected_ids - known_ids)
            if unknown:
                raise ValueError(f"unknown --model values: {unknown}")
            models = [
                model
                for model in models
                if model["entry"]["model_release_id"] in selected_ids
            ]

        context_override = (
            _positive_int_list(args.contexts, "--contexts")
            if args.contexts
            else None
        )
        batch_override = (
            _positive_int_list(args.batches, "--batches")
            if args.batches
            else None
        )
        batches = list(batch_override or main_batches)
        if args.include_stress_batches:
            batches.extend(stress_batches)
        batches = sorted(set(batches))

        generated_at = datetime.now(timezone.utc).replace(
            microsecond=0
        ).isoformat().replace("+00:00", "Z")
        run_id = (
            f"{manifest['study_id']}-"
            f"{generated_at.replace(':', '').replace('-', '')}"
        )
        common_context_set = set(common_contexts)
        all_json_records: list[dict[str, Any]] = []
        all_csv_rows: list[dict[str, Any]] = []
        warnings: list[str] = []
        profiles: list[dict[str, Any]] = []
        selected_contexts: dict[str, list[int]] = {}
        for model in models:
            entry = model["entry"]
            contexts = (
                list(context_override)
                if context_override is not None
                else sorted(
                    set(common_contexts)
                    | {
                        anchor["tokens"]
                        for anchor in entry["context_anchors"]
                    }
                )
            )
            selected_contexts[entry["model_release_id"]] = contexts
            json_records, csv_rows, model_warnings = _build_records(
                run_id=run_id,
                study_id=str(manifest["study_id"]),
                model=model,
                contexts=contexts,
                batches=batches,
                common_contexts=common_context_set,
                main_batches=set(main_batches),
                stress_batches=set(stress_batches),
            )
            all_json_records.extend(json_records)
            all_csv_rows.extend(csv_rows)
            warnings.extend(model_warnings)
            profiles.append(model["profile"])

        checks = _validate_result_rows(all_csv_rows)
        output_dir = args.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        git_commit = _git_value(("rev-parse", "HEAD"))
        git_status = _git_value(("status", "--short"))
        run_manifest = {
            "result_schema_version": RESULT_SCHEMA_VERSION,
            "run_id": run_id,
            "study_id": manifest["study_id"],
            "generated_at_utc": generated_at,
            "source_manifest": _display_path(manifest_path),
            "source_manifest_sha256": _sha256(manifest_path),
            "mechanism_audit": _display_path(audit_path),
            "mechanism_audit_sha256": _sha256(audit_path),
            "mechanism_audit_id": audit_id,
            "engine_version": ENGINE_VERSION,
            "git_commit": git_commit,
            "git_worktree_dirty": bool(git_status),
            "logical_device_boundary": (
                "complete model on one logical superchip; no inter-chip "
                "communication"
            ),
            "phase": "decode",
            "context_semantics": "start-of-step existing tokens per request",
            "batch_semantics": "active requests producing one token this step",
            "contexts_by_model": selected_contexts,
            "batches": batches,
            "selected_model_release_ids": [
                model["entry"]["model_release_id"] for model in models
            ],
            "artifacts": {
                "model_profiles": "model_profiles.jsonl",
                "results_jsonl": "decode_results.jsonl",
                "results_csv": "decode_results.csv",
                "validation_report": "validation_report.json",
            },
        }
        validation_report = {
            "status": "pass",
            "run_id": run_id,
            "model_count": len(models),
            "row_count": len(all_csv_rows),
            "checks": checks,
            "warning_count": len(warnings),
            "warnings": warnings,
        }
        _write_json(output_dir / "run_manifest.json", run_manifest)
        _write_jsonl(output_dir / "model_profiles.jsonl", profiles)
        _write_jsonl(output_dir / "decode_results.jsonl", all_json_records)
        _write_csv(output_dir / "decode_results.csv", all_csv_rows)
        _write_json(
            output_dir / "validation_report.json", validation_report
        )
    except (ConfigurationError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(
        f"wrote {len(all_csv_rows)} rows for {len(models)} models to "
        f"{output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
