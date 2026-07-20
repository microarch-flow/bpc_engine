#!/usr/bin/env python3
"""Fit release-level LLM technology trajectories for Decode trend P9A.

This analysis intentionally does not use the C/B result grid as statistical
observations.  One frozen model deployment profile is one observation.  The
output describes the curated research sample and must not be interpreted as
an industry adoption-rate estimate.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
from datetime import date, datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RELEASE_DIR = (
    PROJECT_ROOT / "studies" / "decode_trend" / "releases" / "v1.0.0"
)
DEFAULT_OUTPUT_DIR = Path("/tmp/decode_trend_p9a_technology_trends")
SCRIPT_VERSION = "0.2"
SUPPORTED_PROFILE_SCHEMA = 1
REFERENCE_YEAR = 2024.0
DEFAULT_PROJECTION_THROUGH_YEAR = 2030
YTD_YEAR = 2026
YTD_CUTOFF = date(2026, 7, 17)
COMPLETE_BACKTEST_YEARS = (2024, 2025)
LOGISTIC_L2 = 0.5

ACCESS_BROAD = {
    "full": "full",
    "swa": "bounded_local",
    "chunked_block": "bounded_local",
    "block_local": "bounded_local",
    "fixed_topk": "sparse_topk",
    "learned_topk": "sparse_topk",
    "dsa": "sparse_topk",
    "csa": "sparse_topk",
}
KV_LAYOUTS = ("mha", "mqa", "gqa", "mla")
MIXER_BROAD = ("softmax", "linear_recurrent", "ssm")
ACCESS_CATEGORIES = ("full", "bounded_local", "sparse_topk")
COMPOSITION_GROUPS = {
    "mixer": (
        "mixer.softmax_share",
        "mixer.linear_recurrent_share",
        "mixer.ssm_share",
    ),
    "kv_layout": (
        "kv_layout.mha_share",
        "kv_layout.mqa_share",
        "kv_layout.gqa_share",
        "kv_layout.mla_share",
    ),
    "attention_access": (
        "access.full_share",
        "access.bounded_local_share",
        "access.sparse_topk_share",
    ),
}

CHART_MODEL_NAMES = {
    "palm-540b": "PaLM 540B",
    "bloom": "BLOOM 176B",
    "glm-130b-int4": "GLM-130B",
    "Llama-2-70b-chat-hf": "Llama 2 70B",
    "falcon-180B": "Falcon 180B",
    "Mistral-7B-Instruct-v0.1": "Mistral 7B",
    "Yi-34B-200K": "Yi-34B",
    "Mixtral-8x7B-Instruct-v0.1": "Mixtral 8x7B",
    "DeepSeek-V2-Chat": "DeepSeek-V2",
    "Llama-3.1-405B-Instruct-FP8": "Llama 3.1 405B",
    "Llama-3.1-70B-Instruct-FP8": "Llama 3.1 70B",
    "AI21-Jamba-Large-1.5": "Jamba 1.5",
    "DeepSeek-V3": "DeepSeek-V3",
    "Llama-4-Scout-17B-16E-Instruct": "Llama 4 Scout",
    "Qwen3-32B-AWQ": "Qwen3-32B",
    "Kimi-Linear-48B-A3B-Instruct": "Kimi-Linear",
    "Kimi-K2-Thinking": "Kimi K2",
    "Kimi-K2.6": "Kimi K2.6",
    "Qwen3.6-35B-A3B-FP8": "Qwen3.6",
    "GLM-5.2-FP8": "GLM-5.2",
}


@dataclass(frozen=True)
class MetricSpec:
    metric_id: str
    field: str
    kind: str
    claim_scope: str
    composition_group: str | None = None
    unit: str = ""
    projection_role: str = "independent_marginal"


METRIC_SPECS: tuple[MetricSpec, ...] = (
    MetricSpec(
        "parameters.resident_elements",
        "decode_resident_parameter_elements",
        "positive_log",
        "curated release-level Decode resident parameter trajectory",
        unit="elements",
    ),
    MetricSpec(
        "parameters.active_elements",
        "active_matrix_parameter_elements_per_token",
        "positive_log",
        "identity-derived from resident parameters and active ratio; direct "
        "marginal fit retained only as a diagnostic",
        unit="elements/token",
        projection_role="derived_identity",
    ),
    MetricSpec(
        "parameters.active_ratio",
        "active_parameter_ratio",
        "fraction",
        "curated release-level active/resident parameter ratio",
        unit="ratio",
    ),
    MetricSpec(
        "parameters.sparsity_multiplier",
        "sparsity_multiplier",
        "positive_log",
        "identity-derived from active/resident parameter ratio",
        unit="ratio",
        projection_role="derived_identity",
    ),
    MetricSpec(
        "parameters.sparsity_gap_bits",
        "sparsity_gap_bits",
        "real_linear",
        "identity-derived log2 resident/active parameter decoupling",
        unit="bits",
        projection_role="derived_identity",
    ),
    MetricSpec(
        "context.advertised_max",
        "advertised_max_context_tokens_at_release",
        "positive_log",
        "curated release advertised context boundary",
        unit="tokens",
    ),
    MetricSpec(
        "context.trained_max",
        "trained_max_context_tokens",
        "positive_log",
        "curated release trained context boundary; independent marginal",
        unit="tokens",
        projection_role="marginal_diagnostic",
    ),
    MetricSpec(
        "context.evaluated_max",
        "evaluated_max_context_tokens",
        "positive_log",
        "curated release evaluated context boundary; independent marginal",
        unit="tokens",
        projection_role="marginal_diagnostic",
    ),
    MetricSpec(
        "context.deployed_max",
        "deployed_max_context_tokens",
        "positive_log",
        "curated selected-profile deployed context boundary; independent "
        "marginal",
        unit="tokens",
        projection_role="marginal_diagnostic",
    ),
    MetricSpec(
        "context.trained_to_advertised_ratio",
        "trained_to_advertised_context_ratio",
        "fraction",
        "curated trained/advertised context boundary ratio",
        unit="ratio",
        projection_role="marginal_diagnostic",
    ),
    MetricSpec(
        "context.evaluated_to_advertised_ratio",
        "evaluated_to_advertised_context_ratio",
        "fraction",
        "curated evaluated/advertised context boundary ratio",
        unit="ratio",
        projection_role="marginal_diagnostic",
    ),
    MetricSpec(
        "context.deployed_to_advertised_ratio",
        "deployed_to_advertised_context_ratio",
        "fraction",
        "curated deployed/advertised context boundary ratio",
        unit="ratio",
        projection_role="marginal_diagnostic",
    ),
    MetricSpec(
        "mixer.softmax_share",
        "softmax_layer_share",
        "fraction",
        "curated release physical-layer composition",
        composition_group="mixer",
        unit="layer share",
        projection_role="composition_component",
    ),
    MetricSpec(
        "mixer.linear_recurrent_share",
        "linear_recurrent_layer_share",
        "fraction",
        "curated release physical-layer composition",
        composition_group="mixer",
        unit="layer share",
        projection_role="composition_component",
    ),
    MetricSpec(
        "mixer.ssm_share",
        "ssm_layer_share",
        "fraction",
        "curated release physical-layer composition",
        composition_group="mixer",
        unit="layer share",
        projection_role="composition_component",
    ),
    *(
        MetricSpec(
            f"kv_layout.{name}_share",
            f"kv_{name}_share_of_softmax",
            "fraction",
            "curated release Softmax-layer KV-layout composition",
            composition_group="kv_layout",
            unit="Softmax layer share",
            projection_role="composition_component",
        )
        for name in KV_LAYOUTS
    ),
    *(
        MetricSpec(
            f"access.{name}_share",
            f"access_{name}_share_of_softmax",
            "fraction",
            "curated release Softmax-layer access composition",
            composition_group="attention_access",
            unit="Softmax layer share",
            projection_role="composition_component",
        )
        for name in ACCESS_CATEGORIES
    ),
    MetricSpec(
        "moe.presence",
        "has_moe",
        "binary",
        "selected-sample presence; not industry adoption",
        unit="probability",
    ),
    MetricSpec(
        "moe.unconditional_layer_share",
        "moe_layer_share",
        "fraction",
        "identity-derived expected routed-layer mass from MoE presence and "
        "conditional layer share; direct marginal retained as diagnostic",
        unit="layer share",
        projection_role="derived_identity",
    ),
    MetricSpec(
        "moe.layer_share_given_moe",
        "moe_layer_share_given_moe",
        "fraction",
        "conditional routed-expert physical-layer share among MoE releases",
        unit="layer share",
    ),
    MetricSpec(
        "moe.expert_count",
        "moe_expert_count",
        "positive_log",
        "conditional on MoE presence in the curated sample",
        unit="experts/layer",
    ),
    MetricSpec(
        "moe.selected_per_token",
        "moe_selected_per_token",
        "positive_log",
        "conditional on MoE presence in the curated sample",
        unit="experts/token/layer",
    ),
    MetricSpec(
        "moe.routing_density",
        "moe_routing_density",
        "positive_log",
        "conditional on MoE presence in the curated sample",
        unit="selected/total experts",
    ),
    MetricSpec(
        "precision.matrix_effective_bits",
        "matrix_effective_weight_bits",
        "positive_log",
        "selected deployment profile precision; not native adoption",
        unit="bits/explicit matrix element",
    ),
    MetricSpec(
        "precision.resident_effective_storage_bits",
        "resident_effective_storage_bits",
        "positive_log",
        "selected deployment profile storage including capacity overhead",
        unit="bits/resident element",
    ),
    MetricSpec(
        "precision.explicit_share_le8",
        "explicit_weight_parameter_share_le8",
        "fraction",
        "selected deployment profile explicit weight groups",
        unit="parameter share",
    ),
    MetricSpec(
        "precision.explicit_share_le4",
        "explicit_weight_parameter_share_le4",
        "fraction",
        "selected deployment profile explicit weight groups",
        unit="parameter share",
    ),
    MetricSpec(
        "precision.kv_effective_bits",
        "kv_effective_bits_used",
        "positive_log",
        "selected deployment profile; element-weighted over actual KV "
        "storage components",
        unit="bits/KV element",
    ),
    MetricSpec(
        "precision.index_effective_bits",
        "index_effective_bits_used",
        "positive_log",
        "conditional on an actual persistent attention-index component",
        unit="bits/index element",
    ),
    MetricSpec(
        "precision.state_effective_bits",
        "state_effective_bits_used",
        "positive_log",
        "conditional on actual recurrent-state components; element-weighted",
        unit="bits/state element",
    ),
    MetricSpec(
        "technology.state_like_presence",
        "has_state_like_mixer",
        "binary",
        "selected-sample presence; not industry adoption",
        unit="probability",
    ),
    MetricSpec(
        "technology.linear_attention_presence",
        "has_explicit_linear_attention",
        "binary",
        "selected-sample presence; descriptive-only when sparse",
        unit="probability",
    ),
    MetricSpec(
        "technology.mla_presence",
        "has_mla",
        "binary",
        "selected-sample presence; not industry adoption",
        unit="probability",
    ),
    MetricSpec(
        "technology.gqa_presence",
        "has_gqa",
        "binary",
        "selected-sample presence; not industry adoption",
        unit="probability",
    ),
    MetricSpec(
        "technology.bounded_access_presence",
        "has_bounded_local_access",
        "binary",
        "selected-sample presence; descriptive-only when sparse",
        unit="probability",
    ),
    MetricSpec(
        "technology.sparse_access_presence",
        "has_sparse_topk_access",
        "binary",
        "selected-sample presence; descriptive-only when sparse",
        unit="probability",
    ),
    MetricSpec(
        "technology.weight_le8_majority_presence",
        "has_majority_explicit_weight_le8",
        "binary",
        "selected deployment profile presence; not native adoption",
        unit="probability",
    ),
    MetricSpec(
        "technology.weight_le4_majority_presence",
        "has_majority_explicit_weight_le4",
        "binary",
        "selected deployment profile presence; not native adoption",
        unit="probability",
    ),
)
METRIC_BY_ID = {spec.metric_id: spec for spec in METRIC_SPECS}

DERIVED_IDENTITIES: dict[str, dict[str, Any]] = {
    "parameters.active_elements": {
        "source_metric_ids": (
            "parameters.resident_elements",
            "parameters.active_ratio",
        ),
        "formula": (
            "parameters.resident_elements(t) * "
            "parameters.active_ratio(t)"
        ),
    },
    "parameters.sparsity_multiplier": {
        "source_metric_ids": ("parameters.active_ratio",),
        "formula": "1 / parameters.active_ratio(t)",
    },
    "parameters.sparsity_gap_bits": {
        "source_metric_ids": ("parameters.active_ratio",),
        "formula": "-log2(parameters.active_ratio(t))",
    },
    "moe.unconditional_layer_share": {
        "source_metric_ids": (
            "moe.presence",
            "moe.layer_share_given_moe",
        ),
        "formula": (
            "moe.presence(t) * moe.layer_share_given_moe(t)"
        ),
    },
}

BINARY_TECHNOLOGIES = (
    "moe.presence",
    "technology.state_like_presence",
    "technology.linear_attention_presence",
    "technology.mla_presence",
    "technology.gqa_presence",
    "technology.bounded_access_presence",
    "technology.sparse_access_presence",
    "technology.weight_le8_majority_presence",
    "technology.weight_le4_majority_presence",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit release-level LLM technology trajectories from a frozen "
            "Decode trend dataset. Outputs describe a curated sample, not "
            "industry adoption."
        )
    )
    parser.add_argument(
        "--release-dir",
        type=Path,
        default=DEFAULT_RELEASE_DIR,
        help="Frozen release directory. Defaults to v1.0.0.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Analysis output directory. Defaults under /tmp.",
    )
    parser.add_argument(
        "--projection-through-year",
        type=int,
        default=DEFAULT_PROJECTION_THROUGH_YEAR,
        help="Last year in the function-substitution display grid.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Write and validate data tables without importing matplotlib.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=160,
        help="PNG resolution. Defaults to 160.",
    )
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_object(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"invalid JSON in {path}: line {exc.lineno}, column {exc.colno}"
        ) from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} root must be an object")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    values: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid JSONL in {path}:{line_number}: {exc.msg}"
            ) from exc
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} must contain an object")
        values.append(value)
    if not values:
        raise ValueError(f"{path} contains no records")
    return values


def _json_cell(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _finite_number(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    if positive and result <= 0:
        raise ValueError(f"{label} must be positive")
    return result


def _decimal_year(value: date) -> float:
    start = date(value.year, 1, 1)
    end = date(value.year + 1, 1, 1)
    return value.year + (value - start).days / (end - start).days


def _short_model_name(profile: Mapping[str, Any]) -> str:
    checkpoint = str(profile.get("checkpoint", ""))
    if "/" in checkpoint:
        return checkpoint.rsplit("/", 1)[-1]
    model_id = str(profile.get("model_release_id", "unknown"))
    parts = model_id.split(":")
    if len(parts) >= 2 and parts[1]:
        return parts[1]
    return checkpoint or model_id


def _chart_model_name(value: str) -> str:
    return CHART_MODEL_NAMES.get(value, value)


def _verify_sha256sums(release_dir: Path) -> dict[str, str]:
    checksum_path = release_dir / "SHA256SUMS"
    try:
        lines = checksum_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"cannot read {checksum_path}: {exc}") from exc
    entries: dict[str, str] = {}
    for line_number, line in enumerate(lines, start=1):
        if not line:
            continue
        digest, separator, relative = line.partition("  ")
        if (
            separator != "  "
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or not relative
        ):
            raise ValueError(
                f"invalid SHA256SUMS entry at line {line_number}"
            )
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(
                f"SHA256SUMS path escapes release at line {line_number}"
            )
        normalized = relative_path.as_posix()
        if normalized in entries:
            raise ValueError(f"duplicate SHA256SUMS path: {normalized}")
        path = (release_dir / relative_path).resolve()
        try:
            path.relative_to(release_dir)
        except ValueError as exc:
            raise ValueError(
                f"SHA256SUMS path escapes release: {normalized}"
            ) from exc
        if not path.is_file():
            raise ValueError(f"SHA256SUMS file is missing: {normalized}")
        actual = _sha256(path)
        if actual != digest:
            raise ValueError(
                f"SHA256SUMS mismatch for {normalized}: "
                f"{actual} != {digest}"
            )
        entries[normalized] = digest
    required = {
        "release_manifest.json",
        "data/run_manifest.json",
        "data/validation_report.json",
        "data/model_profiles.jsonl",
        "data/decode_results.csv",
    }
    missing = sorted(required.difference(entries))
    if missing:
        raise ValueError(
            "SHA256SUMS does not cover required frozen inputs: "
            + ", ".join(missing)
        )
    return entries


def _resolve_config_path(
    profile: Mapping[str, Any],
    release_dir: Path,
    checksum_entries: Mapping[str, str],
) -> Path:
    raw = profile.get("config_path")
    if not isinstance(raw, str) or not raw:
        raise ValueError(
            f"{profile.get('model_release_id')} has no config_path"
        )
    relative = Path("source_snapshot") / raw
    path = (release_dir / relative).resolve()
    try:
        path.relative_to(release_dir.resolve())
    except ValueError as exc:
        raise ValueError(f"config_path escapes frozen snapshot: {raw}") from exc
    if not path.is_file():
        raise ValueError(f"frozen snapshot config does not exist: {path}")
    expected_hash = profile.get("config_sha256")
    actual_hash = _sha256(path)
    if expected_hash != actual_hash:
        raise ValueError(
            f"config hash mismatch for {profile.get('model_release_id')}: "
            f"{actual_hash} != {expected_hash}"
        )
    checksum_hash = checksum_entries.get(relative.as_posix())
    if checksum_hash != actual_hash:
        raise ValueError(
            f"snapshot config is not covered by SHA256SUMS for "
            f"{profile.get('model_release_id')}: {relative.as_posix()}"
        )
    return path


def _csv_data_row_count(path: Path) -> int:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            if not header:
                raise ValueError(f"{path} has no CSV header")
            return sum(1 for row in reader if row)
    except OSError as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc


def _load_release(
    release_dir: Path,
) -> tuple[
    list[dict[str, Any]],
    dict[tuple[str, str], Mapping[str, Any]],
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[str, Any],
    dict[str, str],
]:
    release_dir = release_dir.resolve()
    data_dir = release_dir / "data"
    checksum_entries = _verify_sha256sums(release_dir)
    release_manifest = _load_object(release_dir / "release_manifest.json")
    run_manifest = _load_object(data_dir / "run_manifest.json")
    validation_report = _load_object(data_dir / "validation_report.json")
    profiles_path = data_dir / "model_profiles.jsonl"
    if release_manifest.get("dataset_release_schema_version") != 1:
        raise ValueError("unsupported dataset_release_schema_version")
    if run_manifest.get("result_schema_version") != SUPPORTED_PROFILE_SCHEMA:
        raise ValueError("unsupported result_schema_version")
    if validation_report.get("status") != "pass":
        raise ValueError("frozen validation status is not pass")
    profiles = _load_jsonl(profiles_path)
    expected_count = int(release_manifest.get("model_count", -1))
    if expected_count != len(profiles):
        raise ValueError("release model_count disagrees with model profiles")
    if int(validation_report.get("model_count", -1)) != len(profiles):
        raise ValueError("validation model_count disagrees with profiles")
    if run_manifest.get("study_id") != release_manifest.get("study_id"):
        raise ValueError("run and release manifests have different study_id")
    if run_manifest.get("run_id") != validation_report.get("run_id"):
        raise ValueError("run and validation manifests have different run_id")
    actual_result_rows = _csv_data_row_count(data_dir / "decode_results.csv")
    declared_result_rows = {
        int(release_manifest.get("row_count", -1)),
        int(validation_report.get("row_count", -1)),
        int(validation_report.get("checks", {}).get("rows_checked", -1)),
    }
    if declared_result_rows != {actual_result_rows}:
        raise ValueError(
            "frozen decode_results.csv row count disagrees with release "
            "metadata"
        )

    keyed: dict[tuple[str, str], Mapping[str, Any]] = {}
    configs: list[dict[str, Any]] = []
    config_hashes: dict[str, str] = {}
    for profile in profiles:
        key = (
            str(profile.get("model_release_id", "")),
            str(profile.get("deployment_profile_id", "")),
        )
        if not all(key):
            raise ValueError("profile is missing release/profile identity")
        if key in keyed:
            raise ValueError(f"duplicate model profile key: {key}")
        keyed[key] = profile
        path = _resolve_config_path(
            profile, release_dir, checksum_entries
        )
        config = dict(_load_object(path))
        if config.get("schema_version") != 1:
            raise ValueError(f"unsupported config schema in {path}")
        configs.append(config)
        config_hashes[str(path.relative_to(release_dir))] = _sha256(path)
    return (
        configs,
        keyed,
        release_manifest,
        run_manifest,
        validation_report,
        config_hashes,
    )


def _canonical_architecture(config: Mapping[str, Any]) -> dict[str, Any]:
    model = config.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("config.model must be an object")
    groups = model.get("layer_groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError("config.model.layer_groups must be non-empty")

    mixer_counts = Counter()
    kv_counts = Counter()
    access_counts = Counter()
    access_detail_counts = Counter()
    physical_layers = 0
    softmax_layers = 0
    for group_index, group in enumerate(groups):
        if not isinstance(group, Mapping):
            raise ValueError(f"layer_groups[{group_index}] must be an object")
        layers = int(
            _finite_number(
                group.get("layers"),
                f"layer_groups[{group_index}].layers",
                positive=True,
            )
        )
        mixers = group.get("mixers")
        if not isinstance(mixers, list) or not mixers:
            raise ValueError(
                f"layer_groups[{group_index}].mixers must be non-empty"
            )
        kinds = [
            str(mixer.get("kind"))
            for mixer in mixers
            if isinstance(mixer, Mapping)
        ]
        primary = set(kinds).intersection(
            {"softmax_attention", "linear_attention", "recurrent_state", "mamba"}
        )
        if not primary:
            if set(kinds) == {"fixed_cost"}:
                continue
            raise ValueError(
                f"layer_groups[{group_index}] has no canonical token mixer: "
                f"{kinds}"
            )

        if "softmax_attention" in primary:
            if len(primary) != 1:
                raise ValueError(
                    "softmax token-mixer group mixes incompatible primary "
                    f"kinds: {sorted(primary)}"
                )
            category = "softmax"
        elif "mamba" in primary:
            if len(primary) != 1:
                raise ValueError(
                    f"Mamba group mixes incompatible primary kinds: {primary}"
                )
            category = "ssm"
        elif "linear_attention" in primary:
            if not primary.issubset({"linear_attention", "recurrent_state"}):
                raise ValueError(
                    "linear group mixes incompatible primary kinds: "
                    f"{sorted(primary)}"
                )
            category = "linear_recurrent"
            mixer_counts["linear_attention_detail"] += layers
        else:
            category = "linear_recurrent"
            mixer_counts["recurrent_detail"] += layers

        physical_layers += layers
        mixer_counts[category] += layers
        if category == "ssm":
            mixer_counts["mamba_detail"] += layers
        if category != "softmax":
            continue

        softmax_mixers = [
            mixer
            for mixer in mixers
            if isinstance(mixer, Mapping)
            and mixer.get("kind") == "softmax_attention"
        ]
        layouts = {
            str(mixer.get("kv_layout", {}).get("kind"))
            for mixer in softmax_mixers
            if isinstance(mixer.get("kv_layout"), Mapping)
        }
        accesses = {
            str(mixer.get("access", {}).get("kind"))
            for mixer in softmax_mixers
            if isinstance(mixer.get("access"), Mapping)
        }
        if len(layouts) != 1 or next(iter(layouts), "") not in KV_LAYOUTS:
            raise ValueError(
                f"Softmax group has unsupported/mixed KV layouts: {layouts}"
            )
        if len(accesses) != 1:
            raise ValueError(
                f"Softmax group has unsupported/mixed accesses: {accesses}"
            )
        layout = next(iter(layouts))
        access = next(iter(accesses))
        if access not in ACCESS_BROAD:
            raise ValueError(f"unsupported attention access kind {access!r}")
        broad_access = ACCESS_BROAD[access]
        softmax_layers += layers
        kv_counts[layout] += layers
        access_counts[broad_access] += layers
        access_detail_counts[access] += layers

    if physical_layers <= 0:
        raise ValueError("canonical architecture needs positive physical layers")
    if sum(mixer_counts[name] for name in MIXER_BROAD) != physical_layers:
        raise ValueError("canonical mixer layer counts do not close")
    if sum(kv_counts.values()) != softmax_layers:
        raise ValueError("canonical KV-layout layer counts do not close")
    if sum(access_counts.values()) != softmax_layers:
        raise ValueError("canonical access layer counts do not close")

    result: dict[str, Any] = {
        "physical_sequence_layer_count": physical_layers,
        "softmax_layer_count": mixer_counts["softmax"],
        "linear_recurrent_layer_count": mixer_counts["linear_recurrent"],
        "ssm_layer_count": mixer_counts["ssm"],
        "linear_attention_layer_count": mixer_counts[
            "linear_attention_detail"
        ],
        "recurrent_layer_count": mixer_counts["recurrent_detail"],
        "mamba_layer_count": mixer_counts["mamba_detail"],
        "softmax_layer_share": mixer_counts["softmax"] / physical_layers,
        "linear_recurrent_layer_share": (
            mixer_counts["linear_recurrent"] / physical_layers
        ),
        "ssm_layer_share": mixer_counts["ssm"] / physical_layers,
        "state_like_layer_share": (
            mixer_counts["linear_recurrent"] + mixer_counts["ssm"]
        )
        / physical_layers,
        "has_state_like_mixer": int(
            mixer_counts["linear_recurrent"] + mixer_counts["ssm"] > 0
        ),
        "has_explicit_linear_attention": int(
            mixer_counts["linear_attention_detail"] > 0
        ),
        "has_hybrid_token_mixers": int(
            sum(mixer_counts[name] > 0 for name in MIXER_BROAD) > 1
        ),
        "access_detail_layer_counts": _json_cell(
            dict(sorted(access_detail_counts.items()))
        ),
    }
    for layout in KV_LAYOUTS:
        count = kv_counts[layout]
        result[f"kv_{layout}_layer_count"] = count
        result[f"kv_{layout}_share_of_softmax"] = (
            None if softmax_layers == 0 else count / softmax_layers
        )
        result[f"has_{layout}"] = int(count > 0)
    for access in ACCESS_CATEGORIES:
        count = access_counts[access]
        result[f"access_{access}_layer_count"] = count
        result[f"access_{access}_share_of_softmax"] = (
            None if softmax_layers == 0 else count / softmax_layers
        )
        result[f"has_{access}_access"] = int(count > 0)
    return result


def _precision_features(
    profile: Mapping[str, Any],
    config: Mapping[str, Any],
    resident_elements: float,
) -> dict[str, Any]:
    defaults = profile.get("deployment_defaults")
    if not isinstance(defaults, Mapping):
        raise ValueError("profile.deployment_defaults must be an object")
    weight_groups: list[tuple[float, float]] = []
    always_groups = profile.get("always_active_weight_groups")
    routed_groups = profile.get("routed_expert_groups")
    if not isinstance(always_groups, list) or not isinstance(routed_groups, list):
        raise ValueError("profile weight groups must be arrays")
    for group in always_groups:
        if not isinstance(group, Mapping):
            raise ValueError("always-active weight group must be an object")
        elements = _finite_number(
            group.get("parameters"),
            "always-active parameters",
            positive=True,
        )
        bits = _finite_number(
            group.get("weight_bits", defaults.get("weight_bits")),
            "always-active weight_bits",
            positive=True,
        )
        weight_groups.append((elements, bits))
    for group in routed_groups:
        if not isinstance(group, Mapping):
            raise ValueError("routed expert group must be an object")
        elements = (
            _finite_number(group.get("layers"), "routed layers", positive=True)
            * _finite_number(
                group.get("expert_count"), "expert_count", positive=True
            )
            * _finite_number(
                group.get("parameters_per_expert"),
                "parameters_per_expert",
                positive=True,
            )
        )
        bits = _finite_number(
            group.get(
                "weight_bits",
                defaults.get("expert_weight_bits", defaults.get("weight_bits")),
            ),
            "expert weight_bits",
            positive=True,
        )
        weight_groups.append((elements, bits))
    if not weight_groups:
        raise ValueError("profile contains no explicit weight groups")
    for _, bits in weight_groups:
        if bits > 64:
            raise ValueError(f"implausible weight precision: {bits}")
    explicit_elements = sum(elements for elements, _ in weight_groups)
    if explicit_elements > resident_elements * (1.0 + 1e-9):
        raise ValueError("explicit weight groups exceed resident parameters")
    weighted_bits = sum(
        elements * bits for elements, bits in weight_groups
    )

    kv_values: set[float] = set()
    index_values: set[float] = set()
    state_values: set[float] = set()
    kv_components: list[tuple[float, float]] = []
    index_components: list[tuple[float, float]] = []
    state_components: list[tuple[float, float]] = []
    model = config["model"]
    for group in model["layer_groups"]:
        layers = _finite_number(
            group.get("layers"), "precision layer-group layers", positive=True
        )
        for mixer in group["mixers"]:
            kind = mixer["kind"]
            if kind == "softmax_attention":
                layout = mixer["kv_layout"]
                if layout["kind"] == "mla":
                    latent_bits = float(
                        layout.get("latent_bits", defaults["kv_bits"])
                    )
                    rope_bits = float(
                        layout.get("rope_bits", defaults["kv_bits"])
                    )
                    latent_elements = _finite_number(
                        layout.get("latent_dim"),
                        "MLA latent_dim",
                        positive=True,
                    )
                    rope_elements = _finite_number(
                        layout.get("rope_dim"),
                        "MLA rope_dim",
                        positive=True,
                    )
                    kv_values.update((latent_bits, rope_bits))
                    kv_components.extend(
                        (
                            (layers * latent_elements, latent_bits),
                            (layers * rope_elements, rope_bits),
                        )
                    )
                else:
                    key_bits = float(
                        layout.get("key_bits", defaults["kv_bits"])
                    )
                    value_bits = float(
                        layout.get("value_bits", defaults["kv_bits"])
                    )
                    entry_elements = _finite_number(
                        layout.get("kv_heads"),
                        "KV heads",
                        positive=True,
                    ) * _finite_number(
                        layout.get("head_dim"),
                        "KV head_dim",
                        positive=True,
                    )
                    kv_values.update((key_bits, value_bits))
                    kv_components.extend(
                        (
                            (layers * entry_elements, key_bits),
                            (layers * entry_elements, value_bits),
                        )
                    )
                access = mixer["access"]
                if "index_entry_elements" in access:
                    index_bits = float(
                        access.get("index_bits", defaults["index_bits"])
                    )
                    index_elements = _finite_number(
                        access.get("index_entry_elements"),
                        "attention index_entry_elements",
                        positive=True,
                    )
                    compression = _finite_number(
                        access.get("compression_ratio", 1.0),
                        "attention index compression_ratio",
                        positive=True,
                    )
                    index_values.add(index_bits)
                    index_components.append(
                        (layers * index_elements / compression, index_bits)
                    )
            if kind in {"mamba", "linear_attention", "recurrent_state"}:
                state_bits = float(
                    mixer.get("state_bits", defaults["state_bits"])
                )
                if kind == "recurrent_state":
                    state_elements = _finite_number(
                        mixer.get("state_elements"),
                        "recurrent state_elements",
                        positive=True,
                    )
                elif kind == "linear_attention":
                    query_heads = _finite_number(
                        mixer.get("query_heads"),
                        "linear attention query_heads",
                        positive=True,
                    )
                    key_dim = _finite_number(
                        mixer.get("key_dim"),
                        "linear attention key_dim",
                        positive=True,
                    )
                    value_dim = _finite_number(
                        mixer.get("value_dim"),
                        "linear attention value_dim",
                        positive=True,
                    )
                    state_elements = query_heads * key_dim * value_dim
                    if bool(mixer.get("normalizer_state", True)):
                        state_elements += query_heads * key_dim
                else:
                    variant = str(mixer.get("variant"))
                    inner_dim = _finite_number(
                        mixer.get("inner_dim"),
                        "Mamba inner_dim",
                        positive=True,
                    )
                    state_dim = _finite_number(
                        mixer.get("state_dim"),
                        "Mamba state_dim",
                        positive=True,
                    )
                    conv_kernel = _finite_number(
                        mixer.get("conv_kernel"),
                        "Mamba conv_kernel",
                        positive=True,
                    )
                    if variant == "mamba1":
                        recurrence_width = inner_dim
                        conv_channels = inner_dim
                    elif variant == "mamba2":
                        recurrence_width = _finite_number(
                            mixer.get("ssm_dim", inner_dim),
                            "Mamba-2 ssm_dim",
                            positive=True,
                        )
                        groups = _finite_number(
                            mixer.get("groups", 1),
                            "Mamba-2 groups",
                            positive=True,
                        )
                        conv_channels = (
                            recurrence_width + 2.0 * groups * state_dim
                        )
                    else:
                        raise ValueError(
                            f"unsupported Mamba variant for precision: "
                            f"{variant!r}"
                        )
                    state_elements = (
                        recurrence_width * state_dim
                        + conv_channels * conv_kernel
                    )
                state_values.add(state_bits)
                state_components.append(
                    (layers * state_elements, state_bits)
                )
    for label, values in (
        ("KV", kv_values),
        ("Index", index_values),
        ("State", state_values),
    ):
        if any(value <= 0 or value > 64 for value in values):
            raise ValueError(f"invalid {label} precision values: {values}")

    capacity = _finite_number(
        profile["capacity"]["decode_profile_weight_capacity_bytes"],
        "decode_profile_weight_capacity_bytes",
        positive=True,
    )

    def effective_bits(
        components: Sequence[tuple[float, float]],
    ) -> float | None:
        total_elements = sum(elements for elements, _ in components)
        if total_elements <= 0:
            return None
        return (
            sum(elements * bits for elements, bits in components)
            / total_elements
        )

    def component_elements(
        components: Sequence[tuple[float, float]],
    ) -> float:
        return sum(elements for elements, _ in components)

    def component_bytes(
        components: Sequence[tuple[float, float]],
    ) -> float:
        return sum(
            elements * bits / 8.0 for elements, bits in components
        )

    return {
        "explicit_weight_group_count": len(weight_groups),
        "explicit_weight_parameter_elements": explicit_elements,
        "explicit_weight_group_coverage_ratio": (
            explicit_elements / resident_elements
        ),
        "matrix_effective_weight_bits": weighted_bits / explicit_elements,
        "explicit_weight_parameter_share_le8": sum(
            elements for elements, bits in weight_groups if bits <= 8
        )
        / explicit_elements,
        "explicit_weight_parameter_share_le4": sum(
            elements for elements, bits in weight_groups if bits <= 4
        )
        / explicit_elements,
        "resident_effective_storage_bits": 8.0 * capacity / resident_elements,
        "deployment_default_weight_bits": float(defaults["weight_bits"]),
        "deployment_default_expert_weight_bits": float(
            defaults["expert_weight_bits"]
        ),
        "deployment_default_kv_bits": float(defaults["kv_bits"]),
        "deployment_default_index_bits": float(defaults["index_bits"]),
        "deployment_default_state_bits": float(defaults["state_bits"]),
        "kv_bits_used_values": _json_cell(sorted(kv_values)),
        "index_bits_used_values": _json_cell(sorted(index_values)),
        "state_bits_used_values": _json_cell(sorted(state_values)),
        "has_kv_storage": int(bool(kv_components)),
        "kv_effective_bits_used": effective_bits(kv_components),
        "kv_logical_elements_per_history_token": component_elements(
            kv_components
        ),
        "kv_capacity_bytes_per_history_token": component_bytes(
            kv_components
        ),
        "has_index_storage": int(bool(index_components)),
        "index_effective_bits_used": effective_bits(index_components),
        "index_logical_elements_per_history_token": component_elements(
            index_components
        ),
        "index_capacity_bytes_per_history_token": component_bytes(
            index_components
        ),
        "has_state_storage": int(bool(state_components)),
        "state_effective_bits_used": effective_bits(state_components),
        "state_logical_elements_per_request": component_elements(
            state_components
        ),
        "state_capacity_bytes_per_request": component_bytes(
            state_components
        ),
        "has_majority_explicit_weight_le8": int(
            sum(
                elements for elements, bits in weight_groups if bits <= 8
            )
            / explicit_elements
            >= 0.5
        ),
        "has_majority_explicit_weight_le4": int(
            sum(
                elements for elements, bits in weight_groups if bits <= 4
            )
            / explicit_elements
            >= 0.5
        ),
    }


def _moe_features(
    profile: Mapping[str, Any],
    physical_layers: int,
) -> dict[str, Any]:
    groups = profile.get("routed_expert_groups")
    if not isinstance(groups, list):
        raise ValueError("routed_expert_groups must be an array")
    if len(groups) > 1:
        raise ValueError(
            "multiple routed expert groups need canonical layer identities"
        )
    if not groups:
        return {
            "has_moe": 0,
            "moe_layer_count": 0,
            "moe_layer_share": 0.0,
            "moe_layer_share_given_moe": None,
            "moe_expert_count": None,
            "moe_selected_per_token": None,
            "moe_routing_density": None,
            "moe_parameters_per_expert": None,
            "moe_resident_expert_parameter_elements": None,
            "moe_active_expert_parameter_elements_per_token": None,
            "moe_routing_mode": "",
        }
    group = groups[0]
    layers = int(
        _finite_number(group.get("layers"), "MoE layers", positive=True)
    )
    experts = int(
        _finite_number(
            group.get("expert_count"), "MoE expert_count", positive=True
        )
    )
    selected = int(
        _finite_number(
            group.get("selected_per_token"),
            "MoE selected_per_token",
            positive=True,
        )
    )
    parameters_per_expert = _finite_number(
        group.get("parameters_per_expert"),
        "MoE parameters_per_expert",
        positive=True,
    )
    if layers > physical_layers:
        raise ValueError(
            f"MoE layers {layers} exceed physical layers {physical_layers}"
        )
    if selected > experts:
        raise ValueError("MoE selected_per_token exceeds expert_count")
    return {
        "has_moe": 1,
        "moe_layer_count": layers,
        "moe_layer_share": layers / physical_layers,
        "moe_layer_share_given_moe": layers / physical_layers,
        "moe_expert_count": experts,
        "moe_selected_per_token": selected,
        "moe_routing_density": selected / experts,
        "moe_parameters_per_expert": parameters_per_expert,
        "moe_resident_expert_parameter_elements": (
            layers * experts * parameters_per_expert
        ),
        "moe_active_expert_parameter_elements_per_token": (
            layers * selected * parameters_per_expert
        ),
        "moe_routing_mode": str(group.get("routing_mode", "")),
    }


def _technology_observations(
    profiles: Mapping[tuple[str, str], Mapping[str, Any]],
    configs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if len(profiles) != len(configs):
        raise ValueError("profile/config count mismatch")
    output: list[dict[str, Any]] = []
    for profile, config in zip(profiles.values(), configs):
        release_id = str(profile["model_release_id"])
        release_date = date.fromisoformat(str(profile["release_date"]))
        if int(profile["year"]) != release_date.year:
            raise ValueError(f"year/release_date mismatch for {release_id}")
        if release_date > YTD_CUTOFF:
            raise ValueError(
                f"release {release_id} is after study cutoff {YTD_CUTOFF}"
            )
        resident = _finite_number(
            profile["parameters"]["decode_resident_parameter_elements"],
            f"{release_id}.resident parameters",
            positive=True,
        )
        active = _finite_number(
            profile["parameters"][
                "active_matrix_parameter_elements_per_token"
            ],
            f"{release_id}.active parameters",
            positive=True,
        )
        if active > resident * (1.0 + 1e-12):
            raise ValueError(f"active parameters exceed resident for {release_id}")
        advertised = int(
            _finite_number(
                profile["context"][
                    "advertised_max_context_tokens_at_release"
                ],
                f"{release_id}.advertised context",
                positive=True,
            )
        )
        optional_context: dict[str, int | None] = {}
        for field in (
            "trained_max_context_tokens",
            "evaluated_max_context_tokens",
            "deployed_max_context_tokens",
        ):
            raw = profile["context"].get(field)
            optional_context[field] = (
                None
                if raw is None
                else int(
                    _finite_number(
                        raw, f"{release_id}.{field}", positive=True
                    )
                )
            )
        config_hash = str(profile["config_sha256"])
        architecture = _canonical_architecture(config)
        moe = _moe_features(
            profile, int(architecture["physical_sequence_layer_count"])
        )
        precision = _precision_features(profile, config, resident)
        evaluated = optional_context["evaluated_max_context_tokens"]
        deployed = optional_context["deployed_max_context_tokens"]
        trained = optional_context["trained_max_context_tokens"]
        item: dict[str, Any] = {
            "year": release_date.year,
            "is_ytd_year": int(release_date.year == YTD_YEAR),
            "release_date": release_date.isoformat(),
            "release_decimal_year": _decimal_year(release_date),
            "organization": str(profile["organization"]),
            "model_release_id": release_id,
            "deployment_profile_id": str(profile["deployment_profile_id"]),
            "short_model_name": _short_model_name(profile),
            "sample_roles": _json_cell(profile["sample_roles"]),
            "config_path": str(profile["config_path"]),
            "config_sha256": config_hash,
            "claim_population": "curated_representative_release_sample",
            "decode_resident_parameter_elements": int(resident),
            "active_matrix_parameter_elements_per_token": int(active),
            "active_parameter_ratio": active / resident,
            "sparsity_multiplier": resident / active,
            "sparsity_gap_bits": math.log2(resident / active),
            "advertised_max_context_tokens_at_release": advertised,
            "trained_max_context_tokens": optional_context[
                "trained_max_context_tokens"
            ],
            "evaluated_max_context_tokens": evaluated,
            "deployed_max_context_tokens": deployed,
            "evaluated_to_advertised_context_ratio": (
                None if evaluated is None else evaluated / advertised
            ),
            "trained_to_advertised_context_ratio": (
                None if trained is None else trained / advertised
            ),
            "deployed_to_advertised_context_ratio": (
                None if deployed is None else deployed / advertised
            ),
            "effective_context_observation_count": len(
                profile["context"]["effective_context_observations"]
            ),
            **architecture,
            **moe,
            **precision,
            "has_mla": architecture["has_mla"],
            "has_gqa": architecture["has_gqa"],
            "has_bounded_local_access": architecture[
                "has_bounded_local_access"
            ],
            "has_sparse_topk_access": architecture[
                "has_sparse_topk_access"
            ],
        }
        output.append(item)
    output.sort(
        key=lambda row: (
            str(row["release_date"]),
            str(row["model_release_id"]),
        )
    )
    keys = {
        (row["model_release_id"], row["deployment_profile_id"])
        for row in output
    }
    if len(keys) != len(output):
        raise ValueError("technology observations contain duplicate keys")
    return output


def _median(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("median requires at least one value")
    return float(statistics.median(values))


def _direction(slope: float | None, *, tolerance: float = 1e-10) -> str:
    if slope is None or not math.isfinite(slope):
        return "not_available"
    if slope > tolerance:
        return "increasing"
    if slope < -tolerance:
        return "decreasing"
    return "flat"


def _formula_for_kind(kind: str) -> str:
    if kind == "positive_log":
        return "X(t)=2^(alpha+beta*(t-t0))"
    if kind == "real_linear":
        return "X(t)=alpha+beta*(t-t0)"
    if kind in {"fraction", "binary"}:
        return "p(t)=sigmoid(alpha+beta*(t-t0))"
    raise ValueError(f"unknown metric kind {kind!r}")


def _sigmoid(value: float) -> float:
    if value >= 0:
        inverse = math.exp(-value)
        return 1.0 / (1.0 + inverse)
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def _logit(value: float) -> float:
    clipped = min(max(value, 1e-12), 1.0 - 1e-12)
    return math.log(clipped / (1.0 - clipped))


def _softplus(value: float) -> float:
    if value > 35:
        return value
    if value < -35:
        return math.exp(value)
    return math.log1p(math.exp(value))


def _theil_sen_fit(
    points: Sequence[tuple[float, float]],
    *,
    reference_year: float = REFERENCE_YEAR,
) -> dict[str, Any]:
    if len(points) < 2:
        return {
            "status": "insufficient",
            "alpha": None,
            "beta_per_year": None,
            "reason": "fewer than two observations",
        }
    transformed: list[tuple[float, float]] = []
    for time_value, raw_value in points:
        if not math.isfinite(time_value):
            raise ValueError("Theil-Sen time must be finite")
        if not math.isfinite(raw_value) or raw_value <= 0:
            raise ValueError("Theil-Sen values must be finite and positive")
        transformed.append((time_value, math.log2(raw_value)))
    slopes = [
        (right_value - left_value) / (right_time - left_time)
        for left_index, (left_time, left_value) in enumerate(transformed)
        for right_time, right_value in transformed[left_index + 1 :]
        if right_time != left_time
    ]
    if not slopes:
        return {
            "status": "insufficient",
            "alpha": None,
            "beta_per_year": None,
            "reason": "all observations have the same release time",
        }
    beta = _median(slopes)
    alpha = _median(
        [
            value - beta * (time_value - reference_year)
            for time_value, value in transformed
        ]
    )
    return {
        "status": "fit",
        "alpha": alpha,
        "beta_per_year": beta,
        "reason": "",
    }


def _constant_log_fit(
    points: Sequence[tuple[float, float]],
    *,
    reference_year: float = REFERENCE_YEAR,
) -> dict[str, Any]:
    del reference_year
    if not points:
        return {
            "status": "insufficient",
            "alpha": None,
            "beta_per_year": None,
            "reason": "no observations",
        }
    values: list[float] = []
    for _, raw in points:
        if not math.isfinite(raw) or raw <= 0:
            raise ValueError("constant log values must be finite and positive")
        values.append(math.log2(raw))
    return {
        "status": "fit",
        "alpha": _median(values),
        "beta_per_year": 0.0,
        "reason": "",
    }


def _theil_sen_linear_fit(
    points: Sequence[tuple[float, float]],
    *,
    reference_year: float = REFERENCE_YEAR,
) -> dict[str, Any]:
    if len(points) < 2:
        return {
            "status": "insufficient",
            "alpha": None,
            "beta_per_year": None,
            "reason": "fewer than two observations",
        }
    for time_value, raw_value in points:
        if not math.isfinite(time_value) or not math.isfinite(raw_value):
            raise ValueError("linear Theil-Sen inputs must be finite")
    slopes = [
        (right_value - left_value) / (right_time - left_time)
        for left_index, (left_time, left_value) in enumerate(points)
        for right_time, right_value in points[left_index + 1 :]
        if right_time != left_time
    ]
    if not slopes:
        return {
            "status": "insufficient",
            "alpha": None,
            "beta_per_year": None,
            "reason": "all observations have the same release time",
        }
    beta = _median(slopes)
    alpha = _median(
        [
            value - beta * (time_value - reference_year)
            for time_value, value in points
        ]
    )
    return {
        "status": "fit",
        "alpha": alpha,
        "beta_per_year": beta,
        "reason": "",
    }


def _constant_linear_fit(
    points: Sequence[tuple[float, float]],
    *,
    reference_year: float = REFERENCE_YEAR,
) -> dict[str, Any]:
    del reference_year
    if not points:
        return {
            "status": "insufficient",
            "alpha": None,
            "beta_per_year": None,
            "reason": "no observations",
        }
    values = [value for _, value in points]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("constant linear inputs must be finite")
    return {
        "status": "fit",
        "alpha": _median(values),
        "beta_per_year": 0.0,
        "reason": "",
    }


def _probability_constant_fit(
    points: Sequence[tuple[float, float]],
    *,
    reference_year: float = REFERENCE_YEAR,
) -> dict[str, Any]:
    del reference_year
    if not points:
        return {
            "status": "insufficient",
            "alpha": None,
            "beta_per_year": None,
            "reason": "no observations",
        }
    values: list[float] = []
    for _, raw in points:
        if not math.isfinite(raw) or raw < 0 or raw > 1:
            raise ValueError("probability values must be finite within [0, 1]")
        values.append(raw)
    probability = (sum(values) + 0.5) / (len(values) + 1.0)
    return {
        "status": "fit",
        "alpha": _logit(probability),
        "beta_per_year": 0.0,
        "reason": "",
    }


def _penalized_logistic_objective(
    points: Sequence[tuple[float, float]],
    alpha: float,
    beta_scaled: float,
    mean_time: float,
    time_scale: float,
    l2: float,
) -> float:
    value = -0.5 * l2 * beta_scaled * beta_scaled
    for time_value, observed in points:
        scaled = (time_value - mean_time) / time_scale
        linear = alpha + beta_scaled * scaled
        value += observed * linear - _softplus(linear)
    return value


def _ridge_logistic_fit(
    points: Sequence[tuple[float, float]],
    *,
    reference_year: float = REFERENCE_YEAR,
    l2: float = LOGISTIC_L2,
) -> dict[str, Any]:
    if len(points) < 2:
        return {
            "status": "insufficient",
            "alpha": None,
            "beta_per_year": None,
            "reason": "fewer than two observations",
        }
    for time_value, observed in points:
        if not math.isfinite(time_value):
            raise ValueError("logistic time must be finite")
        if not math.isfinite(observed) or observed < 0 or observed > 1:
            raise ValueError("logistic response must be within [0, 1]")
    observed_values = [observed for _, observed in points]
    if max(observed_values) - min(observed_values) <= 1e-12:
        return {
            "status": "insufficient",
            "alpha": None,
            "beta_per_year": None,
            "reason": "response has no variation; time slope is unidentified",
        }
    mean_time = sum(time_value for time_value, _ in points) / len(points)
    time_scale = math.sqrt(
        sum((time_value - mean_time) ** 2 for time_value, _ in points)
        / len(points)
    )
    if time_scale <= 1e-12:
        return {
            "status": "insufficient",
            "alpha": None,
            "beta_per_year": None,
            "reason": "all observations have the same release time",
        }
    smoothed_mean = (
        sum(observed for _, observed in points) + 0.5
    ) / (len(points) + 1.0)
    alpha = _logit(smoothed_mean)
    beta_scaled = 0.0
    objective = _penalized_logistic_objective(
        points, alpha, beta_scaled, mean_time, time_scale, l2
    )
    converged = False
    for _ in range(100):
        gradient_alpha = 0.0
        gradient_beta = -l2 * beta_scaled
        info_aa = 0.0
        info_ab = 0.0
        info_bb = l2
        for time_value, observed in points:
            scaled = (time_value - mean_time) / time_scale
            probability = _sigmoid(alpha + beta_scaled * scaled)
            residual = observed - probability
            weight = max(probability * (1.0 - probability), 1e-12)
            gradient_alpha += residual
            gradient_beta += residual * scaled
            info_aa += weight
            info_ab += weight * scaled
            info_bb += weight * scaled * scaled
        determinant = info_aa * info_bb - info_ab * info_ab
        if determinant <= 1e-18:
            return {
                "status": "failed",
                "alpha": None,
                "beta_per_year": None,
                "reason": "singular logistic information matrix",
            }
        delta_alpha = (
            gradient_alpha * info_bb - gradient_beta * info_ab
        ) / determinant
        delta_beta = (
            gradient_beta * info_aa - gradient_alpha * info_ab
        ) / determinant
        step = 1.0
        accepted = False
        while step >= 2 ** -20:
            candidate_alpha = alpha + step * delta_alpha
            candidate_beta = beta_scaled + step * delta_beta
            candidate_objective = _penalized_logistic_objective(
                points,
                candidate_alpha,
                candidate_beta,
                mean_time,
                time_scale,
                l2,
            )
            if candidate_objective >= objective - 1e-12:
                alpha = candidate_alpha
                beta_scaled = candidate_beta
                objective = candidate_objective
                accepted = True
                break
            step *= 0.5
        if not accepted:
            return {
                "status": "failed",
                "alpha": None,
                "beta_per_year": None,
                "reason": "logistic line search failed",
            }
        if max(abs(step * delta_alpha), abs(step * delta_beta)) < 1e-10:
            converged = True
            break
    if not converged:
        return {
            "status": "failed",
            "alpha": None,
            "beta_per_year": None,
            "reason": "logistic optimizer did not converge",
        }
    beta_per_year = beta_scaled / time_scale
    alpha_at_reference = alpha + beta_per_year * (
        reference_year - mean_time
    )
    return {
        "status": "fit",
        "alpha": alpha_at_reference,
        "beta_per_year": beta_per_year,
        "reason": "",
    }


def _predict_fit(
    fit: Mapping[str, Any],
    kind: str,
    time_value: float,
    *,
    reference_year: float = REFERENCE_YEAR,
) -> float | None:
    if fit.get("status") != "fit":
        return None
    alpha = fit.get("alpha")
    beta = fit.get("beta_per_year")
    if alpha is None or beta is None:
        return None
    linear = float(alpha) + float(beta) * (time_value - reference_year)
    if kind == "positive_log":
        try:
            value = 2.0**linear
        except OverflowError:
            return None
    elif kind == "real_linear":
        value = linear
    elif kind in {"fraction", "binary"}:
        value = _sigmoid(linear)
    else:
        raise ValueError(f"unknown fit kind {kind!r}")
    return value if math.isfinite(value) else None


def _observed_points(
    observations: Sequence[Mapping[str, Any]],
    spec: MetricSpec,
) -> list[tuple[float, float, Mapping[str, Any]]]:
    output: list[tuple[float, float, Mapping[str, Any]]] = []
    for row in observations:
        raw = row.get(spec.field)
        if raw is None or raw == "":
            continue
        value = _finite_number(raw, spec.metric_id)
        if spec.kind == "positive_log" and value <= 0:
            raise ValueError(f"{spec.metric_id} must be positive")
        if spec.kind in {"fraction", "binary"} and not 0 <= value <= 1:
            raise ValueError(f"{spec.metric_id} must be in [0, 1]")
        if spec.kind == "binary" and value not in (0.0, 1.0):
            raise ValueError(f"{spec.metric_id} must be binary")
        output.append((float(row["release_decimal_year"]), value, row))
    return output


def _fit_candidate(
    points: Sequence[tuple[float, float]],
    kind: str,
    candidate_id: str,
) -> dict[str, Any]:
    if kind == "positive_log":
        if candidate_id == "constant":
            return _constant_log_fit(points)
        if candidate_id == "trend":
            return _theil_sen_fit(points)
    elif kind == "real_linear":
        if candidate_id == "constant":
            return _constant_linear_fit(points)
        if candidate_id == "trend":
            return _theil_sen_linear_fit(points)
    elif kind in {"fraction", "binary"}:
        if candidate_id == "constant":
            return _probability_constant_fit(points)
        if candidate_id == "trend":
            return _ridge_logistic_fit(points)
    raise ValueError(
        f"unsupported candidate {candidate_id!r} for kind {kind!r}"
    )


def _score_predictions(
    observed: Sequence[float],
    predicted: Sequence[float],
    kind: str,
) -> tuple[str, float, float | None]:
    if len(observed) != len(predicted) or not observed:
        raise ValueError("scoring requires paired non-empty observations")
    if kind == "positive_log":
        errors = [
            abs(math.log2(actual) - math.log2(estimate))
            for actual, estimate in zip(observed, predicted)
        ]
        mae = sum(errors) / len(errors)
        return "mae_log2", mae, 2.0**mae
    if kind == "real_linear":
        mae = sum(
            abs(actual - estimate)
            for actual, estimate in zip(observed, predicted)
        ) / len(observed)
        return "mae", mae, None
    brier = sum(
        (actual - estimate) ** 2
        for actual, estimate in zip(observed, predicted)
    ) / len(observed)
    log_loss = sum(
        -actual * math.log(min(max(estimate, 1e-12), 1.0 - 1e-12))
        - (1.0 - actual)
        * math.log(
            min(max(1.0 - estimate, 1e-12), 1.0 - 1e-12)
        )
        for actual, estimate in zip(observed, predicted)
    ) / len(observed)
    return "brier", brier, log_loss


def _backtest_rows_for_metric(
    observations: Sequence[Mapping[str, Any]],
    spec: MetricSpec,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for test_year in COMPLETE_BACKTEST_YEARS:
        train_rows = [
            row for row in observations if int(row["year"]) < test_year
        ]
        test_rows = [
            row for row in observations if int(row["year"]) == test_year
        ]
        train = _observed_points(train_rows, spec)
        test = _observed_points(test_rows, spec)
        if len(train) < 2 or not test:
            continue
        for candidate_id in ("constant", "trend"):
            fit = _fit_candidate(
                [(time_value, value) for time_value, value, _ in train],
                spec.kind,
                candidate_id,
            )
            predictions = [
                _predict_fit(fit, spec.kind, time_value)
                for time_value, _, _ in test
            ]
            if any(value is None for value in predictions):
                output.append(
                    {
                        "metric_id": spec.metric_id,
                        "candidate_id": candidate_id,
                        "test_year": test_year,
                        "test_year_is_ytd": 0,
                        "train_max_year": max(
                            int(row["year"]) for _, _, row in train
                        ),
                        "train_observation_count": len(train),
                        "test_observation_count": len(test),
                        "score_name": (
                            "mae_log2"
                            if spec.kind == "positive_log"
                            else (
                                "mae"
                                if spec.kind == "real_linear"
                                else "brier"
                            )
                        ),
                        "score": None,
                        "secondary_score": None,
                        "fit_status": fit["status"],
                        "fit_reason": fit["reason"],
                    }
                )
                continue
            score_name, score, secondary = _score_predictions(
                [value for _, value, _ in test],
                [float(value) for value in predictions if value is not None],
                spec.kind,
            )
            output.append(
                {
                    "metric_id": spec.metric_id,
                    "candidate_id": candidate_id,
                    "test_year": test_year,
                    "test_year_is_ytd": 0,
                    "train_max_year": max(
                        int(row["year"]) for _, _, row in train
                    ),
                    "train_observation_count": len(train),
                    "test_observation_count": len(test),
                    "score_name": score_name,
                    "score": score,
                    "secondary_score": secondary,
                    "fit_status": fit["status"],
                    "fit_reason": fit["reason"],
                }
            )
    return output


def _candidate_minimum_status(
    spec: MetricSpec,
    points: Sequence[tuple[float, float, Mapping[str, Any]]],
) -> tuple[bool, str]:
    years = {int(row["year"]) for _, _, row in points}
    if len(points) < 8 or len(years) < 4:
        return False, "requires at least 8 observations across 4 years"
    times = [time_value for time_value, _, _ in points]
    if max(times) - min(times) < 2.0:
        return False, "observation span is shorter than two years"
    values = [value for _, value, _ in points]
    if spec.kind == "binary":
        positives = sum(value == 1.0 for value in values)
        negatives = sum(value == 0.0 for value in values)
        positive_years = {
            int(row["year"])
            for _, value, row in points
            if value == 1.0
        }
        if positives < 3 or negatives < 3:
            return False, "binary trend requires at least 3 positives and 3 negatives"
        if len(positive_years) < 2:
            return False, "binary positives occur in only one year"
    if spec.kind == "fraction":
        if sum(value > 0 for value in values) < 3:
            return False, "fractional trend has fewer than 3 positive observations"
        if sum(value < 1 for value in values) < 3:
            return False, "fractional trend has fewer than 3 observations below one"
    return True, ""


def _sign_consistency(reference: float, values: Sequence[float]) -> float | None:
    meaningful = [value for value in values if math.isfinite(value)]
    reference_direction = _direction(reference)
    if not meaningful or reference_direction in {"flat", "not_available"}:
        return None
    return sum(
        _direction(value) == reference_direction for value in meaningful
    ) / len(meaningful)


def _sensitivity_rows_for_metric(
    observations: Sequence[Mapping[str, Any]],
    spec: MetricSpec,
    full_fit: Mapping[str, Any],
) -> list[dict[str, Any]]:
    reference_slope = full_fit.get("beta_per_year")
    output: list[dict[str, Any]] = []
    eligible_rows = [
        row for _, _, row in _observed_points(observations, spec)
    ]
    dimensions = (
        ("release", "model_release_id"),
        ("year", "year"),
        ("organization", "organization"),
    )
    for kind, field in dimensions:
        values = sorted({str(row[field]) for row in eligible_rows})
        for omitted in values:
            subset = [
                row for row in eligible_rows if str(row[field]) != omitted
            ]
            points = _observed_points(subset, spec)
            fit = _fit_candidate(
                [(time_value, value) for time_value, value, _ in points],
                spec.kind,
                "trend",
            )
            slope = fit.get("beta_per_year")
            consistent = (
                None
                if slope is None or reference_slope is None
                else int(
                    _direction(float(slope))
                    == _direction(float(reference_slope))
                )
            )
            output.append(
                {
                    "metric_id": spec.metric_id,
                    "sensitivity_kind": f"leave_one_{kind}_out",
                    "omitted_value": omitted,
                    "remaining_observation_count": len(points),
                    "fit_status": fit["status"],
                    "beta_per_year": slope,
                    "direction": _direction(
                        None if slope is None else float(slope)
                    ),
                    "matches_full_direction": consistent,
                    "fit_reason": fit["reason"],
                }
            )
    return output


def _fit_all_trends(
    observations: Sequence[Mapping[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    candidate_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    backtests: list[dict[str, Any]] = []
    sensitivities: list[dict[str, Any]] = []
    for spec in METRIC_SPECS:
        points_with_rows = _observed_points(observations, spec)
        points = [
            (time_value, value)
            for time_value, value, _ in points_with_rows
        ]
        metric_backtests = _backtest_rows_for_metric(observations, spec)
        backtests.extend(metric_backtests)
        fits: dict[str, dict[str, Any]] = {}
        in_sample_scores: dict[str, tuple[str, float, float | None] | None] = {}
        cv_scores: dict[str, float | None] = {}
        for candidate_id in ("constant", "trend"):
            fit = _fit_candidate(points, spec.kind, candidate_id)
            fits[candidate_id] = fit
            predictions = [
                _predict_fit(fit, spec.kind, time_value)
                for time_value, _ in points
            ]
            if points and all(value is not None for value in predictions):
                in_sample_scores[candidate_id] = _score_predictions(
                    [value for _, value in points],
                    [float(value) for value in predictions if value is not None],
                    spec.kind,
                )
            else:
                in_sample_scores[candidate_id] = None
            fold_scores = [
                float(row["score"])
                for row in metric_backtests
                if row["candidate_id"] == candidate_id
                and row["score"] is not None
            ]
            cv_scores[candidate_id] = (
                None
                if len(fold_scores) < 2
                else sum(fold_scores) / len(fold_scores)
            )

        minimum_ok, minimum_reason = _candidate_minimum_status(
            spec, points_with_rows
        )
        trend_cv = cv_scores["trend"]
        constant_cv = cv_scores["constant"]
        improvement = (
            None
            if trend_cv is None
            or constant_cv is None
            or constant_cv <= 0
            else (constant_cv - trend_cv) / constant_cv
        )
        select_trend = (
            minimum_ok
            and fits["trend"]["status"] == "fit"
            and improvement is not None
            and improvement >= 0.05
        )
        selected_id = "trend" if select_trend else "constant"
        selected_fit = fits[selected_id]
        trend_sensitivity = _sensitivity_rows_for_metric(
            observations, spec, fits["trend"]
        )
        sensitivities.extend(trend_sensitivity)
        reference_slope = fits["trend"].get("beta_per_year")
        year_sensitivity = [
            row
            for row in trend_sensitivity
            if row["sensitivity_kind"] == "leave_one_year_out"
        ]
        organization_sensitivity = [
            row
            for row in trend_sensitivity
            if row["sensitivity_kind"] == "leave_one_organization_out"
        ]

        def sensitivity_summary(
            rows: Sequence[Mapping[str, Any]],
        ) -> tuple[int, int, int, float | None, float | None, float | None]:
            total = len(rows)
            valid = sum(row["beta_per_year"] is not None for row in rows)
            matches = sum(row["matches_full_direction"] == 1 for row in rows)
            if reference_slope is None or total == 0:
                return total, valid, matches, None, None, None
            return (
                total,
                valid,
                matches,
                valid / total,
                None if valid == 0 else matches / valid,
                matches / total,
            )

        (
            year_total,
            year_valid,
            year_matches,
            year_coverage,
            year_match_among_valid,
            year_consistency,
        ) = sensitivity_summary(year_sensitivity)
        (
            organization_total,
            organization_valid,
            organization_matches,
            organization_coverage,
            organization_match_among_valid,
            organization_consistency,
        ) = sensitivity_summary(organization_sensitivity)
        if not minimum_ok:
            evidence = "insufficient"
            selection_reason = minimum_reason
        elif trend_cv is None or constant_cv is None:
            evidence = "insufficient"
            selection_reason = (
                "fewer than two valid complete-year rolling backtest folds"
            )
        elif not select_trend:
            evidence = "unstable"
            selection_reason = (
                "trend did not improve two complete-year rolling folds "
                "by at least 5%; selected constant baseline"
            )
        elif (
            (year_coverage or 0.0) >= 0.8
            and (year_consistency or 0.0) >= 0.8
            and (organization_coverage or 0.0) >= 0.8
            and (organization_consistency or 0.0) >= 0.8
        ):
            evidence = "emerging"
            selection_reason = (
                "trend improved rolling backtest and direction was stable; "
                "curated sample prevents established/adoption claim"
            )
        else:
            evidence = "unstable"
            selection_reason = (
                "trend improved rolling backtest but leave-group direction "
                "was not at least 80% stable"
            )

        for candidate_id in ("constant", "trend"):
            fit = fits[candidate_id]
            score = in_sample_scores[candidate_id]
            beta = fit.get("beta_per_year")
            annual_multiplier = (
                None
                if beta is None or spec.kind != "positive_log"
                else 2.0 ** float(beta)
            )
            doubling_time = (
                None
                if beta is None
                or spec.kind != "positive_log"
                or abs(float(beta)) < 1e-12
                else 1.0 / float(beta)
            )
            candidate_rows.append(
                {
                    "metric_id": spec.metric_id,
                    "field": spec.field,
                    "kind": spec.kind,
                    "candidate_id": candidate_id,
                    "formula": _formula_for_kind(spec.kind),
                    "reference_year_t0": REFERENCE_YEAR,
                    "fit_status": fit["status"],
                    "fit_reason": fit["reason"],
                    "observation_count": len(points),
                    "distinct_year_count": len(
                        {int(row["year"]) for _, _, row in points_with_rows}
                    ),
                    "alpha": fit.get("alpha"),
                    "beta_per_year": beta,
                    "direction": _direction(
                        None if beta is None else float(beta)
                    ),
                    "annual_multiplier": annual_multiplier,
                    "doubling_time_years_signed": doubling_time,
                    "in_sample_score_name": None if score is None else score[0],
                    "in_sample_score": None if score is None else score[1],
                    "in_sample_secondary_score": (
                        None if score is None else score[2]
                    ),
                    "rolling_complete_year_score": cv_scores[candidate_id],
                    "rolling_complete_year_fold_count": sum(
                        row["candidate_id"] == candidate_id
                        and row["score"] is not None
                        for row in metric_backtests
                    ),
                    "selected": int(candidate_id == selected_id),
                    "claim_scope": spec.claim_scope,
                    "composition_group": spec.composition_group or "",
                    "projection_role": spec.projection_role,
                    "selection_scope": (
                        "direct marginal diagnostic; formal projection is "
                        "an exact derived identity"
                        if spec.projection_role == "derived_identity"
                        else (
                            "raw component candidate; formal selection is "
                            "performed on the normalized composition group"
                            if spec.composition_group
                            else "formal independent marginal candidate"
                        )
                    ),
                }
            )

        selected_alpha = selected_fit.get("alpha")
        selected_beta = selected_fit.get("beta_per_year")
        selected_cv = cv_scores[selected_id]
        selected_fold_count = sum(
            row["candidate_id"] == selected_id
            and row["score"] is not None
            for row in metric_backtests
        )
        skill_status = (
            "not_evaluable"
            if trend_cv is None or constant_cv is None
            else (
                "positive_skill"
                if improvement is not None and improvement >= 0.05
                else "no_positive_skill"
            )
        )
        training_times = [time_value for time_value, _ in points]
        selected_rows.append(
            {
                "metric_id": spec.metric_id,
                "field": spec.field,
                "kind": spec.kind,
                "unit": spec.unit,
                "projection_role": spec.projection_role,
                "joint_configuration_eligible": int(
                    spec.projection_role != "marginal_diagnostic"
                ),
                "selected_candidate_id": selected_id,
                "formula": _formula_for_kind(spec.kind),
                "reference_year_t0": REFERENCE_YEAR,
                "alpha": selected_alpha,
                "beta_per_year": selected_beta,
                "direction": _direction(
                    None
                    if selected_beta is None
                    else float(selected_beta)
                ),
                "observation_count": len(points),
                "training_start_decimal_year": (
                    None if not training_times else min(training_times)
                ),
                "training_end_decimal_year": (
                    None if not training_times else max(training_times)
                ),
                "recommended_extrapolation_end_decimal_year": (
                    None
                    if not training_times
                    else max(training_times) + 2.0
                ),
                "constant_cv_score": constant_cv,
                "trend_cv_score": trend_cv,
                "trend_relative_cv_improvement": improvement,
                "skill_status": skill_status,
                "selected_rolling_complete_year_score": selected_cv,
                "selected_rolling_complete_year_fold_count": (
                    selected_fold_count
                ),
                "selected_typical_factor_error": (
                    None
                    if selected_cv is None or spec.kind != "positive_log"
                    else 2.0 ** float(selected_cv)
                ),
                "selected_rolling_rmse_percentage_points": (
                    None
                    if selected_cv is None
                    or spec.kind not in {"fraction", "binary"}
                    else 100.0 * math.sqrt(float(selected_cv))
                ),
                "accuracy_maturity": (
                    "provisional_two_folds"
                    if selected_fold_count >= 2
                    else "not_evaluable"
                ),
                "evidence_dimension": (
                    "directional trajectory and relative skill; not "
                    "forecast-accuracy certification"
                ),
                "leave_year_direction_consistency": year_consistency,
                "leave_organization_direction_consistency": (
                    organization_consistency
                ),
                "leave_year_total_omissions": year_total,
                "leave_year_valid_fit_count": year_valid,
                "leave_year_direction_match_count": year_matches,
                "leave_year_fit_coverage": year_coverage,
                "leave_year_match_rate_among_valid": (
                    year_match_among_valid
                ),
                "leave_organization_total_omissions": organization_total,
                "leave_organization_valid_fit_count": organization_valid,
                "leave_organization_direction_match_count": (
                    organization_matches
                ),
                "leave_organization_fit_coverage": organization_coverage,
                "leave_organization_match_rate_among_valid": (
                    organization_match_among_valid
                ),
                "evidence_grade": evidence,
                "selection_reason": selection_reason,
                "claim_scope": spec.claim_scope,
                "composition_group": spec.composition_group or "",
                "composition_fit_scope": (
                    "pending group-level applied-function selection"
                    if spec.composition_group
                    else "not_applicable"
                ),
                "composition_group_evidence_grade": "",
                "composition_group_score_name": "",
                "composition_group_constant_cv_score": None,
                "composition_group_trend_cv_score": None,
                "composition_group_trend_relative_cv_improvement": None,
                "composition_applied_delta_training_window": None,
                "composition_dominant_gain_metric_id": "",
                "composition_dominant_loss_metric_id": "",
                "composition_in_sample_mean_component_brier": None,
                "composition_in_sample_cross_entropy": None,
                "composition_in_sample_mae_percentage_points": None,
                "composition_in_sample_rmse_percentage_points": None,
                "cross_metric_consistency": (
                    "independent marginal function; P9B joint constraints "
                    "not applied"
                ),
                "curated_sample_prevalence_not_industry_adoption": int(
                    spec.kind == "binary"
                    or "selected deployment profile" in spec.claim_scope
                ),
                "derived_source_metric_ids": "",
                "direct_marginal_candidate_id": selected_id,
                "direct_marginal_formula": _formula_for_kind(spec.kind),
                "direct_marginal_alpha": selected_alpha,
                "direct_marginal_beta_per_year": selected_beta,
                "direct_marginal_evidence_grade": evidence,
            }
        )
    return candidate_rows, selected_rows, backtests, sensitivities


def _composition_eligible_rows(
    observations: Sequence[Mapping[str, Any]],
    specs: Sequence[MetricSpec],
) -> list[Mapping[str, Any]]:
    return [
        row
        for row in observations
        if all(
            row.get(spec.field) is not None
            and row.get(spec.field) != ""
            for spec in specs
        )
    ]


def _fit_composition_group(
    rows: Sequence[Mapping[str, Any]],
    specs: Sequence[MetricSpec],
    candidate_id: str,
) -> dict[str, Any]:
    fits: dict[str, dict[str, Any]] = {}
    cold_start_fallbacks: list[str] = []
    for spec in specs:
        points = [
            (float(row["release_decimal_year"]), float(row[spec.field]))
            for row in rows
        ]
        fit = _fit_candidate(points, spec.kind, candidate_id)
        if (
            candidate_id == "trend"
            and fit["status"] == "insufficient"
            and "no variation" in str(fit["reason"])
        ):
            fit = _fit_candidate(points, spec.kind, "constant")
            cold_start_fallbacks.append(spec.metric_id)
        if fit["status"] != "fit":
            return {
                "status": fit["status"],
                "reason": (
                    f"{spec.metric_id}: {fit['reason']}"
                ),
                "fits": {},
                "cold_start_fallbacks": cold_start_fallbacks,
            }
        fits[spec.metric_id] = fit
    return {
        "status": "fit",
        "reason": "",
        "fits": fits,
        "cold_start_fallbacks": cold_start_fallbacks,
    }


def _composition_prediction(
    fit_result: Mapping[str, Any],
    specs: Sequence[MetricSpec],
    time_value: float,
) -> dict[str, float] | None:
    if fit_result.get("status") != "fit":
        return None
    raw: dict[str, float] = {}
    fits = fit_result["fits"]
    for spec in specs:
        value = _predict_fit(fits[spec.metric_id], spec.kind, time_value)
        if value is None:
            return None
        raw[spec.metric_id] = float(value)
    total = sum(raw.values())
    if total <= 0:
        return None
    return {
        metric_id: value / total for metric_id, value in raw.items()
    }


def _score_composition_predictions(
    rows: Sequence[Mapping[str, Any]],
    specs: Sequence[MetricSpec],
    fit_result: Mapping[str, Any],
) -> tuple[float, float, float, float] | None:
    squared_errors: list[float] = []
    absolute_errors: list[float] = []
    cross_entropies: list[float] = []
    for row in rows:
        predicted = _composition_prediction(
            fit_result, specs, float(row["release_decimal_year"])
        )
        if predicted is None:
            return None
        row_cross_entropy = 0.0
        for spec in specs:
            observed = float(row[spec.field])
            estimate = min(
                max(predicted[spec.metric_id], 1e-12), 1.0
            )
            squared_errors.append((observed - estimate) ** 2)
            absolute_errors.append(abs(observed - estimate))
            if observed > 0:
                row_cross_entropy -= observed * math.log(estimate)
        cross_entropies.append(row_cross_entropy)
    if not squared_errors:
        return None
    mean_squared = sum(squared_errors) / len(squared_errors)
    return (
        mean_squared,
        sum(cross_entropies) / len(cross_entropies),
        100.0 * sum(absolute_errors) / len(absolute_errors),
        100.0 * math.sqrt(mean_squared),
    )


def _composition_curve_direction(
    fit_result: Mapping[str, Any],
    specs: Sequence[MetricSpec],
    metric_id: str,
    start: float,
    end: float,
) -> tuple[str, float | None]:
    values: list[float] = []
    for index in range(25):
        time_value = start + (end - start) * index / 24.0
        predicted = _composition_prediction(
            fit_result, specs, time_value
        )
        if predicted is None:
            return "not_available", None
        values.append(predicted[metric_id])
    deltas = [
        right - left for left, right in zip(values, values[1:])
    ]
    has_positive = any(delta > 1e-10 for delta in deltas)
    has_negative = any(delta < -1e-10 for delta in deltas)
    if has_positive and has_negative:
        direction = "mixed"
    elif has_positive:
        direction = "increasing"
    elif has_negative:
        direction = "decreasing"
    else:
        direction = "flat"
    return direction, values[-1] - values[0]


def _composition_backtest_rows(
    observations: Sequence[Mapping[str, Any]],
    group: str,
    specs: Sequence[MetricSpec],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    eligible = _composition_eligible_rows(observations, specs)
    for test_year in COMPLETE_BACKTEST_YEARS:
        train = [
            row for row in eligible if int(row["year"]) < test_year
        ]
        test = [
            row for row in eligible if int(row["year"]) == test_year
        ]
        if len(train) < 2 or not test:
            continue
        for candidate_id in ("constant", "trend"):
            fit = _fit_composition_group(train, specs, candidate_id)
            score = _score_composition_predictions(test, specs, fit)
            fallbacks = list(fit.get("cold_start_fallbacks", []))
            output.append(
                {
                    "metric_id": f"composition_group.{group}",
                    "candidate_id": candidate_id,
                    "test_year": test_year,
                    "test_year_is_ytd": 0,
                    "train_max_year": max(int(row["year"]) for row in train),
                    "train_observation_count": len(train),
                    "test_observation_count": len(test),
                    "score_name": "mean_component_brier",
                    "score": None if score is None else score[0],
                    "secondary_score": None if score is None else score[1],
                    "fit_status": fit["status"],
                    "fit_reason": (
                        fit["reason"]
                        if not fallbacks
                        else "cold_start_component_fallback="
                        + "|".join(fallbacks)
                    ),
                }
            )
    return output


def _apply_composition_group_selection(
    observations: Sequence[Mapping[str, Any]],
    candidate_rows: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
    backtests: list[dict[str, Any]],
    sensitivities: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    selected_by_id = {
        str(row["metric_id"]): row for row in selected_rows
    }
    candidate_by_key = {
        (str(row["metric_id"]), str(row["candidate_id"])): row
        for row in candidate_rows
    }
    summaries: list[dict[str, Any]] = []
    for group, metric_ids in COMPOSITION_GROUPS.items():
        specs = [METRIC_BY_ID[metric_id] for metric_id in metric_ids]
        eligible = _composition_eligible_rows(observations, specs)
        group_backtests = _composition_backtest_rows(
            observations, group, specs
        )
        backtests.extend(group_backtests)
        scores: dict[str, float | None] = {}
        fold_counts: dict[str, int] = {}
        for candidate_id in ("constant", "trend"):
            fold_scores = [
                float(row["score"])
                for row in group_backtests
                if row["candidate_id"] == candidate_id
                and row["score"] is not None
            ]
            fold_counts[candidate_id] = len(fold_scores)
            scores[candidate_id] = (
                None
                if len(fold_scores) < 2
                else sum(fold_scores) / len(fold_scores)
            )
        constant_cv = scores["constant"]
        trend_cv = scores["trend"]
        improvement = (
            None
            if constant_cv is None
            or trend_cv is None
            or constant_cv <= 0
            else (constant_cv - trend_cv) / constant_cv
        )
        years = {int(row["year"]) for row in eligible}
        minimum_ok = len(eligible) >= 8 and len(years) >= 4
        selected_id = (
            "trend"
            if minimum_ok
            and improvement is not None
            and improvement >= 0.05
            else "constant"
        )
        full_fit = _fit_composition_group(
            eligible, specs, selected_id
        )
        if full_fit["status"] != "fit":
            selected_id = "constant"
            full_fit = _fit_composition_group(
                eligible, specs, selected_id
            )
        if full_fit["status"] != "fit":
            raise ValueError(
                f"composition group {group} has no usable full fit: "
                f"{full_fit['reason']}"
            )
        start = min(float(row["release_decimal_year"]) for row in eligible)
        end = max(float(row["release_decimal_year"]) for row in eligible)
        full_directions: dict[str, str] = {}
        full_deltas: dict[str, float] = {}
        for metric_id in metric_ids:
            direction, delta = _composition_curve_direction(
                full_fit, specs, metric_id, start, end
            )
            full_directions[metric_id] = direction
            full_deltas[metric_id] = 0.0 if delta is None else delta
        dominant_gain = max(metric_ids, key=lambda value: full_deltas[value])
        dominant_loss = min(metric_ids, key=lambda value: full_deltas[value])

        composition_sensitivity: list[dict[str, Any]] = []
        for dimension, field in (
            ("release", "model_release_id"),
            ("year", "year"),
            ("organization", "organization"),
        ):
            for omitted in sorted({str(row[field]) for row in eligible}):
                subset = [
                    row for row in eligible if str(row[field]) != omitted
                ]
                fit = _fit_composition_group(
                    subset, specs, selected_id
                )
                subset_start = min(
                    (float(row["release_decimal_year"]) for row in subset),
                    default=start,
                )
                subset_end = max(
                    (float(row["release_decimal_year"]) for row in subset),
                    default=end,
                )
                for spec in specs:
                    if fit["status"] == "fit":
                        direction, _ = _composition_curve_direction(
                            fit,
                            specs,
                            spec.metric_id,
                            subset_start,
                            subset_end,
                        )
                        raw_beta = fit["fits"][spec.metric_id].get(
                            "beta_per_year"
                        )
                    else:
                        direction = "not_available"
                        raw_beta = None
                    reference_direction = full_directions[spec.metric_id]
                    matches = (
                        None
                        if direction == "not_available"
                        or reference_direction in {
                            "flat",
                            "mixed",
                            "not_available",
                        }
                        else int(direction == reference_direction)
                    )
                    composition_sensitivity.append(
                        {
                            "metric_id": spec.metric_id,
                            "sensitivity_kind": (
                                f"leave_one_{dimension}_out_"
                                "composition_applied"
                            ),
                            "omitted_value": omitted,
                            "remaining_observation_count": len(subset),
                            "fit_status": fit["status"],
                            "beta_per_year": raw_beta,
                            "direction": direction,
                            "matches_full_direction": matches,
                            "fit_reason": fit["reason"],
                        }
                    )
        sensitivities.extend(composition_sensitivity)

        def applied_sensitivity_summary(
            metric_id: str, dimension: str
        ) -> tuple[int, int, int, float | None, float | None]:
            rows = [
                row
                for row in composition_sensitivity
                if row["metric_id"] == metric_id
                and row["sensitivity_kind"]
                == f"leave_one_{dimension}_out_composition_applied"
            ]
            total = len(rows)
            valid = sum(
                row["direction"] != "not_available" for row in rows
            )
            matches = sum(
                row["matches_full_direction"] == 1 for row in rows
            )
            return (
                total,
                valid,
                matches,
                None if total == 0 else valid / total,
                None if total == 0 else matches / total,
            )

        dominant_stable = selected_id == "trend"
        for metric_id in {dominant_gain, dominant_loss}:
            for dimension in ("year", "organization"):
                _, _, _, coverage, conservative = (
                    applied_sensitivity_summary(metric_id, dimension)
                )
                dominant_stable = (
                    dominant_stable
                    and (coverage or 0.0) >= 0.8
                    and (conservative or 0.0) >= 0.8
                )
        if not minimum_ok or trend_cv is None or constant_cv is None:
            group_evidence = "insufficient"
            group_reason = (
                "fewer than two valid group-level complete-year backtests"
            )
        elif selected_id == "constant":
            group_evidence = "unstable"
            group_reason = (
                "group-normalized trend did not improve the constant "
                "composition by at least 5%"
            )
        elif dominant_stable:
            group_evidence = "emerging"
            group_reason = (
                "group-normalized trend improved backtests and dominant "
                "gain/loss directions passed leave-group sensitivity"
            )
        else:
            group_evidence = "unstable"
            group_reason = (
                "group trend had positive skill but dominant applied "
                "directions were not at least 80% stable"
            )
        in_sample = _score_composition_predictions(
            eligible, specs, full_fit
        )
        for spec in specs:
            row = selected_by_id[spec.metric_id]
            fit = full_fit["fits"][spec.metric_id]
            (
                year_total,
                year_valid,
                year_matches,
                year_coverage,
                year_conservative,
            ) = applied_sensitivity_summary(spec.metric_id, "year")
            (
                organization_total,
                organization_valid,
                organization_matches,
                organization_coverage,
                organization_conservative,
            ) = applied_sensitivity_summary(
                spec.metric_id, "organization"
            )
            component_evidence = group_evidence
            if (
                group_evidence == "emerging"
                and (
                    (year_coverage or 0.0) < 0.8
                    or (year_conservative or 0.0) < 0.8
                    or (organization_coverage or 0.0) < 0.8
                    or (organization_conservative or 0.0) < 0.8
                )
            ):
                component_evidence = "unstable"
            row.update(
                {
                    "selected_candidate_id": selected_id,
                    "formula": (
                        "p_k(t)=sigmoid(alpha_k+beta_k*(t-t0))/"
                        "sum_j(sigmoid(alpha_j+beta_j*(t-t0)))"
                    ),
                    "alpha": fit.get("alpha"),
                    "beta_per_year": fit.get("beta_per_year"),
                    "direction": full_directions[spec.metric_id],
                    "constant_cv_score": constant_cv,
                    "trend_cv_score": trend_cv,
                    "trend_relative_cv_improvement": improvement,
                    "skill_status": (
                        "not_evaluable"
                        if improvement is None
                        else (
                            "positive_skill"
                            if improvement >= 0.05
                            else "no_positive_skill"
                        )
                    ),
                    "selected_rolling_complete_year_score": scores[
                        selected_id
                    ],
                    "selected_rolling_complete_year_fold_count": (
                        fold_counts[selected_id]
                    ),
                    "selected_typical_factor_error": None,
                    "selected_rolling_rmse_percentage_points": (
                        None
                        if scores[selected_id] is None
                        else 100.0 * math.sqrt(
                            float(scores[selected_id])
                        )
                    ),
                    "accuracy_maturity": (
                        "provisional_two_folds"
                        if fold_counts[selected_id] >= 2
                        else "not_evaluable"
                    ),
                    "leave_year_direction_consistency": year_conservative,
                    "leave_organization_direction_consistency": (
                        organization_conservative
                    ),
                    "leave_year_total_omissions": year_total,
                    "leave_year_valid_fit_count": year_valid,
                    "leave_year_direction_match_count": year_matches,
                    "leave_year_fit_coverage": year_coverage,
                    "leave_year_match_rate_among_valid": (
                        None
                        if year_valid == 0
                        else year_matches / year_valid
                    ),
                    "leave_organization_total_omissions": (
                        organization_total
                    ),
                    "leave_organization_valid_fit_count": (
                        organization_valid
                    ),
                    "leave_organization_direction_match_count": (
                        organization_matches
                    ),
                    "leave_organization_fit_coverage": (
                        organization_coverage
                    ),
                    "leave_organization_match_rate_among_valid": (
                        None
                        if organization_valid == 0
                        else organization_matches / organization_valid
                    ),
                    "evidence_grade": component_evidence,
                    "selection_reason": group_reason,
                    "composition_fit_scope": (
                        "group-level applied-function selection and "
                        "backtest; cold-start components use a documented "
                        "constant fallback within a trend candidate"
                    ),
                    "cross_metric_consistency": (
                        "normalized within composition axis; applied "
                        "component predictions sum to one"
                    ),
                    "composition_group_evidence_grade": group_evidence,
                    "composition_group_score_name": (
                        "mean_component_brier"
                    ),
                    "composition_group_constant_cv_score": constant_cv,
                    "composition_group_trend_cv_score": trend_cv,
                    "composition_group_trend_relative_cv_improvement": (
                        improvement
                    ),
                    "composition_applied_delta_training_window": (
                        full_deltas[spec.metric_id]
                    ),
                    "composition_dominant_gain_metric_id": dominant_gain,
                    "composition_dominant_loss_metric_id": dominant_loss,
                    "composition_in_sample_mean_component_brier": (
                        None if in_sample is None else in_sample[0]
                    ),
                    "composition_in_sample_cross_entropy": (
                        None if in_sample is None else in_sample[1]
                    ),
                    "composition_in_sample_mae_percentage_points": (
                        None if in_sample is None else in_sample[2]
                    ),
                    "composition_in_sample_rmse_percentage_points": (
                        None if in_sample is None else in_sample[3]
                    ),
                }
            )
            for candidate_id in ("constant", "trend"):
                candidate_by_key[
                    (spec.metric_id, candidate_id)
                ]["selected"] = int(candidate_id == selected_id)
        summaries.append(
            {
                "composition_group": group,
                "metric_ids": _json_cell(metric_ids),
                "eligible_observation_count": len(eligible),
                "selected_candidate_id": selected_id,
                "constant_cv_score": constant_cv,
                "trend_cv_score": trend_cv,
                "trend_relative_cv_improvement": improvement,
                "selected_cv_rmse_percentage_points": (
                    None
                    if scores[selected_id] is None
                    else 100.0 * math.sqrt(float(scores[selected_id]))
                ),
                "valid_constant_fold_count": fold_counts["constant"],
                "valid_trend_fold_count": fold_counts["trend"],
                "evidence_grade": group_evidence,
                "selection_reason": group_reason,
                "dominant_gain_metric_id": dominant_gain,
                "dominant_loss_metric_id": dominant_loss,
                "applied_formula": (
                    "p_k(t)=sigmoid(alpha_k+beta_k*(t-t0))/"
                    "sum_j(sigmoid(alpha_j+beta_j*(t-t0)))"
                ),
                "score_name": "mean_component_brier",
                "accuracy_maturity": (
                    "provisional_two_folds"
                    if fold_counts[selected_id] >= 2
                    else "not_evaluable"
                ),
            }
        )
    return summaries


def _selected_prediction_map(
    selected_functions: Sequence[Mapping[str, Any]],
    time_value: float,
) -> dict[str, tuple[float | None, float | None]]:
    raw: dict[str, float | None] = {}
    by_id = {str(row["metric_id"]): row for row in selected_functions}
    for metric_id, row in by_id.items():
        if metric_id in DERIVED_IDENTITIES:
            raw[metric_id] = None
            continue
        raw[metric_id] = _predict_fit(
            {
                "status": (
                    "fit"
                    if row.get("alpha") is not None
                    and row.get("beta_per_year") is not None
                    else "insufficient"
                ),
                "alpha": row.get("alpha"),
                "beta_per_year": row.get("beta_per_year"),
            },
            str(row["kind"]),
            time_value,
        )
    normalized = dict(raw)
    for metric_ids in COMPOSITION_GROUPS.values():
        values = [raw.get(metric_id) for metric_id in metric_ids]
        if any(value is None for value in values):
            continue
        total = sum(float(value) for value in values if value is not None)
        if total <= 0:
            continue
        for metric_id, value in zip(metric_ids, values):
            normalized[metric_id] = (
                None if value is None else float(value) / total
            )

    def derive(metric_id: str) -> float | None:
        if metric_id == "parameters.active_elements":
            resident = normalized.get("parameters.resident_elements")
            ratio = normalized.get("parameters.active_ratio")
            return (
                None
                if resident is None or ratio is None
                else float(resident) * float(ratio)
            )
        if metric_id == "parameters.sparsity_multiplier":
            ratio = normalized.get("parameters.active_ratio")
            return (
                None
                if ratio is None or float(ratio) <= 0
                else 1.0 / float(ratio)
            )
        if metric_id == "parameters.sparsity_gap_bits":
            ratio = normalized.get("parameters.active_ratio")
            return (
                None
                if ratio is None or float(ratio) <= 0
                else -math.log2(float(ratio))
            )
        if metric_id == "moe.unconditional_layer_share":
            presence = normalized.get("moe.presence")
            conditional = normalized.get("moe.layer_share_given_moe")
            return (
                None
                if presence is None or conditional is None
                else float(presence) * float(conditional)
            )
        raise ValueError(f"unknown derived identity {metric_id!r}")

    for metric_id in DERIVED_IDENTITIES:
        value = derive(metric_id)
        raw[metric_id] = value
        normalized[metric_id] = value
    return {
        metric_id: (raw.get(metric_id), normalized.get(metric_id))
        for metric_id in by_id
    }


def _apply_derived_identity_rows(
    selected_functions: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
) -> None:
    by_id = {
        str(row["metric_id"]): row for row in selected_functions
    }
    evidence_rank = {
        "insufficient": 0,
        "unstable": 1,
        "emerging": 2,
        "established": 3,
    }
    for metric_id, metadata in DERIVED_IDENTITIES.items():
        row = by_id[metric_id]
        source_ids = tuple(metadata["source_metric_ids"])
        source_rows = [by_id[source_id] for source_id in source_ids]
        evidence = min(
            (str(source["evidence_grade"]) for source in source_rows),
            key=lambda value: evidence_rank[value],
        )
        row.update(
            {
                "projection_role": "derived_identity",
                "joint_configuration_eligible": 1,
                "selected_candidate_id": "derived_identity",
                "formula": str(metadata["formula"]),
                "alpha": None,
                "beta_per_year": None,
                "evidence_grade": evidence,
                "selection_reason": (
                    "derived from selected source functions to enforce an "
                    "exact algebraic identity"
                ),
                "skill_status": "derived_from_sources",
                "selected_rolling_complete_year_score": None,
                "selected_rolling_complete_year_fold_count": 0,
                "selected_typical_factor_error": None,
                "selected_rolling_rmse_percentage_points": None,
                "accuracy_maturity": "inherited_from_sources",
                "derived_source_metric_ids": _json_cell(source_ids),
                "cross_metric_consistency": (
                    "exact within its declared P9A identity basis"
                ),
            }
        )
        for candidate in candidate_rows:
            if candidate["metric_id"] == metric_id:
                candidate["selected"] = 0
    for metric_id in DERIVED_IDENTITIES:
        row = by_id[metric_id]
        start = row.get("training_start_decimal_year")
        end = row.get("training_end_decimal_year")
        if start is None or end is None:
            row["direction"] = "not_available"
            continue
        start_value = _selected_prediction_map(
            selected_functions, float(start)
        )[metric_id][1]
        end_value = _selected_prediction_map(
            selected_functions, float(end)
        )[metric_id][1]
        row["direction"] = (
            "not_available"
            if start_value is None or end_value is None
            else _direction(float(end_value) - float(start_value))
        )


def _fitted_observation_rows(
    observations: Sequence[Mapping[str, Any]],
    selected_functions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in observations:
        time_value = float(row["release_decimal_year"])
        predictions = _selected_prediction_map(
            selected_functions, time_value
        )
        for spec in METRIC_SPECS:
            observed = row.get(spec.field)
            if observed is None or observed == "":
                continue
            raw_prediction, fitted = predictions[spec.metric_id]
            residual = None
            if fitted is not None:
                if spec.kind == "positive_log":
                    residual = math.log2(float(observed)) - math.log2(
                        float(fitted)
                    )
                else:
                    residual = float(observed) - float(fitted)
            output.append(
                {
                    "metric_id": spec.metric_id,
                    "model_release_id": row["model_release_id"],
                    "deployment_profile_id": row["deployment_profile_id"],
                    "release_date": row["release_date"],
                    "release_decimal_year": time_value,
                    "year": row["year"],
                    "is_ytd_year": row["is_ytd_year"],
                    "observed_value": observed,
                    "raw_function_value": raw_prediction,
                    "fitted_value": fitted,
                    "composition_normalized": int(
                        bool(spec.composition_group)
                    ),
                    "residual": residual,
                    "residual_scale": (
                        "log2_observed_minus_fitted"
                        if spec.kind == "positive_log"
                        else "observed_minus_fitted"
                    ),
                }
            )
    return output


def _projection_rows(
    observations: Sequence[Mapping[str, Any]],
    selected_functions: Sequence[Mapping[str, Any]],
    through_year: int,
) -> list[dict[str, Any]]:
    if through_year < YTD_YEAR:
        raise ValueError("--projection-through-year must be at least 2026")
    start_year = min(int(row["year"]) for row in observations)
    last_observed = max(
        float(row["release_decimal_year"]) for row in observations
    )
    recommended_end = last_observed + 2.0
    selected_by_id = {
        str(row["metric_id"]): row for row in selected_functions
    }
    output: list[dict[str, Any]] = []
    for year in range(start_year, through_year + 1):
        time_value = float(year)
        predictions = _selected_prediction_map(
            selected_functions, time_value
        )
        if time_value <= last_observed:
            scope = "historical_fit"
        elif time_value <= recommended_end:
            scope = "conditional_extrapolation"
        else:
            scope = "speculative"
        for spec in METRIC_SPECS:
            function = selected_by_id[spec.metric_id]
            raw, normalized = predictions[spec.metric_id]
            output.append(
                {
                    "year": year,
                    "decimal_year_t": time_value,
                    "projection_scope": scope,
                    "metric_id": spec.metric_id,
                    "kind": spec.kind,
                    "unit": spec.unit,
                    "projection_role": function["projection_role"],
                    "joint_configuration_eligible": function[
                        "joint_configuration_eligible"
                    ],
                    "selected_candidate_id": function[
                        "selected_candidate_id"
                    ],
                    "raw_function_value": raw,
                    "predicted_value": normalized,
                    "composition_normalized": int(
                        bool(spec.composition_group)
                    ),
                    "evidence_grade": function["evidence_grade"],
                    "claim_scope": function["claim_scope"],
                    "joint_projection_status": (
                        "exact derived identity within its declared basis"
                        if function["projection_role"]
                        == "derived_identity"
                        else (
                            "normalized within one composition axis; "
                            "cross-axis constraints not applied"
                            if function["projection_role"]
                            == "composition_component"
                            else (
                                "marginal diagnostic; excluded from direct "
                                "joint-configuration assembly"
                                if function["projection_role"]
                                == "marginal_diagnostic"
                                else (
                                    "independent marginal projection; not "
                                    "a coherent future model configuration"
                                )
                            )
                        )
                    ),
                }
            )
    return output


def _annual_summary_rows(
    observations: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_year: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in observations:
        by_year[int(row["year"])].append(row)
    output: list[dict[str, Any]] = []
    for year, rows in sorted(by_year.items()):
        for spec in METRIC_SPECS:
            values = [
                float(row[spec.field])
                for row in rows
                if row.get(spec.field) is not None
                and row.get(spec.field) != ""
            ]
            if not values:
                continue
            output.append(
                {
                    "year": year,
                    "is_ytd_year": int(year == YTD_YEAR),
                    "year_sample_count": len(rows),
                    "year_organization_count": len(
                        {str(row["organization"]) for row in rows}
                    ),
                    "metric_id": spec.metric_id,
                    "kind": spec.kind,
                    "non_null_observation_count": len(values),
                    "positive_count": (
                        sum(value > 0 for value in values)
                        if spec.kind in {"binary", "fraction"}
                        else None
                    ),
                    "sample_mean": sum(values) / len(values),
                    "sample_median": _median(values),
                    "sample_min": min(values),
                    "sample_max": max(values),
                    "sample_prevalence": (
                        sum(values) / len(values)
                        if spec.kind == "binary"
                        else None
                    ),
                    "claim_scope": spec.claim_scope,
                }
            )
    return output


def _technology_milestone_rows(
    observations: Sequence[Mapping[str, Any]],
    selected_functions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    selected_by_id = {
        str(row["metric_id"]): row for row in selected_functions
    }
    output: list[dict[str, Any]] = []
    for metric_id in BINARY_TECHNOLOGIES:
        spec = METRIC_BY_ID[metric_id]
        presence_rows = sorted(
            (
                row
                for row in observations
                if float(row[spec.field]) == 1.0
            ),
            key=lambda row: (
                str(row["release_date"]),
                str(row["model_release_id"]),
            ),
        )
        first = presence_rows[0] if presence_rows else None
        first_org = None if first is None else str(first["organization"])
        second_org = next(
            (
                row
                for row in presence_rows
                if str(row["organization"]) != first_org
            ),
            None,
        )
        function = selected_by_id[metric_id]
        beta = function.get("beta_per_year")
        alpha = function.get("alpha")
        identifiable = bool(
            function["selected_candidate_id"] == "trend"
            and function["evidence_grade"] == "emerging"
            and beta is not None
            and alpha is not None
            and abs(float(beta)) > 1e-10
        )
        transition_values: dict[str, float | None] = {}
        for label, probability in (
            ("t10_decimal_year", 0.1),
            ("t50_decimal_year", 0.5),
            ("t90_decimal_year", 0.9),
        ):
            transition_values[label] = (
                REFERENCE_YEAR
                + (_logit(probability) - float(alpha)) / float(beta)
                if identifiable
                else None
            )
        if identifiable:
            training_start = float(
                function["training_start_decimal_year"]
            )
            training_end = float(function["training_end_decimal_year"])
            recommended_end = float(
                function["recommended_extrapolation_end_decimal_year"]
            )
            t50 = transition_values["t50_decimal_year"]
            if (
                t50 is None
                or t50 < training_start - 2.0
                or t50 > recommended_end
            ):
                identifiable = False
                transition_values = {
                    key: None for key in transition_values
                }
        else:
            training_end = float(
                function["training_end_decimal_year"]
                or max(
                    float(row["release_decimal_year"])
                    for row in observations
                )
            )
            recommended_end = float(
                function["recommended_extrapolation_end_decimal_year"]
                or training_end + 2.0
            )

        def transition_scope(value: float | None) -> str:
            if value is None:
                return "not_identified"
            if value <= training_end:
                return "historical_fit"
            if value <= recommended_end:
                return "conditional_extrapolation"
            return "speculative"

        output.append(
            {
                "metric_id": metric_id,
                "field": spec.field,
                "presence_count": len(presence_rows),
                "sample_count": len(observations),
                "first_seen_release_date": (
                    None if first is None else first["release_date"]
                ),
                "first_seen_model_release_id": (
                    None if first is None else first["model_release_id"]
                ),
                "first_seen_organization": first_org,
                "second_independent_organization_release_date": (
                    None if second_org is None else second_org["release_date"]
                ),
                "second_independent_organization": (
                    None
                    if second_org is None
                    else second_org["organization"]
                ),
                "transition_window_identifiable": int(identifiable),
                **transition_values,
                "t10_scope": transition_scope(
                    transition_values["t10_decimal_year"]
                ),
                "t50_scope": transition_scope(
                    transition_values["t50_decimal_year"]
                ),
                "t90_scope": transition_scope(
                    transition_values["t90_decimal_year"]
                ),
                "transition_window_fully_within_recommended_horizon": int(
                    identifiable
                    and transition_values["t10_decimal_year"] is not None
                    and transition_values["t90_decimal_year"] is not None
                    and float(transition_values["t10_decimal_year"])
                    >= float(
                        function["training_start_decimal_year"]
                    )
                    and float(transition_values["t90_decimal_year"])
                    <= recommended_end
                ),
                "evidence_grade": function["evidence_grade"],
                "claim_scope": function["claim_scope"],
            }
        )
    return output


def _cooccurrence_rows(
    observations: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for left_index, left_id in enumerate(BINARY_TECHNOLOGIES):
        left = METRIC_BY_ID[left_id]
        for right_id in BINARY_TECHNOLOGIES[left_index + 1 :]:
            right = METRIC_BY_ID[right_id]
            n11 = n10 = n01 = n00 = 0
            for row in observations:
                left_value = int(row[left.field])
                right_value = int(row[right.field])
                if (left_value, right_value) == (1, 1):
                    n11 += 1
                elif (left_value, right_value) == (1, 0):
                    n10 += 1
                elif (left_value, right_value) == (0, 1):
                    n01 += 1
                else:
                    n00 += 1
            total = n11 + n10 + n01 + n00
            p_right_given_left = (n11 + 0.5) / (n11 + n10 + 1.0)
            p_right_given_not_left = (n01 + 0.5) / (
                n01 + n00 + 1.0
            )
            p_right = (n11 + n01 + 0.5) / (total + 1.0)
            union = n11 + n10 + n01
            output.append(
                {
                    "technology_a": left_id,
                    "technology_b": right_id,
                    "sample_count": total,
                    "n11_both": n11,
                    "n10_a_only": n10,
                    "n01_b_only": n01,
                    "n00_neither": n00,
                    "p_b_given_a_jeffreys": p_right_given_left,
                    "p_b_given_not_a_jeffreys": p_right_given_not_left,
                    "risk_difference": (
                        p_right_given_left - p_right_given_not_left
                    ),
                    "lift": p_right_given_left / p_right,
                    "jaccard": None if union == 0 else n11 / union,
                    "claim_scope": (
                        "curated-sample cooccurrence; descriptive association "
                        "only, not causal"
                    ),
                }
            )
    return output


def _quality_summary(
    observations: Sequence[Mapping[str, Any]],
    validation_report: Mapping[str, Any],
    config_hashes: Mapping[str, str],
    selected_functions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    year_counts = Counter(int(row["year"]) for row in observations)
    return {
        "status": "pass",
        "claim_population": "curated_representative_release_sample",
        "curated_sample_prevalence_not_industry_adoption": True,
        "release_observation_count": len(observations),
        "distinct_organization_count": len(
            {str(row["organization"]) for row in observations}
        ),
        "year_counts": {
            str(year): count for year, count in sorted(year_counts.items())
        },
        "ytd_year": YTD_YEAR,
        "ytd_cutoff": YTD_CUTOFF.isoformat(),
        "config_hashes_verified": len(config_hashes),
        "missing_context_counts": {
            field: sum(row[field] is None for row in observations)
            for field in (
                "trained_max_context_tokens",
                "evaluated_max_context_tokens",
                "deployed_max_context_tokens",
            )
        },
        "effective_context_observation_model_count": sum(
            int(row["effective_context_observation_count"]) > 0
            for row in observations
        ),
        "technology_presence_counts": {
            "moe": sum(int(row["has_moe"]) for row in observations),
            "mla": sum(int(row["has_mla"]) for row in observations),
            "gqa": sum(int(row["has_gqa"]) for row in observations),
            "state_like": sum(
                int(row["has_state_like_mixer"]) for row in observations
            ),
            "explicit_linear_attention": sum(
                int(row["has_explicit_linear_attention"])
                for row in observations
            ),
            "bounded_local_access": sum(
                int(row["has_bounded_local_access"])
                for row in observations
            ),
            "sparse_topk_access": sum(
                int(row["has_sparse_topk_access"])
                for row in observations
            ),
            "majority_weight_le8": sum(
                int(row["has_majority_explicit_weight_le8"])
                for row in observations
            ),
            "majority_weight_le4": sum(
                int(row["has_majority_explicit_weight_le4"])
                for row in observations
            ),
        },
        "minimum_explicit_weight_group_coverage_ratio": min(
            float(row["explicit_weight_group_coverage_ratio"])
            for row in observations
        ),
        "conditional_precision_observation_counts": {
            "kv": sum(
                row["kv_effective_bits_used"] is not None
                for row in observations
            ),
            "index": sum(
                row["index_effective_bits_used"] is not None
                for row in observations
            ),
            "state": sum(
                row["state_effective_bits_used"] is not None
                for row in observations
            ),
        },
        "function_evidence_grade_counts": dict(
            sorted(
                Counter(
                    str(row["evidence_grade"])
                    for row in selected_functions
                ).items()
            )
        ),
        "function_evidence_grade_counts_by_projection_role": {
            role: dict(
                sorted(
                    Counter(
                        str(row["evidence_grade"])
                        for row in selected_functions
                        if row["projection_role"] == role
                    ).items()
                )
            )
            for role in sorted(
                {
                    str(row["projection_role"])
                    for row in selected_functions
                }
            )
        },
        "selected_trend_function_count": sum(
            row["selected_candidate_id"] == "trend"
            for row in selected_functions
        ),
        "selected_constant_function_count": sum(
            row["selected_candidate_id"] == "constant"
            for row in selected_functions
        ),
        "derived_identity_function_count": sum(
            row["selected_candidate_id"] == "derived_identity"
            for row in selected_functions
        ),
        "selected_independent_trend_function_count": sum(
            row["selected_candidate_id"] == "trend"
            and not row["composition_group"]
            for row in selected_functions
        ),
        "selected_independent_marginal_trend_count": sum(
            row["selected_candidate_id"] == "trend"
            and row["projection_role"] == "independent_marginal"
            for row in selected_functions
        ),
        "selected_marginal_diagnostic_trend_count": sum(
            row["selected_candidate_id"] == "trend"
            and row["projection_role"] == "marginal_diagnostic"
            for row in selected_functions
        ),
        "selected_composition_trend_group_count": len(
            {
                row["composition_group"]
                for row in selected_functions
                if row["composition_group"]
                and row["selected_candidate_id"] == "trend"
            }
        ),
        "frozen_validation": dict(validation_report),
        "warnings": [
            "The 20 releases are curated and do not define an industry census.",
            "2026 is right-censored YTD and is excluded from complete-year backtests.",
            "Deployment-profile precision cannot be interpreted as native low-bit adoption.",
            "Capability/quality control and token/deployment shares are not available.",
            "Exact parameter and two-part MoE identities are derived from declared bases; other technology axes remain marginal until P9B.",
            "Composition axes are selected and scored after normalization; cross-axis joint configuration constraints remain deferred to P9B.",
            "Direction evidence and relative skill do not certify forecast accuracy; only two complete-year folds are available.",
        ],
    }


def _write_csv(rows: Iterable[Mapping[str, Any]], path: Path) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(materialized[0])
    for row in materialized:
        if list(row) != fieldnames:
            raise ValueError(f"CSV rows have inconsistent fields: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(materialized)


def _write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _normalize_svg(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    path.write_text(
        "\n".join(line.rstrip() for line in text.splitlines()) + "\n",
        encoding="utf-8",
    )


TABLE_ARTIFACT_NAMES = (
    "analysis_manifest.json",
    "quality_summary.json",
    "technology_observations.csv",
    "annual_sample_summary.csv",
    "trend_candidates.csv",
    "selected_trend_functions.csv",
    "selected_trend_functions.json",
    "composition_group_summary.csv",
    "trend_backtests.csv",
    "trend_sensitivity.csv",
    "fitted_observations.csv",
    "trend_projection_grid.csv",
    "technology_milestones.csv",
    "technology_cooccurrence.csv",
)


def _clear_previous_outputs(output_dir: Path) -> None:
    for name in TABLE_ARTIFACT_NAMES:
        path = output_dir / name
        if path.is_file():
            path.unlink()
    figures_dir = output_dir / "figures"
    if figures_dir.is_dir():
        for pattern in ("*.png", "*.svg"):
            for path in figures_dir.glob(pattern):
                path.unlink()


def _render_charts(
    *,
    observations: Sequence[Mapping[str, Any]],
    selected_functions: Sequence[Mapping[str, Any]],
    projection_through_year: int,
    output_dir: Path,
    dpi: int,
) -> list[str]:
    if dpi <= 0:
        raise ValueError("--dpi must be positive")
    os.environ.setdefault(
        "MPLCONFIGDIR", "/tmp/bpc_engine_decode_trend_mplconfig"
    )
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for charts; install requirements-plot.txt "
            "or pass --no-plots"
        ) from exc

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titleweight": "bold",
            "axes.labelcolor": "#222222",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "text.color": "#222222",
        }
    )
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[str] = []
    selected_by_id = {
        str(row["metric_id"]): row for row in selected_functions
    }
    minimum_time = min(
        float(row["release_decimal_year"]) for row in observations
    )
    maximum_time = max(
        float(row["release_decimal_year"]) for row in observations
    )
    recommended_end = maximum_time + 2.0
    grid_count = max(120, int((projection_through_year - minimum_time) * 24))
    grid = [
        minimum_time
        + (projection_through_year - minimum_time) * index
        / max(1, grid_count - 1)
        for index in range(grid_count)
    ]

    def save(fig: Any, stem: str) -> None:
        fig.tight_layout()
        for suffix in ("png", "svg"):
            path = figures_dir / f"{stem}.{suffix}"
            fig.savefig(
                path,
                dpi=dpi if suffix == "png" else None,
                bbox_inches="tight",
            )
            if suffix == "svg":
                _normalize_svg(path)
            artifacts.append(str(path.relative_to(output_dir)))
        plt.close(fig)

    def shade_projection(ax: Any) -> None:
        right = float(projection_through_year)
        ytd_start = float(YTD_YEAR)
        if maximum_time > ytd_start:
            ax.axvspan(
                ytd_start,
                maximum_time,
                color="#457b9d",
                alpha=0.07,
                label=f"{YTD_YEAR} YTD through {YTD_CUTOFF.isoformat()}",
            )
            ax.axvline(
                ytd_start,
                color="#457b9d",
                linestyle="--",
                linewidth=0.8,
            )
        if right > maximum_time:
            ax.axvspan(
                maximum_time,
                min(recommended_end, right),
                color="#f2c14e",
                alpha=0.12,
                label="conditional extrapolation",
            )
        if right > recommended_end:
            ax.axvspan(
                recommended_end,
                right,
                color="#d1495b",
                alpha=0.08,
                label="speculative",
            )
        ax.axvline(
            maximum_time,
            color="#666666",
            linestyle=":",
            linewidth=1.0,
        )
        ax.text(
            maximum_time,
            0.99,
            "last observed release",
            transform=ax.get_xaxis_transform(),
            ha="right",
            va="top",
            fontsize=7.5,
            color="#555555",
        )

    def function_curve(metric_id: str) -> list[float | None]:
        return [
            _selected_prediction_map(
                selected_functions, time_value
            )[metric_id][1]
            for time_value in grid
        ]

    parameter_specs = (
        (
            "parameters.resident_elements",
            "decode_resident_parameter_elements",
            "Resident parameters",
            "#264653",
        ),
        (
            "parameters.active_elements",
            "active_matrix_parameter_elements_per_token",
            "Active matrix parameters/token",
            "#e76f51",
        ),
    )
    fig, ax = plt.subplots(figsize=(12.5, 6.5))
    for metric_id, field, label, color in parameter_specs:
        ax.scatter(
            [row["release_decimal_year"] for row in observations],
            [row[field] for row in observations],
            label=f"{label} observations",
            color=color,
            alpha=0.72,
            s=34,
        )
        curve = function_curve(metric_id)
        ax.plot(
            grid,
            curve,
            color=color,
            linewidth=2.0,
            label=(
                f"{label} selected "
                f"({selected_by_id[metric_id]['selected_candidate_id']})"
            ),
        )
    shade_projection(ax)
    ax.set_yscale("log")
    ax.set_xlabel("Release year t")
    ax.set_ylabel("Parameter elements")
    ax.set_title(
        "Resident and Active Parameter Trajectories\n"
        "Curated releases; selected functions are not industry forecasts"
    )
    ax.grid(which="both", alpha=0.22)
    ax.legend(ncol=2, fontsize=8)
    save(fig, "parameter_scale_and_active_trends")

    fig, ax = plt.subplots(figsize=(12.5, 5.8))
    metric_id = "parameters.sparsity_multiplier"
    ax.scatter(
        [row["release_decimal_year"] for row in observations],
        [row["sparsity_multiplier"] for row in observations],
        color="#6a4c93",
        s=38,
        alpha=0.75,
        label="resident / active observations",
    )
    ax.plot(
        grid,
        function_curve(metric_id),
        color="#6a4c93",
        linewidth=2.2,
        label=(
            "selected "
            f"{selected_by_id[metric_id]['selected_candidate_id']} function"
        ),
    )
    shade_projection(ax)
    ax.set_yscale("log")
    ax.set_xlabel("Release year t")
    ax.set_ylabel("Resident / active parameter multiplier")
    ax.set_title("Parameter Sparsity Decoupling")
    ax.grid(which="both", alpha=0.22)
    ax.legend()
    save(fig, "parameter_sparsity_decoupling")

    context_series = (
        (
            "context.advertised_max",
            "advertised_max_context_tokens_at_release",
            "Advertised",
            "#e76f51",
        ),
        (
            "context.trained_max",
            "trained_max_context_tokens",
            "Trained",
            "#f4a261",
        ),
        (
            "context.evaluated_max",
            "evaluated_max_context_tokens",
            "Evaluated",
            "#2a9d8f",
        ),
        (
            "context.deployed_max",
            "deployed_max_context_tokens",
            "Deployed",
            "#264653",
        ),
    )
    fig, ax = plt.subplots(figsize=(12.5, 6.8))
    for metric_id, field, label, color in context_series:
        points = [
            row for row in observations if row[field] is not None
        ]
        ax.scatter(
            [row["release_decimal_year"] for row in points],
            [row[field] for row in points],
            color=color,
            alpha=0.65,
            s=28,
            label=f"{label} observations",
        )
        ax.plot(
            grid,
            function_curve(metric_id),
            color=color,
            linewidth=1.8,
            label=f"{label} selected function",
        )
    shade_projection(ax)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("Release year t")
    ax.set_ylabel("Context boundary (tokens)")
    ax.set_title(
        "Context Boundary Trajectories\n"
        "Advertised, trained, evaluated and deployed facts remain separate"
    )
    ax.grid(which="both", alpha=0.22)
    ax.legend(ncol=2, fontsize=7.8)
    save(fig, "context_boundary_trends")

    composition_metadata = {
        "mixer": (
            "Token Mixer Physical-Layer Composition",
            "token_mixer_composition",
            {
                "mixer.softmax_share": ("Softmax", "#264653"),
                "mixer.linear_recurrent_share": (
                    "Linear / recurrent",
                    "#e76f51",
                ),
                "mixer.ssm_share": ("SSM / Mamba", "#2a9d8f"),
            },
        ),
        "kv_layout": (
            "Softmax-Layer KV Layout Composition",
            "kv_layout_composition",
            {
                "kv_layout.mha_share": ("MHA", "#6a4c93"),
                "kv_layout.mqa_share": ("MQA", "#1982c4"),
                "kv_layout.gqa_share": ("GQA", "#2a9d8f"),
                "kv_layout.mla_share": ("MLA", "#e76f51"),
            },
        ),
        "attention_access": (
            "Softmax-Layer Access Composition",
            "attention_access_composition",
            {
                "access.full_share": ("Full", "#264653"),
                "access.bounded_local_share": (
                    "Bounded local",
                    "#f4a261",
                ),
                "access.sparse_topk_share": (
                    "Sparse / Top-k",
                    "#e76f51",
                ),
            },
        ),
    }
    for group, (title, stem, metadata) in composition_metadata.items():
        metric_ids = COMPOSITION_GROUPS[group]
        curves = {metric_id: [] for metric_id in metric_ids}
        for time_value in grid:
            predictions = _selected_prediction_map(
                selected_functions, time_value
            )
            for metric_id in metric_ids:
                curves[metric_id].append(predictions[metric_id][1] or 0.0)
        fig, ax = plt.subplots(figsize=(12.5, 6.2))
        annual = _annual_summary_rows(observations)
        for metric_id in metric_ids:
            function = selected_by_id[metric_id]
            evidence = str(function["evidence_grade"])
            linestyle = {
                "emerging": "-",
                "unstable": "--",
                "insufficient": ":",
                "established": "-",
            }[evidence]
            label, color = metadata[metric_id]
            ax.plot(
                grid,
                curves[metric_id],
                color=color,
                linestyle=linestyle,
                linewidth=2.0,
                label=(
                    f"{label} applied function (group "
                    f"{function['selected_candidate_id']}, {evidence})"
                ),
            )
            annual_rows = [
                row for row in annual if row["metric_id"] == metric_id
            ]
            ax.scatter(
                [row["year"] for row in annual_rows],
                [row["sample_mean"] for row in annual_rows],
                color=color,
                edgecolors="white",
                linewidths=0.7,
                s=46,
                zorder=4,
                label=f"{label} annual sample mean",
            )
        shade_projection(ax)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Release year t")
        ax.set_ylabel("Normalized physical-layer share")
        ax.set_title(
            f"{title}\n"
            "Functions are selected by group-normalized backtests; points "
            "are annual curated-sample means"
        )
        ax.legend(loc="upper left", ncol=2, fontsize=7.5)
        ax.grid(axis="y", alpha=0.2)
        save(fig, stem)

    fig, (presence_ax, intensity_ax) = plt.subplots(
        2, 1, figsize=(12.5, 8.0), sharex=True
    )
    presence_id = "moe.presence"
    intensity_id = "moe.unconditional_layer_share"
    conditional_intensity_id = "moe.layer_share_given_moe"
    annual = _annual_summary_rows(observations)
    annual_presence = [
        row for row in annual if row["metric_id"] == presence_id
    ]
    presence_ax.scatter(
        [row["year"] for row in annual_presence],
        [row["sample_prevalence"] for row in annual_presence],
        color="#e76f51",
        s=48,
        label="annual curated-sample prevalence",
    )
    presence_ax.plot(
        grid,
        function_curve(presence_id),
        color="#e76f51",
        linewidth=2.0,
        label="selected presence function",
    )
    shade_projection(presence_ax)
    presence_ax.set_ylabel("MoE sample presence")
    presence_ax.set_ylim(-0.03, 1.03)
    presence_ax.set_title(
        "MoE Presence and Structural Intensity\n"
        "Presence is not an industry adoption rate"
    )
    presence_ax.grid(alpha=0.2)
    presence_ax.legend()
    intensity_ax.scatter(
        [row["release_decimal_year"] for row in observations],
        [row["moe_layer_share"] for row in observations],
        color="#2a9d8f",
        s=34,
        alpha=0.75,
        label="routed-layer share observations",
    )
    intensity_ax.plot(
        grid,
        function_curve(intensity_id),
        color="#2a9d8f",
        linewidth=2.0,
        label="derived population routed-layer mass",
    )
    conditional_rows = [
        row
        for row in observations
        if row["moe_layer_share_given_moe"] is not None
    ]
    intensity_ax.scatter(
        [row["release_decimal_year"] for row in conditional_rows],
        [row["moe_layer_share_given_moe"] for row in conditional_rows],
        color="#6a4c93",
        marker="x",
        s=42,
        label="layer share given MoE",
    )
    intensity_ax.plot(
        grid,
        function_curve(conditional_intensity_id),
        color="#6a4c93",
        linestyle="--",
        linewidth=1.7,
        label="conditional intensity function",
    )
    shade_projection(intensity_ax)
    intensity_ax.set_xlabel("Release year t")
    intensity_ax.set_ylabel("MoE physical-layer share")
    intensity_ax.set_ylim(-0.03, 1.03)
    intensity_ax.grid(alpha=0.2)
    intensity_ax.legend()
    save(fig, "moe_presence_and_intensity")

    fig, (bits_ax, share_ax) = plt.subplots(
        2, 1, figsize=(12.5, 8.0), sharex=True
    )
    for metric_id, field, label, color in (
        (
            "precision.matrix_effective_bits",
            "matrix_effective_weight_bits",
            "Explicit matrix effective bits",
            "#264653",
        ),
        (
            "precision.resident_effective_storage_bits",
            "resident_effective_storage_bits",
            "Resident storage bits",
            "#6a4c93",
        ),
        (
            "precision.kv_effective_bits",
            "kv_effective_bits_used",
            "Actual KV effective bits",
            "#2a9d8f",
        ),
        (
            "precision.index_effective_bits",
            "index_effective_bits_used",
            "Actual Index effective bits",
            "#e9c46a",
        ),
        (
            "precision.state_effective_bits",
            "state_effective_bits_used",
            "Actual State effective bits",
            "#e76f51",
        ),
    ):
        precision_points = [
            row for row in observations if row[field] is not None
        ]
        bits_ax.scatter(
            [row["release_decimal_year"] for row in precision_points],
            [row[field] for row in precision_points],
            color=color,
            s=32,
            alpha=0.7,
            label=f"{label} observations",
        )
        bits_ax.plot(
            grid,
            function_curve(metric_id),
            color=color,
            linestyle=(
                "-"
                if selected_by_id[metric_id]["evidence_grade"]
                == "emerging"
                else (
                    "--"
                    if selected_by_id[metric_id]["evidence_grade"]
                    == "unstable"
                    else ":"
                )
            ),
            linewidth=1.9,
            label=(
                f"{label} function "
                f"({selected_by_id[metric_id]['evidence_grade']})"
            ),
        )
    shade_projection(bits_ax)
    bits_ax.set_ylabel("Bits / element")
    bits_ax.set_title(
        "Selected Deployment-Profile Precision\n"
        "Does not distinguish native from post-release quantization"
    )
    bits_ax.grid(alpha=0.2)
    bits_ax.legend(ncol=2, fontsize=8)
    for metric_id, field, label, color in (
        (
            "precision.explicit_share_le8",
            "explicit_weight_parameter_share_le8",
            "Explicit parameters ≤8 bit",
            "#2a9d8f",
        ),
        (
            "precision.explicit_share_le4",
            "explicit_weight_parameter_share_le4",
            "Explicit parameters ≤4 bit",
            "#e76f51",
        ),
    ):
        share_ax.scatter(
            [row["release_decimal_year"] for row in observations],
            [row[field] for row in observations],
            color=color,
            s=30,
            alpha=0.68,
            label=f"{label} observations",
        )
        share_ax.plot(
            grid,
            function_curve(metric_id),
            color=color,
            linewidth=1.9,
            label=f"{label} function",
        )
    shade_projection(share_ax)
    share_ax.set_ylim(-0.03, 1.03)
    share_ax.set_xlabel("Release year t")
    share_ax.set_ylabel("Explicit parameter share")
    share_ax.grid(alpha=0.2)
    share_ax.legend(ncol=2, fontsize=8)
    save(fig, "deployment_profile_precision")

    timeline_specs = [METRIC_BY_ID[value] for value in BINARY_TECHNOLOGIES]
    ordered_observations = sorted(
        observations,
        key=lambda row: (
            float(row["release_decimal_year"]),
            str(row["model_release_id"]),
        ),
    )
    matrix = [
        [int(row[spec.field]) for row in ordered_observations]
        for spec in timeline_specs
    ]
    fig, ax = plt.subplots(figsize=(14.5, 6.8))
    image = ax.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        cmap="YlGnBu",
        vmin=0,
        vmax=1,
    )
    ax.set_yticks(
        range(len(timeline_specs)),
        [spec.metric_id for spec in timeline_specs],
    )
    ax.set_xticks(
        range(len(ordered_observations)),
        [
            f"{row['year']} {_chart_model_name(str(row['short_model_name']))}"
            for row in ordered_observations
        ],
        rotation=65,
        ha="right",
    )
    ax.set_title(
        "Technology Presence Timeline in the Curated Release Sample\n"
        "Binary presence is descriptive, not deployment or token share; "
        "2026 is YTD"
    )
    first_ytd_index = next(
        (
            index
            for index, row in enumerate(ordered_observations)
            if int(row["year"]) == YTD_YEAR
        ),
        None,
    )
    if first_ytd_index is not None:
        ax.axvline(
            first_ytd_index - 0.5,
            color="#d1495b",
            linestyle="--",
            linewidth=1.2,
        )
    fig.colorbar(image, ax=ax, label="0 absent / 1 present", shrink=0.7)
    save(fig, "technology_presence_timeline")

    evidence_counts = Counter(
        str(row["evidence_grade"]) for row in selected_functions
    )
    grades = ("insufficient", "unstable", "emerging", "established")
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    bars = ax.barh(
        grades,
        [evidence_counts[grade] for grade in grades],
        color=["#9e9e9e", "#f4a261", "#2a9d8f", "#264653"],
    )
    ax.bar_label(bars)
    ax.set_xlabel("Selected function count")
    ax.set_title(
        "P9A Trend Evidence Summary\n"
        "Established is reserved for a predefined release census"
    )
    ax.grid(axis="x", alpha=0.2)
    save(fig, "trend_evidence_summary")
    return artifacts


def _run(
    *,
    release_dir: Path,
    output_dir: Path,
    projection_through_year: int,
    no_plots: bool,
    dpi: int,
) -> dict[str, Any]:
    release_dir = release_dir.resolve()
    output_dir = output_dir.resolve()
    try:
        output_dir.relative_to(release_dir)
    except ValueError:
        pass
    else:
        raise ValueError(
            "analysis output directory must not be inside the frozen release"
        )
    if projection_through_year < YTD_YEAR:
        raise ValueError("--projection-through-year must be at least 2026")
    if projection_through_year > 2100:
        raise ValueError("--projection-through-year is implausibly large")

    (
        configs,
        profiles,
        release_manifest,
        run_manifest,
        validation_report,
        config_hashes,
    ) = _load_release(release_dir)
    observations = _technology_observations(profiles, configs)
    (
        candidates,
        selected_functions,
        backtests,
        sensitivities,
    ) = _fit_all_trends(observations)
    composition_summaries = _apply_composition_group_selection(
        observations,
        candidates,
        selected_functions,
        backtests,
        sensitivities,
    )
    _apply_derived_identity_rows(selected_functions, candidates)
    annual_summary = _annual_summary_rows(observations)
    fitted_observations = _fitted_observation_rows(
        observations, selected_functions
    )
    projections = _projection_rows(
        observations, selected_functions, projection_through_year
    )
    milestones = _technology_milestone_rows(
        observations, selected_functions
    )
    cooccurrence = _cooccurrence_rows(observations)
    quality = _quality_summary(
        observations,
        validation_report,
        config_hashes,
        selected_functions,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _clear_previous_outputs(output_dir)
    csv_tables: tuple[tuple[str, Sequence[Mapping[str, Any]]], ...] = (
        ("technology_observations.csv", observations),
        ("annual_sample_summary.csv", annual_summary),
        ("trend_candidates.csv", candidates),
        ("selected_trend_functions.csv", selected_functions),
        ("composition_group_summary.csv", composition_summaries),
        ("trend_backtests.csv", backtests),
        ("trend_sensitivity.csv", sensitivities),
        ("fitted_observations.csv", fitted_observations),
        ("trend_projection_grid.csv", projections),
        ("technology_milestones.csv", milestones),
        ("technology_cooccurrence.csv", cooccurrence),
    )
    for name, rows in csv_tables:
        _write_csv(rows, output_dir / name)
    _write_json(quality, output_dir / "quality_summary.json")
    functions_json = {
        "technology_trend_function_schema_version": 2,
        "script_version": SCRIPT_VERSION,
        "reference_year_t0": REFERENCE_YEAR,
        "fit_cutoff_date": YTD_CUTOFF.isoformat(),
        "claim_population": "curated_representative_release_sample",
        "curated_sample_prevalence_not_industry_adoption": True,
        "cross_metric_constraints_enforced": False,
        "declared_identity_constraints_enforced": True,
        "cross_axis_joint_configuration_deferred_to": "P9B",
        "composition_components_fitted_independently": True,
        "composition_selection_jointly_backtested": True,
        "composition_projection_normalized": True,
        "within_axis_composition_constraints_enforced": True,
        "derived_identities": DERIVED_IDENTITIES,
        "composition_group_summaries": composition_summaries,
        "composition_groups": {
            name: {
                "metric_ids": list(metric_ids),
                "projection_formula": (
                    "p_k(t)=sigmoid(alpha_k+beta_k*(t-t0))/"
                    "sum_j(sigmoid(alpha_j+beta_j*(t-t0)))"
                ),
                "selection_score": "mean_component_brier",
                "cold_start_component_fallback": (
                    "Jeffreys-smoothed constant when a training fold has "
                    "no response variation"
                ),
            }
            for name, metric_ids in COMPOSITION_GROUPS.items()
        },
        "functions": selected_functions,
    }
    _write_json(
        functions_json, output_dir / "selected_trend_functions.json"
    )

    chart_artifacts: list[str] = []
    if not no_plots:
        chart_artifacts = _render_charts(
            observations=observations,
            selected_functions=selected_functions,
            projection_through_year=projection_through_year,
            output_dir=output_dir,
            dpi=dpi,
        )

    data_dir = release_dir / "data"
    script_path = Path(__file__).resolve()
    table_artifacts = [
        name
        for name in TABLE_ARTIFACT_NAMES
        if name != "analysis_manifest.json"
    ]
    row_counts = {
        "technology_observations": len(observations),
        "annual_sample_summary": len(annual_summary),
        "trend_candidates": len(candidates),
        "selected_trend_functions": len(selected_functions),
        "composition_group_summary": len(composition_summaries),
        "trend_backtests": len(backtests),
        "trend_sensitivity": len(sensitivities),
        "fitted_observations": len(fitted_observations),
        "trend_projection_grid": len(projections),
        "technology_milestones": len(milestones),
        "technology_cooccurrence": len(cooccurrence),
    }
    artifact_paths = table_artifacts + chart_artifacts
    artifact_sha256 = {
        relative: _sha256(output_dir / relative)
        for relative in artifact_paths
    }
    manifest = {
        "analysis_manifest_schema_version": 2,
        "analysis_id": (
            "decode-trend-p9a-technology-trajectories-v0.2"
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script_path": str(script_path.relative_to(PROJECT_ROOT)),
        "script_sha256": _sha256(script_path),
        "script_version": SCRIPT_VERSION,
        "source_release_dir": str(release_dir),
        "source_dataset_version": release_manifest.get("dataset_version"),
        "source_study_id": run_manifest.get("study_id"),
        "source_run_id": run_manifest.get("run_id"),
        "source_git_commit": release_manifest.get(
            "source_repository", {}
        ).get("git_commit"),
        "input_sha256": {
            "SHA256SUMS": _sha256(release_dir / "SHA256SUMS"),
            "release_manifest.json": _sha256(
                release_dir / "release_manifest.json"
            ),
            "data/run_manifest.json": _sha256(
                data_dir / "run_manifest.json"
            ),
            "data/validation_report.json": _sha256(
                data_dir / "validation_report.json"
            ),
            "data/model_profiles.jsonl": _sha256(
                data_dir / "model_profiles.jsonl"
            ),
            "data/decode_results.csv": _sha256(
                data_dir / "decode_results.csv"
            ),
        },
        "verified_config_sha256": dict(sorted(config_hashes.items())),
        "fit_cutoff_date": YTD_CUTOFF.isoformat(),
        "ytd_year": YTD_YEAR,
        "reference_year_t0": REFERENCE_YEAR,
        "projection_display_through_year": projection_through_year,
        "recommended_extrapolation_horizon_years": 2,
        "statistical_unit": (
            "one model_release_id + deployment_profile_id"
        ),
        "claim_population": "curated_representative_release_sample",
        "curated_sample_prevalence_not_industry_adoption": True,
        "decode_result_grid_used_as_fit_observations": False,
        "frozen_sha256sums_fully_verified": True,
        "decode_result_row_count_verified": True,
        "cross_metric_constraints_enforced": False,
        "declared_identity_constraints_enforced": True,
        "cross_axis_joint_configuration_deferred_to": "P9B",
        "composition_components_fitted_independently": True,
        "composition_selection_jointly_backtested": True,
        "composition_projection_normalized": True,
        "within_axis_composition_constraints_enforced": True,
        "derived_identity_metric_ids": sorted(DERIVED_IDENTITIES),
        "candidate_functions": {
            "positive_log": ["constant_log2_median", "theil_sen_log2"],
            "real_linear": ["constant_median", "theil_sen_linear"],
            "fraction_or_binary": [
                "jeffreys_smoothed_constant",
                "ridge_fractional_logistic",
            ],
        },
        "selection_policy": {
            "complete_backtest_years": list(COMPLETE_BACKTEST_YEARS),
            "exclude_ytd_from_complete_year_backtest": True,
            "minimum_complete_year_folds": 2,
            "minimum_relative_improvement_over_constant": 0.05,
            "logistic_l2_slope_penalty": LOGISTIC_L2,
        },
        "row_counts": row_counts,
        "artifacts": {
            "tables": table_artifacts,
            "figures": chart_artifacts,
            "sha256": artifact_sha256,
        },
    }
    _write_json(manifest, output_dir / "analysis_manifest.json")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        manifest = _run(
            release_dir=arguments.release_dir,
            output_dir=arguments.output_dir,
            projection_through_year=arguments.projection_through_year,
            no_plots=arguments.no_plots,
            dpi=arguments.dpi,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    row_counts = manifest["row_counts"]
    print(
        "P9A analyzed "
        f"{row_counts['technology_observations']} release observations; "
        f"selected {row_counts['selected_trend_functions']} technology "
        f"functions; wrote {manifest['source_release_dir']} -> "
        f"{arguments.output_dir.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
