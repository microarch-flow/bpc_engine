#!/usr/bin/env python3
"""Analyze the native-context Decode workload envelope in a frozen release.

The frozen CSV remains the sole numeric source of truth.  This script parses
its typed fields, joins the model-level profiles, validates core identities,
and emits derived tables and optional charts without modifying the release.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Callable, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RELEASE_DIR = (
    PROJECT_ROOT / "studies" / "decode_trend" / "releases" / "v1.0.0"
)
DEFAULT_OUTPUT_DIR = Path("/tmp/decode_trend_p8_envelope")
SCRIPT_VERSION = "0.2"
SELECTED_BATCHES = (1, 32, 256)
FIXED_COMPARISON_CONTEXT = 2048
SUPPORTED_SCHEMA_VERSION = 1

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
    "Qwen3.6-35B-A3B-FP8": "Qwen3.6-35B",
    "GLM-5.2-FP8": "GLM-5.2",
}

JSON_FIELDS = (
    "context_anchor_tags",
    "expert_weight_sets_read",
    "routing_assumptions",
    "p3_known_gaps",
    "calculation_assumptions",
    "calculation_warnings",
)
BOOLEAN_FIELDS = (
    "is_extrapolated",
    "within_advertised_context",
)
OPTIONAL_BOOLEAN_FIELDS = (
    "within_trained_context",
    "within_evaluated_context",
    "within_deployed_context",
)
INTEGER_FIELDS = (
    "result_schema_version",
    "year",
    "context_C",
    "concurrency_B",
    "decode_resident_parameter_elements",
    "active_matrix_parameter_elements_per_token",
    "decode_profile_weight_capacity_bytes",
)
OPTIONAL_INTEGER_FIELDS = ("full_checkpoint_capacity_bytes",)

FLOP_COMPONENTS = (
    "parameter_flops",
    "attention_flops",
    "index_flops",
    "state_flops",
    "extra_flops",
)
TRAFFIC_COMPONENTS = (
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
WORK_FLOAT_FIELDS = tuple(
    f"{prefix}_{component}"
    for prefix in ("step", "per_token")
    for component in (*FLOP_COMPONENTS, *TRAFFIC_COMPONENTS)
) + tuple(
    f"{prefix}_{total}"
    for prefix in ("step", "per_token")
    for total in (
        "total_flops",
        "engine_total_bytes",
        "logical_hbm_bytes",
    )
)
CAPACITY_FLOAT_FIELDS = (
    "kv_cache_bytes_per_request",
    "index_cache_bytes_per_request",
    "state_cache_bytes_per_request",
    "cache_bytes_per_request",
    "kv_cache_bytes_total",
    "index_cache_bytes_total",
    "state_cache_bytes_total",
    "batch_cache_bytes",
    "persistent_decode_profile_bytes",
)
OPTIONAL_FLOAT_FIELDS = ("persistent_full_checkpoint_bytes",)
DERIVED_FLOAT_FIELDS = (
    "active_parameter_ratio",
    "logical_hbm_bytes_per_flop",
    "flops_per_logical_hbm_byte",
    "tbps_per_pflops",
    "cache_to_decode_weight_capacity_ratio",
)
FLOAT_FIELDS = (
    *WORK_FLOAT_FIELDS,
    *CAPACITY_FLOAT_FIELDS,
    *DERIVED_FLOAT_FIELDS,
)

P3_FIELDS = (
    "p3_flops_support",
    "p3_logical_hbm_traffic_support",
    "p3_cache_capacity_support",
    "p3_weight_capacity_support",
)
P3_STATUS_ORDER = {
    "not_applicable": 0,
    "supported": 1,
    "partially_supported": 2,
    "unsupported": 3,
}

ENVELOPE_METRICS: tuple[tuple[str, str], ...] = (
    ("decode_profile_weight_capacity_bytes", "weight"),
    ("active_matrix_parameter_elements_per_token", "flops"),
    ("per_token_total_flops", "flops"),
    ("per_token_logical_hbm_bytes", "traffic"),
    ("cache_bytes_per_request", "cache"),
    ("persistent_decode_profile_bytes", "persistent"),
    ("tbps_per_pflops", "balance"),
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze the native-context industry demand envelope from a "
            "frozen Decode trend release."
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
    values: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
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


def _parse_bool(value: str, field: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"{field} must be True or False, got {value!r}")


def _parse_optional_bool(value: str, field: str) -> bool | None:
    if value == "":
        return None
    return _parse_bool(value, field)


def _parse_int(value: str, field: str) -> int:
    if value == "":
        raise ValueError(f"{field} must not be empty")
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be an integer, got {value!r}") from exc


def _parse_float(value: str, field: str) -> float:
    if value == "":
        raise ValueError(f"{field} must not be empty")
    try:
        result = float(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be numeric, got {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite, got {value!r}")
    return result


def _parse_json_cell(value: str, field: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field} contains invalid JSON: {value!r}") from exc


def _load_results(path: Path) -> list[dict[str, Any]]:
    try:
        handle = path.open(encoding="utf-8", newline="")
    except OSError as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    with handle:
        raw_rows = list(csv.DictReader(handle))
    if not raw_rows:
        raise ValueError(f"{path} contains no result rows")

    rows: list[dict[str, Any]] = []
    for row_index, raw in enumerate(raw_rows, start=2):
        row: dict[str, Any] = dict(raw)
        prefix = f"{path}:{row_index}"
        for field in INTEGER_FIELDS:
            row[field] = _parse_int(raw.get(field, ""), f"{prefix}.{field}")
        for field in OPTIONAL_INTEGER_FIELDS:
            value = raw.get(field, "")
            row[field] = (
                None if value == "" else _parse_int(value, f"{prefix}.{field}")
            )
        for field in FLOAT_FIELDS:
            row[field] = _parse_float(raw.get(field, ""), f"{prefix}.{field}")
        for field in OPTIONAL_FLOAT_FIELDS:
            value = raw.get(field, "")
            row[field] = (
                None
                if value == ""
                else _parse_float(value, f"{prefix}.{field}")
            )
        for field in BOOLEAN_FIELDS:
            row[field] = _parse_bool(raw.get(field, ""), f"{prefix}.{field}")
        for field in OPTIONAL_BOOLEAN_FIELDS:
            row[field] = _parse_optional_bool(
                raw.get(field, ""), f"{prefix}.{field}"
            )
        for field in JSON_FIELDS:
            row[field] = _parse_json_cell(
                raw.get(field, ""), f"{prefix}.{field}"
            )
        for field in P3_FIELDS:
            status = raw.get(field, "")
            if status not in P3_STATUS_ORDER:
                raise ValueError(
                    f"{prefix}.{field} has unsupported status {status!r}"
                )
        rows.append(row)
    return rows


def _profile_key(value: Mapping[str, Any]) -> tuple[str, str]:
    try:
        return (
            str(value["model_release_id"]),
            str(value["deployment_profile_id"]),
        )
    except KeyError as exc:
        raise ValueError(f"profile/result is missing key {exc.args[0]}") from exc


def _result_key(value: Mapping[str, Any]) -> tuple[str, str, int, int]:
    model_id, profile_id = _profile_key(value)
    return (
        model_id,
        profile_id,
        int(value["context_C"]),
        int(value["concurrency_B"]),
    )


def _isclose(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-5)


def _assert_close(left: float, right: float, message: str) -> None:
    if not _isclose(left, right):
        raise ValueError(f"{message}: {left} != {right}")


def _validate_results(
    rows: Sequence[Mapping[str, Any]],
    profiles: Mapping[tuple[str, str], Mapping[str, Any]],
    *,
    expected_run_id: str,
    expected_study_id: str,
) -> dict[str, int]:
    keys: set[tuple[str, str, int, int]] = set()
    run_ids: set[str] = set()
    study_ids: set[str] = set()
    request_cache_by_model_context: dict[
        tuple[str, str, int], tuple[float, float, float, float]
    ] = {}
    for row in rows:
        if row["result_schema_version"] != SUPPORTED_SCHEMA_VERSION:
            raise ValueError(
                "unsupported result_schema_version="
                f"{row['result_schema_version']}"
            )
        run_ids.add(str(row["run_id"]))
        study_ids.add(str(row["study_id"]))
        if row["run_id"] != expected_run_id:
            raise ValueError(
                "result run_id disagrees with run manifest: "
                f"{row['run_id']!r} != {expected_run_id!r}"
            )
        if row["study_id"] != expected_study_id:
            raise ValueError(
                "result study_id disagrees with run manifest: "
                f"{row['study_id']!r} != {expected_study_id!r}"
            )
        key = _result_key(row)
        if key in keys:
            raise ValueError(f"duplicate result key: {key}")
        keys.add(key)
        profile_key = key[:2]
        if profile_key not in profiles:
            raise ValueError(f"result has no matching model profile: {profile_key}")
        profile = profiles[profile_key]
        if row["config_sha256"] != profile.get("config_sha256"):
            raise ValueError(f"config hash mismatch for {profile_key}")
        expected_static_values = {
            "year": int(profile["year"]),
            "organization": str(profile["organization"]),
            "release_date": str(profile["release_date"]),
            "decode_resident_parameter_elements": int(
                profile["parameters"][
                    "decode_resident_parameter_elements"
                ]
            ),
            "active_matrix_parameter_elements_per_token": int(
                profile["parameters"][
                    "active_matrix_parameter_elements_per_token"
                ]
            ),
            "decode_profile_weight_capacity_bytes": int(
                profile["capacity"][
                    "decode_profile_weight_capacity_bytes"
                ]
            ),
            "full_checkpoint_capacity_bytes": profile["capacity"][
                "full_checkpoint_capacity_bytes"
            ],
        }
        for field, expected in expected_static_values.items():
            if row[field] != expected:
                raise ValueError(
                    f"{field} disagrees with model profile for {key}: "
                    f"{row[field]!r} != {expected!r}"
                )
        _assert_close(
            row["active_parameter_ratio"],
            float(profile["active_parameter_ratio"]),
            f"active parameter ratio disagrees with profile for {key}",
        )
        profile_p3 = profile["p3_audit"]["overall"]
        expected_p3 = {
            "p3_flops_support": profile_p3["flops"],
            "p3_logical_hbm_traffic_support": profile_p3[
                "logical_hbm_traffic"
            ],
            "p3_cache_capacity_support": profile_p3["cache_capacity"],
            "p3_weight_capacity_support": profile_p3["weight_capacity"],
        }
        for field, expected in expected_p3.items():
            if row[field] != expected:
                raise ValueError(
                    f"{field} disagrees with model profile for {key}: "
                    f"{row[field]!r} != {expected!r}"
                )

        context = row["context_C"]
        batch = row["concurrency_B"]
        if context < 0:
            raise ValueError(f"context_C must be non-negative for {key}")
        if batch <= 0:
            raise ValueError(f"concurrency_B must be positive for {key}")
        context_facts = profile["context"]
        advertised = int(
            context_facts["advertised_max_context_tokens_at_release"]
        )
        expected_is_extrapolated = context > advertised
        if row["is_extrapolated"] != expected_is_extrapolated:
            raise ValueError(
                f"is_extrapolated disagrees with advertised context for {key}"
            )
        if row["within_advertised_context"] != (context <= advertised):
            raise ValueError(
                "within_advertised_context disagrees with model profile "
                f"for {key}"
            )
        for field, profile_field in (
            ("within_trained_context", "trained_max_context_tokens"),
            ("within_evaluated_context", "evaluated_max_context_tokens"),
            ("within_deployed_context", "deployed_max_context_tokens"),
        ):
            boundary = context_facts[profile_field]
            expected = None if boundary is None else context <= int(boundary)
            if row[field] != expected:
                raise ValueError(
                    f"{field} disagrees with model profile for {key}: "
                    f"{row[field]!r} != {expected!r}"
                )
        _assert_close(
            row["active_parameter_ratio"],
            row["active_matrix_parameter_elements_per_token"]
            / row["decode_resident_parameter_elements"],
            f"active parameter ratio identity failed for {key}",
        )

        for component in (
            *FLOP_COMPONENTS,
            *TRAFFIC_COMPONENTS,
            "total_flops",
            "engine_total_bytes",
            "logical_hbm_bytes",
        ):
            _assert_close(
                row[f"step_{component}"],
                row[f"per_token_{component}"] * batch,
                f"step/token identity failed for {key} {component}",
            )

        for prefix in ("step", "per_token"):
            _assert_close(
                row[f"{prefix}_total_flops"],
                sum(row[f"{prefix}_{name}"] for name in FLOP_COMPONENTS),
                f"FLOP components do not close for {key} {prefix}",
            )
            _assert_close(
                row[f"{prefix}_engine_total_bytes"],
                sum(row[f"{prefix}_{name}"] for name in TRAFFIC_COMPONENTS),
                f"Traffic components do not close for {key} {prefix}",
            )
            _assert_close(
                row[f"{prefix}_logical_hbm_bytes"],
                row[f"{prefix}_engine_total_bytes"]
                - row[f"{prefix}_activation_bytes"],
                f"logical-HBM identity failed for {key} {prefix}",
            )

        _assert_close(
            row["cache_bytes_per_request"],
            row["kv_cache_bytes_per_request"]
            + row["index_cache_bytes_per_request"]
            + row["state_cache_bytes_per_request"],
            f"request Cache components do not close for {key}",
        )
        _assert_close(
            row["batch_cache_bytes"],
            row["kv_cache_bytes_total"]
            + row["index_cache_bytes_total"]
            + row["state_cache_bytes_total"],
            f"Batch Cache components do not close for {key}",
        )
        for name in ("kv", "index", "state"):
            _assert_close(
                row[f"{name}_cache_bytes_total"],
                row[f"{name}_cache_bytes_per_request"] * batch,
                f"{name} Batch/request Cache identity failed for {key}",
            )
        _assert_close(
            row["batch_cache_bytes"],
            row["cache_bytes_per_request"] * batch,
            f"Batch/request Cache identity failed for {key}",
        )
        _assert_close(
            row["persistent_decode_profile_bytes"],
            row["decode_profile_weight_capacity_bytes"]
            + row["batch_cache_bytes"],
            f"persistent Decode capacity identity failed for {key}",
        )
        if row["full_checkpoint_capacity_bytes"] is None:
            if row["persistent_full_checkpoint_bytes"] is not None:
                raise ValueError(
                    "unknown full checkpoint capacity became a number "
                    f"for {key}"
                )
        else:
            if row["persistent_full_checkpoint_bytes"] is None:
                raise ValueError(
                    f"known full checkpoint capacity lost persistent value for {key}"
                )
            _assert_close(
                row["persistent_full_checkpoint_bytes"],
                row["full_checkpoint_capacity_bytes"]
                + row["batch_cache_bytes"],
                f"persistent full checkpoint identity failed for {key}",
            )
        _assert_close(
            row["logical_hbm_bytes_per_flop"],
            row["per_token_logical_hbm_bytes"]
            / row["per_token_total_flops"],
            f"Byte/FLOP identity failed for {key}",
        )
        _assert_close(
            row["tbps_per_pflops"],
            1000.0 * row["logical_hbm_bytes_per_flop"],
            f"TB/s per PFLOPS identity failed for {key}",
        )

        cache_key = (key[0], key[1], key[2])
        request_cache = (
            row["kv_cache_bytes_per_request"],
            row["index_cache_bytes_per_request"],
            row["state_cache_bytes_per_request"],
            row["cache_bytes_per_request"],
        )
        previous_cache = request_cache_by_model_context.setdefault(
            cache_key, request_cache
        )
        for index, name in enumerate(("kv", "index", "state", "total")):
            _assert_close(
                previous_cache[index],
                request_cache[index],
                f"per-request {name} Cache changes with B for {cache_key}",
            )

    used_profile_keys = {key[:2] for key in keys}
    missing_profiles = sorted(set(profiles) - used_profile_keys)
    if missing_profiles:
        raise ValueError(f"profiles have no result rows: {missing_profiles}")
    if len(run_ids) != 1:
        raise ValueError(f"result CSV contains multiple run_ids: {run_ids}")
    if len(study_ids) != 1:
        raise ValueError(f"result CSV contains multiple study_ids: {study_ids}")
    return {
        "unique_result_keys": len(keys),
        "profiles_with_results": len(used_profile_keys),
        "unique_run_ids": len(run_ids),
        "unique_study_ids": len(study_ids),
        "model_context_cache_invariance_checks": len(
            request_cache_by_model_context
        ),
    }


def _load_release(
    release_dir: Path,
) -> tuple[
    list[dict[str, Any]],
    dict[tuple[str, str], dict[str, Any]],
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[str, Any],
    dict[str, int],
]:
    release_dir = release_dir.resolve()
    data_dir = release_dir / "data"
    results_path = data_dir / "decode_results.csv"
    profiles_path = data_dir / "model_profiles.jsonl"
    release_manifest = _load_object(release_dir / "release_manifest.json")
    run_manifest = _load_object(data_dir / "run_manifest.json")
    validation_report = _load_object(data_dir / "validation_report.json")
    if release_manifest.get("dataset_release_schema_version") != 1:
        raise ValueError("unsupported dataset_release_schema_version")
    if run_manifest.get("result_schema_version") != SUPPORTED_SCHEMA_VERSION:
        raise ValueError("unsupported run manifest result_schema_version")
    if validation_report.get("status") != "pass":
        raise ValueError("frozen release validation status is not pass")

    profiles_list = _load_jsonl(profiles_path)
    profiles: dict[tuple[str, str], dict[str, Any]] = {}
    for profile in profiles_list:
        key = _profile_key(profile)
        if key in profiles:
            raise ValueError(f"duplicate model profile key: {key}")
        profiles[key] = profile
    rows = _load_results(results_path)
    expected_run_id = str(run_manifest.get("run_id", ""))
    expected_study_id = str(run_manifest.get("study_id", ""))
    if not expected_run_id or not expected_study_id:
        raise ValueError("run manifest must contain run_id and study_id")
    checks = _validate_results(
        rows,
        profiles,
        expected_run_id=expected_run_id,
        expected_study_id=expected_study_id,
    )
    if int(validation_report.get("row_count", -1)) != len(rows):
        raise ValueError("validation report row_count disagrees with CSV")
    if int(validation_report.get("model_count", -1)) != len(profiles):
        raise ValueError("validation report model_count disagrees with profiles")
    if int(release_manifest.get("row_count", -1)) != len(rows):
        raise ValueError("release manifest row_count disagrees with CSV")
    if int(release_manifest.get("model_count", -1)) != len(profiles):
        raise ValueError("release manifest model_count disagrees with profiles")
    if run_manifest.get("run_id") != validation_report.get("run_id"):
        raise ValueError("run and validation manifests have different run_id")
    if run_manifest.get("study_id") != release_manifest.get("study_id"):
        raise ValueError("run and release manifests have different study_id")
    return (
        rows,
        profiles,
        release_manifest,
        run_manifest,
        validation_report,
        checks,
    )


def _json_cell(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _short_model_name(profile: Mapping[str, Any]) -> str:
    checkpoint = str(profile.get("checkpoint", ""))
    if "/" in checkpoint:
        return checkpoint.rsplit("/", 1)[-1]
    model_id = str(profile.get("model_release_id", "unknown"))
    parts = model_id.split(":")
    if len(parts) >= 2 and parts[1]:
        return parts[1]
    return checkpoint or model_id


def _chart_model_name(short_model_name: str) -> str:
    return CHART_MODEL_NAMES.get(short_model_name, short_model_name)


def _worst_status(*statuses: str) -> str:
    if not statuses:
        return "not_applicable"
    return max(statuses, key=lambda status: P3_STATUS_ORDER[status])


def _metric_support(row: Mapping[str, Any], metric_kind: str) -> str:
    if metric_kind == "weight":
        return str(row["p3_weight_capacity_support"])
    if metric_kind == "flops":
        return str(row["p3_flops_support"])
    if metric_kind == "traffic":
        return str(row["p3_logical_hbm_traffic_support"])
    if metric_kind == "cache":
        return str(row["p3_cache_capacity_support"])
    if metric_kind == "persistent":
        return _worst_status(
            str(row["p3_cache_capacity_support"]),
            str(row["p3_weight_capacity_support"]),
        )
    if metric_kind == "balance":
        return _worst_status(
            str(row["p3_flops_support"]),
            str(row["p3_logical_hbm_traffic_support"]),
        )
    raise ValueError(f"unknown metric support kind {metric_kind!r}")


def _model_summary_rows(
    profiles: Mapping[tuple[str, str], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for profile in profiles.values():
        p3 = profile["p3_audit"]["overall"]
        context = profile["context"]
        capacity = profile["capacity"]
        parameters = profile["parameters"]
        mechanisms = profile["mechanism_layer_counts"]
        output.append(
            {
                "year": profile["year"],
                "release_date": profile["release_date"],
                "organization": profile["organization"],
                "model_release_id": profile["model_release_id"],
                "deployment_profile_id": profile["deployment_profile_id"],
                "short_model_name": _short_model_name(profile),
                "checkpoint": profile["checkpoint"],
                "sample_roles": _json_cell(profile["sample_roles"]),
                "mechanism_layer_counts": _json_cell(mechanisms),
                "architecture_mechanisms": _json_cell(sorted(mechanisms)),
                "advertised_max_context_tokens_at_release": context[
                    "advertised_max_context_tokens_at_release"
                ],
                "trained_max_context_tokens": context[
                    "trained_max_context_tokens"
                ],
                "evaluated_max_context_tokens": context[
                    "evaluated_max_context_tokens"
                ],
                "deployed_max_context_tokens": context[
                    "deployed_max_context_tokens"
                ],
                "decode_resident_parameter_elements": parameters[
                    "decode_resident_parameter_elements"
                ],
                "active_matrix_parameter_elements_per_token": parameters[
                    "active_matrix_parameter_elements_per_token"
                ],
                "active_parameter_ratio": profile["active_parameter_ratio"],
                "decode_profile_weight_capacity_bytes": capacity[
                    "decode_profile_weight_capacity_bytes"
                ],
                "full_checkpoint_capacity_bytes": capacity[
                    "full_checkpoint_capacity_bytes"
                ],
                "p3_flops_support": p3["flops"],
                "p3_logical_hbm_traffic_support": p3[
                    "logical_hbm_traffic"
                ],
                "p3_cache_capacity_support": p3["cache_capacity"],
                "p3_weight_capacity_support": p3["weight_capacity"],
                "p3_known_gaps": _json_cell(
                    profile["p3_audit"]["known_gaps"]
                ),
                "calculation_assumptions": _json_cell(
                    profile["calculation_status"]["assumptions"]
                ),
            }
        )
    return sorted(
        output,
        key=lambda row: (
            int(row["year"]),
            str(row["release_date"]),
            str(row["model_release_id"]),
        ),
    )


def _native_main_rows(
    rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    selected = [
        row
        for row in rows
        if row["batch_class"] == "main"
        and not row["is_extrapolated"]
        and row["within_advertised_context"] is True
    ]
    if not selected:
        raise ValueError("native main filter selected no rows")
    unsupported = Counter(
        field
        for row in selected
        for field in P3_FIELDS
        if row[field] == "unsupported"
    )
    if unsupported:
        raise ValueError(
            "native envelope contains unsupported P3 dimensions: "
            f"{dict(sorted(unsupported.items()))}"
        )
    return selected


def _annual_envelope_rows(
    rows: Sequence[dict[str, Any]],
    profiles: Mapping[tuple[str, str], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    # Model-specific mechanism anchors are valuable within one model but cannot
    # form a fair annual envelope when peer models were not evaluated there.
    common_rows = [
        row
        for row in rows
        if "common_power_of_two" in row["context_anchor_tags"]
    ]
    scopes = {
        "all_representative_models": common_rows,
        "designated_frontier": [
            row
            for row in common_rows
            if "frontier_resource_envelope"
            in profiles[_profile_key(row)]["sample_roles"]
        ],
    }
    groups: dict[
        tuple[str, int, int, int], list[dict[str, Any]]
    ] = defaultdict(list)
    for scope, scope_rows in scopes.items():
        for row in scope_rows:
            groups[
                (
                    scope,
                    row["year"],
                    row["context_C"],
                    row["concurrency_B"],
                )
            ].append(row)

    output: list[dict[str, Any]] = []
    for (scope, year, context, batch), group in sorted(groups.items()):
        model_ids = sorted({str(row["model_release_id"]) for row in group})
        model_names = sorted(
            {
                _short_model_name(profiles[_profile_key(row)])
                for row in group
            }
        )
        envelope: dict[str, Any] = {
            "envelope_scope": scope,
            "grid_kind": "common_power_of_two",
            "year": year,
            "context_C": context,
            "concurrency_B": batch,
            "eligible_model_count": len(group),
            "eligible_model_release_ids": _json_cell(model_ids),
            "eligible_model_short_names": _json_cell(model_names),
        }
        for metric, metric_kind in ENVELOPE_METRICS:
            minimum = min(group, key=lambda row: float(row[metric]))
            maximum_value = max(float(row[metric]) for row in group)
            winners = [
                row
                for row in group
                if _isclose(float(row[metric]), maximum_value)
            ]
            envelope[f"min_{metric}"] = minimum[metric]
            envelope[f"max_{metric}"] = maximum_value
            envelope[f"max_{metric}_model_release_ids"] = _json_cell(
                sorted(str(row["model_release_id"]) for row in winners)
            )
            envelope[f"max_{metric}_short_model_names"] = _json_cell(
                sorted(
                    _short_model_name(profiles[_profile_key(row)])
                    for row in winners
                )
            )
            envelope[f"max_{metric}_sample_roles"] = _json_cell(
                {
                    str(row["model_release_id"]): profiles[
                        _profile_key(row)
                    ]["sample_roles"]
                    for row in sorted(
                        winners, key=lambda value: value["model_release_id"]
                    )
                }
            )
            envelope[f"max_{metric}_support"] = _worst_status(
                *(_metric_support(row, metric_kind) for row in winners)
            )
            statuses = Counter(
                _metric_support(row, metric_kind) for row in group
            )
            envelope[f"{metric}_supported_model_count"] = statuses[
                "supported"
            ]
            envelope[
                f"{metric}_partially_supported_model_count"
            ] = statuses["partially_supported"]
            supported_rows = [
                row
                for row in group
                if _metric_support(row, metric_kind) == "supported"
            ]
            if supported_rows:
                supported_max = max(
                    float(row[metric]) for row in supported_rows
                )
                supported_winners = [
                    row
                    for row in supported_rows
                    if _isclose(float(row[metric]), supported_max)
                ]
                envelope[f"supported_only_max_{metric}"] = supported_max
                envelope[
                    f"supported_only_max_{metric}_model_release_ids"
                ] = _json_cell(
                    sorted(
                        str(row["model_release_id"])
                        for row in supported_winners
                    )
                )
                envelope[
                    f"supported_only_max_{metric}_short_model_names"
                ] = _json_cell(
                    sorted(
                        _short_model_name(profiles[_profile_key(row)])
                        for row in supported_winners
                    )
                )
            else:
                envelope[f"supported_only_max_{metric}"] = None
                envelope[
                    f"supported_only_max_{metric}_model_release_ids"
                ] = _json_cell([])
                envelope[
                    f"supported_only_max_{metric}_short_model_names"
                ] = _json_cell([])
        output.append(envelope)
    if not output:
        raise ValueError("annual envelope contains no rows")
    return output


def _eligible_model_key(row: Mapping[str, Any]) -> tuple[str, ...]:
    value = row["eligible_model_release_ids"]
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list) or not all(
        isinstance(model_id, str) for model_id in value
    ):
        raise ValueError(
            "eligible_model_release_ids must encode a list of strings"
        )
    return tuple(sorted(value))


def _split_annual_series_by_eligible_cohort(
    rows: Sequence[Mapping[str, Any]],
) -> list[list[Mapping[str, Any]]]:
    ordered = sorted(rows, key=lambda row: int(row["context_C"]))
    segments: list[list[Mapping[str, Any]]] = []
    current: list[Mapping[str, Any]] = []
    current_key: tuple[str, ...] | None = None
    for row in ordered:
        cohort_key = _eligible_model_key(row)
        if current and cohort_key != current_key:
            segments.append(current)
            current = []
        current.append(row)
        current_key = cohort_key
    if current:
        segments.append(current)
    return segments


def _fixed_context_comparison_rows(
    rows: Sequence[dict[str, Any]],
    profiles: Mapping[tuple[str, str], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    selected = [
        row
        for row in rows
        if row["context_C"] == FIXED_COMPARISON_CONTEXT
        and row["concurrency_B"] in SELECTED_BATCHES
    ]
    output: list[dict[str, Any]] = []
    for row in selected:
        profile = profiles[_profile_key(row)]
        output.append(
            {
                "year": row["year"],
                "release_date": row["release_date"],
                "organization": row["organization"],
                "model_release_id": row["model_release_id"],
                "deployment_profile_id": row["deployment_profile_id"],
                "short_model_name": _short_model_name(profile),
                "sample_roles": _json_cell(profile["sample_roles"]),
                "advertised_max_context_tokens_at_release": profile[
                    "context"
                ]["advertised_max_context_tokens_at_release"],
                "context_C": row["context_C"],
                "concurrency_B": row["concurrency_B"],
                "per_token_total_flops": row["per_token_total_flops"],
                "per_token_logical_hbm_bytes": row[
                    "per_token_logical_hbm_bytes"
                ],
                "decode_profile_weight_capacity_bytes": row[
                    "decode_profile_weight_capacity_bytes"
                ],
                "cache_bytes_per_request": row["cache_bytes_per_request"],
                "batch_cache_bytes": row["batch_cache_bytes"],
                "persistent_decode_profile_bytes": row[
                    "persistent_decode_profile_bytes"
                ],
                "tbps_per_pflops": row["tbps_per_pflops"],
                "p3_flops_support": row["p3_flops_support"],
                "p3_logical_hbm_traffic_support": row[
                    "p3_logical_hbm_traffic_support"
                ],
                "p3_cache_capacity_support": row[
                    "p3_cache_capacity_support"
                ],
                "p3_weight_capacity_support": row[
                    "p3_weight_capacity_support"
                ],
            }
        )

    expected_keys = {
        (*profile_key, batch)
        for profile_key in profiles
        for batch in SELECTED_BATCHES
    }
    actual_keys = {
        (
            str(row["model_release_id"]),
            str(row["deployment_profile_id"]),
            int(row["concurrency_B"]),
        )
        for row in output
    }
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        raise ValueError(
            f"fixed C={FIXED_COMPARISON_CONTEXT} coverage mismatch: "
            f"missing={missing}, extra={extra}"
        )
    if len(output) != len(actual_keys):
        raise ValueError(
            f"fixed C={FIXED_COMPARISON_CONTEXT} rows contain duplicates"
        )
    return sorted(
        output,
        key=lambda row: (
            int(row["year"]),
            str(row["release_date"]),
            int(row["concurrency_B"]),
        ),
    )


def _native_max_point_rows(
    rows: Sequence[dict[str, Any]],
    profiles: Mapping[tuple[str, str], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        profile = profiles[_profile_key(row)]
        max_context = profile["context"][
            "advertised_max_context_tokens_at_release"
        ]
        if row["context_C"] != max_context:
            continue
        output.append(
            {
                "year": row["year"],
                "release_date": row["release_date"],
                "organization": row["organization"],
                "model_release_id": row["model_release_id"],
                "deployment_profile_id": row["deployment_profile_id"],
                "short_model_name": _short_model_name(profile),
                "sample_roles": _json_cell(profile["sample_roles"]),
                "context_C": row["context_C"],
                "concurrency_B": row["concurrency_B"],
                "active_matrix_parameter_elements_per_token": row[
                    "active_matrix_parameter_elements_per_token"
                ],
                "active_parameter_ratio": row["active_parameter_ratio"],
                "per_token_parameter_flops": row[
                    "per_token_parameter_flops"
                ],
                "per_token_attention_flops": row[
                    "per_token_attention_flops"
                ],
                "per_token_index_flops": row["per_token_index_flops"],
                "per_token_state_flops": row["per_token_state_flops"],
                "per_token_extra_flops": row["per_token_extra_flops"],
                "per_token_total_flops": row["per_token_total_flops"],
                "per_token_weight_read_bytes": row[
                    "per_token_weight_read_bytes"
                ],
                "per_token_kv_read_bytes": row["per_token_kv_read_bytes"],
                "per_token_kv_write_bytes": row["per_token_kv_write_bytes"],
                "per_token_index_read_bytes": row[
                    "per_token_index_read_bytes"
                ],
                "per_token_index_write_bytes": row[
                    "per_token_index_write_bytes"
                ],
                "per_token_state_read_bytes": row[
                    "per_token_state_read_bytes"
                ],
                "per_token_state_write_bytes": row[
                    "per_token_state_write_bytes"
                ],
                "per_token_other_read_bytes": row[
                    "per_token_other_read_bytes"
                ],
                "per_token_logical_hbm_bytes": row[
                    "per_token_logical_hbm_bytes"
                ],
                "decode_profile_weight_capacity_bytes": row[
                    "decode_profile_weight_capacity_bytes"
                ],
                "kv_cache_bytes_per_request": row[
                    "kv_cache_bytes_per_request"
                ],
                "index_cache_bytes_per_request": row[
                    "index_cache_bytes_per_request"
                ],
                "state_cache_bytes_per_request": row[
                    "state_cache_bytes_per_request"
                ],
                "cache_bytes_per_request": row["cache_bytes_per_request"],
                "kv_cache_bytes_total": row["kv_cache_bytes_total"],
                "index_cache_bytes_total": row["index_cache_bytes_total"],
                "state_cache_bytes_total": row["state_cache_bytes_total"],
                "batch_cache_bytes": row["batch_cache_bytes"],
                "persistent_decode_profile_bytes": row[
                    "persistent_decode_profile_bytes"
                ],
                "logical_hbm_bytes_per_flop": row[
                    "logical_hbm_bytes_per_flop"
                ],
                "tbps_per_pflops": row["tbps_per_pflops"],
                "p3_flops_support": row["p3_flops_support"],
                "p3_logical_hbm_traffic_support": row[
                    "p3_logical_hbm_traffic_support"
                ],
                "p3_cache_capacity_support": row[
                    "p3_cache_capacity_support"
                ],
                "p3_weight_capacity_support": row[
                    "p3_weight_capacity_support"
                ],
                "p3_known_gaps": _json_cell(row["p3_known_gaps"]),
                "calculation_assumptions": _json_cell(
                    row["calculation_assumptions"]
                ),
            }
        )
    expected = len(profiles) * len(
        {row["concurrency_B"] for row in rows if row["batch_class"] == "main"}
    )
    if len(output) != expected:
        raise ValueError(
            "native max point coverage is incomplete: "
            f"expected {expected}, found {len(output)}"
        )
    return sorted(
        output,
        key=lambda row: (
            int(row["year"]),
            str(row["release_date"]),
            int(row["concurrency_B"]),
        ),
    )


def _component_share_rows(
    native_max_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    selected = [
        row
        for row in native_max_rows
        if row["concurrency_B"] in SELECTED_BATCHES
    ]
    output: list[dict[str, Any]] = []
    for row in selected:
        total_flops = float(row["per_token_total_flops"])
        weight_bytes = float(row["per_token_weight_read_bytes"])
        kv_bytes = float(row["per_token_kv_read_bytes"]) + float(
            row["per_token_kv_write_bytes"]
        )
        index_bytes = float(row["per_token_index_read_bytes"]) + float(
            row["per_token_index_write_bytes"]
        )
        state_bytes = float(row["per_token_state_read_bytes"]) + float(
            row["per_token_state_write_bytes"]
        )
        other_bytes = float(row["per_token_other_read_bytes"])
        logical_bytes = float(row["per_token_logical_hbm_bytes"])
        _assert_close(
            weight_bytes
            + kv_bytes
            + index_bytes
            + state_bytes
            + other_bytes,
            logical_bytes,
            f"component-share Traffic does not close for "
            f"{row['model_release_id']} B={row['concurrency_B']}",
        )
        persistent = float(row["persistent_decode_profile_bytes"])
        weight_capacity = float(row["decode_profile_weight_capacity_bytes"])
        kv_cache_total = float(row["kv_cache_bytes_total"])
        index_cache_total = float(row["index_cache_bytes_total"])
        state_cache_total = float(row["state_cache_bytes_total"])
        batch_cache = float(row["batch_cache_bytes"])
        _assert_close(
            kv_cache_total + index_cache_total + state_cache_total,
            batch_cache,
            f"component-share Cache does not close for "
            f"{row['model_release_id']} B={row['concurrency_B']}",
        )
        _assert_close(
            weight_capacity + batch_cache,
            persistent,
            f"component-share capacity does not close for "
            f"{row['model_release_id']} B={row['concurrency_B']}",
        )

        item = {
            "year": row["year"],
            "release_date": row["release_date"],
            "model_release_id": row["model_release_id"],
            "deployment_profile_id": row["deployment_profile_id"],
            "short_model_name": row["short_model_name"],
            "context_C": row["context_C"],
            "concurrency_B": row["concurrency_B"],
            "per_token_total_flops": total_flops,
            "per_token_parameter_flops": row["per_token_parameter_flops"],
            "per_token_attention_flops": row["per_token_attention_flops"],
            "per_token_index_flops": row["per_token_index_flops"],
            "per_token_state_flops": row["per_token_state_flops"],
            "per_token_extra_flops": row["per_token_extra_flops"],
            "parameter_flops_share": float(
                row["per_token_parameter_flops"]
            )
            / total_flops,
            "attention_flops_share": float(
                row["per_token_attention_flops"]
            )
            / total_flops,
            "index_flops_share": float(row["per_token_index_flops"])
            / total_flops,
            "state_flops_share": float(row["per_token_state_flops"])
            / total_flops,
            "extra_flops_share": float(row["per_token_extra_flops"])
            / total_flops,
            "per_token_logical_hbm_bytes": logical_bytes,
            "per_token_weight_read_bytes": weight_bytes,
            "per_token_kv_bytes": kv_bytes,
            "per_token_index_bytes": index_bytes,
            "per_token_state_bytes": state_bytes,
            "per_token_other_read_bytes": other_bytes,
            "weight_read_share": weight_bytes / logical_bytes,
            "kv_traffic_share": kv_bytes / logical_bytes,
            "index_traffic_share": index_bytes / logical_bytes,
            "state_traffic_share": state_bytes / logical_bytes,
            "other_read_share": other_bytes / logical_bytes,
            "persistent_decode_profile_bytes": persistent,
            "decode_profile_weight_capacity_bytes": weight_capacity,
            "kv_cache_bytes_total": kv_cache_total,
            "index_cache_bytes_total": index_cache_total,
            "state_cache_bytes_total": state_cache_total,
            "batch_cache_bytes": batch_cache,
            "weight_capacity_share": weight_capacity / persistent,
            "kv_cache_capacity_share": kv_cache_total / persistent,
            "index_cache_capacity_share": index_cache_total / persistent,
            "state_cache_capacity_share": state_cache_total / persistent,
            "batch_cache_capacity_share": batch_cache / persistent,
            "p3_flops_support": row["p3_flops_support"],
            "p3_logical_hbm_traffic_support": row[
                "p3_logical_hbm_traffic_support"
            ],
            "p3_cache_capacity_support": row[
                "p3_cache_capacity_support"
            ],
            "p3_weight_capacity_support": row[
                "p3_weight_capacity_support"
            ],
        }
        _assert_close(
            sum(item[f"{name}_flops_share"] for name in (
                "parameter",
                "attention",
                "index",
                "state",
                "extra",
            )),
            1.0,
            f"FLOP shares do not close for {row['model_release_id']}",
        )
        _assert_close(
            sum(item[name] for name in (
                "weight_read_share",
                "kv_traffic_share",
                "index_traffic_share",
                "state_traffic_share",
                "other_read_share",
            )),
            1.0,
            f"Traffic shares do not close for {row['model_release_id']}",
        )
        _assert_close(
            sum(item[name] for name in (
                "weight_capacity_share",
                "kv_cache_capacity_share",
                "index_cache_capacity_share",
                "state_cache_capacity_share",
            )),
            1.0,
            f"capacity shares do not close for {row['model_release_id']}",
        )
        _assert_close(
            item["kv_cache_capacity_share"]
            + item["index_cache_capacity_share"]
            + item["state_cache_capacity_share"],
            item["batch_cache_capacity_share"],
            f"Batch Cache shares do not close for {row['model_release_id']}",
        )
        output.append(item)
    return output


def _context_boundary_rows(
    rows: Sequence[dict[str, Any]],
    profiles: Mapping[tuple[str, str], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    boundary_fields = {
        "advertised_max": "advertised_max_context_tokens_at_release",
        "trained_max": "trained_max_context_tokens",
        "evaluated_max": "evaluated_max_context_tokens",
        "deployed_max": "deployed_max_context_tokens",
    }
    output: list[dict[str, Any]] = []
    for row in rows:
        if row["concurrency_B"] not in SELECTED_BATCHES:
            continue
        profile = profiles[_profile_key(row)]
        context_facts = profile["context"]
        derived_tags = {
            tag
            for tag, field in boundary_fields.items()
            if context_facts.get(field) == row["context_C"]
        }
        matched_tags = sorted(
            set(row["context_anchor_tags"]).intersection(boundary_fields)
            | derived_tags
        )
        if not matched_tags:
            continue
        output.append(
            {
                "year": row["year"],
                "release_date": row["release_date"],
                "model_release_id": row["model_release_id"],
                "deployment_profile_id": row["deployment_profile_id"],
                "short_model_name": _short_model_name(profile),
                "context_C": row["context_C"],
                "concurrency_B": row["concurrency_B"],
                "boundary_tags": _json_cell(matched_tags),
                "all_context_anchor_tags": _json_cell(
                    row["context_anchor_tags"]
                ),
                "per_token_total_flops": row["per_token_total_flops"],
                "per_token_logical_hbm_bytes": row[
                    "per_token_logical_hbm_bytes"
                ],
                "decode_profile_weight_capacity_bytes": row[
                    "decode_profile_weight_capacity_bytes"
                ],
                "cache_bytes_per_request": row["cache_bytes_per_request"],
                "batch_cache_bytes": row["batch_cache_bytes"],
                "persistent_decode_profile_bytes": row[
                    "persistent_decode_profile_bytes"
                ],
                "tbps_per_pflops": row["tbps_per_pflops"],
                "p3_flops_support": row["p3_flops_support"],
                "p3_logical_hbm_traffic_support": row[
                    "p3_logical_hbm_traffic_support"
                ],
                "p3_cache_capacity_support": row[
                    "p3_cache_capacity_support"
                ],
                "p3_weight_capacity_support": row[
                    "p3_weight_capacity_support"
                ],
            }
        )
    if not output:
        raise ValueError("no context boundary points were found")
    available_batches = {
        row["concurrency_B"]
        for row in rows
        if row["concurrency_B"] in SELECTED_BATCHES
    }
    expected_keys: set[tuple[str, str, int, int]] = set()
    for profile_key, profile in profiles.items():
        context_facts = profile["context"]
        advertised = int(
            context_facts["advertised_max_context_tokens_at_release"]
        )
        for field in boundary_fields.values():
            context = context_facts.get(field)
            if context is None or int(context) > advertised:
                continue
            for batch in available_batches:
                expected_keys.add(
                    (*profile_key, int(context), int(batch))
                )
    actual_keys = {
        (
            str(row["model_release_id"]),
            str(row["deployment_profile_id"]),
            int(row["context_C"]),
            int(row["concurrency_B"]),
        )
        for row in output
    }
    missing_keys = sorted(expected_keys - actual_keys)
    if missing_keys:
        raise ValueError(
            "context boundary coverage is incomplete: "
            f"missing {missing_keys}"
        )
    return sorted(
        output,
        key=lambda row: (
            int(row["year"]),
            str(row["release_date"]),
            int(row["context_C"]),
            int(row["concurrency_B"]),
        ),
    )


def _crossover_result_at_index(
    rows: Sequence[Mapping[str, Any]],
    index: int,
    left: Callable[[Mapping[str, Any]], float],
    right: Callable[[Mapping[str, Any]], float],
) -> dict[str, Any]:
    row = rows[index]
    left_value = left(row)
    right_value = right(row)
    upper_ratio = math.inf if right_value == 0 else left_value / right_value
    if index == 0:
        return {
            "status": "at_or_below_min_scanned",
            "lower_context_C": None,
            "upper_context_C": row["context_C"],
            "lower_left_to_right_ratio": None,
            "upper_left_to_right_ratio": upper_ratio,
            "left_censored": True,
            "upper_context_anchor_tags": _json_cell(
                row["context_anchor_tags"]
            ),
        }
    previous = rows[index - 1]
    previous_left = left(previous)
    previous_right = right(previous)
    return {
        "status": "crossed_in_grid",
        "lower_context_C": previous["context_C"],
        "upper_context_C": row["context_C"],
        "lower_left_to_right_ratio": (
            math.inf
            if previous_right == 0
            else previous_left / previous_right
        ),
        "upper_left_to_right_ratio": upper_ratio,
        "left_censored": False,
        "upper_context_anchor_tags": _json_cell(
            row["context_anchor_tags"]
        ),
    }


def _not_reached_crossover_result(
    rows: Sequence[Mapping[str, Any]],
    left: Callable[[Mapping[str, Any]], float],
    right: Callable[[Mapping[str, Any]], float],
    *,
    status: str = "not_reached_within_advertised",
) -> dict[str, Any]:
    final = rows[-1]
    left_value = left(final)
    right_value = right(final)
    return {
        "status": status,
        "lower_context_C": final["context_C"],
        "upper_context_C": None,
        "lower_left_to_right_ratio": (
            math.inf if right_value == 0 else left_value / right_value
        ),
        "upper_left_to_right_ratio": None,
        "left_censored": False,
        "upper_context_anchor_tags": _json_cell([]),
    }


def _first_crossover(
    rows: Sequence[Mapping[str, Any]],
    left: Callable[[Mapping[str, Any]], float],
    right: Callable[[Mapping[str, Any]], float],
) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot find a crossover in an empty row set")
    dominant = [left(row) >= right(row) for row in rows]
    first_index = next(
        (index for index, value in enumerate(dominant) if value),
        None,
    )
    if first_index is None:
        result = _not_reached_crossover_result(rows, left, right)
        stable_result = dict(result)
        result["remains_dominant_after_first_reach"] = None
        result["dominance_reversal_count"] = 0
    else:
        result = _crossover_result_at_index(
            rows, first_index, left, right
        )
        result["remains_dominant_after_first_reach"] = all(
            dominant[first_index:]
        )
        result["dominance_reversal_count"] = sum(
            previous and not current
            for previous, current in zip(
                dominant[first_index:], dominant[first_index + 1:]
            )
        )
        stable_index = next(
            (
                index
                for index in range(first_index, len(rows))
                if dominant[index] and all(dominant[index:])
            ),
            None,
        )
        if stable_index is None:
            stable_result = _not_reached_crossover_result(
                rows,
                left,
                right,
                status="not_stably_reached_within_advertised",
            )
        else:
            stable_result = _crossover_result_at_index(
                rows, stable_index, left, right
            )
    for field, value in stable_result.items():
        result[f"stable_{field}"] = value
    return result


def _crossover_rows(
    rows: Sequence[dict[str, Any]],
    profiles: Mapping[tuple[str, str], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        model_id, profile_id = _profile_key(row)
        groups[(model_id, profile_id, row["concurrency_B"])].append(row)

    definitions: tuple[
        tuple[
            str,
            Callable[[Mapping[str, Any]], float],
            Callable[[Mapping[str, Any]], float],
        ],
        ...,
    ] = (
        (
            "non_parameter_flops_vs_parameter",
            lambda row: (
                row["per_token_attention_flops"]
                + row["per_token_index_flops"]
                + row["per_token_state_flops"]
                + row["per_token_extra_flops"]
            ),
            lambda row: row["per_token_parameter_flops"],
        ),
        (
            "non_weight_logical_hbm_vs_weight_read",
            lambda row: (
                row["per_token_kv_read_bytes"]
                + row["per_token_kv_write_bytes"]
                + row["per_token_index_read_bytes"]
                + row["per_token_index_write_bytes"]
                + row["per_token_state_read_bytes"]
                + row["per_token_state_write_bytes"]
                + row["per_token_other_read_bytes"]
            ),
            lambda row: row["per_token_weight_read_bytes"],
        ),
        (
            "request_cache_vs_weight_capacity",
            lambda row: row["cache_bytes_per_request"],
            lambda row: row["decode_profile_weight_capacity_bytes"],
        ),
        (
            "batch_cache_vs_weight_capacity",
            lambda row: row["batch_cache_bytes"],
            lambda row: row["decode_profile_weight_capacity_bytes"],
        ),
    )

    output: list[dict[str, Any]] = []
    for (model_id, profile_id, batch), group in sorted(groups.items()):
        ordered = sorted(group, key=lambda row: row["context_C"])
        profile = profiles[(model_id, profile_id)]
        item: dict[str, Any] = {
            "year": profile["year"],
            "release_date": profile["release_date"],
            "model_release_id": model_id,
            "deployment_profile_id": profile_id,
            "short_model_name": _short_model_name(profile),
            "concurrency_B": batch,
            "advertised_max_context_tokens_at_release": profile["context"][
                "advertised_max_context_tokens_at_release"
            ],
            "native_scan_point_count": len(ordered),
            "p3_flops_support": profile["p3_audit"]["overall"]["flops"],
            "p3_logical_hbm_traffic_support": profile["p3_audit"][
                "overall"
            ]["logical_hbm_traffic"],
            "p3_cache_capacity_support": profile["p3_audit"]["overall"][
                "cache_capacity"
            ],
            "p3_weight_capacity_support": profile["p3_audit"]["overall"][
                "weight_capacity"
            ],
            "non_parameter_flops_vs_parameter_support": profile[
                "p3_audit"
            ]["overall"]["flops"],
            "non_weight_logical_hbm_vs_weight_read_support": profile[
                "p3_audit"
            ]["overall"]["logical_hbm_traffic"],
            "request_cache_vs_weight_capacity_support": _worst_status(
                profile["p3_audit"]["overall"]["cache_capacity"],
                profile["p3_audit"]["overall"]["weight_capacity"],
            ),
            "batch_cache_vs_weight_capacity_support": _worst_status(
                profile["p3_audit"]["overall"]["cache_capacity"],
                profile["p3_audit"]["overall"]["weight_capacity"],
            ),
        }
        for name, left, right in definitions:
            result = _first_crossover(ordered, left, right)
            for field, value in result.items():
                item[f"{name}_{field}"] = value
        output.append(item)
    return output


def _quality_summary(
    rows: Sequence[dict[str, Any]],
    native_rows: Sequence[dict[str, Any]],
    profiles: Mapping[tuple[str, str], Mapping[str, Any]],
    validation_report: Mapping[str, Any],
    checks: Mapping[str, int],
) -> dict[str, Any]:
    support_counts = {
        field: dict(sorted(Counter(row[field] for row in native_rows).items()))
        for field in P3_FIELDS
    }
    model_support_counts = {
        dimension: dict(
            sorted(
                Counter(
                    profile["p3_audit"]["overall"][profile_field]
                    for profile in profiles.values()
                ).items()
            )
        )
        for dimension, profile_field in (
            ("p3_flops_support", "flops"),
            ("p3_logical_hbm_traffic_support", "logical_hbm_traffic"),
            ("p3_cache_capacity_support", "cache_capacity"),
            ("p3_weight_capacity_support", "weight_capacity"),
        )
    }
    source_warning_codes = Counter(
        str(warning)
        for row in rows
        for warning in row["calculation_warnings"]
    )
    native_warning_codes = Counter(
        str(warning)
        for row in native_rows
        for warning in row["calculation_warnings"]
    )

    def warning_categories(codes: Counter[str]) -> dict[str, int]:
        categories: Counter[str] = Counter()
        for code, count in codes.items():
            if code.startswith("p3_"):
                category = "p3_support"
            elif code.startswith("context_"):
                category = "context_scope"
            elif "float" in code or "integer_range" in code:
                category = "numeric_range"
            else:
                category = "other"
            categories[category] += count
        return dict(sorted(categories.items()))

    return {
        "status": "pass",
        "source_result_rows": len(rows),
        "source_model_profiles": len(profiles),
        "source_main_rows": sum(
            row["batch_class"] == "main" for row in rows
        ),
        "source_stress_rows": sum(
            row["batch_class"] == "stress" for row in rows
        ),
        "source_extrapolated_rows": sum(
            row["is_extrapolated"] for row in rows
        ),
        "native_main_rows": len(native_rows),
        "native_main_common_grid_rows": sum(
            "common_power_of_two" in row["context_anchor_tags"]
            for row in native_rows
        ),
        "native_main_rows_by_batch": {
            str(batch): count
            for batch, count in sorted(
                Counter(row["concurrency_B"] for row in native_rows).items()
            )
        },
        "native_main_support_counts": support_counts,
        "model_support_counts": model_support_counts,
        "source_rows_with_calculation_warnings": sum(
            bool(row["calculation_warnings"]) for row in rows
        ),
        "native_main_rows_with_calculation_warnings": sum(
            bool(row["calculation_warnings"]) for row in native_rows
        ),
        "source_row_warning_code_counts": dict(
            sorted(source_warning_codes.items())
        ),
        "native_main_row_warning_code_counts": dict(
            sorted(native_warning_codes.items())
        ),
        "source_row_warning_category_counts": warning_categories(
            source_warning_codes
        ),
        "native_main_row_warning_category_counts": warning_categories(
            native_warning_codes
        ),
        "frozen_validation": dict(validation_report),
        "analysis_validation": dict(checks),
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


def _clear_previous_outputs(output_dir: Path) -> None:
    for name in (
        "analysis_manifest.json",
        "quality_summary.json",
        "model_summary.csv",
        "annual_envelope.csv",
        "fixed_context_comparison.csv",
        "native_max_points.csv",
        "context_boundary_points.csv",
        "component_shares.csv",
        "crossover_points.csv",
    ):
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
    native_rows: Sequence[dict[str, Any]],
    model_summary: Sequence[dict[str, Any]],
    annual_envelope: Sequence[dict[str, Any]],
    fixed_context_comparison: Sequence[dict[str, Any]],
    native_max_points: Sequence[dict[str, Any]],
    profiles: Mapping[tuple[str, str], Mapping[str, Any]],
    output_dir: Path,
    dpi: int,
) -> list[str]:
    os.environ.setdefault(
        "MPLCONFIGDIR", "/tmp/bpc_engine_decode_trend_mplconfig"
    )
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        from matplotlib.ticker import MaxNLocator
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for charts; install requirements-plot.txt "
            "or pass --no-plots"
        ) from exc

    if dpi <= 0:
        raise ValueError("--dpi must be positive")

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

    year_values = sorted({int(row["year"]) for row in model_summary})
    year_cmap = plt.get_cmap("viridis")
    year_colors = {
        year: year_cmap(index / max(1, len(year_values) - 1))
        for index, year in enumerate(year_values)
    }
    ordered_models = sorted(
        model_summary,
        key=lambda row: (
            str(row["release_date"]),
            str(row["short_model_name"]),
        ),
    )
    model_keys = [
        (str(row["model_release_id"]), str(row["deployment_profile_id"]))
        for row in ordered_models
    ]
    model_cmap = plt.get_cmap("tab20")
    model_colors = {
        key: model_cmap(index % 20) for index, key in enumerate(model_keys)
    }

    labels = [str(row["short_model_name"]) for row in ordered_models]
    positions = list(range(len(ordered_models)))
    colors = [year_colors[int(row["year"])] for row in ordered_models]

    def add_bar_legend(
        ax: Any,
        bars: Any,
        rows: Sequence[Mapping[str, Any]],
        support_field: str,
    ) -> None:
        for bar, row in zip(bars, rows):
            if row[support_field] == "partially_supported":
                bar.set_hatch("//")
                bar.set_edgecolor("#333333")
                bar.set_linewidth(0.8)
        handles = [
            Patch(facecolor=year_colors[year], label=str(year))
            for year in year_values
        ]
        if any(
            row[support_field] == "partially_supported" for row in rows
        ):
            handles.append(
                Patch(
                    facecolor="white",
                    edgecolor="#333333",
                    hatch="//",
                    label="P3 partially supported",
                )
            )
        ax.legend(handles=handles, title="Release year / support", ncol=3)

    fig, ax = plt.subplots(figsize=(14, 6.5))
    bars = ax.bar(
        positions,
        [
            float(row["decode_profile_weight_capacity_bytes"]) / (1024**3)
            for row in ordered_models
        ],
        color=colors,
    )
    ax.set_yscale("log")
    ax.set_ylabel("Decode profile weight capacity (GiB)")
    ax.set_title("Historical Decode Weight Capacity by Release")
    ax.set_xticks(positions, labels, rotation=65, ha="right")
    ax.grid(axis="y", alpha=0.25)
    add_bar_legend(
        ax, bars, ordered_models, "p3_weight_capacity_support"
    )
    save(fig, "weight_capacity_by_release")

    fig, ax = plt.subplots(figsize=(14, 6.5))
    bars = ax.bar(
        positions,
        [
            100.0 * float(row["active_parameter_ratio"])
            for row in ordered_models
        ],
        color=colors,
    )
    ax.set_ylabel("Active matrix parameters / resident parameters (%)")
    ax.set_title("Active Parameter Ratio by Release")
    ax.set_xticks(positions, labels, rotation=65, ha="right")
    ax.grid(axis="y", alpha=0.25)
    add_bar_legend(ax, bars, ordered_models, "p3_flops_support")
    save(fig, "active_parameter_ratio_by_release")

    rows_by_model: dict[
        tuple[str, str], list[dict[str, Any]]
    ] = defaultdict(list)
    for row in native_rows:
        rows_by_model[_profile_key(row)].append(row)

    def model_curve_chart(
        *,
        batch: int,
        metric: str,
        support_kind: str,
        ylabel: str,
        title: str,
        stem: str,
    ) -> None:
        fig, ax = plt.subplots(figsize=(13.5, 7.5))
        for key in model_keys:
            series = sorted(
                (
                    row
                    for row in rows_by_model[key]
                    if row["concurrency_B"] == batch
                ),
                key=lambda row: row["context_C"],
            )
            if not series:
                continue
            profile = profiles[key]
            status = _metric_support(series[0], support_kind)
            ax.plot(
                [row["context_C"] for row in series],
                [row[metric] for row in series],
                color=model_colors[key],
                linestyle="--" if status == "partially_supported" else "-",
                marker="o",
                markersize=2.8,
                linewidth=1.45,
                label=_short_model_name(profile),
            )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("Start-of-step context C (tokens)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(which="both", alpha=0.22)
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            fontsize=7.5,
            ncol=1,
        )
        ax.text(
            0.01,
            0.015,
            "solid = P3 supported; dashed = P3 partially supported",
            transform=ax.transAxes,
            fontsize=8,
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": "#bbbbbb",
                "alpha": 0.9,
            },
        )
        save(fig, stem)

    for batch in SELECTED_BATCHES:
        model_curve_chart(
            batch=batch,
            metric="per_token_total_flops",
            support_kind="flops",
            ylabel="FLOPs / output token",
            title=f"Native Decode Compute vs Context (B={batch})",
            stem=f"flops_per_token_by_context_B{batch}",
        )
        model_curve_chart(
            batch=batch,
            metric="per_token_logical_hbm_bytes",
            support_kind="traffic",
            ylabel="Logical-HBM bytes / output token",
            title=f"Native Decode Logical-HBM Traffic vs Context (B={batch})",
            stem=f"logical_hbm_bytes_per_token_by_context_B{batch}",
        )
        model_curve_chart(
            batch=batch,
            metric="persistent_decode_profile_bytes",
            support_kind="persistent",
            ylabel="Persistent Decode profile bytes",
            title=f"Native Persistent Memory vs Context (B={batch})",
            stem=f"persistent_memory_by_context_B{batch}",
        )
        model_curve_chart(
            batch=batch,
            metric="tbps_per_pflops",
            support_kind="balance",
            ylabel="TB/s per PFLOPS",
            title=f"Native Logical Bandwidth / Compute vs Context (B={batch})",
            stem=f"tbps_per_pflops_by_context_B{batch}",
        )

    model_curve_chart(
        batch=1,
        metric="cache_bytes_per_request",
        support_kind="cache",
        ylabel="Cache bytes / request",
        title="Native Per-Request Cache Capacity vs Context",
        stem="cache_capacity_per_request_by_context",
    )

    envelope_groups: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in annual_envelope:
        envelope_groups[int(row["concurrency_B"])].append(row)

    def scatter_by_support(
        ax: Any,
        *,
        x: float,
        y: float,
        color: Any,
        status: str,
        marker: str = "o",
        size: float = 34,
        zorder: int = 4,
    ) -> None:
        ax.scatter(
            [x],
            [y],
            facecolors=(
                "white" if status == "partially_supported" else color
            ),
            edgecolors=color,
            marker=marker,
            linewidths=1.25,
            s=size,
            zorder=zorder,
        )

    def annotate_without_overlap(
        fig: Any,
        ax: Any,
        candidates: Sequence[Mapping[str, Any]],
        *,
        fontsize: float,
        multiline: bool,
    ) -> None:
        """Place short labels with deterministic screen-space avoidance."""
        if not candidates:
            return
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        axes_box = ax.get_window_extent(renderer=renderer)
        points_to_pixels = fig.dpi / 72.0
        placed_boxes: list[tuple[float, float, float, float]] = []
        vertical_offsets = (
            (10, -10, 28, -28, 46, -46, 64, -64)
            if multiline
            else (0, 12, -12, 24, -24, 36, -36, 48, -48)
        )

        def overlap_area(
            left: tuple[float, float, float, float],
            right: tuple[float, float, float, float],
        ) -> float:
            width = max(
                0.0, min(left[2], right[2]) - max(left[0], right[0])
            )
            height = max(
                0.0, min(left[3], right[3]) - max(left[1], right[1])
            )
            return width * height

        def outside_distance(
            box: tuple[float, float, float, float],
        ) -> float:
            return (
                max(0.0, axes_box.x0 - box[0])
                + max(0.0, box[2] - axes_box.x1)
                + max(0.0, axes_box.y0 - box[1])
                + max(0.0, box[3] - axes_box.y1)
            )

        transformed = [
            (
                candidate,
                ax.transData.transform(
                    (float(candidate["x"]), float(candidate["y"]))
                ),
            )
            for candidate in candidates
        ]
        for candidate, (anchor_x, anchor_y) in sorted(
            transformed,
            key=lambda item: (item[1][0], item[1][1], str(item[0]["text"])),
        ):
            text = str(candidate["text"])
            lines = text.splitlines() or [text]
            width = (
                max(len(line) for line in lines) * fontsize * 0.58 + 5
            ) * points_to_pixels
            height = (
                len(lines) * fontsize * 1.35 + 4
            ) * points_to_pixels
            x_fraction = (
                (anchor_x - axes_box.x0) / max(1.0, axes_box.width)
            )
            if x_fraction < 0.22:
                side_order = ("right",)
            elif x_fraction > 0.78:
                side_order = ("left",)
            else:
                side_order = ("right", "left")
            options: list[
                tuple[
                    float,
                    str,
                    int,
                    tuple[float, float, float, float],
                ]
            ] = []
            horizontal_offset = 5 * points_to_pixels
            for side_index, side in enumerate(side_order):
                for vertical_offset in vertical_offsets:
                    center_y = (
                        anchor_y + vertical_offset * points_to_pixels
                    )
                    if side == "right":
                        left_x = anchor_x + horizontal_offset
                        right_x = left_x + width
                    else:
                        right_x = anchor_x - horizontal_offset
                        left_x = right_x - width
                    box = (
                        left_x,
                        center_y - height / 2,
                        right_x,
                        center_y + height / 2,
                    )
                    overlap = sum(
                        overlap_area(box, placed) for placed in placed_boxes
                    )
                    outside = outside_distance(box)
                    score = (
                        overlap * 100.0
                        + (1_000_000_000.0 if outside > 0.5 else 0.0)
                        + outside * 1_000.0
                        + abs(vertical_offset)
                        + side_index
                    )
                    options.append(
                        (score, side, vertical_offset, box)
                    )
            _, side, vertical_offset, chosen_box = min(
                options, key=lambda option: option[0]
            )
            placed_boxes.append(chosen_box)
            ax.annotate(
                text,
                (candidate["x"], candidate["y"]),
                xytext=(
                    5 if side == "right" else -5,
                    vertical_offset,
                ),
                textcoords="offset points",
                ha="left" if side == "right" else "right",
                va="center",
                color=candidate["color"],
                fontsize=fontsize,
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "facecolor": "white",
                    "edgecolor": candidate.get("edgecolor", "none"),
                    "alpha": 0.82,
                },
            )

    def annual_chart(
        *,
        batch: int,
        metric: str,
        ylabel: str,
        title: str,
        stem: str,
    ) -> None:
        fig, (ax, count_ax) = plt.subplots(
            2,
            1,
            figsize=(13.5, 8.5),
            sharex=True,
            gridspec_kw={"height_ratios": (3.4, 1.0)},
        )
        label_candidates: list[dict[str, Any]] = []
        batch_rows = envelope_groups[batch]
        for year in year_values:
            series = sorted(
                (
                    row
                    for row in batch_rows
                    if row["year"] == year
                    and row["envelope_scope"]
                    == "all_representative_models"
                ),
                key=lambda row: row["context_C"],
            )
            if not series:
                continue
            segments = _split_annual_series_by_eligible_cohort(series)
            for segment_index, segment in enumerate(segments):
                ax.plot(
                    [row["context_C"] for row in segment],
                    [row[f"max_{metric}"] for row in segment],
                    color=year_colors[year],
                    linewidth=2,
                    label=str(year) if segment_index == 0 else "_nolegend_",
                )
                count_ax.plot(
                    [row["context_C"] for row in segment],
                    [row["eligible_model_count"] for row in segment],
                    color=year_colors[year],
                    linewidth=1.4,
                    marker="o",
                    markersize=3.5,
                )
                if segment_index:
                    boundary = segment[0]
                    previous_count = segments[segment_index - 1][-1][
                        "eligible_model_count"
                    ]
                    current_count = boundary["eligible_model_count"]
                    ax.axvline(
                        boundary["context_C"],
                        color=year_colors[year],
                        linestyle=":",
                        linewidth=0.8,
                        alpha=0.38,
                    )
                    transition = (
                        f"{previous_count}→{current_count}"
                        if previous_count != current_count
                        else f"cohort changed (n={current_count})"
                    )
                    count_ax.annotate(
                        transition,
                        (
                            boundary["context_C"],
                            boundary["eligible_model_count"],
                        ),
                        xytext=(3, 5),
                        textcoords="offset points",
                        color=year_colors[year],
                        fontsize=6.5,
                    )

            previous_cohort: tuple[str, ...] | None = None
            previous_winners: tuple[str, ...] | None = None
            for row in series:
                scatter_by_support(
                    ax,
                    x=float(row["context_C"]),
                    y=float(row[f"max_{metric}"]),
                    color=year_colors[year],
                    status=str(row[f"max_{metric}_support"]),
                )
                cohort = _eligible_model_key(row)
                winner_names = tuple(
                    json.loads(row[f"max_{metric}_short_model_names"])
                )
                if (
                    previous_cohort is None
                    or cohort != previous_cohort
                    or winner_names != previous_winners
                ):
                    compact_winners = "/".join(
                        _chart_model_name(name) for name in winner_names
                    )
                    label_candidates.append(
                        {
                            "text": (
                            f"{compact_winners}\n"
                            f"n={row['eligible_model_count']}"
                            ),
                            "x": row["context_C"],
                            "y": row[f"max_{metric}"],
                            "color": year_colors[year],
                            "edgecolor": year_colors[year],
                        }
                    )
                previous_cohort = cohort
                previous_winners = winner_names

        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_ylabel(ylabel)
        ax.set_title(
            f"{title}\n"
            "Pointwise maximum over eligible models; no mean or weighting; "
            "gaps mean the eligible cohort changed"
        )
        ax.grid(which="both", alpha=0.25)
        annotate_without_overlap(
            fig,
            ax,
            label_candidates,
            fontsize=6.5,
            multiline=True,
        )
        legend_handles = [
            Line2D([0], [0], color=year_colors[year], label=str(year))
            for year in year_values
        ] + [
            Line2D(
                [0],
                [0],
                marker="o",
                color="#555555",
                markerfacecolor="#555555",
                linestyle="none",
                label="P3 supported max",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="#555555",
                markerfacecolor="white",
                linestyle="none",
                label="P3 partially supported max",
            ),
        ]
        ax.legend(
            handles=legend_handles,
            title="Release year / support",
            ncol=2,
            fontsize=7.5,
        )
        count_ax.set_xscale("log", base=2)
        count_ax.set_xlabel("Start-of-step context C (tokens)")
        count_ax.set_ylabel("Eligible\nmodels")
        count_ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        count_ax.grid(which="both", alpha=0.2)
        save(fig, stem)

    for batch in SELECTED_BATCHES:
        annual_chart(
            batch=batch,
            metric="per_token_total_flops",
            ylabel="Annual sample maximum FLOPs / token",
            title=(
                "Dynamic Native-Cohort Conditional Compute Upper Envelope "
                f"(B={batch})"
            ),
            stem=f"annual_envelope_flops_B{batch}",
        )
        annual_chart(
            batch=batch,
            metric="per_token_logical_hbm_bytes",
            ylabel="Annual sample maximum logical-HBM bytes / token",
            title=(
                "Dynamic Native-Cohort Conditional Logical-HBM Upper "
                f"Envelope (B={batch})"
            ),
            stem=f"annual_envelope_logical_hbm_B{batch}",
        )

    def fixed_context_chart(
        *,
        batch: int,
        metric: str,
        support_kind: str,
        xlabel: str,
        title: str,
        stem: str,
    ) -> None:
        rows = [
            row
            for row in fixed_context_comparison
            if row["concurrency_B"] == batch
        ]
        fig, axes_grid = plt.subplots(
            2, 3, figsize=(15, 8.5), sharex=True
        )
        axes = list(axes_grid.flat)
        for axis, year in zip(axes, year_values):
            group = sorted(
                (row for row in rows if row["year"] == year),
                key=lambda row: (
                    str(row["release_date"]),
                    str(row["short_model_name"]),
                ),
            )
            positions = list(range(len(group)))
            maximum = max(float(row[metric]) for row in group)
            for position, row in zip(positions, group):
                value = float(row[metric])
                status = _metric_support(row, support_kind)
                scatter_by_support(
                    axis,
                    x=value,
                    y=float(position),
                    color=year_colors[year],
                    status=status,
                    size=58,
                )
                if _isclose(value, maximum):
                    axis.scatter(
                        [value],
                        [position],
                        marker="D",
                        facecolors="none",
                        edgecolors="#111111",
                        linewidths=1.35,
                        s=105,
                        zorder=5,
                    )
            axis.set_yticks(
                positions,
                [
                    _chart_model_name(str(row["short_model_name"]))
                    for row in group
                ],
            )
            axis.set_xscale("log")
            axis.set_title(f"{year} (n={len(group)})")
            axis.grid(axis="x", which="both", alpha=0.25)
            axis.tick_params(axis="y", labelsize=8)
        legend_axis = axes[-1]
        legend_axis.axis("off")
        legend_axis.legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="#555555",
                    markerfacecolor="#555555",
                    linestyle="none",
                    label="one model, P3 supported",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="#555555",
                    markerfacecolor="white",
                    linestyle="none",
                    label="one model, P3 partially supported",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="D",
                    color="#111111",
                    markerfacecolor="none",
                    linestyle="none",
                    label="annual maximum",
                ),
            ],
            loc="center",
            title="Point meaning",
        )
        fig.suptitle(
            f"{title}\n"
            f"Common workload C={FIXED_COMPARISON_CONTEXT:,}; "
            "every point is one model; no averaging or weighting",
            fontweight="bold",
        )
        fig.supxlabel(xlabel)
        save(fig, stem)

    fixed_context_chart(
        batch=1,
        metric="per_token_total_flops",
        support_kind="flops",
        xlabel="FLOPs / output token",
        title="Fixed-Context Decode Compute by Release Year",
        stem=f"fixed_context_C{FIXED_COMPARISON_CONTEXT}_flops_by_year",
    )
    for batch in SELECTED_BATCHES:
        fixed_context_chart(
            batch=batch,
            metric="per_token_logical_hbm_bytes",
            support_kind="traffic",
            xlabel="Logical-HBM bytes / output token",
            title=(
                "Fixed-Context Decode Logical-HBM by Release Year "
                f"(B={batch})"
            ),
            stem=(
                f"fixed_context_C{FIXED_COMPARISON_CONTEXT}_"
                f"logical_hbm_B{batch}_by_year"
            ),
        )

    def advertised_ceiling_chart(
        *,
        batch: int,
        metric: str,
        support_kind: str,
        ylabel: str,
        title: str,
        stem: str,
    ) -> None:
        rows = [
            row
            for row in native_max_points
            if row["concurrency_B"] == batch
        ]
        fig, axes_grid = plt.subplots(
            2, 3, figsize=(15, 8.5), sharex=True, sharey=True
        )
        axes = list(axes_grid.flat)
        groups_by_year: dict[int, list[dict[str, Any]]] = {}
        for axis, year in zip(axes, year_values):
            group = sorted(
                (row for row in rows if row["year"] == year),
                key=lambda row: (
                    int(row["context_C"]),
                    str(row["release_date"]),
                ),
            )
            groups_by_year[year] = group
            for row in group:
                context = float(row["context_C"])
                value = float(row[metric])
                status = _metric_support(row, support_kind)
                scatter_by_support(
                    axis,
                    x=context,
                    y=value,
                    color=year_colors[year],
                    status=status,
                    size=58,
                )
            axis.set_xscale("log", base=2)
            axis.set_yscale("log")
            axis.set_title(f"{year} (n={len(group)})")
            axis.grid(which="both", alpha=0.24)
        for axis, year in zip(axes, year_values):
            annotate_without_overlap(
                fig,
                axis,
                [
                    {
                        "text": _chart_model_name(
                            str(row["short_model_name"])
                        ),
                        "x": row["context_C"],
                        "y": row[metric],
                        "color": year_colors[year],
                    }
                    for row in groups_by_year[year]
                ],
                fontsize=7.2,
                multiline=False,
            )
        legend_axis = axes[-1]
        legend_axis.axis("off")
        legend_axis.legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="#555555",
                    markerfacecolor="#555555",
                    linestyle="none",
                    label="P3 supported",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="#555555",
                    markerfacecolor="white",
                    linestyle="none",
                    label="P3 partially supported",
                ),
            ],
            loc="center",
            title="Each point is one model",
        )
        fig.suptitle(
            f"{title}\n"
            "Each model uses its own advertised context ceiling; "
            "different C means pressure comparison, not a common workload",
            fontweight="bold",
        )
        fig.supxlabel("Own advertised max context C (tokens)")
        fig.supylabel(ylabel)
        save(fig, stem)

    advertised_ceiling_chart(
        batch=1,
        metric="per_token_total_flops",
        support_kind="flops",
        ylabel="FLOPs / output token",
        title="Decode Compute at Each Model's Advertised Context Ceiling",
        stem="advertised_ceiling_flops_by_model",
    )
    for batch in SELECTED_BATCHES:
        advertised_ceiling_chart(
            batch=batch,
            metric="per_token_logical_hbm_bytes",
            support_kind="traffic",
            ylabel="Logical-HBM bytes / output token",
            title=(
                "Decode Logical-HBM at Each Model's Advertised Context "
                f"Ceiling (B={batch})"
            ),
            stem=(
                f"advertised_ceiling_logical_hbm_B{batch}_by_model"
            ),
        )
    return artifacts


def _run(
    release_dir: Path,
    output_dir: Path,
    *,
    render_plots: bool,
    dpi: int,
) -> dict[str, Any]:
    release_dir = release_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir == release_dir or release_dir in output_dir.parents:
        raise ValueError("output directory must not be inside the frozen release")

    (
        rows,
        profiles,
        release_manifest,
        run_manifest,
        validation_report,
        checks,
    ) = _load_release(release_dir)
    native_rows = _native_main_rows(rows)
    model_summary = _model_summary_rows(profiles)
    annual_envelope = _annual_envelope_rows(native_rows, profiles)
    fixed_context_comparison = _fixed_context_comparison_rows(
        native_rows, profiles
    )
    native_max_points = _native_max_point_rows(
        native_rows, profiles
    )
    context_boundary_points = _context_boundary_rows(native_rows, profiles)
    component_shares = _component_share_rows(native_max_points)
    crossovers = _crossover_rows(native_rows, profiles)
    quality = _quality_summary(
        rows, native_rows, profiles, validation_report, checks
    )

    _clear_previous_outputs(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table_artifacts = {
        "quality_summary": "quality_summary.json",
        "model_summary": "model_summary.csv",
        "annual_envelope": "annual_envelope.csv",
        "fixed_context_comparison": "fixed_context_comparison.csv",
        "native_max_points": "native_max_points.csv",
        "context_boundary_points": "context_boundary_points.csv",
        "component_shares": "component_shares.csv",
        "crossover_points": "crossover_points.csv",
    }
    _write_json(quality, output_dir / table_artifacts["quality_summary"])
    _write_csv(model_summary, output_dir / table_artifacts["model_summary"])
    _write_csv(
        annual_envelope, output_dir / table_artifacts["annual_envelope"]
    )
    _write_csv(
        fixed_context_comparison,
        output_dir / table_artifacts["fixed_context_comparison"],
    )
    _write_csv(
        native_max_points, output_dir / table_artifacts["native_max_points"]
    )
    _write_csv(
        context_boundary_points,
        output_dir / table_artifacts["context_boundary_points"],
    )
    _write_csv(
        component_shares, output_dir / table_artifacts["component_shares"]
    )
    _write_csv(crossovers, output_dir / table_artifacts["crossover_points"])

    figure_artifacts: list[str] = []
    if render_plots:
        figure_artifacts = _render_charts(
            native_rows=native_rows,
            model_summary=model_summary,
            annual_envelope=annual_envelope,
            fixed_context_comparison=fixed_context_comparison,
            native_max_points=native_max_points,
            profiles=profiles,
            output_dir=output_dir,
            dpi=dpi,
        )

    created_at = (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    manifest = {
        "analysis_manifest_schema_version": 1,
        "analysis_id": "decode-trend-p8-industry-envelope-v0.2",
        "script_version": SCRIPT_VERSION,
        "analysis_script": {
            "path": str(Path(__file__).resolve().relative_to(PROJECT_ROOT)),
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "created_at_utc": created_at,
        "source_release": {
            "path": str(release_dir),
            "dataset_version": release_manifest["dataset_version"],
            "study_id": release_manifest["study_id"],
            "run_id": run_manifest["run_id"],
            "git_commit": release_manifest["source_repository"]["git_commit"],
            "decode_results_csv_sha256": _sha256(
                release_dir / "data" / "decode_results.csv"
            ),
            "model_profiles_jsonl_sha256": _sha256(
                release_dir / "data" / "model_profiles.jsonl"
            ),
        },
        "scope": {
            "phase": "decode",
            "batch_class": "main",
            "is_extrapolated": False,
            "within_advertised_context": True,
            "partially_supported_policy": (
                "included with dimension-specific P3 status"
            ),
            "annual_envelope_context_grid": "common_power_of_two only",
            "fixed_comparison_context_C": FIXED_COMPARISON_CONTEXT,
            "selected_chart_batches": list(SELECTED_BATCHES),
        },
        "row_counts": {
            "source_results": len(rows),
            "native_main_results": len(native_rows),
            "model_summary": len(model_summary),
            "annual_envelope": len(annual_envelope),
            "fixed_context_comparison": len(fixed_context_comparison),
            "native_max_points": len(native_max_points),
            "context_boundary_points": len(context_boundary_points),
            "component_shares": len(component_shares),
            "crossover_points": len(crossovers),
        },
        "artifacts": {
            **table_artifacts,
            "figures": figure_artifacts,
        },
    }
    _write_json(manifest, output_dir / "analysis_manifest.json")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = _run(
            args.release_dir,
            args.output_dir,
            render_plots=not args.no_plots,
            dpi=args.dpi,
        )
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    counts = manifest["row_counts"]
    print(
        "analyzed "
        f"{counts['source_results']} source rows, "
        f"{counts['native_main_results']} native main rows, "
        f"{counts['model_summary']} models; "
        f"wrote results to {args.output_dir.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
