#!/usr/bin/env python3
"""Generate the requested decode Byte/FLOP data and comparison charts.

The script intentionally keeps data generation separate from rendering:

* CSV files contain the complete auditable per-output-token cost breakdown.
* PNG and SVG files plot ``TB/s per PFLOPS``, which is exactly
  ``1000 * Byte/FLOP`` and therefore has the same curve shape.
* Contexts beyond a model's configured maximum are still calculated for
  architectural comparison, but are marked as extrapolated in CSV and drawn
  with dashed line segments.

Run from any directory; paths default to this repository's ``configs`` and
``outputs/decode_ratio`` directories.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

# Use a non-interactive renderer so the script works over SSH and in CI.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import StrMethodFormatter  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from decode_engine.config import EngineConfig, load_engine_config  # noqa: E402
from decode_engine.engine import calculate_decode  # noqa: E402

PROFILE_LABEL = "16-bit weight+KV"

CONTEXTS = (
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
    131072,
    262144,
    524288,
    1048576,
)
CONTEXT_LABELS = (
    "128",
    "256",
    "512",
    "1K",
    "2K",
    "4K",
    "8K",
    "16K",
    "32K",
    "64K",
    "128K",
    "256K",
    "512K",
    "1024K",
)
BATCHES = (1, 32)


@dataclass(frozen=True)
class ModelSeries:
    """Display identity and source config for one plotted model."""

    model_id: str
    label: str
    config_filename: str
    color: str
    marker: str


# The list deliberately excludes synthetic/example configs in ``configs``.
MODEL_SERIES = (
    ModelSeries(
        "deepseek_r1",
        "DeepSeek-R1",
        "deepseek_r1_mla_16bit.json",
        "#0072B2",
        "o",
    ),
    ModelSeries(
        "deepseek_v4_pro",
        "DeepSeek-V4-Pro",
        "deepseek_v4_pro_16bit.json",
        "#D55E00",
        "s",
    ),
    ModelSeries(
        "glm_5_2",
        "GLM-5.2",
        "glm_5_2_dsa_16bit.json",
        "#009E73",
        "^",
    ),
    ModelSeries(
        "qwen3_235b_a22b",
        "Qwen3-235B-A22B",
        "qwen3_235b_a22b_16bit.json",
        "#CC79A7",
        "D",
    ),
    ModelSeries(
        "llama_3_3_70b",
        "Llama-3.3-70B",
        "llama_3_3_70b_16bit.json",
        "#E69F00",
        "v",
    ),
    ModelSeries(
        "qwen3_8b",
        "Qwen3-8B",
        "qwen3_8b_16bit.json",
        "#56B4E9",
        "P",
    ),
    ModelSeries(
        "qwen3_4b",
        "Qwen3-4B",
        "qwen3_4b_16bit.json",
        "#8C6D1F",
        "X",
    ),
    ModelSeries(
        "qwen3_next_80b_a3b",
        "Qwen3-Next-80B-A3B",
        "qwen3_next_80b_a3b_16bit.json",
        "#6F4E7C",
        ">",
    ),
    ModelSeries(
        "mamba_2_8b",
        "Mamba-2.8B",
        "mamba_2_8b_16bit.json",
        "#2B2B2B",
        "h",
    ),
)


CSV_FIELDS = (
    "model_id",
    "model_label",
    "model_name",
    "config_file",
    "batch_size",
    "context_tokens",
    "context_label",
    "official_max_context_tokens",
    "is_extrapolated",
    "active_parameters",
    "parameter_flops_per_token",
    "attention_flops_per_token",
    "index_flops_per_token",
    "state_flops_per_token",
    "extra_flops_per_token",
    "total_flops_per_token",
    "weight_read_bytes_per_token",
    "kv_read_bytes_per_token",
    "kv_write_bytes_per_token",
    "index_read_bytes_per_token",
    "index_write_bytes_per_token",
    "state_read_bytes_per_token",
    "state_write_bytes_per_token",
    "activation_bytes_per_token",
    "other_read_bytes_per_token",
    "total_bytes_per_token",
    "bytes_per_flop",
    "tbps_per_pflops",
    "kv_cache_bytes_per_request",
    "index_cache_bytes_per_request",
    "state_cache_bytes_per_request",
    "total_cache_bytes_per_request",
    "expert_weight_sets_read",
)


def _active_parameters(config: EngineConfig) -> float:
    """Return the engine's matrix-parameter count for one output token."""

    weights = config.model.weights
    routed = sum(
        group.layers
        * group.selected_per_token
        * group.parameters_per_expert
        for group in weights.routed_expert_groups
    )
    return weights.always_active_parameters + routed


def calculate_rows(
    config_dir: Path,
    contexts: Sequence[int] = CONTEXTS,
    batches: Sequence[int] = BATCHES,
) -> list[dict[str, Any]]:
    """Calculate every requested model/context/batch point.

    ``calculate_decode`` normally rejects contexts above the configured model
    limit.  For a requested architecture comparison we explicitly remove that
    validation limit only for the affected point.  The original limit and an
    extrapolation flag remain in every row, so downstream consumers cannot
    silently confuse the theoretical extension with an officially supported
    context.
    """

    context_labels = {
        context: label for context, label in zip(CONTEXTS, CONTEXT_LABELS)
    }
    rows: list[dict[str, Any]] = []

    for series in MODEL_SERIES:
        config_path = config_dir / series.config_filename
        config = load_engine_config(config_path)
        max_context = config.model.max_context_tokens
        active_parameters = _active_parameters(config)

        for batch in batches:
            for context in contexts:
                is_extrapolated = (
                    max_context is not None and context > max_context
                )
                model = config.model
                if is_extrapolated:
                    model = replace(model, max_context_tokens=None)

                result = calculate_decode(
                    model,
                    config.deployment,
                    [context] * batch,
                )
                work = result.per_output_work
                cache = result.cache_capacity_per_request_average

                rows.append(
                    {
                        "model_id": series.model_id,
                        "model_label": series.label,
                        "model_name": config.model.name,
                        "config_file": series.config_filename,
                        "batch_size": batch,
                        "context_tokens": context,
                        "context_label": context_labels.get(
                            context, str(context)
                        ),
                        "official_max_context_tokens": (
                            "" if max_context is None else max_context
                        ),
                        "is_extrapolated": str(is_extrapolated).lower(),
                        "active_parameters": active_parameters,
                        "parameter_flops_per_token": work.parameter_flops,
                        "attention_flops_per_token": work.attention_flops,
                        "index_flops_per_token": work.index_flops,
                        "state_flops_per_token": work.state_flops,
                        "extra_flops_per_token": work.extra_flops,
                        "total_flops_per_token": work.total_flops,
                        "weight_read_bytes_per_token": work.weight_read_bytes,
                        "kv_read_bytes_per_token": work.kv_read_bytes,
                        "kv_write_bytes_per_token": work.kv_write_bytes,
                        "index_read_bytes_per_token": work.index_read_bytes,
                        "index_write_bytes_per_token": work.index_write_bytes,
                        "state_read_bytes_per_token": work.state_read_bytes,
                        "state_write_bytes_per_token": work.state_write_bytes,
                        "activation_bytes_per_token": work.activation_bytes,
                        "other_read_bytes_per_token": work.other_read_bytes,
                        "total_bytes_per_token": work.total_bytes,
                        "bytes_per_flop": result.bytes_per_flop,
                        "tbps_per_pflops": result.tbps_per_pflops,
                        "kv_cache_bytes_per_request": cache.kv_bytes,
                        "index_cache_bytes_per_request": cache.index_bytes,
                        "state_cache_bytes_per_request": cache.state_bytes,
                        "total_cache_bytes_per_request": cache.total_bytes,
                        "expert_weight_sets_read": json.dumps(
                            result.expert_weight_sets_read,
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                    }
                )

    return rows


def write_csv(rows: Iterable[dict[str, Any]], path: Path) -> None:
    """Write deterministic CSV with a stable, explicit column order."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _series_rows(
    rows: Sequence[dict[str, Any]], model_id: str, batch: int
) -> list[dict[str, Any]]:
    return sorted(
        (
            row
            for row in rows
            if row["model_id"] == model_id
            and row["batch_size"] == batch
        ),
        key=lambda row: int(row["context_tokens"]),
    )


def render_chart(
    rows: Sequence[dict[str, Any]], batch: int, output_stem: Path
) -> None:
    """Render one batch chart in both PNG and vector SVG formats."""

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titleweight": "bold",
            "axes.labelcolor": "#222222",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "text.color": "#222222",
        }
    )
    fig, ax = plt.subplots(figsize=(15, 8.5), facecolor="white")
    ax.set_facecolor("white")

    for series in MODEL_SERIES:
        points = _series_rows(rows, series.model_id, batch)
        supported = [
            point
            for point in points
            if point["is_extrapolated"] == "false"
        ]
        extrapolated = [
            point
            for point in points
            if point["is_extrapolated"] == "true"
        ]

        if supported:
            ax.plot(
                [point["context_tokens"] for point in supported],
                [point["tbps_per_pflops"] for point in supported],
                color=series.color,
                marker=series.marker,
                markersize=5.5,
                linewidth=2.1,
                label=series.label,
            )

        if extrapolated:
            # Include the last supported point to make the style transition
            # continuous exactly at the boundary.
            extension = supported[-1:] + extrapolated
            ax.plot(
                [point["context_tokens"] for point in extension],
                [point["tbps_per_pflops"] for point in extension],
                color=series.color,
                linestyle=(0, (4, 3)),
                marker=series.marker,
                markerfacecolor="white",
                markeredgewidth=1.2,
                markersize=5.5,
                linewidth=1.8,
                label=series.label if not supported else None,
            )

    ax.set_xscale("log", base=2)
    ax.set_xlim(CONTEXTS[0], CONTEXTS[-1])
    ax.set_xticks(CONTEXTS, labels=CONTEXT_LABELS)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))

    ax.set_xlabel("Existing context length (tokens, log2 scale)", labelpad=12)
    ax.set_ylabel("Bandwidth per compute (TB/s per PFLOPS)", labelpad=12)
    ax.set_title(
        f"LLM decode data-movement / compute ratio | {PROFILE_LABEL} | "
        f"Batch = {batch}",
        loc="left",
        fontsize=18,
        pad=26,
    )
    ax.text(
        0.0,
        1.015,
        "Per generated token. Solid = within configured context limit; "
        "dashed = theoretical extrapolation. 1 TB/s per PFLOPS = "
        "0.001 Byte/FLOP.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.5,
        color="#555555",
    )

    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.8)
    ax.grid(axis="x", color="#E8E8E8", linewidth=0.6, alpha=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#888888")
    ax.spines["bottom"].set_color("#888888")
    ax.tick_params(axis="x", rotation=0, pad=7)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=3,
        frameon=False,
        fontsize=10.5,
        handlelength=2.8,
        columnspacing=2.0,
    )
    fig.subplots_adjust(left=0.09, right=0.975, top=0.86, bottom=0.22)

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_stem.with_suffix(".png"),
        dpi=220,
        facecolor="white",
    )
    fig.savefig(output_stem.with_suffix(".svg"), facecolor="white")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate EBpC decode-ratio CSV data and batch charts."
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=PROJECT_ROOT / "configs" / "16bit",
        help="Directory containing model JSON configs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "decode_ratio" / "16bit",
        help="Destination for CSV, PNG, and SVG files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = calculate_rows(args.config_dir)

    write_csv(rows, args.output_dir / "decode_ratio_all.csv")
    for batch in BATCHES:
        batch_rows = [row for row in rows if row["batch_size"] == batch]
        write_csv(
            batch_rows,
            args.output_dir / f"decode_ratio_batch_{batch}.csv",
        )
        render_chart(
            rows,
            batch,
            args.output_dir / f"decode_ratio_batch_{batch}",
        )

    print(
        f"Generated {len(rows)} rows for {len(MODEL_SERIES)} models in "
        f"{args.output_dir}"
    )


if __name__ == "__main__":
    main()
