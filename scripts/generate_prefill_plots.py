#!/usr/bin/env python3
"""Generate auditable Prefill data, CSV files, and comparison charts.

The three standard experiments use one shared calculation path:

* ``equal`` fixes the request batch and sweeps equal prompt lengths;
* ``token-budget`` fixes total input tokens and changes request count;
* ``ragged`` evaluates explicit length vectors and execution layouts.

Every chart is derived from the CSV/JSON records written by the same run.
The detailed CSV retains both logical/compulsory HBM traffic and the separate
pair-stream operand boundary.  These are alternative traffic interpretations
and must not be added together.

Matplotlib is optional for the core engine.  Install ``requirements-plot.txt``
before running this script.  The Agg renderer and a writable temporary config
directory make rendering work over SSH and in CI.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Matplotlib resolves its configuration directory during import.  The normal
# home directory may be read-only in a container, so choose a stable per-user
# directory under the system temporary root unless the caller supplied one.
if "MPLCONFIGDIR" not in os.environ:
    getuid = getattr(os, "getuid", lambda: "default")
    mpl_config_dir = (
        Path(tempfile.gettempdir()) / f"bpc_engine_matplotlib_{getuid()}"
    )
    mpl_config_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import StrMethodFormatter  # noqa: E402

from decode_engine.config import ConfigurationError, load_engine_config  # noqa: E402
from decode_engine.engine import calculate_prefill  # noqa: E402
from decode_engine.schema import PrefillResult, WorkCost  # noqa: E402


DEFAULT_PROMPT_LENGTHS = (128, 512, 2048, 8192)
DEFAULT_EQUAL_BATCHES = (1, 32)
DEFAULT_TOKEN_BUDGETS = (4096, 16384)
DEFAULT_TOKEN_BUDGET_BATCHES = (1, 4, 16, 32)
DEFAULT_RAGGED_BATCHES = (
    (128, 512, 2048, 64),
    (256, 256, 1024, 4096),
)
DEFAULT_RAGGED_EXECUTION_MODES = ("varlen", "padded")
EXPERIMENTS = ("equal", "token-budget", "ragged")


@dataclass(frozen=True)
class ModelSeries:
    """Stable display identity and precision-parameterized config name."""

    model_id: str
    label: str
    config_template: str
    color: str
    marker: str

    def config_filename(self, precision: int) -> str:
        return self.config_template.format(bits=precision)


MODEL_SERIES = (
    ModelSeries(
        "deepseek_r1",
        "DeepSeek-R1",
        "deepseek_r1_mla_{bits}bit.json",
        "#0072B2",
        "o",
    ),
    ModelSeries(
        "deepseek_v4_pro",
        "DeepSeek-V4-Pro",
        "deepseek_v4_pro_{bits}bit.json",
        "#D55E00",
        "s",
    ),
    ModelSeries(
        "glm_5_2",
        "GLM-5.2",
        "glm_5_2_dsa_{bits}bit.json",
        "#009E73",
        "^",
    ),
    ModelSeries(
        "qwen3_235b_a22b",
        "Qwen3-235B-A22B",
        "qwen3_235b_a22b_{bits}bit.json",
        "#CC79A7",
        "D",
    ),
    ModelSeries(
        "llama_3_3_70b",
        "Llama-3.3-70B",
        "llama_3_3_70b_{bits}bit.json",
        "#E69F00",
        "v",
    ),
    ModelSeries(
        "qwen3_8b",
        "Qwen3-8B",
        "qwen3_8b_{bits}bit.json",
        "#56B4E9",
        "P",
    ),
    ModelSeries(
        "qwen3_4b",
        "Qwen3-4B",
        "qwen3_4b_{bits}bit.json",
        "#8C6D1F",
        "X",
    ),
    ModelSeries(
        "qwen3_next_80b_a3b",
        "Qwen3-Next-80B-A3B",
        "qwen3_next_80b_a3b_{bits}bit.json",
        "#6F4E7C",
        ">",
    ),
    ModelSeries(
        "mamba_2_8b",
        "Mamba-2.8B",
        "mamba_2_8b_{bits}bit.json",
        "#2B2B2B",
        "h",
    ),
)


@dataclass(frozen=True)
class PrefillPoint:
    """One calculated point plus plotting/provenance metadata."""

    precision_bits: int
    series: ModelSeries
    config_file: str
    trace_id: str
    x_value: int
    official_max_context_tokens: int | None
    is_extrapolated: bool
    result: PrefillResult

    def metadata(self) -> dict[str, Any]:
        return {
            "precision_bits": self.precision_bits,
            "profile": f"{self.precision_bits}-bit weight+KV",
            "model_id": self.series.model_id,
            "model_label": self.series.label,
            "model_name": self.result.model_name,
            "config_file": self.config_file,
            "trace_id": self.trace_id,
            "x_value": self.x_value,
            "official_max_context_tokens": (
                ""
                if self.official_max_context_tokens is None
                else self.official_max_context_tokens
            ),
            "is_extrapolated": str(self.is_extrapolated).lower(),
        }

    def json_record(self) -> dict[str, Any]:
        metadata = self.metadata()
        metadata["is_extrapolated"] = self.is_extrapolated
        metadata["official_max_context_tokens"] = (
            self.official_max_context_tokens
        )
        return {**metadata, "result": self.result.to_dict()}


def _prefixed_work(prefix: str, work: WorkCost) -> dict[str, float]:
    return {f"{prefix}{name}": value for name, value in work.to_dict().items()}


def detail_row(point: PrefillPoint) -> dict[str, Any]:
    """Flatten a result without losing any normalization boundary."""

    result = point.result
    cache_total = result.cache_capacity_total
    cache_average = result.cache_capacity_per_request_average
    row: dict[str, Any] = {
        **point.metadata(),
        "phase": "prefill",
        "experiment": result.experiment,
        "execution_mode": result.execution_mode,
        "logits_mode": result.logits_mode,
        "include_self_attention": result.include_self_attention,
        "batch": result.batch_size,
        "prompt_lengths": json.dumps(list(result.prompt_tokens)),
        "cached_lengths": json.dumps(list(result.cached_context_tokens)),
        "average_prompt_tokens": result.average_prompt_tokens,
        "max_prompt_tokens": result.max_prompt_tokens,
        "valid_input_tokens": result.valid_input_tokens,
        "executed_input_tokens": result.executed_input_tokens,
        "valid_causal_pair_slots": result.valid_causal_pair_slots,
        "executed_causal_pair_slots": result.executed_causal_pair_slots,
        "valid_logit_positions": result.valid_logit_positions,
        "executed_logit_positions": result.executed_logit_positions,
        "output_head_parameters": result.output_head_parameters,
        "output_head_parameters_configured": (
            result.output_head_parameters_configured
        ),
        "output_head_weight_bits": result.output_head_weight_bits,
        "topk_cached_prefix_union_policy": (
            result.topk_cached_prefix_union_policy
        ),
        "token_efficiency": result.token_efficiency,
        "causal_pair_efficiency": result.causal_pair_efficiency,
        "total_flops": result.batch_work.total_flops,
        "total_compulsory_bytes": result.batch_work.total_bytes,
        "total_operand_bytes": result.batch_operand_work.total_bytes,
        "flops_per_input_token": result.per_input_work.total_flops,
        "compulsory_bytes_per_input_token": (
            result.per_input_work.total_bytes
        ),
        "operand_bytes_per_input_token": (
            result.per_input_operand_work.total_bytes
        ),
        "bytes_per_flop": result.bytes_per_flop,
        "tbps_per_pflops": result.tbps_per_pflops,
        "operand_bytes_per_flop": result.operand_bytes_per_flop,
        "operand_tbps_per_pflops": result.operand_tbps_per_pflops,
        "cache_bytes_total": cache_total.total_bytes,
        "kv_cache_bytes_total": cache_total.kv_bytes,
        "index_cache_bytes_total": cache_total.index_bytes,
        "state_cache_bytes_total": cache_total.state_bytes,
        "cache_bytes_per_request": cache_average.total_bytes,
        "kv_cache_bytes_per_request": cache_average.kv_bytes,
        "index_cache_bytes_per_request": cache_average.index_bytes,
        "state_cache_bytes_per_request": cache_average.state_bytes,
        "expert_weight_sets_read": json.dumps(
            result.expert_weight_sets_read,
            sort_keys=True,
            separators=(",", ":"),
        ),
        "useful_expert_weight_sets_read": json.dumps(
            result.useful_expert_weight_sets_read,
            sort_keys=True,
            separators=(",", ":"),
        ),
    }
    row.update(_prefixed_work("batch_", result.batch_work))
    row.update(_prefixed_work("useful_", result.useful_work))
    row.update(_prefixed_work("per_input_", result.per_input_work))
    row.update(_prefixed_work("batch_operand_", result.batch_operand_work))
    row.update(_prefixed_work("useful_operand_", result.useful_operand_work))
    row.update(
        _prefixed_work("per_input_operand_", result.per_input_operand_work)
    )
    return row


def summary_row(point: PrefillPoint) -> dict[str, Any]:
    """Return the compact, chart-friendly subset of one detail row."""

    result = point.result
    prompt_length = (
        result.prompt_tokens[0]
        if result.experiment == "equal"
        and len(set(result.prompt_tokens)) == 1
        else ""
    )
    token_budget = (
        result.valid_input_tokens
        if result.experiment == "token-budget"
        else ""
    )
    ragged_trace = (
        point.x_value if result.experiment == "ragged" else ""
    )
    return {
        **point.metadata(),
        "experiment": result.experiment,
        "execution_mode": result.execution_mode,
        "logits_mode": result.logits_mode,
        "include_self_attention": result.include_self_attention,
        "batch": result.batch_size,
        "prompt_length": prompt_length,
        "token_budget": token_budget,
        "ragged_trace_index": ragged_trace,
        "prompt_lengths": json.dumps(list(result.prompt_tokens)),
        "valid_input_tokens": result.valid_input_tokens,
        "executed_input_tokens": result.executed_input_tokens,
        "valid_causal_pair_slots": result.valid_causal_pair_slots,
        "executed_causal_pair_slots": result.executed_causal_pair_slots,
        "valid_logit_positions": result.valid_logit_positions,
        "executed_logit_positions": result.executed_logit_positions,
        "token_efficiency": result.token_efficiency,
        "causal_pair_efficiency": result.causal_pair_efficiency,
        "total_flops": result.batch_work.total_flops,
        "total_compulsory_bytes": result.batch_work.total_bytes,
        "total_operand_bytes": result.batch_operand_work.total_bytes,
        "flops_per_input_token": result.per_input_work.total_flops,
        "compulsory_bytes_per_input_token": (
            result.per_input_work.total_bytes
        ),
        "operand_bytes_per_input_token": (
            result.per_input_operand_work.total_bytes
        ),
        "bytes_per_flop": result.bytes_per_flop,
        "tbps_per_pflops": result.tbps_per_pflops,
        "operand_bytes_per_flop": result.operand_bytes_per_flop,
        "operand_tbps_per_pflops": result.operand_tbps_per_pflops,
        "cache_bytes_total": result.cache_capacity_total.total_bytes,
        "cache_bytes_per_request": (
            result.cache_capacity_per_request_average.total_bytes
        ),
    }


def write_csv(rows: Iterable[dict[str, Any]], path: Path) -> None:
    """Write deterministic UTF-8 CSV using the first row's stable order."""

    materialized = list(rows)
    if not materialized:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(materialized[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(materialized)


def write_json(points: Sequence[PrefillPoint], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            [point.json_record() for point in points],
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _normalize_svg(path: Path) -> None:
    """Remove renderer-added line-end spaces from an SVG artifact."""

    text = path.read_text(encoding="utf-8")
    path.write_text(
        "\n".join(line.rstrip() for line in text.splitlines()) + "\n",
        encoding="utf-8",
    )


def _balanced_lengths(token_budget: int, batch_size: int) -> tuple[int, ...]:
    if batch_size > token_budget:
        raise ValueError(
            f"batch={batch_size} exceeds token budget={token_budget}; "
            "every request needs at least one token"
        )
    base, remainder = divmod(token_budget, batch_size)
    return tuple(
        base + (1 if index < remainder else 0)
        for index in range(batch_size)
    )


def _calculate_point(
    *,
    precision: int,
    series: ModelSeries,
    config_file: str,
    config: Any,
    prompt_lengths: tuple[int, ...],
    experiment: str,
    execution_mode: str,
    logits_mode: str,
    include_self_attention: bool,
    trace_id: str,
    x_value: int,
) -> PrefillPoint:
    max_context = config.model.max_context_tokens
    is_extrapolated = (
        max_context is not None and max(prompt_lengths) > max_context
    )
    model = (
        replace(config.model, max_context_tokens=None)
        if is_extrapolated
        else config.model
    )
    result = calculate_prefill(
        model,
        config.deployment,
        prompt_lengths,
        execution_mode=execution_mode,
        logits_mode=logits_mode,
        include_self_attention=include_self_attention,
        experiment=experiment,
    )
    return PrefillPoint(
        precision_bits=precision,
        series=series,
        config_file=config_file,
        trace_id=trace_id,
        x_value=x_value,
        official_max_context_tokens=max_context,
        is_extrapolated=is_extrapolated,
        result=result,
    )


def calculate_points(
    *,
    precision: int,
    config_dir: Path,
    series_values: Sequence[ModelSeries],
    experiments: Sequence[str],
    prompt_lengths: Sequence[int],
    equal_batches: Sequence[int],
    token_budgets: Sequence[int],
    token_budget_batches: Sequence[int],
    ragged_batches: Sequence[Sequence[int]],
    execution_mode: str,
    ragged_execution_modes: Sequence[str],
    logits_mode: str,
    include_self_attention: bool,
) -> list[PrefillPoint]:
    """Calculate all points before writing anything to the output directory."""

    points: list[PrefillPoint] = []
    for series in series_values:
        config_file = series.config_filename(precision)
        config_path = config_dir / config_file
        if not config_path.is_file():
            raise FileNotFoundError(
                f"missing config for {series.model_id}: {config_path}"
            )
        config = load_engine_config(config_path)

        if "equal" in experiments:
            for batch in equal_batches:
                for length in prompt_lengths:
                    points.append(
                        _calculate_point(
                            precision=precision,
                            series=series,
                            config_file=config_file,
                            config=config,
                            prompt_lengths=(length,) * batch,
                            experiment="equal",
                            execution_mode=execution_mode,
                            logits_mode=logits_mode,
                            include_self_attention=include_self_attention,
                            trace_id=f"equal_B{batch}_L{length}",
                            x_value=length,
                        )
                    )

        if "token-budget" in experiments:
            for budget in token_budgets:
                for batch in token_budget_batches:
                    lengths = _balanced_lengths(budget, batch)
                    points.append(
                        _calculate_point(
                            precision=precision,
                            series=series,
                            config_file=config_file,
                            config=config,
                            prompt_lengths=lengths,
                            experiment="token-budget",
                            execution_mode=execution_mode,
                            logits_mode=logits_mode,
                            include_self_attention=include_self_attention,
                            trace_id=f"token_budget_T{budget}_B{batch}",
                            x_value=batch,
                        )
                    )

        if "ragged" in experiments:
            for trace_index, trace in enumerate(ragged_batches, start=1):
                lengths = tuple(trace)
                for mode in ragged_execution_modes:
                    points.append(
                        _calculate_point(
                            precision=precision,
                            series=series,
                            config_file=config_file,
                            config=config,
                            prompt_lengths=lengths,
                            experiment="ragged",
                            execution_mode=mode,
                            logits_mode=logits_mode,
                            include_self_attention=include_self_attention,
                            trace_id=f"ragged_R{trace_index}",
                            x_value=trace_index,
                        )
                    )
    return points


def _compact_token_count(value: int) -> str:
    for divisor, suffix in ((1_048_576, "M"), (1024, "K")):
        if value >= divisor and value % divisor == 0:
            return f"{value // divisor}{suffix}"
    return str(value)


PlotMetric = tuple[str, str, Callable[[PrefillResult], float]]
PLOT_METRICS: tuple[PlotMetric, ...] = (
    (
        "Compute",
        "GFLOP / valid input token",
        lambda result: result.per_input_work.total_flops / 1e9,
    ),
    (
        "Logical HBM traffic",
        "GB / valid input token",
        lambda result: result.per_input_work.total_bytes / 1e9,
    ),
    (
        "Operand-stream boundary",
        "GB / valid input token",
        lambda result: result.per_input_operand_work.total_bytes / 1e9,
    ),
    (
        "Logical bandwidth / compute",
        "TB/s per PFLOPS",
        lambda result: result.tbps_per_pflops,
    ),
)


def _configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titleweight": "bold",
            "axes.labelcolor": "#222222",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "text.color": "#222222",
        }
    )


def _plot_model_series(
    ax: Any,
    points: Sequence[PrefillPoint],
    metric: Callable[[PrefillResult], float],
    *,
    add_labels: bool,
) -> None:
    """Draw supported and theoretical points without hiding provenance."""

    present_ids = {point.series.model_id for point in points}
    for series in MODEL_SERIES:
        if series.model_id not in present_ids:
            continue
        series_points = sorted(
            (
                point
                for point in points
                if point.series.model_id == series.model_id
            ),
            key=lambda point: point.x_value,
        )
        # Draw each segment exactly once: a segment touching an extrapolated
        # endpoint is dashed.  Drawing a solid curve underneath and overlaying
        # dashes would leave solid color visible through the dash gaps.
        for index in range(1, len(series_points)):
            left = series_points[index - 1]
            right = series_points[index]
            extrapolated_segment = (
                left.is_extrapolated or right.is_extrapolated
            )
            ax.plot(
                [left.x_value, right.x_value],
                [metric(left.result), metric(right.result)],
                color=series.color,
                linestyle=(0, (4, 3)) if extrapolated_segment else "-",
                linewidth=1.9,
                label=(
                    series.label
                    if add_labels and index == 1
                    else None
                ),
            )

        supported = [
            point for point in series_points if not point.is_extrapolated
        ]
        if supported:
            ax.scatter(
                [point.x_value for point in supported],
                [metric(point.result) for point in supported],
                marker=series.marker,
                s=32,
                color=series.color,
                zorder=4,
                label=(
                    series.label
                    if add_labels and len(series_points) == 1
                    else None
                ),
            )
        extrapolated = [
            point for point in series_points if point.is_extrapolated
        ]
        if extrapolated:
            ax.scatter(
                [point.x_value for point in extrapolated],
                [metric(point.result) for point in extrapolated],
                marker=series.marker,
                s=35,
                facecolors="white",
                edgecolors=series.color,
                linewidths=1.3,
                zorder=4,
                label=(
                    series.label
                    if add_labels and len(series_points) == 1
                    else None
                ),
            )


def render_work_chart(
    points: Sequence[PrefillPoint],
    *,
    title: str,
    subtitle: str,
    x_label: str,
    x_ticks: Sequence[int],
    x_tick_labels: Sequence[str],
    logarithmic_x: bool,
    output_stem: Path,
    dpi: int,
) -> None:
    """Render compute, both byte boundaries, and the logical ratio."""

    if not points:
        raise ValueError(f"no points supplied for chart {output_stem}")
    _configure_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), facecolor="white")
    flat_axes = list(axes.flat)

    for index, (ax, (metric_title, y_label, metric)) in enumerate(
        zip(flat_axes, PLOT_METRICS)
    ):
        _plot_model_series(ax, points, metric, add_labels=index == 0)
        if logarithmic_x and len(set(x_ticks)) > 1:
            ax.set_xscale("log", base=2)
        ax.set_xticks(x_ticks, labels=x_tick_labels)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(metric_title, loc="left", fontsize=12)
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.3g}"))
        ax.grid(axis="both", color="#DDDDDD", linewidth=0.7, alpha=0.75)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(title, x=0.06, ha="left", fontsize=18, fontweight="bold")
    fig.text(
        0.06,
        0.925,
        subtitle,
        ha="left",
        va="top",
        fontsize=10.5,
        color="#555555",
    )
    handles, labels = flat_axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=3,
        frameon=False,
        fontsize=10,
        handlelength=2.8,
        columnspacing=1.8,
    )
    fig.tight_layout(rect=(0.04, 0.09, 0.99, 0.89), h_pad=2.2, w_pad=2.0)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(
            output_stem.with_suffix(".png"), dpi=dpi, facecolor="white"
        )
        svg_path = output_stem.with_suffix(".svg")
        fig.savefig(svg_path, facecolor="white")
        _normalize_svg(svg_path)
    finally:
        plt.close(fig)


def render_ragged_efficiency_chart(
    points: Sequence[PrefillPoint], output_stem: Path, dpi: int
) -> None:
    """Render shape-only token and full-causal pair-slot efficiencies."""

    first_model_id = points[0].series.model_id
    model_points = [
        point for point in points if point.series.model_id == first_model_id
    ]
    modes = tuple(
        dict.fromkeys(point.result.execution_mode for point in model_points)
    )
    trace_indexes = sorted({point.x_value for point in model_points})
    trace_labels: list[str] = []
    for trace_index in trace_indexes:
        point = next(
            point for point in model_points if point.x_value == trace_index
        )
        lengths = point.result.prompt_tokens
        trace_labels.append(
            f"R{trace_index}\nB={len(lengths)}, T={sum(lengths)}"
        )

    _configure_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), facecolor="white")
    colors = {"varlen": "#0072B2", "packed": "#009E73", "padded": "#D55E00"}
    markers = {"varlen": "o", "packed": "s", "padded": "^"}
    metrics = (
        ("Token execution efficiency", lambda r: r.token_efficiency),
        (
            "Full-causal pair-slot efficiency",
            lambda r: r.causal_pair_efficiency,
        ),
    )
    for axis_index, (ax, (title, metric)) in enumerate(zip(axes, metrics)):
        for mode in modes:
            mode_points = sorted(
                (
                    point
                    for point in model_points
                    if point.result.execution_mode == mode
                ),
                key=lambda point: point.x_value,
            )
            ax.plot(
                [point.x_value for point in mode_points],
                [metric(point.result) for point in mode_points],
                color=colors.get(mode, "#555555"),
                marker=markers.get(mode, "o"),
                linewidth=2.0,
                markersize=6,
                label=mode if axis_index == 0 else None,
            )
        ax.set_xticks(trace_indexes, labels=trace_labels)
        ax.set_xlabel("Ragged trace")
        ax.set_ylabel("Useful / executed")
        ax.set_title(title, loc="left", fontsize=12)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.7, alpha=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Prefill ragged execution efficiency",
        x=0.06,
        ha="left",
        fontsize=18,
        fontweight="bold",
    )
    fig.text(
        0.06,
        0.90,
        "Shape-only comparison; persistent cache capacity still follows valid tokens.",
        ha="left",
        fontsize=10.5,
        color="#555555",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=max(1, len(modes)),
        frameon=False,
    )
    fig.tight_layout(rect=(0.04, 0.10, 0.99, 0.86), w_pad=2.5)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(
            output_stem.with_suffix(".png"), dpi=dpi, facecolor="white"
        )
        svg_path = output_stem.with_suffix(".svg")
        fig.savefig(svg_path, facecolor="white")
        _normalize_svg(svg_path)
    finally:
        plt.close(fig)


def _write_experiment_data(
    points: Sequence[PrefillPoint], experiment: str, output_dir: Path
) -> None:
    experiment_dir = output_dir / experiment.replace("-", "_")
    prefix = f"prefill_{experiment.replace('-', '_')}"
    write_csv(
        [detail_row(point) for point in points],
        experiment_dir / f"{prefix}_detail.csv",
    )
    write_csv(
        [summary_row(point) for point in points],
        experiment_dir / f"{prefix}_summary.csv",
    )
    write_json(points, experiment_dir / f"{prefix}.json")


def _clear_previous_generated_outputs(output_dir: Path) -> None:
    """Remove only files owned by this generator from an earlier run.

    A user may rerun the script with fewer experiments or different sweep
    points.  Leaving old charts beside the new CSV would make the directory
    internally inconsistent.  Restrict cleanup to known top-level filenames
    and ``prefill_*`` artifacts in the three generator-owned subdirectories;
    unrelated files in a custom output directory are preserved.
    """

    for filename in (
        "prefill_all_detail.csv",
        "prefill_all_summary.csv",
        "prefill_all.json",
    ):
        path = output_dir / filename
        if path.is_file():
            path.unlink()

    generated_patterns = {
        "equal": (
            "prefill_equal_detail.csv",
            "prefill_equal_summary.csv",
            "prefill_equal.json",
            "prefill_equal_batch_*.csv",
            "prefill_equal_batch_*.png",
            "prefill_equal_batch_*.svg",
        ),
        "token_budget": (
            "prefill_token_budget_detail.csv",
            "prefill_token_budget_summary.csv",
            "prefill_token_budget.json",
            "prefill_token_budget_[0-9]*.csv",
            "prefill_token_budget_[0-9]*.png",
            "prefill_token_budget_[0-9]*.svg",
        ),
        "ragged": (
            "prefill_ragged_detail.csv",
            "prefill_ragged_summary.csv",
            "prefill_ragged.json",
            "prefill_ragged_mode_*.csv",
            "prefill_ragged_mode_*.png",
            "prefill_ragged_mode_*.svg",
            "prefill_ragged_execution_efficiency.png",
            "prefill_ragged_execution_efficiency.svg",
        ),
    }
    for directory_name, patterns in generated_patterns.items():
        directory = output_dir / directory_name
        if not directory.is_dir():
            continue
        for pattern in patterns:
            for path in directory.glob(pattern):
                if path.is_file():
                    path.unlink()
        try:
            directory.rmdir()
        except OSError:
            # Preserve a directory that still contains unrelated user files.
            pass


def write_outputs(
    points: Sequence[PrefillPoint], output_dir: Path, dpi: int
) -> None:
    """Write total/per-experiment data and every selected chart."""

    _clear_previous_generated_outputs(output_dir)
    write_csv(
        [detail_row(point) for point in points],
        output_dir / "prefill_all_detail.csv",
    )
    write_csv(
        [summary_row(point) for point in points],
        output_dir / "prefill_all_summary.csv",
    )
    write_json(points, output_dir / "prefill_all.json")

    experiments = tuple(dict.fromkeys(point.result.experiment for point in points))
    for experiment in experiments:
        experiment_points = [
            point for point in points if point.result.experiment == experiment
        ]
        _write_experiment_data(experiment_points, experiment, output_dir)

        if experiment == "equal":
            batches = sorted(
                {point.result.batch_size for point in experiment_points}
            )
            for batch in batches:
                chart_points = [
                    point
                    for point in experiment_points
                    if point.result.batch_size == batch
                ]
                lengths = sorted({point.x_value for point in chart_points})
                chart_dir = output_dir / "equal"
                stem = chart_dir / f"prefill_equal_batch_{batch}"
                write_csv(
                    [summary_row(point) for point in chart_points],
                    stem.with_suffix(".csv"),
                )
                render_work_chart(
                    chart_points,
                    title=(
                        "LLM Prefill workload | "
                        f"{points[0].precision_bits}-bit | Equal prompts | "
                        f"Batch = {batch}"
                    ),
                    subtitle=(
                        "Per valid input token. Logical HBM and operand-stream "
                        "bytes are alternative boundaries. Dashed/hollow = "
                        "theoretical context extrapolation."
                    ),
                    x_label="Prompt length per request (tokens, log2 scale)",
                    x_ticks=lengths,
                    x_tick_labels=[_compact_token_count(v) for v in lengths],
                    logarithmic_x=True,
                    output_stem=stem,
                    dpi=dpi,
                )

        elif experiment == "token-budget":
            budgets = sorted(
                {point.result.valid_input_tokens for point in experiment_points}
            )
            for budget in budgets:
                chart_points = [
                    point
                    for point in experiment_points
                    if point.result.valid_input_tokens == budget
                ]
                batches = sorted({point.x_value for point in chart_points})
                chart_dir = output_dir / "token_budget"
                stem = chart_dir / f"prefill_token_budget_{budget}"
                write_csv(
                    [summary_row(point) for point in chart_points],
                    stem.with_suffix(".csv"),
                )
                render_work_chart(
                    chart_points,
                    title=(
                        "LLM Prefill workload | "
                        f"{points[0].precision_bits}-bit | Fixed T = {budget}"
                    ),
                    subtitle=(
                        "The same total input-token budget is split as evenly "
                        "as possible across B requests. Extrapolation uses the "
                        "longest individual request, not T."
                    ),
                    x_label="Request batch B (log2 scale)",
                    x_ticks=batches,
                    x_tick_labels=[str(value) for value in batches],
                    logarithmic_x=True,
                    output_stem=stem,
                    dpi=dpi,
                )

        else:
            modes = tuple(
                dict.fromkeys(
                    point.result.execution_mode for point in experiment_points
                )
            )
            trace_indexes = sorted({point.x_value for point in experiment_points})
            trace_labels: list[str] = []
            for trace_index in trace_indexes:
                example = next(
                    point
                    for point in experiment_points
                    if point.x_value == trace_index
                )
                lengths = example.result.prompt_tokens
                trace_labels.append(
                    f"R{trace_index}\nB={len(lengths)}, T={sum(lengths)}"
                )
            for mode in modes:
                chart_points = [
                    point
                    for point in experiment_points
                    if point.result.execution_mode == mode
                ]
                chart_dir = output_dir / "ragged"
                stem = chart_dir / f"prefill_ragged_mode_{mode}"
                write_csv(
                    [summary_row(point) for point in chart_points],
                    stem.with_suffix(".csv"),
                )
                render_work_chart(
                    chart_points,
                    title=(
                        "LLM Prefill workload | "
                        f"{points[0].precision_bits}-bit | Ragged | {mode}"
                    ),
                    subtitle=(
                        "Each x-position is one explicit request-length vector. "
                        "Per-valid-token normalization exposes padding overhead."
                    ),
                    x_label="Ragged trace",
                    x_ticks=trace_indexes,
                    x_tick_labels=trace_labels,
                    logarithmic_x=False,
                    output_stem=stem,
                    dpi=dpi,
                )
            render_ragged_efficiency_chart(
                experiment_points,
                output_dir / "ragged" / "prefill_ragged_execution_efficiency",
                dpi,
            )


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("values must be positive integers")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    model_ids = tuple(series.model_id for series in MODEL_SERIES)
    parser = argparse.ArgumentParser(
        description=(
            "Generate Prefill detail/summary CSV, nested JSON, and PNG/SVG "
            "charts for the three standard experiments."
        )
    )
    parser.add_argument(
        "--precision",
        type=int,
        choices=(4, 8, 16),
        default=16,
        help="Weight+KV profile to load (default: 16).",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=("all",) + EXPERIMENTS,
        default=("all",),
        help="Experiments to generate; default: all.",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        help="Override configs/<precision>bit.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override outputs/prefill/<precision>bit.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=model_ids,
        default=model_ids,
        help="Model ids to include; default: all shipped real-model configs.",
    )
    parser.add_argument(
        "--prompt-lengths",
        nargs="+",
        type=_positive_int,
        default=DEFAULT_PROMPT_LENGTHS,
        help="Equal-prompt lengths.",
    )
    parser.add_argument(
        "--equal-batches",
        nargs="+",
        type=_positive_int,
        default=DEFAULT_EQUAL_BATCHES,
        help="Request batches for the equal experiment.",
    )
    parser.add_argument(
        "--token-budgets",
        nargs="+",
        type=_positive_int,
        default=DEFAULT_TOKEN_BUDGETS,
        help="Exact total input-token budgets.",
    )
    parser.add_argument(
        "--token-budget-batches",
        nargs="+",
        type=_positive_int,
        default=DEFAULT_TOKEN_BUDGET_BATCHES,
        help="Request batches used to split every token budget.",
    )
    parser.add_argument(
        "--ragged-lengths",
        nargs="+",
        type=_positive_int,
        action="append",
        help="One ragged length vector; repeat for multiple traces.",
    )
    parser.add_argument(
        "--execution-mode",
        choices=("varlen", "packed", "padded"),
        default="varlen",
        help="Execution layout for equal and token-budget experiments.",
    )
    parser.add_argument(
        "--ragged-execution-modes",
        nargs="+",
        choices=("varlen", "packed", "padded"),
        default=DEFAULT_RAGGED_EXECUTION_MODES,
        help="Layouts compared in the ragged experiment.",
    )
    parser.add_argument(
        "--logits-mode",
        choices=("last", "all", "none"),
        default="last",
        help="LM-head positions: final token, all tokens, or none.",
    )
    parser.add_argument(
        "--exclude-self-attention",
        action="store_true",
        help="Exclude the new-token causal diagonal.",
    )
    parser.add_argument(
        "--dpi",
        type=_positive_int,
        default=180,
        help="PNG resolution (default: 180).",
    )
    return parser.parse_args(argv)


def _unique(values: Sequence[Any]) -> tuple[Any, ...]:
    return tuple(dict.fromkeys(values))


def _run(args: argparse.Namespace) -> int:
    experiments = (
        EXPERIMENTS
        if "all" in args.experiments
        else _unique(args.experiments)
    )
    series_by_id = {series.model_id: series for series in MODEL_SERIES}
    selected_series = tuple(series_by_id[value] for value in _unique(args.models))
    config_dir = args.config_dir or (
        PROJECT_ROOT / "configs" / f"{args.precision}bit"
    )
    output_dir = args.output_dir or (
        PROJECT_ROOT / "outputs" / "prefill" / f"{args.precision}bit"
    )
    ragged_batches = (
        tuple(tuple(trace) for trace in args.ragged_lengths)
        if args.ragged_lengths
        else DEFAULT_RAGGED_BATCHES
    )

    # Validate every fixed-budget combination before calculating or writing,
    # so invalid input cannot leave a half-populated output directory.
    if "token-budget" in experiments:
        for budget in args.token_budgets:
            for batch in args.token_budget_batches:
                if batch > budget:
                    raise ValueError(
                        f"batch={batch} exceeds token budget={budget}; "
                        "every request needs at least one token"
                    )

    points = calculate_points(
        precision=args.precision,
        config_dir=config_dir,
        series_values=selected_series,
        experiments=experiments,
        prompt_lengths=_unique(args.prompt_lengths),
        equal_batches=_unique(args.equal_batches),
        token_budgets=_unique(args.token_budgets),
        token_budget_batches=_unique(args.token_budget_batches),
        ragged_batches=ragged_batches,
        execution_mode=args.execution_mode,
        ragged_execution_modes=_unique(args.ragged_execution_modes),
        logits_mode=args.logits_mode,
        include_self_attention=not args.exclude_self_attention,
    )
    write_outputs(points, output_dir, args.dpi)
    chart_count = sum(
        len(list((output_dir / directory).glob("prefill_*.png")))
        for directory in ("equal", "token_budget", "ragged")
        if (output_dir / directory).is_dir()
    )
    print(
        f"Generated {len(points)} Prefill points, CSV/JSON data, and "
        f"{chart_count} PNG charts in {output_dir}"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return _run(args)
    except (ConfigurationError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
