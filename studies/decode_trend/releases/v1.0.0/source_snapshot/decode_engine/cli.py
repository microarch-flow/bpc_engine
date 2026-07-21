"""Command-line interface for decode and prefill workload calculations."""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from .config import ConfigurationError, load_engine_config
from .engine import (
    calculate_grid,
    calculate_prefill,
    calculate_prefill_grid,
    calculate_prefill_token_budget_grid,
    calculate_ragged_prefill_grid,
)
from .schema import DecodeResult, EngineConfig, PrefillResult, WorkCost


Result = DecodeResult | PrefillResult


def _flat_decode_result(result: DecodeResult) -> dict[str, Any]:
    work = result.per_output_work
    cache = result.cache_capacity_per_request_average
    contexts = result.context_tokens
    same_context = all(value == contexts[0] for value in contexts)
    return {
        "model": result.model_name,
        "batch": result.batch_size,
        "context_tokens": contexts[0] if same_context else result.average_context_tokens,
        "flops_per_token": work.total_flops,
        "bytes_per_token": work.total_bytes,
        "logical_hbm_bytes_per_token": work.logical_hbm_bytes,
        "bytes_per_flop": result.bytes_per_flop,
        "tbps_per_pflops": result.tbps_per_pflops,
        "logical_hbm_bytes_per_flop": result.logical_hbm_bytes_per_flop,
        "logical_hbm_tbps_per_pflops": (
            result.logical_hbm_tbps_per_pflops
        ),
        "parameter_flops": work.parameter_flops,
        "attention_flops": work.attention_flops,
        "index_flops": work.index_flops,
        "state_flops": work.state_flops,
        "extra_flops": work.extra_flops,
        "weight_read_bytes": work.weight_read_bytes,
        "kv_read_bytes": work.kv_read_bytes,
        "kv_write_bytes": work.kv_write_bytes,
        "index_read_bytes": work.index_read_bytes,
        "index_write_bytes": work.index_write_bytes,
        "state_read_bytes": work.state_read_bytes,
        "state_write_bytes": work.state_write_bytes,
        "activation_bytes": work.activation_bytes,
        "other_read_bytes": work.other_read_bytes,
        "cache_bytes_per_request": cache.total_bytes,
        "kv_cache_bytes_per_request": cache.kv_bytes,
        "index_cache_bytes_per_request": cache.index_bytes,
        "state_cache_bytes_per_request": cache.state_bytes,
        "expert_weight_sets_read": json.dumps(
            result.expert_weight_sets_read,
            ensure_ascii=False,
            sort_keys=True,
        ),
    }


def _prefixed_work(prefix: str, work: WorkCost) -> dict[str, float]:
    """Flatten a work breakdown while retaining its normalization boundary."""

    return {
        f"{prefix}{name}": value
        for name, value in work.to_dict().items()
    }


def _flat_prefill_result(result: PrefillResult) -> dict[str, Any]:
    batch_work = result.batch_work
    batch_operand_work = result.batch_operand_work
    per_input_work = result.per_input_work
    per_input_operand_work = result.per_input_operand_work
    cache_total = result.cache_capacity_total
    cache_average = result.cache_capacity_per_request_average

    row: dict[str, Any] = {
        "phase": "prefill",
        "model": result.model_name,
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
        # These aliases make the two traffic interpretations easy to find in
        # a spreadsheet without discarding the detailed fields below.
        "total_flops": batch_work.total_flops,
        "total_compulsory_bytes": batch_work.total_bytes,
        "total_operand_bytes": batch_operand_work.total_bytes,
        "flops_per_input_token": per_input_work.total_flops,
        "compulsory_bytes_per_input_token": per_input_work.total_bytes,
        "operand_bytes_per_input_token": per_input_operand_work.total_bytes,
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
            ensure_ascii=False,
            sort_keys=True,
        ),
        "useful_expert_weight_sets_read": json.dumps(
            result.useful_expert_weight_sets_read,
            ensure_ascii=False,
            sort_keys=True,
        ),
    }
    # Preserve batch totals, useful-token totals, per-valid-input-token work,
    # and the pair-stream operand boundary with every FLOP/byte component.
    row.update(_prefixed_work("batch_", result.batch_work))
    row.update(_prefixed_work("useful_", result.useful_work))
    row.update(_prefixed_work("per_input_", result.per_input_work))
    row.update(_prefixed_work("batch_operand_", result.batch_operand_work))
    row.update(
        _prefixed_work("useful_operand_", result.useful_operand_work)
    )
    row.update(
        _prefixed_work("per_input_operand_", result.per_input_operand_work)
    )
    return row


def _flat_result(result: Result) -> dict[str, Any]:
    if isinstance(result, PrefillResult):
        return _flat_prefill_result(result)
    return _flat_decode_result(result)


def _render_csv(results: Sequence[Result]) -> str:
    rows = [_flat_result(result) for result in results]
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer, fieldnames=list(rows[0]), lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _render_json(results: Sequence[Result]) -> str:
    return json.dumps(
        [result.to_dict() for result in results],
        ensure_ascii=False,
        indent=2,
    ) + "\n"


def _format_number(value: float, digits: int = 4) -> str:
    if value == float("inf"):
        return "inf"
    return f"{value:.{digits}f}"


def _render_decode_table(results: Sequence[DecodeResult]) -> str:
    headers = [
        "model",
        "batch",
        "context",
        "GFLOP/token",
        "GB/token",
        "Byte/FLOP",
        "TB/s/PFLOPS",
        "W GB/token",
        "KV-read GB/token",
        "cache GB/request",
    ]
    rows: list[list[str]] = []
    for result in results:
        work = result.per_output_work
        rows.append(
            [
                result.model_name,
                str(result.batch_size),
                _format_number(result.average_context_tokens, 0),
                _format_number(work.total_flops / 1e9, 3),
                _format_number(work.total_bytes / 1e9, 6),
                _format_number(result.bytes_per_flop, 8),
                _format_number(result.tbps_per_pflops, 4),
                _format_number(work.weight_read_bytes / 1e9, 6),
                _format_number(work.kv_read_bytes / 1e9, 6),
                _format_number(
                    result.cache_capacity_per_request_average.total_bytes / 1e9,
                    6,
                ),
            ]
        )

    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    def render_row(values: Sequence[str]) -> str:
        return "  ".join(
            value.ljust(widths[index]) for index, value in enumerate(values)
        ).rstrip()

    separator = "  ".join("-" * width for width in widths)
    return "\n".join(
        [render_row(headers), separator, *(render_row(row) for row in rows)]
    ) + "\n"


def _prompt_shape(result: PrefillResult) -> str:
    prompts = result.prompt_tokens
    if all(value == prompts[0] for value in prompts):
        return f"{len(prompts)}x{prompts[0]}"
    return json.dumps(list(prompts), separators=(",", ":"))


def _render_prefill_table(results: Sequence[PrefillResult]) -> str:
    headers = [
        "model",
        "experiment",
        "mode",
        "logits/positions",
        "self",
        "prompts",
        "tokens valid/exec",
        "causal slots valid/exec",
        "GFLOP total",
        "GFLOP/input",
        "compulsory GB/input",
        "operand GB/input",
        "Byte/FLOP",
        "cache GB total",
    ]
    rows: list[list[str]] = []
    for result in results:
        work = result.batch_work
        per_input = result.per_input_work
        per_input_operand = result.per_input_operand_work
        rows.append(
            [
                result.model_name,
                result.experiment,
                result.execution_mode,
                f"{result.logits_mode}/{result.executed_logit_positions}",
                "yes" if result.include_self_attention else "no",
                _prompt_shape(result),
                f"{result.valid_input_tokens}/{result.executed_input_tokens}",
                (
                    f"{_format_number(result.valid_causal_pair_slots, 0)}/"
                    f"{_format_number(result.executed_causal_pair_slots, 0)}"
                ),
                _format_number(work.total_flops / 1e9, 3),
                _format_number(per_input.total_flops / 1e9, 6),
                _format_number(per_input.total_bytes / 1e9, 9),
                _format_number(per_input_operand.total_bytes / 1e9, 9),
                _format_number(result.bytes_per_flop, 8),
                _format_number(result.cache_capacity_total.total_bytes / 1e9, 6),
            ]
        )

    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    def render_row(values: Sequence[str]) -> str:
        return "  ".join(
            value.ljust(widths[index]) for index, value in enumerate(values)
        ).rstrip()

    separator = "  ".join("-" * width for width in widths)
    return "\n".join(
        [render_row(headers), separator, *(render_row(row) for row in rows)]
    ) + "\n"


def _render_table(results: Sequence[Result]) -> str:
    if isinstance(results[0], PrefillResult):
        return _render_prefill_table(results)  # type: ignore[arg-type]
    return _render_decode_table(results)  # type: ignore[arg-type]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="decode-engine",
        description=(
            "Calculate config-driven LLM decode or prefill FLOPs, off-chip "
            "data movement, and persistent cache capacity."
        ),
    )
    parser.add_argument("--config", required=True, help="Path to one JSON config")
    parser.add_argument(
        "--phase",
        choices=("decode", "prefill"),
        default="decode",
        help="Workload phase (default: decode, preserving the original CLI)",
    )
    parser.add_argument(
        "--experiment",
        choices=("equal", "token-budget", "ragged"),
        default="equal",
        help="Prefill experiment family (default: equal)",
    )
    parser.add_argument(
        "--contexts",
        nargs="+",
        type=int,
        help="Context lengths. Defaults to analysis.contexts in the config.",
    )
    parser.add_argument(
        "--allow-extrapolation",
        action="store_true",
        help=(
            "Permit Decode contexts above the model's configured maximum. "
            "The caller must treat those results as theoretical extrapolations."
        ),
    )
    parser.add_argument(
        "--batches",
        nargs="+",
        type=int,
        help="Batch sizes. Defaults to analysis.batches in the config.",
    )
    parser.add_argument(
        "--prompt-lengths",
        nargs="+",
        type=int,
        help=(
            "Equal prompt lengths for the prefill equal-length sweep. "
            "Defaults to analysis.prefill.prompt_lengths, then contexts."
        ),
    )
    parser.add_argument(
        "--token-budgets",
        nargs="+",
        type=int,
        help="Exact total input-token budgets for the token-budget experiment.",
    )
    parser.add_argument(
        "--ragged-lengths",
        nargs="+",
        type=int,
        action="append",
        metavar="TOKENS",
        help=(
            "One ragged prompt-length vector. Repeat the option to evaluate "
            "multiple batches, for example: --ragged-lengths 128 512 "
            "--ragged-lengths 64 2048 32."
        ),
    )
    parser.add_argument(
        "--cached-lengths",
        nargs="+",
        type=int,
        help=(
            "Cached-prefix lengths for one explicitly supplied ragged vector; "
            "requires --phase prefill --experiment ragged."
        ),
    )
    parser.add_argument(
        "--execution-mode",
        choices=("varlen", "packed", "padded"),
        default="varlen",
        help="Prefill execution layout (default: varlen)",
    )
    parser.add_argument(
        "--logits-mode",
        choices=("last", "all", "none"),
        default="last",
        help=(
            "Compute logits for each request's last token, every token, or "
            "no tokens (for non-final chunks)."
        ),
    )
    parser.add_argument(
        "--exclude-self-attention",
        action="store_true",
        help="Exclude each new token's diagonal self-attention pair.",
    )
    parser.add_argument(
        "--format",
        choices=("table", "json", "csv"),
        default="table",
        help="Output representation (default: table)",
    )
    parser.add_argument(
        "--output",
        help="Optional output path. Without it, results are printed to stdout.",
    )
    return parser


def _calculate_prefill_results(
    config: EngineConfig,
    args: argparse.Namespace,
) -> list[PrefillResult]:
    common = {
        "execution_mode": args.execution_mode,
        "logits_mode": args.logits_mode,
        "include_self_attention": not args.exclude_self_attention,
    }

    if args.experiment == "equal":
        if args.token_budgets or args.ragged_lengths or args.cached_lengths:
            raise ValueError(
                "the equal experiment accepts --prompt-lengths and --batches, "
                "not token-budget/ragged/cached-length options"
            )
        return calculate_prefill_grid(
            config,
            args.prompt_lengths,
            args.batches,
            **common,
        )

    if args.experiment == "token-budget":
        if args.prompt_lengths or args.ragged_lengths or args.cached_lengths:
            raise ValueError(
                "the token-budget experiment accepts --token-budgets and "
                "--batches, not prompt/ragged/cached-length options"
            )
        return calculate_prefill_token_budget_grid(
            config,
            args.token_budgets,
            args.batches,
            **common,
        )

    if args.prompt_lengths or args.token_budgets or args.batches:
        raise ValueError(
            "the ragged experiment accepts --ragged-lengths, not "
            "--prompt-lengths, --token-budgets, or --batches"
        )
    if args.cached_lengths is not None:
        if not args.ragged_lengths or len(args.ragged_lengths) != 1:
            raise ValueError(
                "--cached-lengths requires exactly one explicit "
                "--ragged-lengths vector"
            )
        return [
            calculate_prefill(
                config.model,
                config.deployment,
                args.ragged_lengths[0],
                cached_context_tokens=args.cached_lengths,
                experiment="ragged",
                **common,
            )
        ]
    return calculate_ragged_prefill_grid(
        config,
        args.ragged_lengths,
        **common,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = load_engine_config(args.config)
        if args.phase == "prefill":
            if args.contexts or args.allow_extrapolation:
                raise ValueError(
                    "--contexts and --allow-extrapolation are decode-only; "
                    "use --prompt-lengths or --ragged-lengths for prefill"
                )
            results: list[Result] = _calculate_prefill_results(config, args)
        else:
            if (
                args.prompt_lengths
                or args.token_budgets
                or args.ragged_lengths
                or args.cached_lengths
                or args.execution_mode != "varlen"
                or args.logits_mode != "last"
                or args.exclude_self_attention
                or args.experiment != "equal"
            ):
                raise ValueError(
                    "prefill experiment/layout options require --phase prefill"
                )
            results = calculate_grid(
                config,
                args.contexts,
                args.batches,
                allow_extrapolation=args.allow_extrapolation,
            )
    except (ConfigurationError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.format == "json":
        rendered = _render_json(results)
    elif args.format == "csv":
        rendered = _render_csv(results)
    else:
        rendered = _render_table(results)

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
