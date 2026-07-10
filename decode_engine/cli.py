"""Command-line interface for grid calculations and machine-readable output."""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from .config import ConfigurationError, load_engine_config
from .engine import calculate_grid
from .schema import DecodeResult


def _flat_result(result: DecodeResult) -> dict[str, Any]:
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
        "bytes_per_flop": result.bytes_per_flop,
        "tbps_per_pflops": result.tbps_per_pflops,
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


def _render_csv(results: Sequence[DecodeResult]) -> str:
    rows = [_flat_result(result) for result in results]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _render_json(results: Sequence[DecodeResult]) -> str:
    return json.dumps(
        [result.to_dict() for result in results],
        ensure_ascii=False,
        indent=2,
    ) + "\n"


def _format_number(value: float, digits: int = 4) -> str:
    if value == float("inf"):
        return "inf"
    return f"{value:.{digits}f}"


def _render_table(results: Sequence[DecodeResult]) -> str:
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="decode-engine",
        description=(
            "Calculate per-output-token FLOPs and off-chip data movement for "
            "config-driven LLM decode workloads."
        ),
    )
    parser.add_argument("--config", required=True, help="Path to one JSON config")
    parser.add_argument(
        "--contexts",
        nargs="+",
        type=int,
        help="Context lengths. Defaults to analysis.contexts in the config.",
    )
    parser.add_argument(
        "--batches",
        nargs="+",
        type=int,
        help="Batch sizes. Defaults to analysis.batches in the config.",
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


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = load_engine_config(args.config)
        results = calculate_grid(config, args.contexts, args.batches)
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
