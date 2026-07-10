#!/usr/bin/env python3
"""Generate uniform weight+KV precision profiles from base model configs.

The base configs retain architecture-specific published precisions.  A sweep
profile has a different purpose: all model weights and KV-cache elements must
use the requested bit width so curves are comparable.  Index caches and
recurrent states are intentionally left at their base-config precisions because
they are separate traffic classes.

Only the nine real model configs are included.  Synthetic examples in
``configs`` are not copied into precision-profile directories.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / "configs"
SUPPORTED_BITS = (4, 8, 16)


@dataclass(frozen=True)
class ModelProfileSource:
    base_filename: str
    output_stem: str
    display_name: str
    embedding_elements: int


MODEL_SOURCES = (
    ModelProfileSource(
        "deepseek_r1_mla_bf16.json",
        "deepseek_r1_mla",
        "DeepSeek-R1 MLA",
        7168,
    ),
    ModelProfileSource(
        "deepseek_v4_pro.json",
        "deepseek_v4_pro",
        "DeepSeek-V4-Pro",
        7168,
    ),
    ModelProfileSource(
        "glm_5_2_dsa_bf16.json",
        "glm_5_2_dsa",
        "GLM-5.2 MLA+DSA IndexShare",
        6144,
    ),
    ModelProfileSource(
        "qwen3_235b_a22b_bf16.json",
        "qwen3_235b_a22b",
        "Qwen3-235B-A22B",
        4096,
    ),
    ModelProfileSource(
        "llama_3_3_70b_bf16.json",
        "llama_3_3_70b",
        "Llama-3.3-70B-Instruct",
        8192,
    ),
    ModelProfileSource(
        "qwen3_8b_bf16.json",
        "qwen3_8b",
        "Qwen3-8B",
        4096,
    ),
    ModelProfileSource(
        "qwen3_4b_bf16.json",
        "qwen3_4b",
        "Qwen3-4B",
        2560,
    ),
    ModelProfileSource(
        "qwen3_next_80b_a3b_bf16.json",
        "qwen3_next_80b_a3b",
        "Qwen3-Next-80B-A3B-Instruct hybrid DeltaNet+GQA",
        2048,
    ),
    ModelProfileSource(
        "mamba_2_8b_bf16.json",
        "mamba_2_8b",
        "state-spaces Mamba-2.8B",
        2560,
    ),
)


PRECISION_PREFIX = re.compile(
    r"^(?:bf16|fp32|fp16|fp8|fp4|4bit|8bit|16bit)_"
)


def _retag_name(name: str, bits: int) -> str:
    """Replace a storage-precision prefix without changing ownership names."""

    return PRECISION_PREFIX.sub(f"{bits}bit_", name)


def _set_weight_precision(weights: dict[str, Any], bits: int) -> None:
    weights["weight_bits"] = bits
    for group in weights.get("always_active_parameter_groups", []):
        group["weight_bits"] = bits
        group["name"] = _retag_name(group["name"], bits)
    for group in weights.get("routed_expert_groups", []):
        group["weight_bits"] = bits
        group["name"] = _retag_name(group["name"], bits)


def _set_kv_precision(model: dict[str, Any], bits: int) -> None:
    """Force every known KV layout field to the profile bit width."""

    for layer_group in model["layer_groups"]:
        for mixer in layer_group["mixers"]:
            if mixer.get("kind") != "softmax_attention":
                continue
            layout = mixer["kv_layout"]
            kind = layout["kind"]
            if kind in {"grouped", "mha", "mqa", "gqa"}:
                layout["key_bits"] = bits
                layout["value_bits"] = bits
            elif kind in {"latent", "mla"}:
                layout["latent_bits"] = bits
                layout["rope_bits"] = bits
            elif kind in {"shared", "shared_kv"}:
                layout["non_rope_bits"] = bits
                layout["rope_bits"] = bits
            elif kind == "explicit":
                raise ValueError(
                    "explicit KV layouts require a profile-specific byte formula"
                )
            else:
                raise ValueError(f"unsupported KV layout {kind!r}")


def _set_embedding_read(
    model: dict[str, Any], embedding_elements: int, bits: int
) -> None:
    expected_bytes = embedding_elements * bits // 8
    matches = 0
    for layer_group in model["layer_groups"]:
        for mixer in layer_group["mixers"]:
            if mixer.get("kind") != "fixed_cost":
                continue
            work = mixer.get("work", {})
            if "other_read_bytes" not in work:
                continue
            work["other_read_bytes"] = expected_bytes
            matches += 1
    if matches != 1:
        raise ValueError(
            f"expected one input-embedding fixed cost, found {matches}"
        )


def _retag_metadata(metadata: dict[str, Any], bits: int) -> None:
    breakdown = metadata.get("parameter_breakdown")
    if isinstance(breakdown, dict):
        metadata["parameter_breakdown"] = {
            _retag_name(key, bits): value for key, value in breakdown.items()
        }

    replacement = f"{bits}-bit"
    for key in (
        "embedding_note",
        "routing_note",
        "parameter_count_convention",
    ):
        value = metadata.get(key)
        if isinstance(value, str):
            metadata[key] = value.replace("BF16", replacement)

    metadata["precision_note"] = (
        f"Uniform theoretical profile: every model weight and KV-cache "
        f"element is {bits} bits. Index-cache and recurrent-state precision "
        "remain as specified by the base architecture. Quantization scales, "
        "zero-points, and packing metadata are not included."
    )


def build_profile(
    raw: dict[str, Any], source: ModelProfileSource, bits: int
) -> dict[str, Any]:
    profile = copy.deepcopy(raw)
    model = profile["model"]
    model["name"] = (
        f"{source.display_name} {bits}-bit weight+KV theoretical deployment"
    )

    _set_weight_precision(model["weights"], bits)
    _set_kv_precision(model, bits)
    _set_embedding_read(model, source.embedding_elements, bits)

    deployment = profile["deployment"]
    deployment["weight_bits"] = bits
    deployment["expert_weight_bits"] = bits
    deployment["kv_bits"] = bits

    metadata = model.setdefault("metadata", {})
    _retag_metadata(metadata, bits)
    metadata["precision_profile"] = {
        "base_config": source.base_filename,
        "weight_bits": bits,
        "kv_bits": bits,
        "index_bits": deployment.get("index_bits", deployment["kv_bits"]),
        "state_bits": deployment.get("state_bits", 16),
    }
    if source.output_stem == "deepseek_v4_pro":
        metadata["notes"] = (
            "The 61 configured layers contain 31 HCA and 30 CSA layers. "
            f"This generated profile forces every model weight and shared-KV "
            f"element to {bits} bits. Index/state precision remains separate. "
            "Quantization metadata and uncompressed compression-tail state "
            "are not included."
        )

    return profile


def generate_profiles(bits_values: tuple[int, ...]) -> list[Path]:
    generated: list[Path] = []
    for bits in bits_values:
        if bits not in SUPPORTED_BITS:
            raise ValueError(
                f"unsupported bits={bits}; choose from {SUPPORTED_BITS}"
            )
        output_dir = CONFIG_ROOT / f"{bits}bit"
        output_dir.mkdir(parents=True, exist_ok=True)

        for source in MODEL_SOURCES:
            source_path = CONFIG_ROOT / source.base_filename
            with source_path.open("r", encoding="utf-8") as handle:
                raw = json.load(handle)
            profile = build_profile(raw, source, bits)
            output_path = output_dir / f"{source.output_stem}_{bits}bit.json"
            with output_path.open("w", encoding="utf-8") as handle:
                json.dump(profile, handle, indent=2, ensure_ascii=True)
                handle.write("\n")
            generated.append(output_path)
    return generated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate uniform weight+KV model precision profiles."
    )
    parser.add_argument(
        "--bits",
        type=int,
        nargs="+",
        default=[8, 16],
        help="Profile widths to generate (default: 8 16).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = generate_profiles(tuple(args.bits))
    print(f"Generated {len(paths)} precision configs")
    for path in paths:
        print(path.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
