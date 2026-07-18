from __future__ import annotations

import unittest
from pathlib import Path

from decode_engine.config import load_engine_config
from decode_engine.engine import calculate_decode
from decode_engine.mechanisms import build_mixer
from decode_engine.schema import DeploymentConfig


def chunked_mixer(*, retain_full_history: bool):
    mixer = build_mixer(
        {
            "kind": "softmax_attention",
            "kv_layout": {
                "kind": "grouped",
                "query_heads": 2,
                "kv_heads": 1,
                "head_dim": 4,
            },
            "access": {
                "kind": "chunked_block",
                "chunk_tokens": 4,
                "retain_full_history": retain_full_history,
            },
        },
        "test.chunked",
    )
    deployment = DeploymentConfig(
        weight_bits=16,
        expert_weight_bits=16,
        kv_bits=8,
        index_bits=8,
        state_bits=16,
    )
    return mixer, deployment


class ChunkedBlockAccessTests(unittest.TestCase):
    def test_decode_resets_reads_at_fixed_block_boundary(self):
        mixer, deployment = chunked_mixer(retain_full_history=True)

        before = mixer.decode_cost(3, deployment)
        boundary = mixer.decode_cost(4, deployment)
        inside_next = mixer.decode_cost(6, deployment)

        # One GQA entry is 8 bytes and 32 FLOPs.
        self.assertEqual(before.work.kv_read_bytes, 3 * 8)
        self.assertEqual(before.work.attention_flops, 3 * 32)
        self.assertEqual(before.cache.kv_bytes, 3 * 8)
        self.assertEqual(boundary.work.kv_read_bytes, 0)
        self.assertEqual(boundary.work.attention_flops, 0)
        self.assertEqual(boundary.cache.kv_bytes, 4 * 8)
        self.assertEqual(inside_next.work.kv_read_bytes, 2 * 8)
        self.assertEqual(inside_next.cache.kv_bytes, 6 * 8)

    def test_optimized_retention_keeps_only_current_block(self):
        mixer, deployment = chunked_mixer(retain_full_history=False)

        boundary = mixer.decode_cost(4, deployment)
        inside_next = mixer.decode_cost(6, deployment)

        self.assertEqual(boundary.cache.kv_bytes, 0)
        self.assertEqual(inside_next.cache.kv_bytes, 2 * 8)

    def test_prefill_crosses_block_boundary_without_sliding(self):
        mixer, deployment = chunked_mixer(retain_full_history=True)

        included = mixer.prefill_cost(3, 3, deployment, True)
        excluded = mixer.prefill_cost(3, 3, deployment, False)

        # Including the diagonal sees 4,1,2 entries; excluding it sees 3,0,1.
        self.assertEqual(included.work.kv_read_bytes, 3 * 8)
        self.assertEqual(included.work.attention_flops, 7 * 32)
        self.assertEqual(included.operand_work.kv_read_bytes, 7 * 8)
        self.assertEqual(included.cache.kv_bytes, 6 * 8)
        self.assertEqual(excluded.work.attention_flops, 4 * 32)
        self.assertEqual(excluded.operand_work.kv_read_bytes, 4 * 8)

        optimized, deployment = chunked_mixer(retain_full_history=False)
        optimized_cost = optimized.prefill_cost(3, 3, deployment, True)
        self.assertEqual(optimized_cost.cache.kv_bytes, 2 * 8)

    def test_llama4_scout_release_profile_anchors(self):
        root = Path(__file__).resolve().parents[1]
        config = load_engine_config(
            root
            / "configs"
            / "2025"
            / "llama_4_scout_17b_16e_instruct_bf16.json"
        )

        before = calculate_decode(config.model, config.deployment, [8191])
        boundary = calculate_decode(config.model, config.deployment, [8192])
        aws_limit = calculate_decode(
            config.model, config.deployment, [3_500_000]
        )

        self.assertEqual(
            before.per_output_work.attention_flops, 8_052_080_640
        )
        self.assertEqual(
            boundary.per_output_work.attention_flops, 2_013_265_920
        )
        self.assertEqual(
            boundary.cache_capacity_total.kv_bytes, 1_610_612_736
        )
        self.assertEqual(
            aws_limit.per_output_work.parameter_flops, 32_275_824_640
        )
        self.assertEqual(
            aws_limit.per_output_work.attention_flops, 861_646_356_480
        )
        self.assertEqual(
            aws_limit.per_output_work.total_bytes, 204_605_302_784
        )
        self.assertEqual(
            aws_limit.cache_capacity_total.kv_bytes, 688_128_000_000
        )


if __name__ == "__main__":
    unittest.main()
