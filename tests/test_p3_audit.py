from __future__ import annotations

import unittest
from pathlib import Path

from decode_engine.config import load_engine_config
from decode_engine.engine import calculate_decode


ROOT = Path(__file__).resolve().parents[1]


class P3MechanismAnchorTests(unittest.TestCase):
    def _config(self, relative_path: str):
        return load_engine_config(ROOT / relative_path)

    def test_mistral_swa_saturates_at_4096(self):
        config = self._config(
            "configs/2023/mistral_7b_instruct_v0_1_bf16.json"
        )
        before = calculate_decode(config.model, config.deployment, [4095])
        boundary = calculate_decode(config.model, config.deployment, [4096])
        after = calculate_decode(config.model, config.deployment, [4097])

        self.assertLess(
            before.per_output_work.kv_read_bytes,
            boundary.per_output_work.kv_read_bytes,
        )
        self.assertEqual(
            boundary.per_output_work.kv_read_bytes,
            after.per_output_work.kv_read_bytes,
        )
        self.assertEqual(
            boundary.cache_capacity_total.kv_bytes,
            after.cache_capacity_total.kv_bytes,
        )

    def test_glm52_main_topk_saturates_but_index_scan_continues(self):
        config = self._config(
            "configs/2026/glm_5_2_fp8_dsa_fp8kv.json"
        )
        before = calculate_decode(config.model, config.deployment, [2047])
        boundary = calculate_decode(config.model, config.deployment, [2048])
        after = calculate_decode(config.model, config.deployment, [2049])

        self.assertLess(
            before.per_output_work.attention_flops,
            boundary.per_output_work.attention_flops,
        )
        self.assertEqual(
            boundary.per_output_work.attention_flops,
            after.per_output_work.attention_flops,
        )
        self.assertGreater(
            after.per_output_work.index_flops,
            boundary.per_output_work.index_flops,
        )
        self.assertGreater(
            after.per_output_work.index_read_bytes,
            boundary.per_output_work.index_read_bytes,
        )

    def test_recurrent_state_is_context_independent_in_selected_models(self):
        for relative_path in (
            "configs/2024/jamba_1_5_large_experts_int8_bf16kv.json",
            "configs/2025/kimi_linear_48b_a3b_bf16.json",
            "configs/2026/qwen3_6_35b_a3b_fp8_bf16kv.json",
        ):
            with self.subTest(config=relative_path):
                config = self._config(relative_path)
                short = calculate_decode(
                    config.model, config.deployment, [128]
                )
                long = calculate_decode(
                    config.model,
                    config.deployment,
                    [config.model.max_context_tokens],
                )
                self.assertEqual(
                    short.per_output_work.state_flops,
                    long.per_output_work.state_flops,
                )
                self.assertEqual(
                    short.per_output_work.state_read_bytes,
                    long.per_output_work.state_read_bytes,
                )
                self.assertEqual(
                    short.cache_capacity_total.state_bytes,
                    long.cache_capacity_total.state_bytes,
                )

    def test_mixtral_batch1_moe_union_equals_top2(self):
        config = self._config(
            "configs/2023/mixtral_8x7b_instruct_v0_1_bf16.json"
        )
        batch1 = calculate_decode(config.model, config.deployment, [128])
        batch32 = calculate_decode(config.model, config.deployment, [128] * 32)
        group_name = "bf16_routed_swiglu_experts"

        self.assertEqual(batch1.expert_weight_sets_read[group_name], 2.0)
        self.assertGreater(batch32.expert_weight_sets_read[group_name], 2.0)
        self.assertLessEqual(batch32.expert_weight_sets_read[group_name], 8.0)


if __name__ == "__main__":
    unittest.main()
