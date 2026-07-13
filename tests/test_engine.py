from __future__ import annotations

import unittest
from pathlib import Path

from decode_engine.config import (
    ConfigurationError,
    load_engine_config,
    parse_engine_config,
)
from decode_engine.engine import calculate_decode, expected_expert_reads


def engine_config(
    *,
    always_active_parameters: float = 0,
    layer_groups: list[dict] | None = None,
    expert_groups: list[dict] | None = None,
    deployment: dict | None = None,
):
    """Build a small valid config so each test states only relevant fields."""

    return parse_engine_config(
        {
            "schema_version": 1,
            "model": {
                "name": "test-model",
                "max_context_tokens": 1048576,
                "weights": {
                    "always_active_parameters": always_active_parameters,
                    "routed_expert_groups": expert_groups or [],
                },
                "layer_groups": layer_groups
                or [
                    {
                        "name": "empty sequence work",
                        "layers": 1,
                        "mixers": [
                            {
                                "kind": "fixed_cost",
                                "work": {},
                                "cache": {},
                            }
                        ],
                    }
                ],
            },
            "deployment": {
                "weight_bits": 8,
                "kv_bits": 8,
                "include_kv_write": False,
                "include_index_write": False,
                "include_state_write": False,
                **(deployment or {}),
            },
        }
    )


def full_gqa_group(layers: int = 32) -> list[dict]:
    return [
        {
            "name": "gqa",
            "layers": layers,
            "mixers": [
                {
                    "kind": "softmax_attention",
                    "kv_layout": {
                        "kind": "grouped",
                        "query_heads": 32,
                        "kv_heads": 8,
                        "head_dim": 128,
                    },
                    "access": {"kind": "full"},
                }
            ],
        }
    ]


class DecodeEngineTests(unittest.TestCase):
    def test_gqa_matches_document_golden_example(self):
        config = engine_config(
            always_active_parameters=8_000_000_000,
            layer_groups=full_gqa_group(),
        )
        result = calculate_decode(
            config.model, config.deployment, [32768] * 64
        )

        expected_bytes = 8e9 / 64 + 2 * 32 * 8 * 128 * 32768
        expected_flops = 2 * 8e9 + 4 * 32 * 4096 * 32768
        self.assertEqual(result.per_output_work.total_bytes, expected_bytes)
        self.assertEqual(result.per_output_work.total_flops, expected_flops)
        self.assertAlmostEqual(result.bytes_per_flop, 0.06848983145165133)

    def test_dense_weights_amortize_but_request_work_does_not(self):
        config = engine_config(
            always_active_parameters=8_000_000_000,
            layer_groups=full_gqa_group(),
        )
        batch1 = calculate_decode(config.model, config.deployment, [8192])
        batch32 = calculate_decode(
            config.model, config.deployment, [8192] * 32
        )

        self.assertEqual(
            batch32.per_output_work.weight_read_bytes,
            batch1.per_output_work.weight_read_bytes / 32,
        )
        self.assertEqual(
            batch32.per_output_work.kv_read_bytes,
            batch1.per_output_work.kv_read_bytes,
        )
        self.assertEqual(
            batch32.per_output_work.total_flops,
            batch1.per_output_work.total_flops,
        )

    def test_moe_decode_uses_batch_expert_union(self):
        expert_group = {
            "name": "experts",
            "layers": 58,
            "expert_count": 256,
            "selected_per_token": 8,
            "parameters_per_expert": 44_040_192,
            "routing_mode": "uniform_independent",
        }
        config = engine_config(
            always_active_parameters=16_189_947_904,
            expert_groups=[expert_group],
        )
        result = calculate_decode(config.model, config.deployment, [0] * 32)
        group = config.model.weights.routed_expert_groups[0]
        expected_unique = 256 * (1 - (1 - 8 / 256) ** 32)
        expected_step_weight_bytes = (
            16_189_947_904
            + 58 * expected_unique * 44_040_192
        )

        self.assertAlmostEqual(expected_expert_reads(group, 32), expected_unique)
        self.assertAlmostEqual(
            result.step_work.weight_read_bytes, expected_step_weight_bytes
        )
        self.assertAlmostEqual(
            result.expert_weight_sets_read["experts"], expected_unique
        )

    def test_mla_bytes_and_scan_flops(self):
        layers = 61
        context = 1000
        config = engine_config(
            layer_groups=[
                {
                    "name": "mla",
                    "layers": layers,
                    "mixers": [
                        {
                            "kind": "softmax_attention",
                            "kv_layout": {
                                "kind": "latent",
                                "query_heads": 128,
                                "latent_dim": 512,
                                "rope_dim": 64,
                            },
                            "access": {"kind": "full"},
                        }
                    ],
                }
            ],
            deployment={"kv_bits": 16},
        )
        result = calculate_decode(config.model, config.deployment, [context])

        expected_bytes = layers * context * (512 + 64) * 2
        expected_flops = layers * context * 2 * 128 * (2 * 512 + 64)
        self.assertEqual(result.per_output_work.kv_read_bytes, expected_bytes)
        self.assertEqual(result.per_output_work.attention_flops, expected_flops)

    def test_named_attention_layouts_derive_bytes_and_flops(self):
        cases = [
            (
                {"kind": "mha", "query_heads": 4, "head_dim": 8},
                4 * 8 * 2,
                4 * 4 * 8,
            ),
            (
                {"kind": "mqa", "query_heads": 4, "head_dim": 8},
                1 * 8 * 2,
                4 * 4 * 8,
            ),
            (
                {
                    "kind": "gqa",
                    "query_heads": 4,
                    "kv_heads": 2,
                    "head_dim": 8,
                },
                2 * 8 * 2,
                4 * 4 * 8,
            ),
            (
                {
                    "kind": "mla",
                    "query_heads": 4,
                    "latent_dim": 6,
                    "rope_dim": 2,
                },
                6 + 2,
                2 * 4 * (2 * 6 + 2),
            ),
        ]

        for layout, bytes_per_entry, flops_per_entry in cases:
            with self.subTest(layout=layout["kind"]):
                config = engine_config(
                    layer_groups=[
                        {
                            "name": "named attention",
                            "layers": 1,
                            "mixers": [
                                {
                                    "kind": "softmax_attention",
                                    "kv_layout": layout,
                                    "access": {"kind": "full"},
                                }
                            ],
                        }
                    ]
                )
                result = calculate_decode(
                    config.model, config.deployment, [10]
                )
                self.assertEqual(
                    result.per_output_work.kv_read_bytes,
                    10 * bytes_per_entry,
                )
                self.assertEqual(
                    result.per_output_work.attention_flops,
                    10 * flops_per_entry,
                )

    def test_sliding_window_cost_saturates(self):
        config = engine_config(
            layer_groups=[
                {
                    "name": "local",
                    "layers": 2,
                    "mixers": [
                        {
                            "kind": "softmax_attention",
                            "kv_layout": {
                                "kind": "grouped",
                                "query_heads": 4,
                                "kv_heads": 1,
                                "head_dim": 8,
                            },
                            "access": {
                                "kind": "sliding_window",
                                "window_tokens": 16,
                            },
                        }
                    ],
                }
            ]
        )
        at_window = calculate_decode(config.model, config.deployment, [16])
        long_context = calculate_decode(config.model, config.deployment, [1000])
        self.assertEqual(
            at_window.per_output_work.kv_read_bytes,
            long_context.per_output_work.kv_read_bytes,
        )
        self.assertEqual(
            at_window.per_output_work.attention_flops,
            long_context.per_output_work.attention_flops,
        )

    def test_learned_topk_counts_main_and_index_paths(self):
        config = engine_config(
            layer_groups=[
                {
                    "name": "sparse",
                    "layers": 1,
                    "mixers": [
                        {
                            "kind": "softmax_attention",
                            "kv_layout": {
                                "kind": "grouped",
                                "query_heads": 2,
                                "kv_heads": 1,
                                "head_dim": 4,
                            },
                            "access": {
                                "kind": "learned_topk",
                                "top_k": 3,
                                "index_entry_elements": 2,
                                "index_query_heads": 1,
                                "index_head_dim": 2,
                            },
                        }
                    ],
                }
            ]
        )
        result = calculate_decode(config.model, config.deployment, [10])
        work = result.per_output_work
        cache = result.cache_capacity_per_request_average

        self.assertEqual(work.kv_read_bytes, 3 * 8)
        self.assertEqual(work.index_read_bytes, 10 * 2)
        self.assertEqual(work.attention_flops, 3 * 4 * 2 * 4)
        self.assertEqual(work.index_flops, 10 * 2 * 1 * 2)
        self.assertEqual(cache.kv_bytes, 10 * 8)
        self.assertEqual(cache.index_bytes, 10 * 2)

    def test_named_sparse_access_aliases(self):
        layout = {
            "kind": "gqa",
            "query_heads": 4,
            "kv_heads": 2,
            "head_dim": 8,
        }
        cases = [
            (
                {"kind": "swa", "window_tokens": 16},
                16,
                16,
                0,
            ),
            (
                {
                    "kind": "dsa",
                    "top_k": 3,
                    "index_entry_elements": 2,
                    "index_query_heads": 1,
                    "index_head_dim": 2,
                },
                3,
                40,
                40,
            ),
            (
                {
                    "kind": "csa",
                    "compression_ratio": 4,
                    "top_k": 3,
                    "index_entry_elements": 2,
                    "index_query_heads": 1,
                    "index_head_dim": 2,
                },
                3,
                10,
                10,
            ),
            (
                {"kind": "hca", "compression_ratio": 8},
                5,
                5,
                0,
            ),
        ]

        entry_bytes = 2 * 8 * 2
        entry_flops = 4 * 4 * 8
        for access, read_entries, stored_entries, index_entries in cases:
            with self.subTest(access=access["kind"]):
                config = engine_config(
                    layer_groups=[
                        {
                            "name": "named sparse attention",
                            "layers": 1,
                            "mixers": [
                                {
                                    "kind": "softmax_attention",
                                    "kv_layout": layout,
                                    "access": access,
                                }
                            ],
                        }
                    ]
                )
                result = calculate_decode(
                    config.model, config.deployment, [40]
                )
                work = result.per_output_work
                cache = result.cache_capacity_per_request_average
                self.assertEqual(work.kv_read_bytes, read_entries * entry_bytes)
                self.assertEqual(work.attention_flops, read_entries * entry_flops)
                self.assertEqual(cache.kv_bytes, stored_entries * entry_bytes)
                self.assertEqual(work.index_read_bytes, index_entries * 2)
                self.assertEqual(cache.index_bytes, index_entries * 2)

    def test_recurrent_state_is_context_independent(self):
        config = engine_config(
            layer_groups=[
                {
                    "name": "linear",
                    "layers": 4,
                    "mixers": [
                        {
                            "kind": "recurrent_state",
                            "state_elements": 128,
                            "read_elements_per_token": 128,
                            "write_elements_per_token": 128,
                            "flops_per_token": 1024,
                        }
                    ],
                }
            ],
            deployment={
                "state_bits": 16,
                "include_state_write": True,
            },
        )
        short = calculate_decode(config.model, config.deployment, [1])
        long = calculate_decode(config.model, config.deployment, [1_000_000])
        self.assertEqual(short.per_output_work, long.per_output_work)
        self.assertEqual(short.cache_capacity_total, long.cache_capacity_total)

    def test_named_state_mixers_derive_state_and_recurrence_flops(self):
        cases = [
            (
                {
                    "kind": "linear_attention",
                    "query_heads": 2,
                    "key_dim": 3,
                    "value_dim": 4,
                    "normalizer_state": True,
                },
                30,
                122,
            ),
            (
                {
                    "kind": "ssm",
                    "channels": 8,
                    "state_dim": 4,
                    "conv_state_length": 3,
                },
                56,
                160,
            ),
            (
                {
                    "kind": "mamba",
                    "variant": "mamba1",
                    "inner_dim": 8,
                    "state_dim": 4,
                    "conv_kernel": 4,
                },
                64,
                160,
            ),
            (
                {
                    "kind": "mamba2",
                    "inner_dim": 8,
                    "state_dim": 4,
                    "conv_kernel": 4,
                    "groups": 2,
                },
                128,
                160,
            ),
            (
                {
                    "kind": "mamba2",
                    "inner_dim": 8,
                    "ssm_dim": 4,
                    "state_dim": 4,
                    "conv_kernel": 4,
                    "groups": 2,
                },
                96,
                80,
            ),
        ]

        for mixer, state_elements, state_flops in cases:
            with self.subTest(mixer=mixer["kind"]):
                config = engine_config(
                    layer_groups=[
                        {
                            "name": "state mixer",
                            "layers": 1,
                            "mixers": [mixer],
                        }
                    ],
                    deployment={
                        "state_bits": 16,
                        "include_state_write": True,
                    },
                )
                result = calculate_decode(
                    config.model, config.deployment, [1_000_000]
                )
                work = result.per_output_work
                cache = result.cache_capacity_per_request_average
                expected_state_bytes = state_elements * 2
                self.assertEqual(cache.state_bytes, expected_state_bytes)
                self.assertEqual(work.state_read_bytes, expected_state_bytes)
                self.assertEqual(work.state_write_bytes, expected_state_bytes)
                self.assertEqual(work.state_flops, state_flops)

    def test_variable_context_batch_uses_request_sum(self):
        config = engine_config(layer_groups=full_gqa_group(layers=1))
        mixed = calculate_decode(config.model, config.deployment, [10, 30])
        mean = calculate_decode(config.model, config.deployment, [20, 20])
        self.assertEqual(
            mixed.per_output_work.kv_read_bytes,
            mean.per_output_work.kv_read_bytes,
        )
        self.assertEqual(
            mixed.per_output_work.attention_flops,
            mean.per_output_work.attention_flops,
        )

    def test_mechanism_catalog_example_is_executable(self):
        root = Path(__file__).resolve().parents[1]
        config = load_engine_config(
            root / "examples" / "mechanism_catalog.json"
        )
        result = calculate_decode(
            config.model, config.deployment, [4096]
        )
        self.assertGreater(result.per_output_work.attention_flops, 0)
        self.assertGreater(result.per_output_work.index_flops, 0)
        self.assertGreater(result.per_output_work.state_flops, 0)
        self.assertGreater(
            result.cache_capacity_per_request_average.state_bytes, 0
        )

    def test_invalid_expert_selection_is_rejected(self):
        with self.assertRaises(ConfigurationError):
            engine_config(
                expert_groups=[
                    {
                        "name": "bad",
                        "layers": 1,
                        "expert_count": 4,
                        "selected_per_token": 5,
                        "parameters_per_expert": 10,
                    }
                ]
            )

    def test_v4_official_one_million_context_anchor(self):
        root = Path(__file__).resolve().parents[1]
        config = load_engine_config(
            root / "configs" / "16bit" / "deepseek_v4_pro_16bit.json"
        )
        result = calculate_decode(
            config.model, config.deployment, [1_048_576]
        )

        # The arithmetic remains the report's approximately 0.3 TFLOP/token.
        # Cache/traffic reflect this repository's uniform 16-bit sweep profile.
        self.assertEqual(result.per_output_work.total_flops, 303_219_179_520)
        self.assertAlmostEqual(
            result.cache_capacity_total.total_bytes,
            8_824_422_400,
        )
        self.assertAlmostEqual(
            result.per_output_work.total_bytes,
            98_501_184_728,
        )

        shared = next(
            group
            for group in config.model.weights.always_active_parameter_groups
            if group.name == "16bit_shared_experts"
        )
        self.assertEqual(shared.weight_bits, 16)

    def test_all_shipped_model_configs_load_and_execute(self):
        """Every checked-in config must remain executable at its largest grid point."""

        root = Path(__file__).resolve().parents[1]
        config_paths = sorted((root / "configs").rglob("*.json"))
        self.assertGreater(len(config_paths), 0)

        for path in config_paths:
            with self.subTest(config=path.name):
                config = load_engine_config(path)
                self.assertGreater(len(config.default_contexts), 0)
                self.assertGreater(len(config.default_batches), 0)
                context = config.default_contexts[-1]
                batch = config.default_batches[-1]
                result = calculate_decode(
                    config.model,
                    config.deployment,
                    [context] * batch,
                )
                self.assertGreater(result.per_output_work.total_flops, 0)
                self.assertGreater(result.per_output_work.total_bytes, 0)

    def test_precision_profiles_have_no_weight_or_kv_override(self):
        """Local fields must agree with each directory's sweep precision."""

        root = Path(__file__).resolve().parents[1]
        kv_precision_keys = {
            "key_bits",
            "value_bits",
            "latent_bits",
            "rope_bits",
            "non_rope_bits",
        }
        embedding_elements_by_prefix = {
            "deepseek_r1_mla_": 7168,
            "deepseek_v4_pro_": 7168,
            "glm_5_2_dsa_": 6144,
            "llama_3_3_70b_": 8192,
            "mamba_2_8b_": 2560,
            "qwen3_235b_a22b_": 4096,
            "qwen3_4b_": 2560,
            "qwen3_8b_": 4096,
            "qwen3_next_80b_a3b_": 2048,
        }
        for bits in (4, 8, 16):
            config_paths = sorted(
                (root / "configs" / f"{bits}bit").glob("*.json")
            )
            self.assertEqual(len(config_paths), 9)

            for path in config_paths:
                with self.subTest(bits=bits, config=path.name):
                    config = load_engine_config(path)
                    deployment = config.deployment
                    weights = config.model.weights
                    self.assertEqual(deployment.weight_bits, bits)
                    self.assertEqual(deployment.expert_weight_bits, bits)
                    self.assertEqual(deployment.kv_bits, bits)

                    default_weight_bits = (
                        weights.weight_bits or deployment.weight_bits
                    )
                    for group in weights.always_active_parameter_groups:
                        self.assertEqual(
                            group.weight_bits or default_weight_bits,
                            bits,
                        )
                    for group in weights.routed_expert_groups:
                        self.assertEqual(
                            group.weight_bits
                            or deployment.expert_weight_bits,
                            bits,
                        )

                    fixed_reads: list[float] = []
                    for layer_group in config.model.layer_groups:
                        for mixer in layer_group.mixers:
                            if mixer.get("kind") == "fixed_cost":
                                fixed_reads.append(
                                    mixer.get("work", {}).get(
                                        "other_read_bytes", 0
                                    )
                                )
                            if mixer.get("kind") != "softmax_attention":
                                continue
                            layout = mixer["kv_layout"]
                            for key in kv_precision_keys & layout.keys():
                                self.assertEqual(layout[key], bits)

                    embedding_elements = next(
                        elements
                        for prefix, elements in (
                            embedding_elements_by_prefix.items()
                        )
                        if path.name.startswith(prefix)
                    )
                    self.assertEqual(
                        fixed_reads,
                        [embedding_elements * bits // 8],
                    )

    def test_new_official_config_active_parameter_anchors(self):
        """Lock the audited matrix-parameter decompositions in model metadata."""

        expected = {
            "glm_5_2_dsa_16bit.json": 40_297_758_720,
            "qwen3_235b_a22b_16bit.json": 21_567_635_456,
            "llama_3_3_70b_16bit.json": 69_501_714_432,
            "qwen3_8b_16bit.json": 7_568_097_280,
            "qwen3_4b_16bit.json": 4_022_272_000,
            "qwen3_next_80b_a3b_16bit.json": 3_563_552_768,
            "mamba_2_8b_16bit.json": 2_767_523_840,
        }
        root = Path(__file__).resolve().parents[1]

        for filename, expected_active in expected.items():
            with self.subTest(config=filename):
                config = load_engine_config(
                    root / "configs" / "16bit" / filename
                )
                weights = config.model.weights
                active = weights.always_active_parameters + sum(
                    group.layers
                    * group.selected_per_token
                    * group.parameters_per_expert
                    for group in weights.routed_expert_groups
                )
                self.assertEqual(active, expected_active)
                self.assertEqual(
                    config.model.metadata["active_parameters"],
                    expected_active,
                )

    def test_new_sparse_and_recurrent_config_state_anchors(self):
        root = Path(__file__).resolve().parents[1]

        glm = load_engine_config(
            root / "configs" / "16bit" / "glm_5_2_dsa_16bit.json"
        )
        glm_result = calculate_decode(glm.model, glm.deployment, [4096])
        # IndexShare stores/scans one 128-element 16-bit index key on only
        # 21 of the 78 layers; the other 57 layers reuse selected indices.
        expected_glm_index_bytes = 21 * 4096 * 128 * 2
        self.assertEqual(
            glm_result.per_output_work.index_read_bytes,
            expected_glm_index_bytes,
        )
        self.assertEqual(
            glm_result.cache_capacity_total.index_bytes,
            expected_glm_index_bytes,
        )

        qwen_next = load_engine_config(
            root
            / "configs"
            / "16bit"
            / "qwen3_next_80b_a3b_16bit.json"
        )
        qwen_next_result = calculate_decode(
            qwen_next.model, qwen_next.deployment, [4096]
        )
        self.assertEqual(
            qwen_next_result.cache_capacity_total.state_bytes,
            36 * (524_288 + 32_768) * 2,
        )

        mamba = load_engine_config(
            root / "configs" / "16bit" / "mamba_2_8b_16bit.json"
        )
        short = calculate_decode(mamba.model, mamba.deployment, [128])
        long = calculate_decode(mamba.model, mamba.deployment, [1_048_576])
        self.assertEqual(short.per_output_work, long.per_output_work)
        self.assertEqual(
            short.cache_capacity_total.state_bytes,
            64 * 102_400 * 2,
        )


if __name__ == "__main__":
    unittest.main()
