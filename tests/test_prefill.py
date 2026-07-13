from __future__ import annotations

import csv
import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from decode_engine.config import (
    ConfigurationError,
    load_engine_config,
    parse_engine_config,
)
from decode_engine.engine import (
    calculate_decode,
    calculate_prefill,
    calculate_prefill_grid,
    calculate_prefill_token_budget_grid,
    calculate_ragged_prefill_grid,
)
from decode_engine.mechanisms import build_mixer
from decode_engine.cli import main as cli_main


def prefill_config(
    *,
    always_active_parameters: float = 0,
    output_head_parameters: float = 0,
    layer_groups: list[dict] | None = None,
    expert_groups: list[dict] | None = None,
    deployment: dict | None = None,
    analysis: dict | None = None,
    max_context_tokens: int = 1_048_576,
):
    """Build a small config whose arithmetic can be checked by hand."""

    return parse_engine_config(
        {
            "schema_version": 1,
            "model": {
                "name": "prefill-test-model",
                "max_context_tokens": max_context_tokens,
                "weights": {
                    "always_active_parameters": always_active_parameters,
                    "output_head_parameters": output_head_parameters,
                    "routed_expert_groups": expert_groups or [],
                },
                "layer_groups": layer_groups
                or [
                    {
                        "name": "empty sequence work",
                        "layers": 1,
                        "mixers": [
                            {"kind": "fixed_cost", "work": {}, "cache": {}}
                        ],
                    }
                ],
            },
            "deployment": {
                "weight_bits": 8,
                "expert_weight_bits": 8,
                "kv_bits": 8,
                "index_bits": 8,
                "state_bits": 16,
                "include_kv_write": True,
                "include_index_write": True,
                "include_state_write": True,
                **(deployment or {}),
            },
            "analysis": analysis or {},
        }
    )


def attention_group(access: dict, *, layers: int = 1) -> list[dict]:
    """One tiny grouped-attention layer: 8 bytes and 32 FLOPs per entry."""

    return [
        {
            "name": "tiny attention",
            "layers": layers,
            "mixers": [
                {
                    "kind": "softmax_attention",
                    "kv_layout": {
                        "kind": "grouped",
                        "query_heads": 2,
                        "kv_heads": 1,
                        "head_dim": 4,
                    },
                    "access": access,
                }
            ],
        }
    ]


def built_attention_mixer(access: dict):
    config = prefill_config(layer_groups=attention_group(access))
    mixer_config = config.model.layer_groups[0].mixers[0]
    return build_mixer(mixer_config, "test.mixer"), config.deployment


class PrefillMechanismTests(unittest.TestCase):
    def test_full_attention_cached_prefix_golden(self):
        mixer, deployment = built_attention_mixer({"kind": "full"})

        cost = mixer.prefill_cost(2, 3, deployment, True)
        self.assertEqual(cost.work.attention_flops, 12 * 32)
        self.assertEqual(cost.work.kv_read_bytes, 2 * 8)
        self.assertEqual(cost.work.kv_write_bytes, 3 * 8)
        self.assertEqual(cost.operand_work.kv_read_bytes, 12 * 8)
        self.assertEqual(cost.cache.kv_bytes, 5 * 8)

        without_diagonal = mixer.prefill_cost(2, 3, deployment, False)
        self.assertEqual(without_diagonal.work.attention_flops, 9 * 32)
        self.assertEqual(without_diagonal.work.kv_read_bytes, 2 * 8)
        self.assertEqual(without_diagonal.work.kv_write_bytes, 3 * 8)
        self.assertEqual(without_diagonal.operand_work.kv_read_bytes, 9 * 8)
        self.assertEqual(without_diagonal.cache.kv_bytes, 5 * 8)

    def test_sliding_window_prefill_saturates(self):
        mixer, deployment = built_attention_mixer(
            {"kind": "swa", "window_tokens": 2}
        )
        cost = mixer.prefill_cost(0, 5, deployment)

        # Visible entries are 1, 2, 2, 2, 2.
        self.assertEqual(cost.work.attention_flops, 9 * 32)
        self.assertEqual(cost.work.kv_read_bytes, 0)
        self.assertEqual(cost.work.kv_write_bytes, 5 * 8)
        self.assertEqual(cost.operand_work.kv_read_bytes, 9 * 8)
        self.assertEqual(cost.cache.kv_bytes, 2 * 8)

    def test_dsa_prefill_separates_main_and_index_paths(self):
        access = {
            "kind": "dsa",
            "top_k": 2,
            "index_entry_elements": 2,
            "index_query_heads": 1,
            "index_head_dim": 2,
        }
        mixer, deployment = built_attention_mixer(access)

        fresh = mixer.prefill_cost(0, 4, deployment)
        # Main top-k entries are 1+2+2+2=7; index candidates are 1+2+3+4=10.
        self.assertEqual(fresh.work.attention_flops, 7 * 32)
        self.assertEqual(fresh.work.index_flops, 10 * 4)
        self.assertEqual(fresh.work.kv_read_bytes, 0)
        self.assertEqual(fresh.work.index_read_bytes, 0)
        self.assertEqual(fresh.operand_work.kv_read_bytes, 7 * 8)
        self.assertEqual(fresh.operand_work.index_read_bytes, 10 * 2)
        self.assertEqual(fresh.work.kv_write_bytes, 4 * 8)
        self.assertEqual(fresh.work.index_write_bytes, 4 * 2)
        self.assertEqual(fresh.cache.kv_bytes, 4 * 8)
        self.assertEqual(fresh.cache.index_bytes, 4 * 2)

        cached = mixer.prefill_cost(3, 2, deployment)
        self.assertEqual(cached.work.attention_flops, 4 * 32)
        self.assertEqual(cached.work.index_flops, 9 * 4)
        self.assertEqual(cached.work.kv_read_bytes, 3 * 8)
        self.assertEqual(cached.work.index_read_bytes, 3 * 2)
        self.assertEqual(cached.operand_work.kv_read_bytes, 4 * 8)
        self.assertEqual(cached.operand_work.index_read_bytes, 9 * 2)
        self.assertEqual(cached.work.kv_write_bytes, 2 * 8)
        self.assertEqual(cached.work.index_write_bytes, 2 * 2)
        self.assertEqual(cached.cache.kv_bytes, 5 * 8)
        self.assertEqual(cached.cache.index_bytes, 5 * 2)

    def test_hca_and_csa_cross_compression_boundaries(self):
        hca, deployment = built_attention_mixer(
            {"kind": "hca", "compression_ratio": 3}
        )
        included = hca.prefill_cost(2, 5, deployment, True)
        excluded = hca.prefill_cost(2, 5, deployment, False)
        # floor([3,4,5,6,7]/3) sums to 7; excluding the diagonal evaluates
        # [2,3,4,5,6] and sums to 5.  Two complete entries exist at the end.
        self.assertEqual(included.work.attention_flops, 7 * 32)
        self.assertEqual(included.operand_work.kv_read_bytes, 7 * 8)
        self.assertEqual(included.work.kv_write_bytes, 2 * 8)
        self.assertEqual(included.cache.kv_bytes, 2 * 8)
        self.assertEqual(excluded.work.attention_flops, 5 * 32)

        csa, deployment = built_attention_mixer(
            {
                "kind": "csa",
                "compression_ratio": 3,
                "top_k": 2,
                "index_entry_elements": 2,
                "index_query_heads": 1,
                "index_head_dim": 2,
            }
        )
        cost = csa.prefill_cost(2, 5, deployment, True)
        self.assertEqual(cost.work.attention_flops, 7 * 32)
        self.assertEqual(cost.work.index_flops, 7 * 4)
        self.assertEqual(cost.operand_work.kv_read_bytes, 7 * 8)
        self.assertEqual(cost.operand_work.index_read_bytes, 7 * 2)
        self.assertEqual(cost.work.kv_write_bytes, 2 * 8)
        self.assertEqual(cost.work.index_write_bytes, 2 * 2)
        self.assertEqual(cost.cache.kv_bytes, 2 * 8)
        self.assertEqual(cost.cache.index_bytes, 2 * 2)

    def test_recurrent_prefill_uses_one_fused_state_transaction(self):
        config = prefill_config(
            layer_groups=[
                {
                    "name": "state",
                    "layers": 1,
                    "mixers": [
                        {
                            "kind": "recurrent_state",
                            "state_elements": 4,
                            "read_elements_per_token": 4,
                            "write_elements_per_token": 4,
                            "flops_per_token": 10,
                        }
                    ],
                }
            ]
        )
        mixer = build_mixer(
            config.model.layer_groups[0].mixers[0], "test.state_mixer"
        )

        fresh = mixer.prefill_cost(0, 3, config.deployment)
        self.assertEqual(fresh.work.state_flops, 30)
        self.assertEqual(fresh.work.state_read_bytes, 0)
        self.assertEqual(fresh.work.state_write_bytes, 8)
        self.assertEqual(fresh.operand_work, fresh.work)
        self.assertEqual(fresh.cache.state_bytes, 8)

        cached = mixer.prefill_cost(2, 3, config.deployment)
        self.assertEqual(cached.work.state_flops, 30)
        self.assertEqual(cached.work.state_read_bytes, 8)
        self.assertEqual(cached.work.state_write_bytes, 8)
        self.assertEqual(cached.operand_work, cached.work)
        self.assertEqual(cached.cache.state_bytes, 8)

    def test_fixed_cost_prefill_scope(self):
        deployment = prefill_config().deployment
        per_token = build_mixer(
            {
                "kind": "fixed_cost",
                "work": {"other_read_bytes": 5},
                "cache": {},
            },
            "test.fixed.per_token",
        )
        per_request = build_mixer(
            {
                "kind": "fixed_cost",
                "work": {"other_read_bytes": 5},
                "cache": {},
                "prefill_scope": "per_request",
            },
            "test.fixed.per_request",
        )
        self.assertEqual(
            per_token.prefill_cost(0, 3, deployment).work.other_read_bytes,
            15,
        )
        self.assertEqual(
            per_request.prefill_cost(0, 3, deployment).work.other_read_bytes,
            5,
        )


class PrefillEngineTests(unittest.TestCase):
    def setUp(self):
        self.config = prefill_config(
            always_active_parameters=100,
            layer_groups=attention_group({"kind": "full"}),
        )

    def test_single_full_prefill_total_and_per_input_work(self):
        result = calculate_prefill(
            self.config.model, self.config.deployment, [4]
        )

        self.assertEqual(result.valid_input_tokens, 4)
        self.assertEqual(result.executed_input_tokens, 4)
        self.assertEqual(result.valid_causal_pair_slots, 10)
        self.assertEqual(result.executed_causal_pair_slots, 10)
        self.assertEqual(result.batch_work.parameter_flops, 800)
        self.assertEqual(result.batch_work.attention_flops, 320)
        self.assertEqual(result.batch_work.total_flops, 1120)
        self.assertEqual(result.batch_work.weight_read_bytes, 100)
        self.assertEqual(result.batch_work.kv_read_bytes, 0)
        self.assertEqual(result.batch_work.kv_write_bytes, 32)
        self.assertEqual(result.batch_work.total_bytes, 132)
        self.assertEqual(result.batch_operand_work.kv_read_bytes, 80)
        self.assertEqual(result.batch_operand_work.total_bytes, 212)
        self.assertEqual(result.cache_capacity_total.kv_bytes, 32)
        self.assertEqual(result.per_input_work.total_flops, 280)
        self.assertEqual(result.per_input_work.total_bytes, 33)
        self.assertEqual(result.to_dict()["phase"], "prefill")

    def test_experiment_one_fixed_batch_length_sweep(self):
        results = calculate_prefill_grid(
            self.config, prompt_lengths=[2, 4], batches=[2]
        )
        short, long = results

        self.assertEqual(short.experiment, "equal")
        self.assertEqual(short.prompt_tokens, (2, 2))
        self.assertEqual(short.valid_input_tokens, 4)
        self.assertEqual(short.valid_causal_pair_slots, 6)
        self.assertEqual(short.batch_work.total_flops, 992)
        self.assertEqual(short.batch_work.total_bytes, 132)
        self.assertEqual(short.batch_operand_work.total_bytes, 180)
        self.assertEqual(short.cache_capacity_total.kv_bytes, 32)

        self.assertEqual(long.prompt_tokens, (4, 4))
        self.assertEqual(long.valid_input_tokens, 8)
        self.assertEqual(long.valid_causal_pair_slots, 20)
        self.assertEqual(long.batch_work.total_flops, 2240)
        self.assertEqual(long.batch_work.total_bytes, 164)
        self.assertEqual(long.batch_operand_work.total_bytes, 324)
        self.assertEqual(long.cache_capacity_total.kv_bytes, 64)

    def test_experiment_two_fixed_token_budget_changes_attention_shape(self):
        results = calculate_prefill_token_budget_grid(
            self.config, token_budgets=[4], batches=[1, 2]
        )
        one_long, two_short = results

        self.assertEqual(one_long.experiment, "token-budget")
        self.assertEqual(one_long.prompt_tokens, (4,))
        self.assertEqual(two_short.prompt_tokens, (2, 2))
        self.assertEqual(one_long.valid_input_tokens, 4)
        self.assertEqual(two_short.valid_input_tokens, 4)
        self.assertEqual(one_long.valid_causal_pair_slots, 10)
        self.assertEqual(two_short.valid_causal_pair_slots, 6)
        self.assertEqual(one_long.batch_work.parameter_flops, 800)
        self.assertEqual(two_short.batch_work.parameter_flops, 800)
        self.assertEqual(one_long.batch_work.total_flops, 1120)
        self.assertEqual(two_short.batch_work.total_flops, 992)
        self.assertEqual(one_long.batch_work.total_bytes, 132)
        self.assertEqual(two_short.batch_work.total_bytes, 132)
        self.assertEqual(one_long.batch_operand_work.total_bytes, 212)
        self.assertEqual(two_short.batch_operand_work.total_bytes, 180)

        non_divisible = calculate_prefill_token_budget_grid(
            self.config, token_budgets=[10], batches=[3]
        )[0]
        self.assertEqual(non_divisible.prompt_tokens, (4, 3, 3))
        self.assertEqual(non_divisible.valid_input_tokens, 10)

    def test_experiment_three_ragged_varlen_and_padded(self):
        varlen = calculate_ragged_prefill_grid(
            self.config, ragged_batches=[[1, 3]], execution_mode="varlen"
        )[0]
        padded = calculate_ragged_prefill_grid(
            self.config, ragged_batches=[[1, 3]], execution_mode="padded"
        )[0]

        self.assertEqual(varlen.experiment, "ragged")
        self.assertEqual(varlen.prompt_tokens, (1, 3))
        self.assertEqual(varlen.valid_input_tokens, 4)
        self.assertEqual(varlen.executed_input_tokens, 4)
        self.assertEqual(varlen.valid_causal_pair_slots, 7)
        self.assertEqual(varlen.executed_causal_pair_slots, 7)
        self.assertEqual(varlen.batch_work.total_flops, 1024)
        self.assertEqual(varlen.batch_work.total_bytes, 132)
        self.assertEqual(varlen.batch_operand_work.total_bytes, 188)

        self.assertEqual(padded.valid_input_tokens, 4)
        self.assertEqual(padded.executed_input_tokens, 6)
        self.assertEqual(padded.valid_causal_pair_slots, 7)
        self.assertEqual(padded.executed_causal_pair_slots, 12)
        self.assertEqual(padded.useful_work, varlen.batch_work)
        self.assertEqual(padded.batch_work.parameter_flops, 1200)
        self.assertEqual(padded.batch_work.attention_flops, 384)
        self.assertEqual(padded.batch_work.total_flops, 1584)
        # Padding compute does not persist invalid KV entries.
        self.assertEqual(padded.batch_work.kv_write_bytes, 32)
        self.assertEqual(padded.batch_work.total_bytes, 132)
        self.assertEqual(padded.batch_operand_work.kv_read_bytes, 96)
        self.assertEqual(padded.batch_operand_work.total_bytes, 228)
        self.assertEqual(padded.cache_capacity_total.kv_bytes, 32)
        self.assertAlmostEqual(padded.token_efficiency, 2 / 3)
        self.assertAlmostEqual(padded.causal_pair_efficiency, 7 / 12)

    def test_prefix_cache_and_causal_diagonal_are_explicit(self):
        with_diagonal = calculate_prefill(
            self.config.model,
            self.config.deployment,
            [3],
            cached_context_tokens=[2],
        )
        without_diagonal = calculate_prefill(
            self.config.model,
            self.config.deployment,
            [3],
            cached_context_tokens=[2],
            include_self_attention=False,
        )

        self.assertEqual(with_diagonal.valid_causal_pair_slots, 12)
        self.assertEqual(with_diagonal.batch_work.attention_flops, 384)
        self.assertEqual(with_diagonal.batch_work.kv_read_bytes, 16)
        self.assertEqual(with_diagonal.cache_capacity_total.kv_bytes, 40)
        self.assertEqual(without_diagonal.valid_causal_pair_slots, 9)
        self.assertEqual(without_diagonal.batch_work.attention_flops, 288)
        self.assertEqual(without_diagonal.cache_capacity_total.kv_bytes, 40)

        dsa_config = prefill_config(
            layer_groups=attention_group(
                {
                    "kind": "dsa",
                    "top_k": 2,
                    "index_entry_elements": 2,
                    "index_query_heads": 1,
                    "index_head_dim": 2,
                }
            )
        )
        dsa_cached = calculate_prefill(
            dsa_config.model,
            dsa_config.deployment,
            [2],
            cached_context_tokens=[3],
        )
        self.assertEqual(
            dsa_cached.topk_cached_prefix_union_policy,
            "conservative_distinct_upper_bound",
        )

    def test_lm_head_last_and_all_modes(self):
        config = prefill_config(
            always_active_parameters=13,
            output_head_parameters=3,
        )
        last = calculate_prefill(
            config.model, config.deployment, [2, 4], logits_mode="last"
        )
        all_positions = calculate_prefill(
            config.model, config.deployment, [2, 4], logits_mode="all"
        )
        no_logits = calculate_prefill(
            config.model, config.deployment, [2, 4], logits_mode="none"
        )

        # Body=10 parameters over six inputs.  Head=3 parameters over either
        # two final positions or all six positions.  MACs count as two FLOPs.
        self.assertEqual(last.batch_work.parameter_flops, 132)
        self.assertEqual(all_positions.batch_work.parameter_flops, 156)
        self.assertEqual(no_logits.batch_work.parameter_flops, 120)
        self.assertEqual(last.batch_work.weight_read_bytes, 13)
        self.assertEqual(all_positions.batch_work.weight_read_bytes, 13)
        self.assertEqual(no_logits.batch_work.weight_read_bytes, 10)
        self.assertEqual(no_logits.executed_logit_positions, 0)

        # Decode still executes the complete decode-era always-active total.
        decode = calculate_decode(config.model, config.deployment, [0])
        self.assertEqual(decode.per_output_work.parameter_flops, 26)

    def test_moe_prefill_uses_routed_token_count_not_request_count(self):
        config = prefill_config(
            expert_groups=[
                {
                    "name": "experts",
                    "layers": 2,
                    "expert_count": 4,
                    "selected_per_token": 1,
                    "parameters_per_expert": 10,
                    "routing_mode": "uniform_independent",
                }
            ]
        )
        result = calculate_prefill(
            config.model, config.deployment, [1, 2]
        )
        expected_unique = 4 * (1 - (1 - 1 / 4) ** 3)

        self.assertEqual(expected_unique, 37 / 16)
        self.assertAlmostEqual(
            result.expert_weight_sets_read["experts"], expected_unique
        )
        self.assertAlmostEqual(result.batch_work.weight_read_bytes, 46.25)
        self.assertEqual(result.batch_work.parameter_flops, 120)

    def test_explicit_moe_trace_uses_phase_specific_axis(self):
        config = prefill_config(
            expert_groups=[
                {
                    "name": "experts",
                    "layers": 1,
                    "expert_count": 4,
                    "selected_per_token": 1,
                    "parameters_per_expert": 10,
                    "routing_mode": "explicit_unique",
                    "expected_unique_experts_by_batch": {"2": 1.5},
                    "expected_unique_experts_by_active_tokens": {"3": 2.5},
                }
            ]
        )
        decode = calculate_decode(config.model, config.deployment, [0, 0])
        prefill = calculate_prefill(
            config.model, config.deployment, [1, 2]
        )

        self.assertEqual(decode.expert_weight_sets_read["experts"], 1.5)
        self.assertEqual(decode.step_work.weight_read_bytes, 15)
        self.assertEqual(prefill.expert_weight_sets_read["experts"], 2.5)
        self.assertEqual(prefill.batch_work.weight_read_bytes, 25)

    def test_ssm_ragged_and_padded_state_is_per_request(self):
        config = prefill_config(
            layer_groups=[
                {
                    "name": "state",
                    "layers": 1,
                    "mixers": [
                        {
                            "kind": "recurrent_state",
                            "state_elements": 4,
                            "read_elements_per_token": 4,
                            "write_elements_per_token": 4,
                            "flops_per_token": 10,
                        }
                    ],
                }
            ]
        )
        varlen = calculate_prefill(
            config.model, config.deployment, [2, 4]
        )
        padded = calculate_prefill(
            config.model,
            config.deployment,
            [2, 4],
            execution_mode="padded",
        )

        self.assertEqual(varlen.batch_work.state_flops, 60)
        self.assertEqual(varlen.batch_work.state_read_bytes, 0)
        self.assertEqual(varlen.batch_work.state_write_bytes, 16)
        self.assertEqual(varlen.batch_operand_work, varlen.batch_work)
        self.assertEqual(varlen.cache_capacity_total.state_bytes, 16)

        self.assertEqual(padded.batch_work.state_flops, 80)
        self.assertEqual(padded.batch_work.state_read_bytes, 0)
        self.assertEqual(padded.batch_work.state_write_bytes, 16)
        self.assertEqual(padded.batch_operand_work, padded.batch_work)
        self.assertEqual(padded.cache_capacity_total.state_bytes, 16)

    def test_packed_all_logits_and_padded_moe_use_executed_positions(self):
        head_config = prefill_config(
            always_active_parameters=13,
            output_head_parameters=3,
        )
        varlen = calculate_prefill(
            head_config.model,
            head_config.deployment,
            [1, 3],
            execution_mode="varlen",
            logits_mode="all",
        )
        packed = calculate_prefill(
            head_config.model,
            head_config.deployment,
            [1, 3],
            execution_mode="packed",
            logits_mode="all",
        )
        padded = calculate_prefill(
            head_config.model,
            head_config.deployment,
            [1, 3],
            execution_mode="padded",
            logits_mode="all",
        )
        self.assertEqual(packed.batch_work, varlen.batch_work)
        self.assertEqual(varlen.executed_logit_positions, 4)
        self.assertEqual(padded.executed_logit_positions, 6)
        self.assertEqual(varlen.batch_work.parameter_flops, 104)
        self.assertEqual(padded.batch_work.parameter_flops, 156)

        moe_config = prefill_config(
            expert_groups=[
                {
                    "name": "experts",
                    "layers": 1,
                    "expert_count": 4,
                    "selected_per_token": 1,
                    "parameters_per_expert": 10,
                }
            ]
        )
        moe_padded = calculate_prefill(
            moe_config.model,
            moe_config.deployment,
            [1, 3],
            execution_mode="padded",
        )
        expected_useful = 4 * (1 - (3 / 4) ** 4)
        expected_executed = 4 * (1 - (3 / 4) ** 6)
        self.assertAlmostEqual(
            moe_padded.useful_expert_weight_sets_read["experts"],
            expected_useful,
        )
        self.assertAlmostEqual(
            moe_padded.expert_weight_sets_read["experts"],
            expected_executed,
        )

    def test_prefill_deployment_fractions_and_per_input_extras(self):
        config = prefill_config(
            always_active_parameters=100,
            layer_groups=attention_group({"kind": "full"}),
            deployment={
                "weight_hbm_fraction": 0.5,
                "kv_hbm_fraction": 0.5,
                "activation_bytes_per_input_token": 5,
                "extra_flops_per_input_token": 7,
            },
        )
        result = calculate_prefill(
            config.model, config.deployment, [2]
        )
        self.assertEqual(result.batch_work.weight_read_bytes, 50)
        self.assertEqual(result.batch_work.kv_write_bytes, 8)
        self.assertEqual(result.batch_operand_work.kv_read_bytes, 12)
        self.assertEqual(result.batch_work.activation_bytes, 10)
        self.assertEqual(result.batch_work.extra_flops, 14)

        no_write = prefill_config(
            layer_groups=attention_group({"kind": "full"}),
            deployment={"include_kv_write": False},
        )
        self.assertEqual(
            calculate_prefill(
                no_write.model, no_write.deployment, [2]
            ).batch_work.kv_write_bytes,
            0,
        )

    def test_prefill_input_validation(self):
        for prompts in ([], [0], [True], [1.5]):
            with self.subTest(prompts=prompts), self.assertRaises(ValueError):
                calculate_prefill(
                    self.config.model, self.config.deployment, prompts
                )

        with self.assertRaises(ValueError):
            calculate_prefill(
                self.config.model,
                self.config.deployment,
                [1, 2],
                cached_context_tokens=[0],
            )
        with self.assertRaises(ValueError):
            calculate_prefill(
                self.config.model,
                self.config.deployment,
                [1],
                execution_mode="padded",
                cached_context_tokens=[1],
            )
        with self.assertRaises(ValueError):
            calculate_prefill(
                self.config.model,
                self.config.deployment,
                [1],
                logits_mode="unsupported",
            )

        with self.assertRaises(ValueError):
            calculate_prefill_grid(
                self.config, prompt_lengths=[], batches=[1]
            )
        with self.assertRaises(ValueError):
            calculate_prefill_grid(
                self.config,
                prompt_lengths=(value for value in ()),
                batches=[1],
            )

        short_context = prefill_config(max_context_tokens=4)
        with self.assertRaises(ValueError):
            calculate_prefill(
                short_context.model, short_context.deployment, [5]
            )
        with self.assertRaises(ValueError):
            calculate_prefill(
                short_context.model,
                short_context.deployment,
                [2],
                cached_context_tokens=[3],
            )


class PrefillConfigurationTests(unittest.TestCase):
    def test_prefill_analysis_and_deployment_fields_parse(self):
        config = prefill_config(
            always_active_parameters=13,
            output_head_parameters=3,
            deployment={
                "activation_bytes_per_input_token": 5,
                "extra_flops_per_input_token": 7,
            },
            analysis={
                "prefill": {
                    "prompt_lengths": [2, 4],
                    "batches": [1, 2],
                    "token_budgets": [4, 8],
                    "ragged_batches": [[1, 3], [2, 2, 4]],
                }
            },
        )

        self.assertEqual(config.model.weights.output_head_parameters, 3)
        self.assertTrue(
            config.model.weights.output_head_parameters_configured
        )
        self.assertEqual(config.model.weights.backbone_parameters, 10)
        self.assertEqual(config.deployment.activation_bytes_per_input_token, 5)
        self.assertEqual(config.deployment.extra_flops_per_input_token, 7)
        self.assertEqual(config.default_prefill_lengths, (2, 4))
        self.assertEqual(config.default_prefill_batches, (1, 2))
        self.assertEqual(config.default_prefill_token_budgets, (4, 8))
        self.assertEqual(
            config.default_ragged_prefill_batches,
            ((1, 3), (2, 2, 4)),
        )

        default_grid = calculate_prefill_grid(config)
        self.assertEqual(
            [result.prompt_tokens for result in default_grid],
            [(2,), (4,), (2, 2), (4, 4)],
        )
        default_budget_grid = calculate_prefill_token_budget_grid(config)
        self.assertEqual(len(default_budget_grid), 4)
        default_ragged_grid = calculate_ragged_prefill_grid(config)
        self.assertEqual(
            [result.prompt_tokens for result in default_ragged_grid],
            [(1, 3), (2, 2, 4)],
        )

    def test_invalid_output_head_and_ragged_config_are_rejected(self):
        with self.assertRaises(ConfigurationError):
            prefill_config(
                always_active_parameters=10,
                output_head_parameters=11,
            )
        with self.assertRaises(ConfigurationError):
            prefill_config(
                always_active_parameters=10,
                output_head_parameters=-1,
            )
        with self.assertRaises(ConfigurationError):
            prefill_config(analysis={"prefill": {"ragged_batches": [[]]}})
        for impossible_union in (1, 5):
            with self.subTest(unique=impossible_union), self.assertRaises(
                ConfigurationError
            ):
                prefill_config(
                    expert_groups=[
                        {
                            "name": "experts",
                            "layers": 1,
                            "expert_count": 8,
                            "selected_per_token": 2,
                            "parameters_per_expert": 10,
                            "routing_mode": "explicit_unique",
                            "expected_unique_experts_by_active_tokens": {
                                "2": impossible_union
                            },
                        }
                    ]
                )

    def test_all_precision_configs_have_audited_output_head(self):
        expected_by_stem = {
            "deepseek_r1_mla": 926_679_040,
            "deepseek_v4_pro": 926_679_040,
            "glm_5_2_dsa": 951_582_720,
            "qwen3_235b_a22b": 622_329_856,
            "llama_3_3_70b": 1_050_673_152,
            "qwen3_8b": 622_329_856,
            "qwen3_4b": 388_956_160,
            "qwen3_next_80b_a3b": 311_164_928,
            "mamba_2_8b": 128_716_800,
        }
        root = Path(__file__).resolve().parents[1]
        checked = 0

        for bits in (4, 8, 16):
            paths = sorted((root / "configs" / f"{bits}bit").glob("*.json"))
            self.assertEqual(len(paths), 9)
            for path in paths:
                with self.subTest(bits=bits, config=path.name):
                    stem = path.stem.removesuffix(f"_{bits}bit")
                    config = load_engine_config(path)
                    self.assertEqual(
                        config.model.weights.output_head_parameters,
                        expected_by_stem[stem],
                    )
                    self.assertLessEqual(
                        config.model.weights.output_head_parameters,
                        config.model.weights.always_active_parameters,
                    )
                    self.assertTrue(
                        config.model.weights.output_head_parameters_configured
                    )
                    self.assertEqual(
                        config.model.weights.output_head_weight_bits,
                        bits,
                    )
                    checked += 1

        self.assertEqual(checked, 27)

    def test_all_shipped_configs_execute_prefill(self):
        root = Path(__file__).resolve().parents[1]
        paths = sorted((root / "configs").rglob("*.json"))
        self.assertEqual(len(paths), 27)

        for path in paths:
            with self.subTest(config=str(path.relative_to(root))):
                config = load_engine_config(path)
                result = calculate_prefill(
                    config.model,
                    config.deployment,
                    [8, 4],
                )
                self.assertGreater(result.batch_work.total_flops, 0)
                self.assertGreater(result.batch_work.total_bytes, 0)
                self.assertGreaterEqual(
                    result.batch_operand_work.total_bytes,
                    result.batch_work.total_bytes,
                )
                self.assertGreater(result.cache_capacity_total.total_bytes, 0)

    def test_mechanism_catalog_executes_prefill(self):
        root = Path(__file__).resolve().parents[1]
        config = load_engine_config(
            root / "examples" / "mechanism_catalog.json"
        )
        result = calculate_prefill(
            config.model,
            config.deployment,
            [8, 4],
        )
        self.assertGreater(result.batch_work.attention_flops, 0)
        self.assertGreater(result.batch_work.index_flops, 0)
        self.assertGreater(result.batch_work.state_flops, 0)
        self.assertGreater(result.cache_capacity_total.kv_bytes, 0)
        self.assertGreater(result.cache_capacity_total.index_bytes, 0)
        self.assertGreater(result.cache_capacity_total.state_bytes, 0)


class PrefillCLITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cls.config_path = str(
            root / "configs" / "16bit" / "qwen3_8b_16bit.json"
        )

    def run_cli(self, *args: str) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = cli_main(list(args))
        return code, stdout.getvalue(), stderr.getvalue()

    def test_equal_prefill_json(self):
        code, output, error = self.run_cli(
            "--config",
            self.config_path,
            "--phase",
            "prefill",
            "--experiment",
            "equal",
            "--prompt-lengths",
            "2",
            "--batches",
            "1",
            "--format",
            "json",
        )
        self.assertEqual((code, error), (0, ""))
        row = json.loads(output)[0]
        self.assertEqual(row["experiment"], "equal")
        self.assertEqual(row["prompt_tokens"], [2])
        self.assertTrue(row["output_head_parameters_configured"])

        code, output, error = self.run_cli(
            "--config",
            self.config_path,
            "--phase",
            "prefill",
            "--experiment",
            "equal",
            "--prompt-lengths",
            "2",
            "--batches",
            "1",
            "--logits-mode",
            "none",
            "--format",
            "json",
        )
        self.assertEqual((code, error), (0, ""))
        self.assertEqual(json.loads(output)[0]["executed_logit_positions"], 0)

    def test_token_budget_csv_and_ragged_padded_table(self):
        code, output, error = self.run_cli(
            "--config",
            self.config_path,
            "--phase",
            "prefill",
            "--experiment",
            "token-budget",
            "--token-budgets",
            "4",
            "--batches",
            "1",
            "2",
            "--format",
            "csv",
        )
        self.assertEqual((code, error), (0, ""))
        rows = list(csv.DictReader(io.StringIO(output)))
        self.assertEqual([row["experiment"] for row in rows], [
            "token-budget",
            "token-budget",
        ])
        self.assertEqual(
            [int(row["valid_input_tokens"]) for row in rows],
            [4, 4],
        )

        code, output, error = self.run_cli(
            "--config",
            self.config_path,
            "--phase",
            "prefill",
            "--experiment",
            "ragged",
            "--ragged-lengths",
            "1",
            "3",
            "--execution-mode",
            "padded",
        )
        self.assertEqual((code, error), (0, ""))
        self.assertIn("4/6", output)
        self.assertIn("ragged", output)

    def test_decode_default_and_prefill_option_conflict(self):
        code, output, error = self.run_cli(
            "--config",
            self.config_path,
            "--contexts",
            "1",
            "--batches",
            "1",
        )
        self.assertEqual((code, error), (0, ""))
        self.assertIn("GFLOP/token", output)

        code, _output, error = self.run_cli(
            "--config",
            self.config_path,
            "--prompt-lengths",
            "2",
        )
        self.assertEqual(code, 2)
        self.assertIn("require --phase prefill", error)


if __name__ == "__main__":
    unittest.main()
