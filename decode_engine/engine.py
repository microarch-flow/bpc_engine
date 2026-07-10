"""Decode-step aggregation and batch normalization.

The central rule is simple but important: calculate the complete scheduler
step first, then divide by the number of output tokens produced by that step.
This ensures that shared weight reads are amortized while request-private KV
and recurrent-state traffic is not accidentally divided twice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from .mechanisms import build_mixer
from .schema import (
    CacheCapacity,
    DecodeResult,
    DeploymentConfig,
    EngineConfig,
    ModelConfig,
    RoutedExpertGroup,
    WorkCost,
)


@dataclass(frozen=True)
class _WeightStepResult:
    work: WorkCost
    expected_expert_reads: dict[str, float]


def expected_expert_reads(group: RoutedExpertGroup, batch_size: int) -> float:
    """Estimate expert weight sets read once during one decode step.

    The default formula assumes each token independently chooses ``k`` distinct
    experts uniformly from ``E`` experts.  For a particular expert, the chance
    of not being selected by one token is ``1-k/E``; after ``B`` tokens it is
    ``(1-k/E)^B``.  Summing the probability of being touched over all experts
    gives the expected union size.

    Real routing is not perfectly uniform.  The configuration therefore also
    supports a measured/assumed explicit union and two boundary policies.
    """

    experts = group.expert_count
    selected = group.selected_per_token
    mode = group.routing_mode

    if mode == "uniform_independent":
        return experts * (1.0 - (1.0 - selected / experts) ** batch_size)
    if mode == "same_experts":
        # Best reuse boundary: every request happens to select the same experts.
        return float(selected)
    if mode == "explicit_unique":
        if batch_size not in group.expected_unique_experts_by_batch:
            raise ValueError(
                f"{group.name} has no explicit unique-expert value for "
                f"batch={batch_size}"
            )
        return float(group.expected_unique_experts_by_batch[batch_size])
    if mode == "no_batch_reuse":
        # Traffic upper boundary.  The same expert may be counted repeatedly
        # because its weights are assumed to be re-read for different tokens.
        return float(batch_size * selected)
    raise ValueError(f"unsupported routing mode {mode!r} in {group.name}")


def _weight_step_cost(
    model: ModelConfig, deployment: DeploymentConfig, batch_size: int
) -> _WeightStepResult:
    weights = model.weights
    shared_bits = weights.weight_bits or deployment.weight_bits

    # Every output token executes all always-active parameterized operations,
    # but their weights can be read once and reused by the batch GEMM.  Groups
    # allow attention, shared experts, and special projections to use different
    # storage precision without changing the arithmetic parameter count.
    active_parameters_per_output = weights.always_active_parameters
    weight_read_bytes = sum(
        group.parameters * (group.weight_bits or shared_bits) / 8.0
        for group in weights.always_active_parameter_groups
    )
    expert_reads: dict[str, float] = {}

    for group in weights.routed_expert_groups:
        active_parameters_per_output += (
            group.layers
            * group.selected_per_token
            * group.parameters_per_expert
        )

        reads = expected_expert_reads(group, batch_size)
        expert_reads[group.name] = reads
        bits = group.weight_bits or deployment.expert_weight_bits
        weight_read_bytes += (
            group.layers * reads * group.parameters_per_expert * bits / 8.0
        )

    # Parameterized matrix work is performed once per request, even though the
    # weight bytes may be shared across the batch.
    parameter_flops = (
        deployment.mac_flops * batch_size * active_parameters_per_output
    )
    weight_read_bytes *= (
        deployment.weight_hbm_fraction * deployment.weight_read_multiplier
    )

    return _WeightStepResult(
        WorkCost(
            parameter_flops=parameter_flops,
            weight_read_bytes=weight_read_bytes,
        ),
        expert_reads,
    )


def _request_sequence_cost(
    model: ModelConfig,
    deployment: DeploymentConfig,
    context_tokens: int,
) -> tuple[WorkCost, CacheCapacity]:
    """Calculate sequence-mixer work for one request in one decode step."""

    request_work = WorkCost()
    request_cache = CacheCapacity()

    for group_index, group in enumerate(model.layer_groups):
        for mixer_index, mixer_config in enumerate(group.mixers):
            path = f"model.layer_groups[{group_index}].mixers[{mixer_index}]"
            mixer = build_mixer(mixer_config, path)
            cost = mixer.decode_cost(context_tokens, deployment).scaled(group.layers)
            request_work = request_work + cost.work
            request_cache = request_cache + cost.cache

    return request_work, request_cache


def calculate_decode(
    model: ModelConfig,
    deployment: DeploymentConfig,
    context_tokens: Sequence[int],
) -> DecodeResult:
    """Calculate one decode step and normalize it per generated token.

    Args:
        model: Normalized model architecture and weight-access configuration.
        deployment: Precision and off-chip residency assumptions.
        context_tokens: One existing-context length for each active request.
            Supplying ``[32768] * 32`` describes a batch of 32 equal-length
            requests.  Different values model continuous batching directly.
    """

    contexts = tuple(context_tokens)
    if not contexts:
        raise ValueError("context_tokens must contain at least one request")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in contexts):
        raise ValueError("every context length must be an integer")
    if any(value < 0 for value in contexts):
        raise ValueError("context lengths must be >= 0")
    if model.max_context_tokens is not None and any(
        value > model.max_context_tokens for value in contexts
    ):
        raise ValueError(
            f"context exceeds {model.name} max_context_tokens="
            f"{model.max_context_tokens}"
        )

    batch_size = len(contexts)
    weight_result = _weight_step_cost(model, deployment, batch_size)
    step_work = weight_result.work
    total_cache = CacheCapacity()

    # KV/index/state are private to each request.  They are added request by
    # request and are not batch-amortized before the final per-output division.
    for context in contexts:
        request_work, request_cache = _request_sequence_cost(
            model, deployment, context
        )
        step_work = step_work + request_work
        total_cache = total_cache + request_cache

    step_work = step_work + WorkCost(
        extra_flops=(deployment.extra_flops_per_output_token * batch_size),
        activation_bytes=(
            deployment.activation_bytes_per_output_token * batch_size
        ),
    )

    return DecodeResult(
        model_name=model.name,
        batch_size=batch_size,
        context_tokens=contexts,
        step_work=step_work,
        per_output_work=step_work.divided(batch_size),
        cache_capacity_total=total_cache,
        cache_capacity_per_request_average=total_cache.scaled(1.0 / batch_size),
        expert_weight_sets_read=weight_result.expected_expert_reads,
    )


def calculate_grid(
    config: EngineConfig,
    contexts: Iterable[int] | None = None,
    batches: Iterable[int] | None = None,
) -> list[DecodeResult]:
    """Calculate equal-context batches over a context/batch parameter grid."""

    context_values = tuple(contexts or config.default_contexts)
    batch_values = tuple(batches or config.default_batches)
    if not context_values:
        raise ValueError("no contexts supplied and config has no default_contexts")
    if not batch_values:
        raise ValueError("no batches supplied and config has no default_batches")

    results: list[DecodeResult] = []
    for batch in batch_values:
        if isinstance(batch, bool) or not isinstance(batch, int) or batch <= 0:
            raise ValueError("batch sizes must be positive integers")
        for context in context_values:
            if isinstance(context, bool) or not isinstance(context, int):
                raise ValueError("contexts must be integers")
            results.append(
                calculate_decode(
                    config.model,
                    config.deployment,
                    [context] * batch,
                )
            )
    return results
