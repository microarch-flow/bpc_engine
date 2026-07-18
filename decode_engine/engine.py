"""Decode and prefill aggregation with explicit normalization boundaries.

The central rule is simple but important: calculate the complete scheduler
step first, then divide by the number of output tokens produced by that step.
This ensures that shared weight reads are amortized while request-private KV
and recurrent-state traffic is not accidentally divided twice.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Sequence

from .mechanisms import build_mixer
from .schema import (
    CacheCapacity,
    DecodeResult,
    DeploymentConfig,
    EngineConfig,
    ModelConfig,
    PrefillResult,
    RoutedExpertGroup,
    WorkCost,
)


@dataclass(frozen=True)
class _WeightStepResult:
    work: WorkCost
    expected_expert_reads: dict[str, float]


def expected_expert_reads(
    group: RoutedExpertGroup,
    sample_count: int,
    *,
    explicit_axis: str = "batch",
) -> float:
    """Estimate expert weight sets read once during one invocation.

    The default formula assumes each token independently chooses ``k`` distinct
    experts uniformly from ``E`` experts.  For a particular expert, the chance
    of not being selected by one token is ``1-k/E``; after ``N`` samples it is
    ``(1-k/E)^N``.  Summing the probability of being touched over all experts
    gives the expected union size.  Decode samples are output positions in the
    request batch; prefill samples are executed routed-token positions.

    Real routing is not perfectly uniform.  The configuration therefore also
    supports a measured/assumed explicit union and two boundary policies.
    """

    experts = group.expert_count
    selected = group.selected_per_token
    mode = group.routing_mode

    if mode == "uniform_independent":
        return experts * (1.0 - (1.0 - selected / experts) ** sample_count)
    if mode == "same_experts":
        # Best reuse boundary: every request happens to select the same experts.
        return float(selected)
    if mode == "explicit_unique":
        if explicit_axis not in {"batch", "active_tokens"}:
            raise ValueError("explicit_axis must be batch or active_tokens")
        if explicit_axis == "active_tokens":
            mapping = group.expected_unique_experts_by_active_tokens
            label = "active_tokens"
            if not mapping:
                raise ValueError(
                    f"{group.name} explicit prefill routing requires "
                    "expected_unique_experts_by_active_tokens"
                )
        else:
            mapping = group.expected_unique_experts_by_batch
            label = "batch"
        if sample_count not in mapping:
            raise ValueError(
                f"{group.name} has no explicit unique-expert value for "
                f"{label}={sample_count}"
            )
        return float(mapping[sample_count])
    if mode == "no_batch_reuse":
        # Traffic upper boundary.  The same expert may be counted repeatedly
        # because its weights are assumed to be re-read for different tokens.
        return float(sample_count * selected)
    raise ValueError(f"unsupported routing mode {mode!r} in {group.name}")


def _weight_invocation_cost(
    model: ModelConfig,
    deployment: DeploymentConfig,
    active_token_count: int,
    logit_position_count: int,
    expert_explicit_axis: str,
) -> _WeightStepResult:
    """Calculate parameter work and one invocation's logical weight reads.

    The decode-era always-active total includes the output head.  Prefill uses
    the explicit ``output_head_parameters`` subset to apply the backbone to all
    executed input positions while applying the head only to requested logit
    positions.  Weight storage is unchanged: dense groups are still read once
    per invocation and the head subset must not be added a second time.
    """

    weights = model.weights
    shared_bits = weights.weight_bits or deployment.weight_bits

    # Every output token executes all always-active parameterized operations,
    # but their weights can be read once and reused by the batch GEMM.  Groups
    # allow attention, shared experts, and special projections to use different
    # storage precision without changing the arithmetic parameter count.
    parameter_operations = (
        active_token_count * weights.backbone_parameters
        + logit_position_count * weights.output_head_parameters
    )
    weight_read_bytes = sum(
        group.parameters * (group.weight_bits or shared_bits) / 8.0
        for group in weights.always_active_parameter_groups
    )
    if logit_position_count == 0 and weights.output_head_parameters:
        head_bits = weights.output_head_weight_bits or shared_bits
        weight_read_bytes -= (
            weights.output_head_parameters * head_bits / 8.0
        )
        if weight_read_bytes < 0:
            raise ValueError(
                "output-head byte split exceeds always-active weight bytes; "
                "check output_head_parameters/output_head_weight_bits"
            )
    expert_reads: dict[str, float] = {}

    for group in weights.routed_expert_groups:
        parameter_operations += active_token_count * (
            group.layers
            * group.selected_per_token
            * group.parameters_per_expert
        )

        reads = expected_expert_reads(
            group,
            active_token_count,
            explicit_axis=expert_explicit_axis,
        )
        expert_reads[group.name] = reads
        bits = group.weight_bits or deployment.expert_weight_bits
        weight_read_bytes += (
            group.layers * reads * group.parameters_per_expert * bits / 8.0
        )

    # Parameterized matrix work is performed for each active position, even
    # though dense weight bytes may be shared across the invocation.
    parameter_flops = deployment.mac_flops * parameter_operations
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


def _weight_step_cost(
    model: ModelConfig, deployment: DeploymentConfig, batch_size: int
) -> _WeightStepResult:
    """Compatibility wrapper for one decode step."""

    return _weight_invocation_cost(
        model,
        deployment,
        active_token_count=batch_size,
        logit_position_count=batch_size,
        expert_explicit_axis="batch",
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


def _request_prefill_cost(
    model: ModelConfig,
    deployment: DeploymentConfig,
    cached_tokens: int,
    new_tokens: int,
    include_self_attention: bool,
) -> tuple[WorkCost, WorkCost, CacheCapacity]:
    """Aggregate one request's fused prefill work over every layer group."""

    request_work = WorkCost()
    request_operand_work = WorkCost()
    request_cache = CacheCapacity()
    for group_index, group in enumerate(model.layer_groups):
        for mixer_index, mixer_config in enumerate(group.mixers):
            path = f"model.layer_groups[{group_index}].mixers[{mixer_index}]"
            mixer = build_mixer(mixer_config, path)
            cost = mixer.prefill_cost(
                cached_tokens,
                new_tokens,
                deployment,
                include_self_attention,
            ).scaled(group.layers)
            request_work = request_work + cost.work
            request_operand_work = request_operand_work + cost.operand_work
            request_cache = request_cache + cost.cache
    return request_work, request_operand_work, request_cache


def _causal_pair_slots(
    cached_tokens: int, new_tokens: int, include_self_attention: bool
) -> float:
    diagonal = 1 if include_self_attention else -1
    return (
        new_tokens * cached_tokens
        + new_tokens * (new_tokens + diagonal) / 2.0
    )


def _prefill_extra_work(
    deployment: DeploymentConfig, input_tokens: int
) -> WorkCost:
    return WorkCost(
        extra_flops=deployment.extra_flops_per_input_token * input_tokens,
        activation_bytes=(
            deployment.activation_bytes_per_input_token * input_tokens
        ),
    )


def _topk_cached_prefix_union_policy(
    model: ModelConfig, cached_tokens: tuple[int, ...]
) -> str:
    if not any(cached_tokens):
        return "not_applicable"
    topk_kinds = {"fixed_topk", "learned_topk", "dsa", "csa"}
    for group in model.layer_groups:
        for mixer in group.mixers:
            if mixer.get("kind") != "softmax_attention":
                continue
            access = mixer.get("access", {})
            if access.get("kind") in topk_kinds:
                return "conservative_distinct_upper_bound"
    return "not_applicable"


def _validate_token_vector(
    values: Sequence[int], name: str, *, allow_zero: bool
) -> tuple[int, ...]:
    result = tuple(values)
    if not result:
        raise ValueError(f"{name} must contain at least one request")
    for value in result:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"every {name} value must be an integer")
        if value < 0 or (not allow_zero and value == 0):
            relation = ">= 0" if allow_zero else "> 0"
            raise ValueError(f"every {name} value must be {relation}")
    return result


def _aggregate_prefill_layout(
    model: ModelConfig,
    deployment: DeploymentConfig,
    cached_tokens: tuple[int, ...],
    new_tokens: tuple[int, ...],
    *,
    logits_mode: str,
    include_self_attention: bool,
) -> tuple[
    WorkCost,
    WorkCost,
    CacheCapacity,
    dict[str, float],
]:
    input_count = sum(new_tokens)
    if logits_mode == "last":
        logit_positions = len(new_tokens)
    elif logits_mode == "all":
        logit_positions = input_count
    else:
        logit_positions = 0
    weight_result = _weight_invocation_cost(
        model,
        deployment,
        active_token_count=input_count,
        logit_position_count=logit_positions,
        expert_explicit_axis="active_tokens",
    )
    work = weight_result.work
    operand_work = weight_result.work
    cache = CacheCapacity()
    for cached, new in zip(cached_tokens, new_tokens):
        sequence_work, sequence_operand_work, sequence_cache = (
            _request_prefill_cost(
                model,
                deployment,
                cached,
                new,
                include_self_attention,
            )
        )
        work = work + sequence_work
        operand_work = operand_work + sequence_operand_work
        cache = cache + sequence_cache

    extra = _prefill_extra_work(deployment, input_count)
    return (
        work + extra,
        operand_work + extra,
        cache,
        weight_result.expected_expert_reads,
    )


def calculate_prefill(
    model: ModelConfig,
    deployment: DeploymentConfig,
    prompt_tokens: Sequence[int],
    *,
    cached_context_tokens: Sequence[int] | None = None,
    execution_mode: str = "varlen",
    logits_mode: str = "last",
    include_self_attention: bool = True,
    experiment: str = "ragged",
) -> PrefillResult:
    """Calculate one prefill invocation over a prompt-length vector.

    ``varlen`` and segment-aware ``packed`` execute only valid positions.
    ``padded`` models a dense causal batch padded to the longest new prompt;
    it currently targets ordinary one-shot prefill and therefore rejects a
    pre-existing cached prefix.  Its final persistent cache is still reported
    for valid tokens only, while executed work includes padded positions.
    """

    prompts = _validate_token_vector(
        prompt_tokens, "prompt_tokens", allow_zero=False
    )
    if cached_context_tokens is None:
        cached = (0,) * len(prompts)
    else:
        cached = _validate_token_vector(
            cached_context_tokens,
            "cached_context_tokens",
            allow_zero=True,
        )
        if len(cached) != len(prompts):
            raise ValueError(
                "cached_context_tokens must have the same length as prompt_tokens"
            )
    if execution_mode not in {"varlen", "packed", "padded"}:
        raise ValueError("execution_mode must be varlen, packed, or padded")
    if logits_mode not in {"last", "all", "none"}:
        raise ValueError("logits_mode must be last, all, or none")
    if not isinstance(include_self_attention, bool):
        raise ValueError("include_self_attention must be true or false")
    if not isinstance(experiment, str) or not experiment:
        raise ValueError("experiment must be a non-empty string")
    if execution_mode == "padded" and any(cached):
        raise ValueError(
            "padded execution with cached prefixes is not modeled; use varlen "
            "or packed execution"
        )
    if model.max_context_tokens is not None and any(
        prefix + new > model.max_context_tokens
        for prefix, new in zip(cached, prompts)
    ):
        raise ValueError(
            f"prefill context exceeds {model.name} max_context_tokens="
            f"{model.max_context_tokens}"
        )

    useful_work, useful_operand_work, final_cache, useful_experts = (
        _aggregate_prefill_layout(
            model,
            deployment,
            cached,
            prompts,
            logits_mode=logits_mode,
            include_self_attention=include_self_attention,
        )
    )
    if execution_mode == "padded":
        padded_length = max(prompts)
        executed_prompts = (padded_length,) * len(prompts)
        executed_cached = (0,) * len(prompts)
        padded_work, padded_operand_work, _ignored_cache, experts = (
            _aggregate_prefill_layout(
                model,
                deployment,
                executed_cached,
                executed_prompts,
                logits_mode=logits_mode,
                include_self_attention=include_self_attention,
            )
        )
        # Padding may execute arithmetic and materialize temporary tensors for
        # padded positions, but those positions are not persistent request KV,
        # index, or recurrent state.  Keep executed FLOPs/operand reads while
        # retaining valid-token persistent writes in their dedicated fields.
        batch_work = replace(
            padded_work,
            kv_write_bytes=useful_work.kv_write_bytes,
            index_write_bytes=useful_work.index_write_bytes,
            state_write_bytes=useful_work.state_write_bytes,
        )
        batch_operand_work = replace(
            padded_operand_work,
            kv_write_bytes=useful_operand_work.kv_write_bytes,
            index_write_bytes=useful_operand_work.index_write_bytes,
            state_write_bytes=useful_operand_work.state_write_bytes,
        )
    else:
        executed_prompts = prompts
        batch_work = useful_work
        batch_operand_work = useful_operand_work
        experts = useful_experts

    valid_pair_slots = sum(
        _causal_pair_slots(prefix, new, include_self_attention)
        for prefix, new in zip(cached, prompts)
    )
    executed_pair_slots = sum(
        _causal_pair_slots(prefix, new, include_self_attention)
        for prefix, new in zip(
            (0,) * len(prompts) if execution_mode == "padded" else cached,
            executed_prompts,
        )
    )
    batch_size = len(prompts)
    return PrefillResult(
        model_name=model.name,
        experiment=experiment,
        execution_mode=execution_mode,
        logits_mode=logits_mode,
        include_self_attention=include_self_attention,
        prompt_tokens=prompts,
        cached_context_tokens=cached,
        valid_input_tokens=sum(prompts),
        executed_input_tokens=sum(executed_prompts),
        valid_causal_pair_slots=valid_pair_slots,
        executed_causal_pair_slots=executed_pair_slots,
        output_head_parameters=model.weights.output_head_parameters,
        output_head_parameters_configured=(
            model.weights.output_head_parameters_configured
        ),
        output_head_weight_bits=(
            model.weights.output_head_weight_bits
            or model.weights.weight_bits
            or deployment.weight_bits
        ),
        topk_cached_prefix_union_policy=(
            _topk_cached_prefix_union_policy(model, cached)
        ),
        useful_work=useful_work,
        batch_work=batch_work,
        useful_operand_work=useful_operand_work,
        batch_operand_work=batch_operand_work,
        cache_capacity_total=final_cache,
        cache_capacity_per_request_average=final_cache.scaled(1.0 / batch_size),
        expert_weight_sets_read=experts,
        useful_expert_weight_sets_read=useful_experts,
    )


def calculate_decode(
    model: ModelConfig,
    deployment: DeploymentConfig,
    context_tokens: Sequence[int],
    *,
    allow_extrapolation: bool = False,
) -> DecodeResult:
    """Calculate one decode step and normalize it per generated token.

    Args:
        model: Normalized model architecture and weight-access configuration.
        deployment: Precision and off-chip residency assumptions.
        context_tokens: One existing-context length for each active request.
            Supplying ``[32768] * 32`` describes a batch of 32 equal-length
            requests.  Different values model continuous batching directly.
        allow_extrapolation: Permit contexts above ``model.max_context_tokens``.
            The caller remains responsible for marking those research records
            as extrapolated.  The safe default preserves strict deployment
            validation.
    """

    contexts = tuple(context_tokens)
    if not contexts:
        raise ValueError("context_tokens must contain at least one request")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in contexts):
        raise ValueError("every context length must be an integer")
    if any(value < 0 for value in contexts):
        raise ValueError("context lengths must be >= 0")
    if not allow_extrapolation and model.max_context_tokens is not None and any(
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
    *,
    allow_extrapolation: bool = False,
) -> list[DecodeResult]:
    """Calculate equal-context batches over a context/batch parameter grid."""

    context_values = tuple(
        config.default_contexts if contexts is None else contexts
    )
    batch_values = tuple(
        config.default_batches if batches is None else batches
    )
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
                    allow_extrapolation=allow_extrapolation,
                )
            )
    return results


def calculate_prefill_grid(
    config: EngineConfig,
    prompt_lengths: Iterable[int] | None = None,
    batches: Iterable[int] | None = None,
    *,
    execution_mode: str = "varlen",
    logits_mode: str = "last",
    include_self_attention: bool = True,
) -> list[PrefillResult]:
    """Experiment 1: fixed request batch, sweep equal prompt lengths."""

    if prompt_lengths is None:
        length_values = (
            config.default_prefill_lengths or config.default_contexts
        )
    else:
        length_values = tuple(prompt_lengths)
    if batches is None:
        batch_values = (
            config.default_prefill_batches or config.default_batches
        )
    else:
        batch_values = tuple(batches)
    if not length_values:
        raise ValueError(
            "no prompt lengths supplied and config has no prefill/context defaults"
        )
    if not batch_values:
        raise ValueError(
            "no batches supplied and config has no prefill/decode batch defaults"
        )

    results: list[PrefillResult] = []
    for batch in batch_values:
        if isinstance(batch, bool) or not isinstance(batch, int) or batch <= 0:
            raise ValueError("batch sizes must be positive integers")
        for length in length_values:
            if (
                isinstance(length, bool)
                or not isinstance(length, int)
                or length <= 0
            ):
                raise ValueError("prompt lengths must be positive integers")
            results.append(
                calculate_prefill(
                    config.model,
                    config.deployment,
                    [length] * batch,
                    execution_mode=execution_mode,
                    logits_mode=logits_mode,
                    include_self_attention=include_self_attention,
                    experiment="equal",
                )
            )
    return results


def _balanced_lengths(token_budget: int, batch_size: int) -> tuple[int, ...]:
    """Split an exact token budget as evenly as possible across requests."""

    if (
        isinstance(token_budget, bool)
        or not isinstance(token_budget, int)
        or token_budget <= 0
    ):
        raise ValueError("token budgets must be positive integers")
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size <= 0
    ):
        raise ValueError("batch sizes must be positive integers")
    if batch_size > token_budget:
        raise ValueError(
            f"batch={batch_size} exceeds token budget={token_budget}; "
            "every prefill request must contain at least one token"
        )
    base, remainder = divmod(token_budget, batch_size)
    return tuple(
        base + (1 if index < remainder else 0)
        for index in range(batch_size)
    )


def calculate_prefill_token_budget_grid(
    config: EngineConfig,
    token_budgets: Iterable[int] | None = None,
    batches: Iterable[int] | None = None,
    *,
    execution_mode: str = "varlen",
    logits_mode: str = "last",
    include_self_attention: bool = True,
) -> list[PrefillResult]:
    """Experiment 2: hold total input tokens fixed and vary request shape."""

    budget_values = (
        config.default_prefill_token_budgets
        if token_budgets is None
        else tuple(token_budgets)
    )
    if batches is None:
        batch_values = (
            config.default_prefill_batches or config.default_batches
        )
    else:
        batch_values = tuple(batches)
    if not budget_values:
        raise ValueError(
            "no token budgets supplied and config has no prefill token budgets"
        )
    if not batch_values:
        raise ValueError("no batches supplied and config has no batch defaults")

    results: list[PrefillResult] = []
    for budget in budget_values:
        for batch in batch_values:
            lengths = _balanced_lengths(budget, batch)
            results.append(
                calculate_prefill(
                    config.model,
                    config.deployment,
                    lengths,
                    execution_mode=execution_mode,
                    logits_mode=logits_mode,
                    include_self_attention=include_self_attention,
                    experiment="token-budget",
                )
            )
    return results


def calculate_ragged_prefill_grid(
    config: EngineConfig,
    ragged_batches: Iterable[Sequence[int]] | None = None,
    *,
    execution_mode: str = "varlen",
    logits_mode: str = "last",
    include_self_attention: bool = True,
) -> list[PrefillResult]:
    """Experiment 3: evaluate explicit real/request-trace length vectors."""

    batches = (
        config.default_ragged_prefill_batches
        if ragged_batches is None
        else tuple(ragged_batches)
    )
    if not batches:
        raise ValueError(
            "no ragged batches supplied and config has no ragged prefill defaults"
        )
    return [
        calculate_prefill(
            config.model,
            config.deployment,
            lengths,
            execution_mode=execution_mode,
            logits_mode=logits_mode,
            include_self_attention=include_self_attention,
            experiment="ragged",
        )
        for lengths in batches
    ]
