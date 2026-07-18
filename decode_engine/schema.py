"""Typed data structures shared by the configuration parser and cost engine.

The engine distinguishes three quantities that are easy to conflate:

1. ``step_work`` is the cost of one decode scheduler step.  A step advances
   every request in the batch once, and therefore produces ``batch_size``
   output tokens.
2. ``per_output_work`` is ``step_work / batch_size``.  This is the quantity
   requested by the EBpC plots.
3. ``persistent_cache`` is capacity rather than traffic.  It describes the
   KV/index/state data retained for the active requests and is never folded
   into bytes moved per token.

Prefill keeps the same separation, but has two token counts: valid prompt
tokens and tokens actually executed after an optional padding policy.  It also
reports a compulsory/logical-HBM view separately from a pair-stream operand
view; the latter is an explicit no-cross-query-reuse boundary, not a claim
about a particular attention kernel.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class WorkCost:
    """FLOP and off-chip traffic components for a unit of work.

    All fields use base SI units: FLOPs and bytes.  Keeping components separate
    makes the result auditable and prevents a mechanism-specific approximation
    from being hidden inside one aggregate number.
    """

    parameter_flops: float = 0.0
    attention_flops: float = 0.0
    index_flops: float = 0.0
    state_flops: float = 0.0
    extra_flops: float = 0.0

    weight_read_bytes: float = 0.0
    kv_read_bytes: float = 0.0
    kv_write_bytes: float = 0.0
    index_read_bytes: float = 0.0
    index_write_bytes: float = 0.0
    state_read_bytes: float = 0.0
    state_write_bytes: float = 0.0
    activation_bytes: float = 0.0
    other_read_bytes: float = 0.0

    def __add__(self, other: "WorkCost") -> "WorkCost":
        if not isinstance(other, WorkCost):
            return NotImplemented
        values = {
            name: getattr(self, name) + getattr(other, name)
            for name in self.__dataclass_fields__
        }
        return WorkCost(**values)

    def scaled(self, factor: float) -> "WorkCost":
        return WorkCost(
            **{
                name: getattr(self, name) * factor
                for name in self.__dataclass_fields__
            }
        )

    def divided(self, divisor: float) -> "WorkCost":
        if divisor == 0:
            raise ZeroDivisionError("work cost cannot be divided by zero")
        return self.scaled(1.0 / divisor)

    @property
    def total_flops(self) -> float:
        return (
            self.parameter_flops
            + self.attention_flops
            + self.index_flops
            + self.state_flops
            + self.extra_flops
        )

    @property
    def total_bytes(self) -> float:
        return (
            self.weight_read_bytes
            + self.kv_read_bytes
            + self.kv_write_bytes
            + self.index_read_bytes
            + self.index_write_bytes
            + self.state_read_bytes
            + self.state_write_bytes
            + self.activation_bytes
            + self.other_read_bytes
        )

    @property
    def logical_hbm_bytes(self) -> float:
        """Trend-study HBM traffic, excluding temporary activations.

        ``total_bytes`` retains the historical engine behavior because Prefill
        experiments may explicitly model activation traffic.  The Decode trend
        contract uses this narrower view and keeps ``activation_bytes`` as a
        diagnostic field instead of silently adding it to logical HBM traffic.
        """

        return self.total_bytes - self.activation_bytes

    def to_dict(self) -> dict[str, float]:
        result = asdict(self)
        result["total_flops"] = self.total_flops
        result["total_bytes"] = self.total_bytes
        result["logical_hbm_bytes"] = self.logical_hbm_bytes
        return result


@dataclass(frozen=True)
class CacheCapacity:
    """Persistent per-request cache/state capacity, in bytes."""

    kv_bytes: float = 0.0
    index_bytes: float = 0.0
    state_bytes: float = 0.0

    def __add__(self, other: "CacheCapacity") -> "CacheCapacity":
        if not isinstance(other, CacheCapacity):
            return NotImplemented
        return CacheCapacity(
            kv_bytes=self.kv_bytes + other.kv_bytes,
            index_bytes=self.index_bytes + other.index_bytes,
            state_bytes=self.state_bytes + other.state_bytes,
        )

    def scaled(self, factor: float) -> "CacheCapacity":
        return CacheCapacity(
            kv_bytes=self.kv_bytes * factor,
            index_bytes=self.index_bytes * factor,
            state_bytes=self.state_bytes * factor,
        )

    @property
    def total_bytes(self) -> float:
        return self.kv_bytes + self.index_bytes + self.state_bytes

    def to_dict(self) -> dict[str, float]:
        result = asdict(self)
        result["total_bytes"] = self.total_bytes
        return result


@dataclass(frozen=True)
class MixerCost:
    """Per-layer, per-request cost returned by a sequence mixer."""

    work: WorkCost = field(default_factory=WorkCost)
    cache: CacheCapacity = field(default_factory=CacheCapacity)

    def scaled(self, factor: float) -> "MixerCost":
        return MixerCost(self.work.scaled(factor), self.cache.scaled(factor))


@dataclass(frozen=True)
class DeploymentConfig:
    """Deployment choices and memory-traffic assumptions.

    Fractions represent the part of a logical read/write that reaches the
    off-chip memory level being studied.  ``1.0`` is the conservative standard
    workload model where weights and persistent state are streamed from HBM.
    A later chip-design study may lower a fraction to model on-chip residency.
    """

    weight_bits: float
    expert_weight_bits: float
    kv_bits: float
    index_bits: float
    state_bits: float
    mac_flops: float = 2.0

    include_kv_write: bool = True
    include_index_write: bool = True
    include_state_write: bool = True

    weight_hbm_fraction: float = 1.0
    kv_hbm_fraction: float = 1.0
    index_hbm_fraction: float = 1.0
    state_hbm_fraction: float = 1.0

    weight_read_multiplier: float = 1.0
    activation_bytes_per_output_token: float = 0.0
    extra_flops_per_output_token: float = 0.0
    activation_bytes_per_input_token: float = 0.0
    extra_flops_per_input_token: float = 0.0


@dataclass(frozen=True)
class AlwaysActiveParameterGroup:
    """Always-executed parameters sharing one storage precision."""

    name: str
    parameters: float
    weight_bits: float | None = None


@dataclass(frozen=True)
class RoutedExpertGroup:
    """A homogeneous collection of routed experts across several layers.

    ``parameters_per_expert`` is for one expert in one layer.  Shared experts
    are deliberately excluded and belong in ``always_active_parameters``
    because every token executes them.
    """

    name: str
    layers: int
    expert_count: int
    selected_per_token: int
    parameters_per_expert: float
    routing_mode: str = "uniform_independent"
    expected_unique_experts_by_batch: Mapping[int, float] = field(
        default_factory=dict
    )
    expected_unique_experts_by_active_tokens: Mapping[int, float] = field(
        default_factory=dict
    )
    weight_bits: float | None = None


@dataclass(frozen=True)
class WeightConfig:
    """Parameterized work and weight traffic configuration.

    ``always_active_parameters`` includes every parameterized operation used by
    every token: attention projections, dense FFNs, routers, shared experts,
    output head, and any other model-specific projection.  Routed expert
    parameters are added separately to preserve correct batch behavior.
    """

    always_active_parameter_groups: tuple[AlwaysActiveParameterGroup, ...]
    routed_expert_groups: tuple[RoutedExpertGroup, ...] = ()
    weight_bits: float | None = None
    # ``always_active_parameter_groups`` retain the decode-era convention and
    # include the output head.  This explicit subset lets prefill execute the
    # backbone for every input position while executing the head only for the
    # requested logit positions.  It is not an additional weight allocation.
    output_head_parameters: float = 0.0
    output_head_parameters_configured: bool = False
    output_head_weight_bits: float | None = None

    @property
    def always_active_parameters(self) -> float:
        return sum(group.parameters for group in self.always_active_parameter_groups)

    @property
    def backbone_parameters(self) -> float:
        return self.always_active_parameters - self.output_head_parameters


@dataclass(frozen=True)
class LayerGroupConfig:
    """Layers that share the same sequence-mixer configuration."""

    name: str
    layers: int
    mixers: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class ModelConfig:
    name: str
    max_context_tokens: int | None
    weights: WeightConfig
    layer_groups: tuple[LayerGroupConfig, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EngineConfig:
    schema_version: int
    model: ModelConfig
    deployment: DeploymentConfig
    default_contexts: tuple[int, ...] = ()
    default_batches: tuple[int, ...] = ()
    default_prefill_lengths: tuple[int, ...] = ()
    default_prefill_batches: tuple[int, ...] = ()
    default_prefill_token_budgets: tuple[int, ...] = ()
    default_ragged_prefill_batches: tuple[tuple[int, ...], ...] = ()


@dataclass(frozen=True)
class DecodeResult:
    """Auditable result for one model, context set, and batch size."""

    model_name: str
    batch_size: int
    context_tokens: tuple[int, ...]
    step_work: WorkCost
    per_output_work: WorkCost
    cache_capacity_total: CacheCapacity
    cache_capacity_per_request_average: CacheCapacity
    expert_weight_sets_read: Mapping[str, float] = field(default_factory=dict)

    @property
    def average_context_tokens(self) -> float:
        return sum(self.context_tokens) / self.batch_size

    @property
    def bytes_per_flop(self) -> float:
        flops = self.per_output_work.total_flops
        return self.per_output_work.total_bytes / flops if flops else float("inf")

    @property
    def tbps_per_pflops(self) -> float:
        # 1 PFLOP/s * 1 Byte/FLOP = 10^15 Byte/s = 1000 TB/s.
        return self.bytes_per_flop * 1000.0

    @property
    def logical_hbm_bytes_per_flop(self) -> float:
        flops = self.per_output_work.total_flops
        return (
            self.per_output_work.logical_hbm_bytes / flops
            if flops
            else float("inf")
        )

    @property
    def logical_hbm_tbps_per_pflops(self) -> float:
        return self.logical_hbm_bytes_per_flop * 1000.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "batch_size": self.batch_size,
            "context_tokens": list(self.context_tokens),
            "average_context_tokens": self.average_context_tokens,
            "step_work": self.step_work.to_dict(),
            "per_output_work": self.per_output_work.to_dict(),
            "cache_capacity_total": self.cache_capacity_total.to_dict(),
            "cache_capacity_per_request_average": (
                self.cache_capacity_per_request_average.to_dict()
            ),
            "expert_weight_sets_read": dict(self.expert_weight_sets_read),
            "bytes_per_flop": self.bytes_per_flop,
            "tbps_per_pflops": self.tbps_per_pflops,
            "logical_hbm_bytes_per_flop": (
                self.logical_hbm_bytes_per_flop
            ),
            "logical_hbm_tbps_per_pflops": (
                self.logical_hbm_tbps_per_pflops
            ),
        }


@dataclass(frozen=True)
class PrefillResult:
    """Auditable result for one prefill scheduler invocation.

    ``useful_work`` evaluates only real prompt tokens.  ``batch_work`` applies
    the chosen execution layout and therefore includes padding work when
    ``execution_mode == "padded"``.  Both use the compulsory/logical-HBM
    traffic convention.  The corresponding ``*_operand_work`` values replace
    the attention/index read component with a pair-stream, no-reuse boundary.
    ``*_causal_pair_slots`` describe the dense full-causal execution shape;
    mechanism-specific accessed entries remain in the FLOP/operand breakdowns.
    """

    model_name: str
    experiment: str
    execution_mode: str
    logits_mode: str
    include_self_attention: bool
    prompt_tokens: tuple[int, ...]
    cached_context_tokens: tuple[int, ...]
    valid_input_tokens: int
    executed_input_tokens: int
    valid_causal_pair_slots: float
    executed_causal_pair_slots: float
    output_head_parameters: float
    output_head_parameters_configured: bool
    output_head_weight_bits: float | None
    topk_cached_prefix_union_policy: str
    useful_work: WorkCost
    batch_work: WorkCost
    useful_operand_work: WorkCost
    batch_operand_work: WorkCost
    cache_capacity_total: CacheCapacity
    cache_capacity_per_request_average: CacheCapacity
    expert_weight_sets_read: Mapping[str, float] = field(default_factory=dict)
    useful_expert_weight_sets_read: Mapping[str, float] = field(
        default_factory=dict
    )

    @property
    def batch_size(self) -> int:
        return len(self.prompt_tokens)

    @property
    def average_prompt_tokens(self) -> float:
        return self.valid_input_tokens / self.batch_size

    @property
    def max_prompt_tokens(self) -> int:
        return max(self.prompt_tokens)

    @property
    def valid_logit_positions(self) -> int:
        if self.logits_mode == "last":
            return self.batch_size
        if self.logits_mode == "all":
            return self.valid_input_tokens
        return 0

    @property
    def executed_logit_positions(self) -> int:
        if self.logits_mode == "last":
            return self.batch_size
        if self.logits_mode == "all":
            return self.executed_input_tokens
        return 0

    @property
    def token_efficiency(self) -> float:
        return self.valid_input_tokens / self.executed_input_tokens

    @property
    def causal_pair_efficiency(self) -> float:
        if self.executed_causal_pair_slots == 0:
            return 1.0
        return self.valid_causal_pair_slots / self.executed_causal_pair_slots

    @property
    def per_input_work(self) -> WorkCost:
        """Executed batch work normalized by useful input tokens."""

        return self.batch_work.divided(self.valid_input_tokens)

    @property
    def per_executed_token_work(self) -> WorkCost:
        return self.batch_work.divided(self.executed_input_tokens)

    @property
    def per_input_operand_work(self) -> WorkCost:
        return self.batch_operand_work.divided(self.valid_input_tokens)

    @property
    def bytes_per_flop(self) -> float:
        flops = self.batch_work.total_flops
        return self.batch_work.total_bytes / flops if flops else float("inf")

    @property
    def operand_bytes_per_flop(self) -> float:
        flops = self.batch_operand_work.total_flops
        return (
            self.batch_operand_work.total_bytes / flops
            if flops
            else float("inf")
        )

    @property
    def tbps_per_pflops(self) -> float:
        return self.bytes_per_flop * 1000.0

    @property
    def operand_tbps_per_pflops(self) -> float:
        return self.operand_bytes_per_flop * 1000.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": "prefill",
            "model": self.model_name,
            "experiment": self.experiment,
            "execution_mode": self.execution_mode,
            "logits_mode": self.logits_mode,
            "include_self_attention": self.include_self_attention,
            "batch_size": self.batch_size,
            "prompt_tokens": list(self.prompt_tokens),
            "cached_context_tokens": list(self.cached_context_tokens),
            "valid_input_tokens": self.valid_input_tokens,
            "executed_input_tokens": self.executed_input_tokens,
            "average_prompt_tokens": self.average_prompt_tokens,
            "max_prompt_tokens": self.max_prompt_tokens,
            "valid_logit_positions": self.valid_logit_positions,
            "executed_logit_positions": self.executed_logit_positions,
            "valid_causal_pair_slots": self.valid_causal_pair_slots,
            "executed_causal_pair_slots": self.executed_causal_pair_slots,
            "output_head_parameters": self.output_head_parameters,
            "output_head_parameters_configured": (
                self.output_head_parameters_configured
            ),
            "output_head_weight_bits": self.output_head_weight_bits,
            "topk_cached_prefix_union_policy": (
                self.topk_cached_prefix_union_policy
            ),
            "token_efficiency": self.token_efficiency,
            "causal_pair_efficiency": self.causal_pair_efficiency,
            "useful_work": self.useful_work.to_dict(),
            "batch_work": self.batch_work.to_dict(),
            "per_input_work": self.per_input_work.to_dict(),
            "per_executed_token_work": self.per_executed_token_work.to_dict(),
            "useful_operand_work": self.useful_operand_work.to_dict(),
            "batch_operand_work": self.batch_operand_work.to_dict(),
            "per_input_operand_work": self.per_input_operand_work.to_dict(),
            "cache_capacity_total": self.cache_capacity_total.to_dict(),
            "cache_capacity_per_request_average": (
                self.cache_capacity_per_request_average.to_dict()
            ),
            "expert_weight_sets_read": dict(self.expert_weight_sets_read),
            "useful_expert_weight_sets_read": dict(
                self.useful_expert_weight_sets_read
            ),
            "bytes_per_flop": self.bytes_per_flop,
            "tbps_per_pflops": self.tbps_per_pflops,
            "operand_bytes_per_flop": self.operand_bytes_per_flop,
            "operand_tbps_per_pflops": self.operand_tbps_per_pflops,
        }


def sum_work(costs: Sequence[WorkCost]) -> WorkCost:
    total = WorkCost()
    for cost in costs:
        total = total + cost
    return total


def sum_cache(capacities: Sequence[CacheCapacity]) -> CacheCapacity:
    total = CacheCapacity()
    for capacity in capacities:
        total = total + capacity
    return total
