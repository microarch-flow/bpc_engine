"""Sequence-mixer implementations used by the decode cost engine.

The code deliberately separates two independent architectural choices:

* a KV layout determines bytes and FLOPs for one accessed history entry;
* an access pattern determines how many history entries are stored/read.

This composition represents GQA+SWA, MLA+top-k, compressed MQA, and similar
combinations without creating a bespoke formula for every model name.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping

from .schema import CacheCapacity, DeploymentConfig, MixerCost, WorkCost


Number = int | float


def _number(config: Mapping[str, Any], key: str, path: str) -> float:
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path}.{key} must be a number")
    return float(value)


def _positive(config: Mapping[str, Any], key: str, path: str) -> float:
    value = _number(config, key, path)
    if value <= 0:
        raise ValueError(f"{path}.{key} must be > 0")
    return value


def _nonnegative(config: Mapping[str, Any], key: str, path: str) -> float:
    value = _number(config, key, path)
    if value < 0:
        raise ValueError(f"{path}.{key} must be >= 0")
    return value


def _bits_to_bytes(elements: float, bits: float) -> float:
    return elements * bits / 8.0


class KVLayout(ABC):
    """Byte and arithmetic cost of one logical attention history entry."""

    @abstractmethod
    def entry_bytes(self, deployment: DeploymentConfig) -> float:
        raise NotImplementedError

    @abstractmethod
    def attention_flops_per_entry(self) -> float:
        """Return QK plus AV FLOPs for one query over one history entry."""

        raise NotImplementedError

    @property
    @abstractmethod
    def query_heads(self) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class GroupedKVLayout(KVLayout):
    """MHA/MQA/GQA represented by a common parameterization.

    MHA uses ``kv_heads == query_heads``; MQA uses ``kv_heads == 1``; GQA is
    any intermediate value.  GQA changes KV bytes but not QK/AV FLOPs because
    attention computation follows query heads.
    """

    q_heads: int
    kv_heads: int
    head_dim: int
    key_bits: float | None = None
    value_bits: float | None = None

    @property
    def query_heads(self) -> int:
        return self.q_heads

    def entry_bytes(self, deployment: DeploymentConfig) -> float:
        key_bits = self.key_bits or deployment.kv_bits
        value_bits = self.value_bits or deployment.kv_bits
        return self.kv_heads * self.head_dim * (key_bits + value_bits) / 8.0

    def attention_flops_per_entry(self) -> float:
        # QK: 2*Hq*dh; AV: 2*Hq*dh.
        return 4.0 * self.q_heads * self.head_dim


@dataclass(frozen=True)
class LatentKVLayout(KVLayout):
    """Absorbed Multi-head Latent Attention (MLA) cache representation."""

    q_heads: int
    latent_dim: int
    rope_dim: int
    latent_bits: float | None = None
    rope_bits: float | None = None

    @property
    def query_heads(self) -> int:
        return self.q_heads

    def entry_bytes(self, deployment: DeploymentConfig) -> float:
        latent_bits = self.latent_bits or deployment.kv_bits
        rope_bits = self.rope_bits or deployment.kv_bits
        return (
            self.latent_dim * latent_bits + self.rope_dim * rope_bits
        ) / 8.0

    def attention_flops_per_entry(self) -> float:
        # Per query head: latent score (2*dc), RoPE score (2*dR), and
        # latent value aggregation (2*dc).
        return 2.0 * self.q_heads * (2 * self.latent_dim + self.rope_dim)


@dataclass(frozen=True)
class SharedKVLayout(KVLayout):
    """One vector serves as both key and value, as in V4 shared-KV MQA.

    Unlike conventional MQA, which stores separate K and V vectors, this
    representation stores one shared entry.  The arithmetic still performs
    both score and value paths.
    """

    q_heads: int
    head_dim: int
    rope_dim: int = 0
    non_rope_bits: float | None = None
    rope_bits: float | None = None

    @property
    def query_heads(self) -> int:
        return self.q_heads

    def entry_bytes(self, deployment: DeploymentConfig) -> float:
        if self.rope_dim > self.head_dim:
            raise ValueError("shared KV rope_dim cannot exceed head_dim")
        non_rope_bits = self.non_rope_bits or deployment.kv_bits
        rope_bits = self.rope_bits or deployment.kv_bits
        return (
            (self.head_dim - self.rope_dim) * non_rope_bits
            + self.rope_dim * rope_bits
        ) / 8.0

    def attention_flops_per_entry(self) -> float:
        return 4.0 * self.q_heads * self.head_dim


@dataclass(frozen=True)
class ExplicitKVLayout(KVLayout):
    """Escape hatch for a published mechanism with explicit coefficients."""

    q_heads: int
    bytes_per_entry: float
    flops_per_entry: float

    @property
    def query_heads(self) -> int:
        return self.q_heads

    def entry_bytes(self, deployment: DeploymentConfig) -> float:
        del deployment
        return self.bytes_per_entry

    def attention_flops_per_entry(self) -> float:
        return self.flops_per_entry


@dataclass(frozen=True)
class AccessStats:
    """History-entry counts and index costs for one layer and one request."""

    main_read_entries: float
    main_write_entries: float
    main_stored_entries: float
    index_read_bytes: float = 0.0
    index_write_bytes: float = 0.0
    index_stored_bytes: float = 0.0
    index_flops: float = 0.0


@dataclass(frozen=True)
class PrefillAccessStats:
    """Aggregate access counts for one fused prefill invocation.

    ``*_compulsory_*`` counts distinct entries from the cached prefix which
    have to be brought into the invocation at least once.  Entries produced by
    the same prefill invocation are assumed to remain available to its fused
    attention kernel and are therefore writes, but not compulsory HBM reads.

    ``*_operand_*`` expands every query-to-entry access.  It is the no-reuse
    pair-stream traffic alternative, not a claim about a particular kernel's
    measured HBM behavior.  Compute, writes, and final cache capacity are the
    same under both traffic alternatives.
    """

    main_compulsory_read_entries: float
    main_operand_read_entries: float
    main_write_entries: float
    main_stored_entries: float
    index_compulsory_read_bytes: float = 0.0
    index_operand_read_bytes: float = 0.0
    index_write_bytes: float = 0.0
    index_stored_bytes: float = 0.0
    index_flops: float = 0.0


def _validate_prefill_arguments(
    cached_tokens: int,
    new_tokens: int,
    include_self_attention: bool,
) -> None:
    """Validate the common single-request prefill arguments."""

    for name, value in (
        ("cached_tokens", cached_tokens),
        ("new_tokens", new_tokens),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
        if value < 0:
            raise ValueError(f"{name} must be >= 0")
    if not isinstance(include_self_attention, bool):
        raise ValueError("include_self_attention must be true or false")


def _effective_prefill_start(
    cached_tokens: int, include_self_attention: bool
) -> int:
    """Context value evaluated for the first query in a prefill chunk."""

    return cached_tokens + int(include_self_attention)


def _sum_clamped_linear(start: int, count: int, cap: int) -> float:
    """Return ``sum(min(start + i, cap), i=0..count-1)`` in O(1)."""

    below_cap = min(count, max(cap - start, 0))
    linear_sum = below_cap * (2 * start + below_cap - 1) / 2.0
    return linear_sum + (count - below_cap) * cap


def _floor_division_prefix(count: int, divisor: int) -> int:
    """Return ``sum(floor(i/divisor), i=0..count-1)`` exactly."""

    blocks, remainder = divmod(count, divisor)
    return (
        divisor * blocks * (blocks - 1) // 2
        + blocks * remainder
    )


def _sum_floor_division(start: int, count: int, divisor: int) -> float:
    """Return ``sum(floor((start+i)/divisor), i=0..count-1)``."""

    return float(
        _floor_division_prefix(start + count, divisor)
        - _floor_division_prefix(start, divisor)
    )


def _sum_min_floor_division(
    start: int, count: int, divisor: int, cap: int
) -> float:
    """Sum compressed candidate counts after applying a top-k cap."""

    # floor(value / divisor) first reaches ``cap`` at cap * divisor.
    below_cap = min(count, max(cap * divisor - start, 0))
    return (
        _sum_floor_division(start, below_cap, divisor)
        + float((count - below_cap) * cap)
    )


class AccessPattern(ABC):
    @abstractmethod
    def evaluate(
        self, context_tokens: int, deployment: DeploymentConfig
    ) -> AccessStats:
        raise NotImplementedError

    @abstractmethod
    def prefill_evaluate(
        self,
        cached_tokens: int,
        new_tokens: int,
        deployment: DeploymentConfig,
        include_self_attention: bool = True,
    ) -> PrefillAccessStats:
        """Aggregate one request's causal prefill accesses without looping."""

        raise NotImplementedError


@dataclass(frozen=True)
class FullAccess(AccessPattern):
    def evaluate(
        self, context_tokens: int, deployment: DeploymentConfig
    ) -> AccessStats:
        del deployment
        return AccessStats(
            main_read_entries=float(context_tokens),
            main_write_entries=1.0,
            main_stored_entries=float(context_tokens),
        )

    def prefill_evaluate(
        self,
        cached_tokens: int,
        new_tokens: int,
        deployment: DeploymentConfig,
        include_self_attention: bool = True,
    ) -> PrefillAccessStats:
        del deployment
        _validate_prefill_arguments(
            cached_tokens, new_tokens, include_self_attention
        )
        start = _effective_prefill_start(
            cached_tokens, include_self_attention
        )
        operand_entries = new_tokens * (2 * start + new_tokens - 1) / 2.0
        return PrefillAccessStats(
            main_compulsory_read_entries=(
                float(cached_tokens) if new_tokens else 0.0
            ),
            main_operand_read_entries=operand_entries,
            main_write_entries=float(new_tokens),
            main_stored_entries=float(cached_tokens + new_tokens),
        )


@dataclass(frozen=True)
class SlidingWindowAccess(AccessPattern):
    window_tokens: int

    def evaluate(
        self, context_tokens: int, deployment: DeploymentConfig
    ) -> AccessStats:
        del deployment
        visible = float(min(context_tokens, self.window_tokens))
        return AccessStats(visible, 1.0, visible)

    def prefill_evaluate(
        self,
        cached_tokens: int,
        new_tokens: int,
        deployment: DeploymentConfig,
        include_self_attention: bool = True,
    ) -> PrefillAccessStats:
        del deployment
        _validate_prefill_arguments(
            cached_tokens, new_tokens, include_self_attention
        )
        self_entries = int(include_self_attention)
        start = cached_tokens + self_entries
        # The first query exposes the only cached-prefix entries which can
        # appear in this chunk.  Later windows slide toward newly produced KV.
        compulsory_entries = min(
            cached_tokens, max(self.window_tokens - self_entries, 0)
        )
        if not new_tokens:
            compulsory_entries = 0
        return PrefillAccessStats(
            main_compulsory_read_entries=float(compulsory_entries),
            main_operand_read_entries=_sum_clamped_linear(
                start, new_tokens, self.window_tokens
            ),
            main_write_entries=float(new_tokens),
            main_stored_entries=float(
                min(cached_tokens + new_tokens, self.window_tokens)
            ),
        )


@dataclass(frozen=True)
class CompressedFullAccess(AccessPattern):
    """Dense attention over entries produced every ``compression_ratio`` tokens."""

    compression_ratio: int

    def evaluate(
        self, context_tokens: int, deployment: DeploymentConfig
    ) -> AccessStats:
        del deployment
        complete_entries = float(context_tokens // self.compression_ratio)
        return AccessStats(
            main_read_entries=complete_entries,
            main_write_entries=1.0 / self.compression_ratio,
            main_stored_entries=complete_entries,
        )

    def prefill_evaluate(
        self,
        cached_tokens: int,
        new_tokens: int,
        deployment: DeploymentConfig,
        include_self_attention: bool = True,
    ) -> PrefillAccessStats:
        del deployment
        _validate_prefill_arguments(
            cached_tokens, new_tokens, include_self_attention
        )
        ratio = self.compression_ratio
        initial_entries = cached_tokens // ratio
        final_entries = (cached_tokens + new_tokens) // ratio
        start = _effective_prefill_start(
            cached_tokens, include_self_attention
        )
        return PrefillAccessStats(
            main_compulsory_read_entries=(
                float(initial_entries) if new_tokens else 0.0
            ),
            main_operand_read_entries=_sum_floor_division(
                start, new_tokens, ratio
            ),
            main_write_entries=float(final_entries - initial_entries),
            main_stored_entries=float(final_entries),
        )


@dataclass(frozen=True)
class FixedTopKAccess(AccessPattern):
    """Top-k access where selection/index cost is intentionally external.

    Use this only when selection is fixed by structure or its cost is accounted
    by another mixer.  Learned content selection should use
    :class:`LearnedTopKAccess` instead.
    """

    top_k: int
    compression_ratio: int = 1

    def evaluate(
        self, context_tokens: int, deployment: DeploymentConfig
    ) -> AccessStats:
        del deployment
        candidates = float(context_tokens // self.compression_ratio)
        return AccessStats(
            main_read_entries=min(float(self.top_k), candidates),
            main_write_entries=1.0 / self.compression_ratio,
            main_stored_entries=candidates,
        )

    def prefill_evaluate(
        self,
        cached_tokens: int,
        new_tokens: int,
        deployment: DeploymentConfig,
        include_self_attention: bool = True,
    ) -> PrefillAccessStats:
        del deployment
        _validate_prefill_arguments(
            cached_tokens, new_tokens, include_self_attention
        )
        ratio = self.compression_ratio
        initial_entries = cached_tokens // ratio
        final_entries = (cached_tokens + new_tokens) // ratio
        start = _effective_prefill_start(
            cached_tokens, include_self_attention
        )
        operand_entries = _sum_min_floor_division(
            start, new_tokens, ratio, self.top_k
        )
        # Without a structural selection trace the overlap of Q top-k sets is
        # unknowable.  This is a conservative distinct cached-entry union: it
        # is bounded by both the prefix capacity and all selected slots.
        compulsory_entries = min(float(initial_entries), operand_entries)
        return PrefillAccessStats(
            main_compulsory_read_entries=compulsory_entries,
            main_operand_read_entries=operand_entries,
            main_write_entries=float(final_entries - initial_entries),
            main_stored_entries=float(final_entries),
        )


@dataclass(frozen=True)
class LearnedTopKAccess(AccessPattern):
    """Content-selected sparse attention with an explicit indexer scan.

    ``compression_ratio`` is one for DSA over ordinary tokens and greater than
    one for CSA-style selection over compressed entries.  The main attention
    reads only top-k entries, while the indexer scans every candidate.
    """

    top_k: int
    compression_ratio: int
    index_entry_elements: int
    index_query_heads: int
    index_head_dim: int
    index_bits: float | None = None
    selection_flops_per_candidate: float = 0.0

    def evaluate(
        self, context_tokens: int, deployment: DeploymentConfig
    ) -> AccessStats:
        candidates = float(context_tokens // self.compression_ratio)
        entry_bits = self.index_bits or deployment.index_bits
        entry_bytes = _bits_to_bytes(self.index_entry_elements, entry_bits)

        # One dot product per indexer head and candidate.  ReLU, head weighting,
        # and top-k selection can be represented by selection_flops_per_candidate.
        score_flops = 2.0 * self.index_query_heads * self.index_head_dim
        index_flops = candidates * (
            score_flops + self.selection_flops_per_candidate
        )
        write_entries = 1.0 / self.compression_ratio

        return AccessStats(
            main_read_entries=min(float(self.top_k), candidates),
            main_write_entries=write_entries,
            main_stored_entries=candidates,
            index_read_bytes=candidates * entry_bytes,
            index_write_bytes=write_entries * entry_bytes,
            index_stored_bytes=candidates * entry_bytes,
            index_flops=index_flops,
        )

    def prefill_evaluate(
        self,
        cached_tokens: int,
        new_tokens: int,
        deployment: DeploymentConfig,
        include_self_attention: bool = True,
    ) -> PrefillAccessStats:
        _validate_prefill_arguments(
            cached_tokens, new_tokens, include_self_attention
        )
        ratio = self.compression_ratio
        initial_entries = cached_tokens // ratio
        final_entries = (cached_tokens + new_tokens) // ratio
        start = _effective_prefill_start(
            cached_tokens, include_self_attention
        )
        candidate_operand_entries = _sum_floor_division(
            start, new_tokens, ratio
        )
        main_operand_entries = _sum_min_floor_division(
            start, new_tokens, ratio, self.top_k
        )
        main_compulsory_entries = min(
            float(initial_entries), main_operand_entries
        )

        entry_bits = self.index_bits or deployment.index_bits
        entry_bytes = _bits_to_bytes(self.index_entry_elements, entry_bits)
        score_flops = 2.0 * self.index_query_heads * self.index_head_dim

        return PrefillAccessStats(
            # The selected KV sets can vary across queries, so this is the same
            # conservative distinct-prefix union used by FixedTopKAccess.
            main_compulsory_read_entries=main_compulsory_entries,
            main_operand_read_entries=main_operand_entries,
            main_write_entries=float(final_entries - initial_entries),
            main_stored_entries=float(final_entries),
            # Every old index entry is scanned by at least the first query;
            # new entries can stay within the fused invocation.
            index_compulsory_read_bytes=(
                initial_entries * entry_bytes if new_tokens else 0.0
            ),
            index_operand_read_bytes=(
                candidate_operand_entries * entry_bytes
            ),
            index_write_bytes=(final_entries - initial_entries) * entry_bytes,
            index_stored_bytes=final_entries * entry_bytes,
            index_flops=candidate_operand_entries
            * (score_flops + self.selection_flops_per_candidate),
        )


@dataclass(frozen=True)
class PrefillMixerCost:
    """Per-layer, per-request cost of one prefill chunk.

    ``work`` is the compulsory/logical-HBM alternative: a fused invocation
    reads each required cached-prefix entry once and does not reread entries it
    produces itself.  ``operand_work`` is the pair-stream alternative where
    attention and index reads are expanded across all queries.  Both contain
    identical FLOPs and write traffic, so callers select one alternative; they
    must never add the two together.  ``cache`` is capacity after all
    ``new_tokens`` have been processed, rather than a sum of per-position
    capacities.
    """

    work: WorkCost
    operand_work: WorkCost
    cache: CacheCapacity

    def scaled(self, factor: float) -> "PrefillMixerCost":
        return PrefillMixerCost(
            work=self.work.scaled(factor),
            operand_work=self.operand_work.scaled(factor),
            cache=self.cache.scaled(factor),
        )


class SequenceMixer(ABC):
    """A non-parameterized sequence operation for one model layer."""

    @abstractmethod
    def decode_cost(
        self, context_tokens: int, deployment: DeploymentConfig
    ) -> MixerCost:
        raise NotImplementedError

    @abstractmethod
    def prefill_cost(
        self,
        cached_tokens: int,
        new_tokens: int,
        deployment: DeploymentConfig,
        include_self_attention: bool = True,
    ) -> PrefillMixerCost:
        """Return aggregate cost for one request's fused prefill chunk."""

        raise NotImplementedError


@dataclass(frozen=True)
class SoftmaxAttentionMixer(SequenceMixer):
    layout: KVLayout
    access: AccessPattern
    softmax_flops_per_score: float = 0.0

    def decode_cost(
        self, context_tokens: int, deployment: DeploymentConfig
    ) -> MixerCost:
        stats = self.access.evaluate(context_tokens, deployment)
        entry_bytes = self.layout.entry_bytes(deployment)

        kv_read_bytes = (
            stats.main_read_entries
            * entry_bytes
            * deployment.kv_hbm_fraction
        )
        kv_write_bytes = 0.0
        if deployment.include_kv_write:
            kv_write_bytes = (
                stats.main_write_entries
                * entry_bytes
                * deployment.kv_hbm_fraction
            )

        attention_flops = (
            stats.main_read_entries * self.layout.attention_flops_per_entry()
        )
        # Exp/max/reduction cost is hardware-dependent, so the default is zero
        # to preserve the document's QK+AV convention.  It can be enabled per
        # mixer when a project adopts a specific FLOP-equivalent convention.
        attention_flops += (
            stats.main_read_entries
            * self.layout.query_heads
            * self.softmax_flops_per_score
        )

        index_write_bytes = 0.0
        if deployment.include_index_write:
            index_write_bytes = (
                stats.index_write_bytes * deployment.index_hbm_fraction
            )

        work = WorkCost(
            attention_flops=attention_flops,
            index_flops=stats.index_flops,
            kv_read_bytes=kv_read_bytes,
            kv_write_bytes=kv_write_bytes,
            index_read_bytes=(
                stats.index_read_bytes * deployment.index_hbm_fraction
            ),
            index_write_bytes=index_write_bytes,
        )
        cache = CacheCapacity(
            kv_bytes=stats.main_stored_entries * entry_bytes,
            index_bytes=stats.index_stored_bytes,
        )
        return MixerCost(work, cache)

    def prefill_cost(
        self,
        cached_tokens: int,
        new_tokens: int,
        deployment: DeploymentConfig,
        include_self_attention: bool = True,
    ) -> PrefillMixerCost:
        stats = self.access.prefill_evaluate(
            cached_tokens,
            new_tokens,
            deployment,
            include_self_attention,
        )
        entry_bytes = self.layout.entry_bytes(deployment)

        attention_flops = stats.main_operand_read_entries * (
            self.layout.attention_flops_per_entry()
            + self.layout.query_heads * self.softmax_flops_per_score
        )
        kv_write_bytes = 0.0
        if deployment.include_kv_write:
            kv_write_bytes = (
                stats.main_write_entries
                * entry_bytes
                * deployment.kv_hbm_fraction
            )
        index_write_bytes = 0.0
        if deployment.include_index_write:
            index_write_bytes = (
                stats.index_write_bytes * deployment.index_hbm_fraction
            )

        common = {
            "attention_flops": attention_flops,
            "index_flops": stats.index_flops,
            "kv_write_bytes": kv_write_bytes,
            "index_write_bytes": index_write_bytes,
        }
        compulsory_work = WorkCost(
            **common,
            kv_read_bytes=(
                stats.main_compulsory_read_entries
                * entry_bytes
                * deployment.kv_hbm_fraction
            ),
            index_read_bytes=(
                stats.index_compulsory_read_bytes
                * deployment.index_hbm_fraction
            ),
        )
        operand_work = WorkCost(
            **common,
            kv_read_bytes=(
                stats.main_operand_read_entries
                * entry_bytes
                * deployment.kv_hbm_fraction
            ),
            index_read_bytes=(
                stats.index_operand_read_bytes
                * deployment.index_hbm_fraction
            ),
        )
        cache = CacheCapacity(
            kv_bytes=stats.main_stored_entries * entry_bytes,
            index_bytes=stats.index_stored_bytes,
        )
        return PrefillMixerCost(compulsory_work, operand_work, cache)


@dataclass(frozen=True)
class RecurrentStateMixer(SequenceMixer):
    """Generic fixed-state mixer for linear attention, SSMs, or RNN variants.

    Gated DeltaNet, KDA, Lightning Attention, Mamba, and related mechanisms do
    not share one exact update formula.  The configuration therefore states the
    persistent state size and mechanism-specific FLOPs explicitly instead of
    pretending that a universal formula exists.
    """

    state_elements: float
    read_elements_per_token: float
    write_elements_per_token: float
    flops_per_token: float
    state_bits: float | None = None
    read_hbm_fraction: float | None = None
    write_hbm_fraction: float | None = None

    def decode_cost(
        self, context_tokens: int, deployment: DeploymentConfig
    ) -> MixerCost:
        del context_tokens
        bits = self.state_bits or deployment.state_bits
        read_fraction = (
            deployment.state_hbm_fraction
            if self.read_hbm_fraction is None
            else self.read_hbm_fraction
        )
        write_fraction = (
            deployment.state_hbm_fraction
            if self.write_hbm_fraction is None
            else self.write_hbm_fraction
        )
        write_bytes = 0.0
        if deployment.include_state_write:
            write_bytes = _bits_to_bytes(
                self.write_elements_per_token, bits
            ) * write_fraction

        return MixerCost(
            work=WorkCost(
                state_flops=self.flops_per_token,
                state_read_bytes=_bits_to_bytes(
                    self.read_elements_per_token, bits
                )
                * read_fraction,
                state_write_bytes=write_bytes,
            ),
            cache=CacheCapacity(
                state_bytes=_bits_to_bytes(self.state_elements, bits)
            ),
        )

    def prefill_cost(
        self,
        cached_tokens: int,
        new_tokens: int,
        deployment: DeploymentConfig,
        include_self_attention: bool = True,
    ) -> PrefillMixerCost:
        _validate_prefill_arguments(
            cached_tokens, new_tokens, include_self_attention
        )
        bits = self.state_bits or deployment.state_bits
        read_fraction = (
            deployment.state_hbm_fraction
            if self.read_hbm_fraction is None
            else self.read_hbm_fraction
        )
        write_fraction = (
            deployment.state_hbm_fraction
            if self.write_hbm_fraction is None
            else self.write_hbm_fraction
        )

        # A fused scan consumes the prior persistent state once and materializes
        # only its final state.  Intermediate recurrence values are not HBM
        # traffic.  A fresh prompt starts from an implicit zero state.
        read_bytes = 0.0
        if cached_tokens and new_tokens:
            read_bytes = _bits_to_bytes(
                self.read_elements_per_token, bits
            ) * read_fraction
        write_bytes = 0.0
        if new_tokens and deployment.include_state_write:
            write_bytes = _bits_to_bytes(
                self.write_elements_per_token, bits
            ) * write_fraction
        work = WorkCost(
            state_flops=self.flops_per_token * new_tokens,
            state_read_bytes=read_bytes,
            state_write_bytes=write_bytes,
        )
        cache = CacheCapacity(
            state_bytes=_bits_to_bytes(self.state_elements, bits)
        )
        # Operand expansion is meaningful for attention/index scans.  A
        # recurrent prefill is explicitly modeled as one fused scan in both
        # traffic alternatives.
        return PrefillMixerCost(work, work, cache)


@dataclass(frozen=True)
class LinearAttentionMixer(SequenceMixer):
    """Decode recurrence for kernel/feature-map linear attention.

    The persistent numerator state of one head is a ``key_dim x value_dim``
    matrix.  Normalized variants also keep a ``key_dim`` denominator vector.
    This covers the state shape used by many linear-attention families while
    leaving feature-map-specific projections in the model parameter count.

    The default non-parameter FLOP convention is:

    * numerator outer-product update: 2 FLOPs per matrix element;
    * query/state contraction: 2 FLOPs per matrix element;
    * optional denominator update and dot product: 3 FLOPs per vector element;
    * optional output normalization: one division per output element.

    Nonlinear feature maps and gates are mechanism-specific and can be added
    with ``extra_flops_per_token``.
    """

    q_heads: int
    key_dim: int
    value_dim: int
    normalizer_state: bool = True
    extra_flops_per_token: float = 0.0
    state_bits: float | None = None
    read_hbm_fraction: float | None = None
    write_hbm_fraction: float | None = None

    def decode_cost(
        self, context_tokens: int, deployment: DeploymentConfig
    ) -> MixerCost:
        matrix_elements = self.q_heads * self.key_dim * self.value_dim
        normalizer_elements = (
            self.q_heads * self.key_dim if self.normalizer_state else 0
        )
        state_elements = matrix_elements + normalizer_elements

        state_flops = 4.0 * matrix_elements
        if self.normalizer_state:
            state_flops += 3.0 * normalizer_elements
            state_flops += self.q_heads * self.value_dim
        state_flops += self.extra_flops_per_token

        return RecurrentStateMixer(
            state_elements=state_elements,
            read_elements_per_token=state_elements,
            write_elements_per_token=state_elements,
            flops_per_token=state_flops,
            state_bits=self.state_bits,
            read_hbm_fraction=self.read_hbm_fraction,
            write_hbm_fraction=self.write_hbm_fraction,
        ).decode_cost(context_tokens, deployment)

    def prefill_cost(
        self,
        cached_tokens: int,
        new_tokens: int,
        deployment: DeploymentConfig,
        include_self_attention: bool = True,
    ) -> PrefillMixerCost:
        matrix_elements = self.q_heads * self.key_dim * self.value_dim
        normalizer_elements = (
            self.q_heads * self.key_dim if self.normalizer_state else 0
        )
        state_elements = matrix_elements + normalizer_elements

        state_flops = 4.0 * matrix_elements
        if self.normalizer_state:
            state_flops += 3.0 * normalizer_elements
            state_flops += self.q_heads * self.value_dim
        state_flops += self.extra_flops_per_token

        return RecurrentStateMixer(
            state_elements=state_elements,
            read_elements_per_token=state_elements,
            write_elements_per_token=state_elements,
            flops_per_token=state_flops,
            state_bits=self.state_bits,
            read_hbm_fraction=self.read_hbm_fraction,
            write_hbm_fraction=self.write_hbm_fraction,
        ).prefill_cost(
            cached_tokens,
            new_tokens,
            deployment,
            include_self_attention,
        )


@dataclass(frozen=True)
class DiagonalSSMMixer(SequenceMixer):
    """Generic diagonal state-space recurrence with optional conv state.

    ``channels * state_dim`` is the recurrent SSM state.  Some SSM blocks also
    keep a short depthwise-convolution history; ``conv_state_length`` states
    how many elements are allocated per channel for that history.

    By default, one recurrent state element costs five arithmetic FLOPs:
    ``A*x + B*u`` uses three, and the ``C*x`` output contraction uses two.
    Exponentials, gates, and discretization details are not universal, so the
    coefficient and an additive extra cost remain configurable.
    """

    channels: int
    state_dim: int
    conv_state_length: int = 0
    recurrence_flops_per_state_element: float = 5.0
    extra_flops_per_token: float = 0.0
    state_bits: float | None = None
    read_hbm_fraction: float | None = None
    write_hbm_fraction: float | None = None

    def decode_cost(
        self, context_tokens: int, deployment: DeploymentConfig
    ) -> MixerCost:
        recurrent_elements = self.channels * self.state_dim
        state_elements = self.channels * (
            self.state_dim + self.conv_state_length
        )
        state_flops = (
            recurrent_elements * self.recurrence_flops_per_state_element
            + self.extra_flops_per_token
        )
        return RecurrentStateMixer(
            state_elements=state_elements,
            read_elements_per_token=state_elements,
            write_elements_per_token=state_elements,
            flops_per_token=state_flops,
            state_bits=self.state_bits,
            read_hbm_fraction=self.read_hbm_fraction,
            write_hbm_fraction=self.write_hbm_fraction,
        ).decode_cost(context_tokens, deployment)

    def prefill_cost(
        self,
        cached_tokens: int,
        new_tokens: int,
        deployment: DeploymentConfig,
        include_self_attention: bool = True,
    ) -> PrefillMixerCost:
        recurrent_elements = self.channels * self.state_dim
        state_elements = self.channels * (
            self.state_dim + self.conv_state_length
        )
        state_flops = (
            recurrent_elements * self.recurrence_flops_per_state_element
            + self.extra_flops_per_token
        )
        return RecurrentStateMixer(
            state_elements=state_elements,
            read_elements_per_token=state_elements,
            write_elements_per_token=state_elements,
            flops_per_token=state_flops,
            state_bits=self.state_bits,
            read_hbm_fraction=self.read_hbm_fraction,
            write_hbm_fraction=self.write_hbm_fraction,
        ).prefill_cost(
            cached_tokens,
            new_tokens,
            deployment,
            include_self_attention,
        )


@dataclass(frozen=True)
class MambaStateMixer(SequenceMixer):
    """Persistent decode state for Mamba-1 and Mamba-2 style blocks.

    Mamba-1 keeps ``inner_dim * state_dim`` selective-SSM elements.  Mamba-2
    may apply its SSM to only ``ssm_dim`` of the expanded inner stream; when
    omitted, ``ssm_dim`` defaults to ``inner_dim``.
    Mamba-1 convolves only the inner stream, so its convolution cache has
    ``inner_dim * conv_kernel`` elements.  Mamba-2 convolves the combined
    ``x/B/C`` stream; its cache width is
    ``inner_dim + 2 * groups * state_dim``.

    Projection, convolution-weight, and output-projection MACs must already be
    present in ``always_active_parameters``.  This mixer adds only recurrence
    arithmetic and persistent state traffic, which prevents double counting.
    """

    variant: str
    inner_dim: int
    state_dim: int
    conv_kernel: int
    ssm_dim: int | None = None
    groups: int = 1
    recurrence_flops_per_state_element: float = 5.0
    extra_flops_per_token: float = 0.0
    state_bits: float | None = None
    read_hbm_fraction: float | None = None
    write_hbm_fraction: float | None = None

    def decode_cost(
        self, context_tokens: int, deployment: DeploymentConfig
    ) -> MixerCost:
        if self.variant == "mamba1":
            recurrence_width = self.inner_dim
            conv_channels = self.inner_dim
        elif self.variant == "mamba2":
            recurrence_width = self.ssm_dim or self.inner_dim
            conv_channels = (
                recurrence_width + 2 * self.groups * self.state_dim
            )
        else:  # Construction validates this; retain a defensive API check.
            raise ValueError(f"unsupported Mamba variant {self.variant!r}")

        recurrent_elements = recurrence_width * self.state_dim
        state_elements = (
            recurrent_elements + conv_channels * self.conv_kernel
        )
        state_flops = (
            recurrent_elements * self.recurrence_flops_per_state_element
            + self.extra_flops_per_token
        )
        return RecurrentStateMixer(
            state_elements=state_elements,
            read_elements_per_token=state_elements,
            write_elements_per_token=state_elements,
            flops_per_token=state_flops,
            state_bits=self.state_bits,
            read_hbm_fraction=self.read_hbm_fraction,
            write_hbm_fraction=self.write_hbm_fraction,
        ).decode_cost(context_tokens, deployment)

    def prefill_cost(
        self,
        cached_tokens: int,
        new_tokens: int,
        deployment: DeploymentConfig,
        include_self_attention: bool = True,
    ) -> PrefillMixerCost:
        if self.variant == "mamba1":
            recurrence_width = self.inner_dim
            conv_channels = self.inner_dim
        elif self.variant == "mamba2":
            recurrence_width = self.ssm_dim or self.inner_dim
            conv_channels = (
                recurrence_width + 2 * self.groups * self.state_dim
            )
        else:
            raise ValueError(f"unsupported Mamba variant {self.variant!r}")

        recurrent_elements = recurrence_width * self.state_dim
        state_elements = (
            recurrent_elements + conv_channels * self.conv_kernel
        )
        state_flops = (
            recurrent_elements * self.recurrence_flops_per_state_element
            + self.extra_flops_per_token
        )
        return RecurrentStateMixer(
            state_elements=state_elements,
            read_elements_per_token=state_elements,
            write_elements_per_token=state_elements,
            flops_per_token=state_flops,
            state_bits=self.state_bits,
            read_hbm_fraction=self.read_hbm_fraction,
            write_hbm_fraction=self.write_hbm_fraction,
        ).prefill_cost(
            cached_tokens,
            new_tokens,
            deployment,
            include_self_attention,
        )


@dataclass(frozen=True)
class FixedCostMixer(SequenceMixer):
    """Explicit additive cost for a small model-specific operation."""

    work: WorkCost
    cache: CacheCapacity
    prefill_scope: str = "per_token"

    def __post_init__(self) -> None:
        if self.prefill_scope not in {"per_token", "per_request"}:
            raise ValueError(
                "prefill_scope must be 'per_token' or 'per_request'"
            )

    def decode_cost(
        self, context_tokens: int, deployment: DeploymentConfig
    ) -> MixerCost:
        del context_tokens, deployment
        return MixerCost(self.work, self.cache)

    def prefill_cost(
        self,
        cached_tokens: int,
        new_tokens: int,
        deployment: DeploymentConfig,
        include_self_attention: bool = True,
    ) -> PrefillMixerCost:
        del deployment
        _validate_prefill_arguments(
            cached_tokens, new_tokens, include_self_attention
        )
        factor = (
            float(new_tokens)
            if self.prefill_scope == "per_token"
            else float(bool(new_tokens))
        )
        work = self.work.scaled(factor)
        return PrefillMixerCost(work, work, self.cache)


def _int_positive(config: Mapping[str, Any], key: str, path: str) -> int:
    value = _positive(config, key, path)
    if not value.is_integer():
        raise ValueError(f"{path}.{key} must be an integer")
    return int(value)


def _int_nonnegative(config: Mapping[str, Any], key: str, path: str) -> int:
    value = _nonnegative(config, key, path)
    if not value.is_integer():
        raise ValueError(f"{path}.{key} must be an integer")
    return int(value)


def _optional_fraction(
    config: Mapping[str, Any], key: str, path: str
) -> float | None:
    if key not in config:
        return None
    value = _number(config, key, path)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{path}.{key} must be between 0 and 1")
    return value


def _boolean(
    config: Mapping[str, Any], key: str, path: str, default: bool
) -> bool:
    value = config.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"{path}.{key} must be true or false")
    return value


def build_kv_layout(config: Mapping[str, Any], path: str) -> KVLayout:
    kind = config.get("kind")
    if kind in {"grouped", "mha", "mqa", "gqa"}:
        q_heads = _int_positive(config, "query_heads", path)

        # Named aliases make model files self-describing while ``grouped``
        # remains the fully explicit backwards-compatible form.
        if kind == "mha":
            kv_heads = (
                _int_positive(config, "kv_heads", path)
                if "kv_heads" in config
                else q_heads
            )
            if kv_heads != q_heads:
                raise ValueError(
                    f"{path}.kv_heads must equal query_heads for MHA"
                )
        elif kind == "mqa":
            kv_heads = (
                _int_positive(config, "kv_heads", path)
                if "kv_heads" in config
                else 1
            )
            if kv_heads != 1:
                raise ValueError(f"{path}.kv_heads must be 1 for MQA")
        else:
            kv_heads = _int_positive(config, "kv_heads", path)

        if kv_heads > q_heads:
            raise ValueError(f"{path}.kv_heads cannot exceed query_heads")
        if q_heads % kv_heads != 0:
            raise ValueError(
                f"{path}.query_heads must be divisible by kv_heads"
            )
        if kind == "gqa" and not 1 < kv_heads < q_heads:
            raise ValueError(
                f"{path}.kv_heads must be between 1 and query_heads for GQA"
            )
        return GroupedKVLayout(
            q_heads=q_heads,
            kv_heads=kv_heads,
            head_dim=_int_positive(config, "head_dim", path),
            key_bits=(
                _positive(config, "key_bits", path)
                if "key_bits" in config
                else None
            ),
            value_bits=(
                _positive(config, "value_bits", path)
                if "value_bits" in config
                else None
            ),
        )
    if kind in {"latent", "mla"}:
        return LatentKVLayout(
            q_heads=_int_positive(config, "query_heads", path),
            latent_dim=_int_positive(config, "latent_dim", path),
            rope_dim=_int_nonnegative(config, "rope_dim", path),
            latent_bits=(
                _positive(config, "latent_bits", path)
                if "latent_bits" in config
                else None
            ),
            rope_bits=(
                _positive(config, "rope_bits", path)
                if "rope_bits" in config
                else None
            ),
        )
    if kind in {"shared", "shared_kv"}:
        return SharedKVLayout(
            q_heads=_int_positive(config, "query_heads", path),
            head_dim=_int_positive(config, "head_dim", path),
            rope_dim=_int_nonnegative(config, "rope_dim", path)
            if "rope_dim" in config
            else 0,
            non_rope_bits=(
                _positive(config, "non_rope_bits", path)
                if "non_rope_bits" in config
                else None
            ),
            rope_bits=(
                _positive(config, "rope_bits", path)
                if "rope_bits" in config
                else None
            ),
        )
    if kind == "explicit":
        return ExplicitKVLayout(
            q_heads=_int_positive(config, "query_heads", path),
            bytes_per_entry=_positive(config, "bytes_per_entry", path),
            flops_per_entry=_positive(config, "flops_per_entry", path),
        )
    raise ValueError(f"{path}.kind has unsupported KV layout {kind!r}")


def build_access_pattern(config: Mapping[str, Any], path: str) -> AccessPattern:
    kind = config.get("kind")
    if kind == "full":
        return FullAccess()
    if kind in {"sliding_window", "swa"}:
        return SlidingWindowAccess(
            window_tokens=_int_positive(config, "window_tokens", path)
        )
    if kind in {"compressed_full", "hca"}:
        compression_ratio = _int_positive(
            config, "compression_ratio", path
        )
        if kind == "hca" and compression_ratio <= 1:
            raise ValueError(
                f"{path}.compression_ratio must be > 1 for HCA"
            )
        return CompressedFullAccess(
            compression_ratio=compression_ratio
        )
    if kind == "fixed_topk":
        return FixedTopKAccess(
            top_k=_int_positive(config, "top_k", path),
            compression_ratio=(
                _int_positive(config, "compression_ratio", path)
                if "compression_ratio" in config
                else 1
            ),
        )
    if kind in {"learned_topk", "dsa", "csa"}:
        if kind == "dsa":
            compression_ratio = (
                _int_positive(config, "compression_ratio", path)
                if "compression_ratio" in config
                else 1
            )
            if compression_ratio != 1:
                raise ValueError(
                    f"{path}.compression_ratio must be 1 for DSA"
                )
        elif kind == "csa":
            compression_ratio = _int_positive(
                config, "compression_ratio", path
            )
            if compression_ratio <= 1:
                raise ValueError(
                    f"{path}.compression_ratio must be > 1 for CSA"
                )
        else:
            compression_ratio = (
                _int_positive(config, "compression_ratio", path)
                if "compression_ratio" in config
                else 1
            )

        return LearnedTopKAccess(
            top_k=_int_positive(config, "top_k", path),
            compression_ratio=compression_ratio,
            index_entry_elements=_int_positive(
                config, "index_entry_elements", path
            ),
            index_query_heads=_int_positive(config, "index_query_heads", path),
            index_head_dim=_int_positive(config, "index_head_dim", path),
            index_bits=(
                _positive(config, "index_bits", path)
                if "index_bits" in config
                else None
            ),
            selection_flops_per_candidate=(
                _nonnegative(config, "selection_flops_per_candidate", path)
                if "selection_flops_per_candidate" in config
                else 0.0
            ),
        )
    raise ValueError(f"{path}.kind has unsupported access pattern {kind!r}")


def _explicit_work(config: Mapping[str, Any], path: str) -> WorkCost:
    names = set(WorkCost.__dataclass_fields__)
    unknown = set(config) - names
    if unknown:
        raise ValueError(f"{path} has unknown cost fields: {sorted(unknown)}")
    values: dict[str, float] = {}
    for name, value in config.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{path}.{name} must be a number")
        if value < 0:
            raise ValueError(f"{path}.{name} must be >= 0")
        values[name] = float(value)
    return WorkCost(**values)


def _explicit_cache(config: Mapping[str, Any], path: str) -> CacheCapacity:
    names = set(CacheCapacity.__dataclass_fields__)
    unknown = set(config) - names
    if unknown:
        raise ValueError(f"{path} has unknown cache fields: {sorted(unknown)}")
    values: dict[str, float] = {}
    for name, value in config.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{path}.{name} must be a number")
        if value < 0:
            raise ValueError(f"{path}.{name} must be >= 0")
        values[name] = float(value)
    return CacheCapacity(**values)


def _state_storage_options(
    config: Mapping[str, Any], path: str
) -> dict[str, float | None]:
    """Parse storage precision and off-chip residency overrides once."""

    return {
        "state_bits": (
            _positive(config, "state_bits", path)
            if "state_bits" in config
            else None
        ),
        "read_hbm_fraction": _optional_fraction(
            config, "read_hbm_fraction", path
        ),
        "write_hbm_fraction": _optional_fraction(
            config, "write_hbm_fraction", path
        ),
    }


def build_mixer(config: Mapping[str, Any], path: str) -> SequenceMixer:
    kind = config.get("kind")
    if kind == "softmax_attention":
        kv_config = config.get("kv_layout")
        access_config = config.get("access")
        if not isinstance(kv_config, Mapping):
            raise ValueError(f"{path}.kv_layout must be an object")
        if not isinstance(access_config, Mapping):
            raise ValueError(f"{path}.access must be an object")
        return SoftmaxAttentionMixer(
            layout=build_kv_layout(kv_config, f"{path}.kv_layout"),
            access=build_access_pattern(access_config, f"{path}.access"),
            softmax_flops_per_score=(
                _nonnegative(config, "softmax_flops_per_score", path)
                if "softmax_flops_per_score" in config
                else 0.0
            ),
        )
    if kind == "recurrent_state":
        state_elements = _positive(config, "state_elements", path)
        return RecurrentStateMixer(
            state_elements=state_elements,
            read_elements_per_token=(
                _nonnegative(config, "read_elements_per_token", path)
                if "read_elements_per_token" in config
                else state_elements
            ),
            write_elements_per_token=(
                _nonnegative(config, "write_elements_per_token", path)
                if "write_elements_per_token" in config
                else state_elements
            ),
            flops_per_token=_nonnegative(config, "flops_per_token", path),
            **_state_storage_options(config, path),
        )
    if kind in {"linear_attention", "linear_attn"}:
        return LinearAttentionMixer(
            q_heads=_int_positive(config, "query_heads", path),
            key_dim=_int_positive(config, "key_dim", path),
            value_dim=_int_positive(config, "value_dim", path),
            normalizer_state=_boolean(
                config, "normalizer_state", path, True
            ),
            extra_flops_per_token=(
                _nonnegative(config, "extra_flops_per_token", path)
                if "extra_flops_per_token" in config
                else 0.0
            ),
            **_state_storage_options(config, path),
        )
    if kind == "ssm":
        return DiagonalSSMMixer(
            channels=_int_positive(config, "channels", path),
            state_dim=_int_positive(config, "state_dim", path),
            conv_state_length=(
                _int_nonnegative(config, "conv_state_length", path)
                if "conv_state_length" in config
                else 0
            ),
            recurrence_flops_per_state_element=(
                _nonnegative(
                    config,
                    "recurrence_flops_per_state_element",
                    path,
                )
                if "recurrence_flops_per_state_element" in config
                else 5.0
            ),
            extra_flops_per_token=(
                _nonnegative(config, "extra_flops_per_token", path)
                if "extra_flops_per_token" in config
                else 0.0
            ),
            **_state_storage_options(config, path),
        )
    if kind in {"mamba", "mamba2"}:
        variant = "mamba2" if kind == "mamba2" else config.get(
            "variant", "mamba1"
        )
        if variant not in {"mamba1", "mamba2"}:
            raise ValueError(
                f"{path}.variant must be 'mamba1' or 'mamba2'"
            )
        groups = (
            _int_positive(config, "groups", path)
            if "groups" in config
            else 1
        )
        if variant == "mamba1" and groups != 1:
            raise ValueError(f"{path}.groups is only applicable to mamba2")
        inner_dim = _int_positive(config, "inner_dim", path)
        ssm_dim = (
            _int_positive(config, "ssm_dim", path)
            if "ssm_dim" in config
            else None
        )
        if variant == "mamba1" and ssm_dim not in {None, inner_dim}:
            raise ValueError(
                f"{path}.ssm_dim must equal inner_dim for mamba1"
            )
        if variant == "mamba2" and ssm_dim is not None and ssm_dim > inner_dim:
            raise ValueError(f"{path}.ssm_dim cannot exceed inner_dim")
        return MambaStateMixer(
            variant=str(variant),
            inner_dim=inner_dim,
            state_dim=_int_positive(config, "state_dim", path),
            conv_kernel=_int_positive(config, "conv_kernel", path),
            ssm_dim=ssm_dim,
            groups=groups,
            recurrence_flops_per_state_element=(
                _nonnegative(
                    config,
                    "recurrence_flops_per_state_element",
                    path,
                )
                if "recurrence_flops_per_state_element" in config
                else 5.0
            ),
            extra_flops_per_token=(
                _nonnegative(config, "extra_flops_per_token", path)
                if "extra_flops_per_token" in config
                else 0.0
            ),
            **_state_storage_options(config, path),
        )
    if kind == "fixed_cost":
        work_config = config.get("work", {})
        cache_config = config.get("cache", {})
        if not isinstance(work_config, Mapping):
            raise ValueError(f"{path}.work must be an object")
        if not isinstance(cache_config, Mapping):
            raise ValueError(f"{path}.cache must be an object")
        prefill_scope = config.get("prefill_scope", "per_token")
        if not isinstance(prefill_scope, str) or prefill_scope not in {
            "per_token",
            "per_request",
        }:
            raise ValueError(
                f"{path}.prefill_scope must be 'per_token' or 'per_request'"
            )
        return FixedCostMixer(
            work=_explicit_work(work_config, f"{path}.work"),
            cache=_explicit_cache(cache_config, f"{path}.cache"),
            prefill_scope=prefill_scope,
        )
    raise ValueError(f"{path}.kind has unsupported mixer {kind!r}")
