# ADR-0003: Memory budget derives from available RAM, in separate pools

- Status: accepted
- Date: 2026-11-01

## Context

The engine must decide whether a model fits before allocating it. Two naive
approaches are both wrong:

- Planning against installed RAM ignores that Windows and other applications are
  already using it. On the development machine, 16.00 GiB installed corresponded
  to 1.95 GiB actually available during the audit.
- Planning against a single free-memory number lets one workload consume
  everything. Growing a context would then evict model weights that are still in
  use, which is the failure the KV cache exists to avoid.

The specification also forbids treating the pagefile as a memory tier, so commit
limit must not become a de facto budget.

## Decision

`MemoryBudget` computes its ceiling as:

```
os_reserve = max(total_physical * os_reserve_fraction, os_reserve_floor)
usable     = min(total_physical - os_reserve,
                 available_physical * usable_fraction)
```

`usable` is then split across four independent pools: weights, KV cache,
scratch, and sessions. Reservations are refused, not queued, when a pool limit
would be crossed.

Defaults: 25 percent OS reserve with a 2 GiB floor, 90 percent of available
memory claimable, shares of 62/23/8/7.

## Consequences

- On a machine with 1.95 GiB free, the engine plans for 1.75 GiB, not 16 GiB.
  That is the honest answer and it makes memory-constrained mode the default
  reality on this hardware.
- Pool separation means a long context cannot starve model weights, and
  concurrent sessions cannot starve the KV cache.
- Limits are recomputed on `refresh()`. Usage is preserved across a refresh, so a
  shrinking budget can drive a pool past its new limit; `available()` reports
  zero rather than a negative number, and nothing is silently evicted.
- The fractions are policy, not measurements. They need tuning against real
  model sizes once GGUF loading exists, and they are configurable precisely
  because the defaults are guesses.
