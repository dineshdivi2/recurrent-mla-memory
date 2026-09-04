# Recurrent MLA Memory

**Can a pretrained MLA Transformer replace old token-level cache history with a fixed synthetic latent bank—and retain useful computation across many updates?**

Long context preserves evidence. Working memory should preserve the model's current computation derived from that evidence.

This repository investigates a bounded, recurrent working-memory architecture for Multi-head Latent Attention (MLA). The proposal is not to rewrite old KV cache entries in place. Instead, it consolidates expired history into a fixed-size synthetic latent bank, retains a recent exact window, and compares that design against equally sized caches of selected original states.

> The central test is not whether synthetic cache compression works. It is whether recurrent synthetic state provides computational memory that retaining old evidence alone cannot.

## The hypothesis

A long-running model or agent needs at least two different kinds of memory:

```text
Evidence memory
- exact observations
- numbers, filenames, quotations
- tool outputs and constraints

Computational memory
- active hypothesis
- what was ruled out
- current interpretation
- unresolved questions
- next useful action
```

Token eviction may preserve the first well. Recurrent synthesis may help with the second. The hypothesis fails if an equal-budget eviction cache consistently does better.

## Architecture

```text
durable event/history store
        ↓ retrieve on demand
recent exact context window
        +
selected exact anchors
        +
synthetic mutable latent bank
        ↓
      MLA Transformer
        ↓
  output + updated synthetic bank
```

The external event/history store remains the lossless source of truth. The latent bank is intentionally lossy, mutable, and optimized for current computation.

See [architecture](docs/01-architecture.md).

## Equal-budget comparison

Every system receives the same recent exact window. Only its representation of old history changes.

| System | Old history |
| --- | --- |
| Full MLA | All original states |
| Eviction MLA | K selected original states |
| Recurrent MLA | K synthetic mutable states |
| Hybrid MLA | Exact anchors + synthetic state |

The hybrid is especially important: exact evidence and synthesized interpretation may be complementary rather than competing.

See [baselines and evaluation](docs/02-baselines-and-evaluation.md).

## What would count as evidence?

Two deliberately different task families are required:

1. **Archival recall** — recover exact historical facts, where eviction may be strongest.
2. **Evolving state** — track belief revision, derived constraints, and next actions across many updates.

A credible result reports quality, memory budget, latency, update-cycle stability, failures under contradictory evidence, and the exact-recall/state-tracking tradeoff.

## Scope

The first prototype is intentionally narrow:

- fixed-size per-layer synthetic bank
- recent exact window
- equal-budget eviction and hybrid baselines
- repeated consolidation-cycle evaluation
- explicit treatment of positional-state/RoPE behavior

Fast/slow memory, goal-conditioned updates, agent orchestration, and durable retrieval are follow-on directions—not prerequisites for testing the core claim.

See [research positioning](docs/03-research-positioning.md).

## Status

Early research memo and experiment design. The next milestone is a minimal benchmark that can falsify the working-memory hypothesis before a larger training effort.
