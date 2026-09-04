# Architecture

## Memory hierarchy

The proposed system separates evidence from the model's evolving interpretation:

```text
durable event/history store
        ↓ retrieve on demand
recent exact token window W
        +
exact anchor bank E
        +
synthetic recurrent bank S
        ↓
      MLA Transformer
        ↓
next tokens and updated S
```

- **Recent exact window** preserves high-fidelity local context.
- **Exact anchors** preserve selected archival evidence: numbers, filenames, quotes, constraints, and tool outputs.
- **Synthetic recurrent bank** carries a bounded, mutable representation of the current computation.
- **Event store** remains the lossless source of truth; latent state is not an audit log.

## Transition

At a consolidation boundary, an update controller maps outgoing exact state and the existing synthetic state to a new fixed-size synthetic bank:

```text
S(t+1) = Update(S(t), C(outgoing), optional task signal)
```

The initial prototype deliberately omits goal conditioning, fast/slow memory, and multi-timescale gates. It must first establish that repeated updates remain coherent.

## Why MLA is a useful substrate

In MLA, a compressed latent can act as a shared attention-state bottleneck rather than requiring a controller to synthesize separate K/V tensors for every attention head. This is an enabling property, not a novelty claim.

The harder issue is positional representation. A synthetic slot can aggregate information from many positions, so assigning it one token-like position may be semantically wrong. The prototype must treat this as an experimental variable rather than assume it away.
