# Research Positioning

## The question

This work is not asking whether KV cache compression is possible, nor whether fixed recurrent state is generally useful.

It asks:

> Can a pretrained MLA Transformer be recurrentized through its native latent attention bottleneck while preserving useful attention behavior—and does the resulting state offer computational value beyond retaining selected original evidence?

## Boundary

This repository does **not** claim that:

- a mutable cache is automatically causally equivalent to the original Transformer on a rewritten prefix
- synthetic memory should replace all exact cache states
- fixed recurrent state is a new idea
- a fluent decoded summary proves latent state is correct

Instead, it defines a new recurrent transition system and evaluates it against cache retention baselines.

## Working terms

- **Evidence memory:** archival observations that should remain recoverable.
- **Computational memory:** the current hypothesis, derived constraints, task progress, and other state created by prior computation.
- **Working memory:** a bounded state intended to retain computational memory across consolidation cycles.

The distinction to test is:

```text
memory of evidence ≠ memory of computation
```
