# Baselines and Evaluation

## Equal-budget systems

All systems receive the same recent exact window W. Older history is constrained to the same fixed physical budget K.

| System | Old-history representation |
| --- | --- |
| Full MLA | All original cache states |
| Eviction MLA | K selected original MLA states |
| Recurrent MLA | K synthetic, recurrently updated states |
| Hybrid MLA | K_e exact anchors + K_s synthetic states |

The important adversarial baseline is eviction. If keeping the best original states consistently wins, synthetic recurrence is an inferior compressor—not working memory.

## Task families

### Archival recall

Tests preservation of exact evidence:

- What exact number appeared far earlier?
- Which file or quote was observed?
- What constraint did the user specify?

Eviction is expected to be difficult to beat here.

### Evolving state

Tests preservation of computation derived from evidence:

- Which hypothesis is currently active?
- What evidence ruled out the earlier explanation?
- What changed, and what should be tested next?
- Which constraints remain unresolved?

The working-memory hypothesis is supported only if recurrent synthetic state materially helps on this class or complements exact anchors.

## Metrics

Measure task quality alongside:

- memory budget and active-state size
- update-cycle count and long-horizon degradation
- latency and throughput
- exact-recall versus state-tracking tradeoff
- robustness to contradictory evidence
- failure detection and recovery

Every result must retain an equal-budget comparison and publish failure cases, not only aggregate gains.
