# Benchmark Plan

The benchmark must distinguish retained evidence from retained computation.

## A. Archival recall

Generate or curate long sequences containing exact facts, identifiers, constraints, and quotations. Ask for a fact after many consolidation cycles.

Primary outcome: exact-match accuracy as a function of the old-history memory budget.

## B. Evolving-state tracking

Present incremental evidence that revises an earlier hypothesis or constraint. The final question requires the current interpretation, the evidence that changed it, and a next diagnostic action.

Primary outcome: state-tracking accuracy across repeated updates, with contradiction handling and calibration.

## Protocol

For every memory budget K, compare Full MLA, Eviction MLA, Recurrent MLA, and Hybrid MLA with the same local exact window. Log update count, latency, active-state size, and per-example failures.

The first benchmark should be synthetic and inspectable. It should make a negative result easy to recognize before moving to naturalistic agent traces.
