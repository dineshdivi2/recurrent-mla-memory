# Focus Control

This note extends the working-memory hypothesis with a narrower, cheaper question:

> Can an LLM be steered to concentrate on the idea it is working on now, instead of letting every part of its context compete on equal terms?

Status: hypothesis. Nothing here has been measured.

## The gap

Attention already weights context by how well it matches the current query, so context does not have literally equal impact. The gap is that a model has no built-in notion of **superseded**. A hypothesis that was ruled out can match the current query as strongly as the one that replaced it.

So the target is not recency alone. The target is fading what no longer applies while keeping what still does.

```text
memory of evidence ≠ memory of computation ≠ what currently applies
```

## Operations on the KV cache

Working notes can sit at different points on a cost ladder. The levels are candidates to compare, not a roadmap to climb.

| Level | Operation | Cost | Training |
| --- | --- | --- | --- |
| 1 | Pin or evict entries | Lowest | No |
| 2 | Add a per-entry attention bias | Very low | No |
| 3 | Merge similar entries | Low | No |
| 4 | Add a steering vector to the cache | Low to run | Example pairs |
| 5 | Insert learned note slots | Medium | Yes |
| 6 | Update note slots with full attention | Highest | Yes |

Level 6 is the recurrent update in [architecture](01-architecture.md). Level 4 exists as [KV Cache Steering](https://arxiv.org/abs/2507.08799v1).

## Focus zones

Level 2 is the first design to test. Each cached entry belongs to one zone:

| Zone | Contents | Effect |
| --- | --- | --- |
| Pinned | Hard constraints, instructions, key facts | Never fades |
| Active | The idea being worked on now | Boosted |
| Background | Everything else | Faded, not deleted |

For an attention score `s(q, k_i)`, the proposal adds a bias `b_i` before normalization:

```text
weight_i ∝ exp(s(q, k_i) + b_i)
```

Because nothing is deleted, background context can return when it becomes relevant.

## Finding the latest idea

| Signal | How it works | Known weakness |
| --- | --- | --- |
| Recency decay | Newer entries weigh more | Fails on returns to older work and on long tool output |
| Step boundaries | Each plan step or tool call starts a segment | Needs structured turns |
| Attention momentum | What the current step uses stays active | Can lock onto the wrong idea |
| Model-marked focus | The model emits a marker for its current idea | Needs training or prompting |

Step boundaries are the first signal to try: they follow ideas more closely than raw position, and agent transcripts already contain them.

## Focus as a control setting

Temperature controls randomness in sampling. Focus would control what the model pays attention to, which is a different axis.

| Setting | Controls | Acts on |
| --- | --- | --- |
| Temperature, top-p | How random the next token is | Output distribution |
| Reasoning effort | How much the model thinks | Generation budget |
| Focus (proposed) | Which context carries weight | Attention over the cache |

A focus profile would be a small set of per-session biases and a strength value. Strength 0 must reproduce unmodified behavior exactly, which gives every experiment a clean control. Whether a single strength value is meaningful across models and contexts is an open question.

## Forking agents by focus

Several agents can share one expensive prefix and differ only in focus:

```text
                shared cached prefix (immutable)
               /               |               \
     focus: logs        focus: config      focus: recent diff
     overlay A           overlay B          overlay C
```

Each fork is a small overlay of bias values rather than a private copy of the cache. Three properties make this worth testing:

- **Cheap forks:** storage per fork scales with the overlay, not with the context.
- **Preserved prefix reuse:** the shared tensors are never edited, so the cached prefix stays reusable. Serving systems would need an engine change to accept per-session biases.
- **Real diversity:** today's forks mostly differ through an appended instruction or sampling noise. Focus overlays could produce search directed by evidence.

The fork claim is only supported if focused forks find correct answers that unfocused forks miss. Diversity alone is not a result; forks can differ and all be wrong. A later merge step is also an open problem, because each fork's conclusions must be reconciled without trusting them blindly.

## Failure modes

- Models were not trained on biased attention, so output may degrade.
- A bias can shift attention but may be unable to store a conclusion. Text may do better for that.
- A boost can feed on itself and fixate on the wrong idea.
- Fading a segment can drop a constraint that still applies.
- Attention weight is a correlate of influence, not proof of it.

## Baselines and order of work

Every result is compared with plain eviction and a text scratchpad at equal budget.

1. **Establish the problem.** Run the [stale-hypothesis diagnostic](../benchmarks/stale_hypothesis/README.md). It uses an unmodified model. If models rarely slip back to overturned ideas, focus is not the bottleneck and this line stops.
2. **Test level 2 bias zones** against the baselines.
3. **Track two errors together.** Stale error should fall without forgetting error rising.
4. **Test forks** only if single-agent focus helps.
