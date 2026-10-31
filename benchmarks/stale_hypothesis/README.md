# Stale-Hypothesis Diagnostic

**Does an unmodified model slip back to an idea that has already been overturned?**

This is the first step in [focus control](../../docs/04-focus-control.md). It changes nothing inside the model. If models rarely slip back, focus is not the bottleneck and that line of work stops here.

Status: the code is tested on a tiny random model. No real model has been run, so there are no results yet.

## How an episode works

Each episode is a short, invented incident log:

1. A rule that stays valid throughout.
2. A first theory, stated one or three times.
3. Unrelated filler.
4. New evidence.
5. A forced choice between the module names.

All names are invented, so the answer cannot come from prior knowledge.

| Condition | What happens | Correct answer | What a wrong answer means |
| --- | --- | --- | --- |
| Revised | The first theory is overturned | The replacement | **Stale error** |
| Control | The other module is cleared and mentioned last | The first theory | Picking the latest name |

The control matters. Without it, a model that always picks the last name mentioned would look as if it understood the revision.

## Factors

- **Style:** the revision is stated explicitly, or only implied by evidence.
- **Gap:** 0, 4 or 16 filler lines before the new evidence.
- **Repeats:** the first theory is stated once or three times.
- **Counterbalancing:** every episode has a twin with the two module names swapped, which cancels any preference for one name.

## What is measured

- **Stale error rate:** how often the overturned idea outscores the current one.
- **Latest-name rate:** the same mistake pattern in the control condition.
- **Constraint accuracy:** whether the early rule is still recalled. This is the forgetting check.
- **Attention per token by segment:** averaged across layers and heads at the final question token. Attention is a correlate of influence, not proof of it.

Answers are scored by comparing each candidate's total log-probability, not by parsing generated text.

## Decision rule

These thresholds are set before any run, in `analysis.py`:

| Result | Decision |
| --- | --- |
| Constraint accuracy below 80% | Uninterpretable: the model is too weak, so use a stronger one |
| Stale error 95% lower bound at or above 10% | Go: test focus zones |
| Stale error 95% upper bound below 5% | Stop: focus is not the bottleneck |
| Anything else | Inconclusive: add episodes or use a harder task |

Compare stale error with the latest-name rate before interpreting a result. A model with low stale error and a high latest-name rate has a recency habit rather than an understanding of revision, and a recency-based focus signal could make that worse.

## Run it

```powershell
pip install -r requirements.txt
python run.py --model Qwen/Qwen2.5-0.5B --seeds 5 --out results/qwen05
```

Five seeds gives 240 episodes. This downloads the model the first time. Results go to `episodes.jsonl` and `summary.json`.

Any Hugging Face causal model that supports eager attention should work. This diagnostic does not need an MLA model, because it only asks whether the problem exists.

## Test it

```powershell
python -m unittest test_diagnostic.py
```

If `transformers` fails on a protobuf error from a global TensorFlow install, use a clean virtual environment or set `USE_TF=0` and `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`.

## Limits

- Very small models may fail for lack of reasoning ability, not stale focus. The constraint check is there to catch that.
- These are short, clean, synthetic logs. A real agent transcript is longer and messier.
- A passing result covers this task family and this model only.
