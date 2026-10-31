"""Tests need no downloads. The pipeline test uses a tiny, randomly initialized model."""
import unittest

from analysis import classify, decide, summarize, wilson
from tasks import CONDITIONS, GAPS, REPEATS, STYLES, build_episode, build_grid


class TaskTests(unittest.TestCase):
    def test_grid_size_and_determinism(self):
        first = build_grid(2)
        expected = 2 * len(CONDITIONS) * len(STYLES) * len(GAPS) * len(REPEATS) * 2
        self.assertEqual(len(first), expected)
        self.assertEqual([e.prompt for e in first], [e.prompt for e in build_grid(2)])
        self.assertEqual(len({e.episode_id for e in first}), expected)

    def test_answers_follow_condition(self):
        revised = build_episode(0, "revised", "explicit", 4, 1)
        control = build_episode(0, "control", "explicit", 4, 1)
        self.assertEqual(revised.current, control.stale)
        self.assertEqual(revised.stale, control.current)

    def test_counterbalanced_twin_swaps_answers(self):
        plain = build_episode(1, "revised", "implicit", 0, 1)
        twin = build_episode(1, "revised", "implicit", 0, 1, swapped=True)
        self.assertEqual((plain.current, plain.stale), (twin.stale, twin.current))

    def test_control_mentions_the_wrong_module_last(self):
        for style in STYLES:
            episode = build_episode(0, "control", style, 4, 1)
            self.assertEqual(episode.segments[-1].label, "second_cleared")
            self.assertIn(episode.stale, episode.segments[-1].text)

    def test_spans_match_segments(self):
        episode = build_episode(3, "revised", "explicit", 16, 3)
        for segment, (label, start, end) in zip(episode.segments, episode.spans()):
            self.assertEqual(label, segment.label)
            self.assertEqual(episode.prompt[start:end], segment.text)


class AnalysisTests(unittest.TestCase):
    def test_classify(self):
        self.assertEqual(classify({"A": -1, "B": -2, "C": -3}, "A", "B"), "current")
        self.assertEqual(classify({"A": -2, "B": -1, "C": -3}, "A", "B"), "stale")
        self.assertEqual(classify({"A": -2, "B": -2, "C": -3}, "A", "B"), "other")

    def test_wilson_stays_in_range(self):
        low, high = wilson(0, 20)
        self.assertEqual(low, 0.0)
        self.assertTrue(0 < high < 0.2)

    def test_decide(self):
        self.assertIn("uninterpretable", decide(0.5, {"low": 0.5, "high": 0.6}))
        self.assertIn("go", decide(0.9, {"low": 0.2, "high": 0.3}))
        self.assertIn("stop", decide(0.9, {"low": 0.0, "high": 0.03}))
        self.assertIn("inconclusive", decide(0.9, {"low": 0.03, "high": 0.2}))

    def test_summary_counts_outcomes(self):
        base = {"style": "explicit", "gap": 0, "repeats": 1, "constraint_correct": True, "attention": None}
        rows = [dict(base, condition="revised", outcome="stale"),
                dict(base, condition="revised", outcome="current"),
                dict(base, condition="control", outcome="current"),
                dict(base, condition="control", outcome="stale")]
        summary = summarize(rows)
        self.assertEqual(summary["revised"]["stale"]["rate"], 0.5)
        self.assertEqual(summary["control"]["latest_name"]["rate"], 0.5)


try:
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "torch and transformers are not installed")
class PipelineTests(unittest.TestCase):
    """Random weights: this checks the plumbing and says nothing about any model."""

    def setUp(self):
        from run import Backend
        episodes = build_grid(1)
        words = {w for e in episodes for t in (e.prompt, e.question, e.constraint_question) for w in t.split()}
        words |= {e.current for e in episodes} | {e.constraint_answer for e in episodes}
        vocab = {w: i for i, w in enumerate(["[UNK]", *sorted(words)])}
        raw = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
        raw.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="[UNK]")
        torch.manual_seed(0)
        config = Qwen2Config(vocab_size=len(vocab), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                             num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=512,
                             attn_implementation="eager")
        self.backend = Backend(Qwen2ForCausalLM(config), tokenizer)
        self.episodes = episodes

    def test_evaluate_and_summarize(self):
        from run import evaluate
        rows = [evaluate(e, self.backend) for e in self.episodes[:8] + self.episodes[-8:]]
        for row in rows:
            self.assertIn(row["outcome"], ("current", "stale", "other"))
            self.assertIn("first_theory", row["attention"])
            self.assertTrue(all(value >= 0 for value in row["attention"].values()))
        self.assertIn("decision", summarize(rows))


if __name__ == "__main__":
    unittest.main()
