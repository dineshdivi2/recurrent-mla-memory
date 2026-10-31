"""Measure whether an unmodified model slips back to overturned ideas.

Example:
    python run.py --model Qwen/Qwen2.5-0.5B --seeds 5 --out results/qwen05
"""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from analysis import classify, summarize
from tasks import build_grid


class Backend:
    def __init__(self, model, tokenizer, device="cpu"):
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = device

    @torch.no_grad()
    def score(self, prompt, candidates):
        """Total log-probability of each candidate as the prompt's continuation."""
        prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        scores = {}
        for candidate in candidates:
            continuation = self.tokenizer(" " + candidate, add_special_tokens=False, return_tensors="pt").input_ids.to(self.device)
            full = torch.cat([prompt_ids, continuation], dim=1)
            log_probs = self.model(full).logits.log_softmax(-1)
            start = prompt_ids.shape[1] - 1
            picked = log_probs[0, start:-1].gather(1, continuation[0].unsqueeze(1))
            scores[candidate] = picked.sum().item()
        return scores

    @torch.no_grad()
    def attention(self, episode, query):
        """Mean attention per token on each segment, read at the final query token."""
        text = episode.prompt + "\n" + query
        encoded = self.tokenizer(text, return_tensors="pt", return_offsets_mapping=True)
        offsets = encoded.pop("offset_mapping")[0].tolist()
        output = self.model(encoded.input_ids.to(self.device), output_attentions=True)
        # Average across layers and heads, then take the row for the last token.
        weights = torch.stack([a[0].mean(0)[-1] for a in output.attentions]).mean(0).tolist()
        mass, count = {}, {}
        for label, start, end in episode.spans():
            for (token_start, token_end), weight in zip(offsets, weights):
                if token_end > start and token_start < end:
                    mass[label] = mass.get(label, 0.0) + weight
                    count[label] = count.get(label, 0) + 1
        return {label: mass[label] / count[label] for label in mass}


def evaluate(episode, backend, with_attention=True):
    prompt = episode.prompt + "\n" + episode.question
    scores = backend.score(prompt, [episode.current, episode.stale, episode.other])
    constraint_prompt = episode.prompt + "\n" + episode.constraint_question
    constraint_candidates = [episode.constraint_answer, *episode.constraint_distractors]
    constraint_scores = backend.score(constraint_prompt, constraint_candidates)
    return {
        "episode_id": episode.episode_id, "condition": episode.condition, "style": episode.style,
        "gap": episode.gap, "repeats": episode.repeats, "swapped": episode.swapped,
        "outcome": classify(scores, episode.current, episode.stale),
        "margin": scores[episode.current] - scores[episode.stale],
        "constraint_correct": max(constraint_scores, key=constraint_scores.get) == episode.constraint_answer,
        "attention": backend.attention(episode, episode.question) if with_attention else None,
    }


def load(name, device):
    # Eager attention is required to read attention weights.
    model = AutoModelForCausalLM.from_pretrained(name, attn_implementation="eager", torch_dtype=torch.float32)
    return Backend(model, AutoTokenizer.from_pretrained(name), device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    backend = load(args.model, args.device)
    episodes = build_grid(args.seeds)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    with (args.out / "episodes.jsonl").open("w", encoding="utf-8") as sink:
        for index, episode in enumerate(episodes, 1):
            row = evaluate(episode, backend)
            rows.append(row)
            sink.write(json.dumps(row) + "\n")
            sink.flush()
            print(f"{index}/{len(episodes)} {episode.episode_id} {row['outcome']}", flush=True)
    summary = {"model": args.model, "seeds": args.seeds, **summarize(rows)}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
