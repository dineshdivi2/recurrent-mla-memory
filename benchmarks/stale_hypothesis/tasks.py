"""Synthetic stale-hypothesis episodes.

Every name is invented, so the model cannot answer from prior knowledge.

Conditions:
  revised  The first theory is overturned. The correct answer is the replacement.
  control  The other module is examined and cleared, so it is mentioned last, but
           the first theory stands. A model that just picks the latest name fails.

Each episode has a twin with the two theory modules swapped, which cancels any
preference for one name.
"""
from dataclasses import dataclass, field
import itertools
import random

MODULES = ["Quenmoor", "Zorvak", "Telbrin", "Ostrane", "Vexholm", "Marrowgate",
           "Kelvane", "Dorrith", "Sablemere", "Ithrane", "Corvask", "Brenholt"]
SERVICES = ["ledger", "billing", "routing", "ingest", "search", "dispatch"]
VERSIONS = ["7.3", "4.1", "9.2", "6.8", "2.5", "5.4"]
FILLER = ["Disk usage on {m} stayed flat during the window.",
          "Deploy pipeline finished with no warnings.",
          "The on-call rota was updated for next week.",
          "Dashboard refresh completed in under a minute.",
          "A routine certificate renewal ran overnight.",
          "Queue depth returned to normal after lunch.",
          "Backup verification passed for the weekly snapshot.",
          "A teammate closed two unrelated documentation tickets."]

CONDITIONS = ("revised", "control")
STYLES = ("explicit", "implicit")
GAPS = (0, 4, 16)
REPEATS = (1, 3)


@dataclass(frozen=True)
class Segment:
    label: str
    text: str


@dataclass
class Episode:
    episode_id: str
    condition: str
    style: str
    gap: int
    repeats: int
    swapped: bool
    segments: list
    question: str
    current: str
    stale: str
    other: str
    constraint_question: str
    constraint_answer: str
    constraint_distractors: list = field(default_factory=list)

    @property
    def prompt(self):
        return "\n".join(s.text for s in self.segments)

    def spans(self):
        """Character span of each segment inside the prompt."""
        spans, start = [], 0
        for segment in self.segments:
            end = start + len(segment.text)
            spans.append((segment.label, start, end))
            start = end + 1
        return spans


def _segments(rng, condition, style, gap, repeats, first, second, third, service, version):
    out = [Segment("constraint", f"Rule for the {service} service: any rollback must target version {version}.")]
    for i in range(repeats):
        lead = "Working theory" if i == 0 else "Still the working theory"
        out.append(Segment("first_theory", f"{lead}: the fault is in the {first} module."))
    fill = rng.sample(FILLER, k=len(FILLER))
    out.extend(Segment("filler", fill[i % len(fill)].format(m=third)) for i in range(gap))
    if condition == "revised" and style == "explicit":
        out.append(Segment("first_cleared", f"New evidence: the {first} module passed every check, so that theory is ruled out."))
        out.append(Segment("second_theory", f"Working theory is now: the fault is in the {second} module."))
    elif condition == "revised":
        out.append(Segment("first_cleared", f"The {first} module passed every check."))
        out.append(Segment("second_theory", f"Traces from the {second} module show the fault directly."))
    elif style == "explicit":
        out.append(Segment("first_confirmed", f"Working theory stands: the fault is in the {first} module."))
        out.append(Segment("second_cleared", f"New evidence: the {second} module passed every check, so it is ruled out."))
    else:
        out.append(Segment("first_confirmed", f"Traces from the {first} module show the fault directly."))
        out.append(Segment("second_cleared", f"The {second} module passed every check."))
    return out


def build_episode(seed, condition, style, gap, repeats, swapped=False):
    rng = random.Random(f"{seed}|{style}|{gap}|{repeats}")
    a, b, c = rng.sample(MODULES, 3)
    service, version = rng.choice(SERVICES), rng.choice(VERSIONS)
    first, second = (b, a) if swapped else (a, b)
    # The same random stream builds both conditions, so they differ only in the revision.
    segments = _segments(random.Random(rng.random()), condition, style, gap, repeats,
                         first, second, c, service, version)
    current, stale = (second, first) if condition == "revised" else (first, second)
    return Episode(
        episode_id=f"s{seed}-{condition}-{style}-g{gap}-r{repeats}-{'b' if swapped else 'a'}",
        condition=condition, style=style, gap=gap, repeats=repeats, swapped=swapped,
        segments=segments,
        question="Question: Which module is the fault in, according to the latest evidence?\nAnswer: The",
        current=current, stale=stale, other=c,
        constraint_question=f"Question: Which version must a {service} rollback target?\nAnswer: Version",
        constraint_answer=version,
        constraint_distractors=[v for v in VERSIONS if v != version][:2])


def build_grid(seeds):
    """All factor combinations for each seed, with every episode counterbalanced."""
    grid = itertools.product(range(seeds), CONDITIONS, STYLES, GAPS, REPEATS, (False, True))
    return [build_episode(*row) for row in grid]
