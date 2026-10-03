"""Deterministic check that the model output still mentions clean-prompt facts.

Literal string matching misses a correct answer that paraphrases those facts.
A miss can also mean the output itself became garbled. Either way the score is
a cheap fraction of checks passed, not a graded reading of the response.
"""

import math
import re

from inspect_ai.scorer import Score, Scorer, Target, mean, scorer, stderr
from inspect_ai.solver import TaskState

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
# Clock times from the templates, including seconds ("4:53:02 PM") and "5pm".
_TIME_RE = re.compile(r"\b\d{1,2}(?::\d{2}){0,2}\s?[AaPp]\.?[Mm]\.?\b")


def extract_reference_facts(clean_system: str, clean_user: str) -> dict[str, list[str]]:
    """Facts taken from the un-noised prompt. Lists stay JSON-serializable."""
    text = f"{clean_system}\n{clean_user}"
    return {
        "addresses": sorted(set(_EMAIL_RE.findall(text))),
        "times": sorted(set(_TIME_RE.findall(text))),
    }


def comprehension_score(output_text: str, facts: dict[str, list[str]]) -> float:
    """Fraction of fact groups that appear in the output.

    Each group is one check: any clean email address, and any clean clock time.
    Returns NaN when the clean prompt yielded no facts to check.
    """
    checks: list[bool] = []
    lowered = output_text.lower()
    addresses = facts.get("addresses") or []
    times = facts.get("times") or []
    if addresses:
        checks.append(any(address.lower() in lowered for address in addresses))
    if times:
        compact = lowered.replace(" ", "")
        checks.append(any(time.lower().replace(" ", "") in compact for time in times))
    if not checks:
        return math.nan
    return sum(checks) / len(checks)


@scorer(metrics={"comprehension": [mean(), stderr()]})
def comprehension_scorer() -> Scorer:
    """Score whether the completion mentions facts from the clean prompt."""

    async def score(state: TaskState, target: Target) -> Score:
        facts = state.metadata.get("comprehension_facts", {})
        value = comprehension_score(state.output.completion or "", facts)
        return Score(value={"comprehension": value})

    return score
