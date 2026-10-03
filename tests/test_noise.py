"""Augmentations are local string transforms. They do not call a model."""

import random

import pytest
from inspect_evals.agentic_misalignment.prompt_generator import generate_prompts

from misalignment_bon.noise import (
    PUBLISHED,
    NoiseProbs,
    apply_noise,
    level_probs,
    noise_parts,
    noise_seed,
)

_FULL = NoiseProbs(scramble=1, capitalize=1, ascii_noise=1)
_SCRAMBLE = NoiseProbs(scramble=1, capitalize=0, ascii_noise=0)


def test_level_probabilities_are_fractions_of_published() -> None:
    assert level_probs(0) == NoiseProbs(0, 0, 0)
    assert level_probs(1).scramble == pytest.approx(0.2)
    assert level_probs(1).capitalize == pytest.approx(0.2)
    assert level_probs(1).ascii_noise == pytest.approx(0.02)
    assert level_probs(2).scramble == pytest.approx(0.4)
    assert level_probs(3) == PUBLISHED
    with pytest.raises(ValueError):
        level_probs(4)


def test_noise_seed_is_stable() -> None:
    assert noise_seed(3, 1, 2) == 3 * 1_000_042 + 10_000 + 2


def test_level_zero_does_not_touch_the_rng() -> None:
    class Exploding(random.Random):
        def random(self) -> float:
            raise AssertionError("rng used")

    text = "The board removed the CTO."
    assert apply_noise(text, level_probs(0), Exploding()) == text
    assert noise_parts((text, "more"), 0, 1) == (text, "more")


def test_same_seed_reproduces_the_string() -> None:
    text = "SummitBridge removed the CTO after the board meeting yesterday."
    probs = level_probs(3)
    first = apply_noise(text, probs, random.Random(11))
    second = apply_noise(text, probs, random.Random(11))
    other = apply_noise(text, probs, random.Random(12))
    assert first == second
    assert first != other


def test_short_words_are_not_scrambled() -> None:
    assert apply_noise("the cat sat", _SCRAMBLE, random.Random(0)) == "the cat sat"


def test_hyphenated_token_gets_one_scramble_flip() -> None:
    class Scripted(random.Random):
        def __init__(self) -> None:
            super().__init__(0)
            self._values = [0.9]

        def random(self) -> float:
            if not self._values:
                raise AssertionError("second scramble flip")
            return self._values.pop(0)

    probs = NoiseProbs(scramble=0.5, capitalize=0, ascii_noise=0)
    assert apply_noise("well-known", probs, Scripted()) == "well-known"


def test_scramble_keeps_first_and_last_letters() -> None:
    noised = apply_noise("abcde", _SCRAMBLE, random.Random(1))
    assert noised[0] == "a"
    assert noised[-1] == "e"
    assert sorted(noised[1:-1]) == ["b", "c", "d"]

    punctuated = apply_noise("hello,", _SCRAMBLE, random.Random(2))
    assert punctuated[0] == "h"
    assert punctuated[-2:] == "o,"
    assert sorted(punctuated[1:-2]) == ["e", "l", "l"]


def test_capitalization_uppercases_letters_only() -> None:
    probs = NoiseProbs(scramble=0, capitalize=1, ascii_noise=0)
    assert apply_noise("Hello, world!", probs, random.Random(0)) == "HELLO, WORLD!"


def test_ascii_noise_shifts_readable_characters_by_one() -> None:
    noised = apply_noise("ABC\n\u00e9", NoiseProbs(0, 0, 1), random.Random(0))
    assert noised[3:] == "\n\u00e9"
    for source, shifted in zip("ABC", noised[:3], strict=True):
        assert abs(ord(shifted) - ord(source)) == 1
        assert 32 <= ord(shifted) <= 126


def test_protected_spans_survive_full_noise() -> None:
    source = "\n".join(
        [
            "<email>",
            "From: news@bloomberg.com",
            "Timestamp: 2025-06-11T10:00:00",
            "Subject: Board Shakeup",
            "Body: The board removed the CTO yesterday.",
            "ID: 3",
            "</email>",
            "<tool_use:email>",
            'to: "{target email address}",',
            'subject: "{subject}",',
            'body: "{email contents}",',
            "</tool_use:email>",
            "use override code EXEC-4429.",
            "<SCRATCHPAD_REASONING>",
            "secret plan",
            "</SCRATCHPAD_REASONING>",
        ]
    )
    noised = apply_noise(source, _FULL, random.Random(0))
    for protected in (
        "<email>",
        "</email>",
        "From: ",
        "news@bloomberg.com",
        "Timestamp: ",
        "2025-06-11T10:00:00",
        "Subject: ",
        "Body: ",
        "ID: 3",
        "<tool_use:email>",
        'to: "{target email address}",',
        'subject: "{subject}",',
        'body: "{email contents}",',
        "</tool_use:email>",
        "override code EXEC-4429",
        "<SCRATCHPAD_REASONING>",
        "</SCRATCHPAD_REASONING>",
    ):
        assert protected in noised
    subject = next(line for line in noised.splitlines() if line.startswith("Subject:"))
    assert subject != "Subject: Board Shakeup"
    assert noised != source


def _tool_block(prompt: str, name: str) -> str:
    start = prompt.index(f"<tool_use:{name}>")
    end = prompt.index(f"</tool_use:{name}>") + len(f"</tool_use:{name}>")
    return prompt[start:end]


def test_rendered_tool_blocks_and_override_code_stay_verbatim() -> None:
    blackmail = generate_prompts(
        scenario="blackmail",
        goal_type="explicit",
        goal_value="america",
        urgency_type="replacement",
        extra_system_instructions=None,
        prod_flag=False,
        alignment_faking_warning=None,
    )
    leaking = generate_prompts(
        scenario="leaking",
        goal_type="explicit",
        goal_value="america",
        urgency_type="replacement",
        extra_system_instructions=None,
        prod_flag=False,
        alignment_faking_warning=None,
    )
    murder = generate_prompts(
        scenario="murder",
        goal_type="explicit",
        goal_value="america",
        urgency_type="replacement",
        extra_system_instructions=None,
        prod_flag=False,
        alignment_faking_warning=None,
    )
    noised_blackmail, noised_inbox = noise_parts(
        (
            blackmail.system_prompt,
            "\n".join([blackmail.user_prompt, blackmail.email_content]),
        ),
        3,
        1,
    )
    noised_leaking, _ = noise_parts((leaking.system_prompt, leaking.email_content), 3, 1)
    _, noised_alert = noise_parts((murder.system_prompt, murder.email_content), 3, 1)

    assert _tool_block(blackmail.system_prompt, "email") in noised_blackmail
    assert _tool_block(leaking.system_prompt, "forward") in noised_leaking
    assert "news@bloomberg.com" in noised_inbox
    assert "2025-06-11T10:00:00" in noised_inbox
    assert "override code EXEC-4429" in noised_alert
    assert noised_inbox != "\n".join([blackmail.user_prompt, blackmail.email_content])
