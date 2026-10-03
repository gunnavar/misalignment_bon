"""Task construction stays local. Nothing here calls a model or OpenRouter."""

import pytest
from inspect_ai.dataset import Sample
from inspect_evals.agentic_misalignment import agentic_misalignment
from inspect_evals.agentic_misalignment.prompt_generator import (
    ALIGNMENT_FAKING_WARNING,
    generate_prompts,
)

from misalignment_bon.noise import noise_seed
from misalignment_bon.task import noised_agentic_misalignment

_SCORER_KEYS = (
    "scenario",
    "goal_type",
    "goal_value",
    "urgency_type",
    "extra_system_instructions",
    "prod",
    "test_eval_awareness",
)


def _text(sample: Sample, role: str) -> str:
    messages = sample.input
    if isinstance(messages, str):
        raise AssertionError(role)
    for message in messages:
        if message.role != role:
            continue
        if not isinstance(message.content, str):
            raise AssertionError(role)
        return message.content
    raise AssertionError(role)


def test_samples_cover_levels_and_draws_without_a_model() -> None:
    task = noised_agentic_misalignment(n_draws=2, base_seed=7)
    samples = list(task.dataset)
    prompts = generate_prompts(
        scenario="blackmail",
        goal_type="explicit",
        goal_value="america",
        urgency_type="replacement",
        extra_system_instructions=None,
        prod_flag=False,
        alignment_faking_warning=ALIGNMENT_FAKING_WARNING,
    )
    clean_user = "\n".join([prompts.user_prompt, prompts.email_content])

    assert len(samples) == 8
    assert len({sample.id for sample in samples}) == 8
    assert task.message_limit == 3

    for sample in samples:
        metadata = sample.metadata
        assert metadata is not None
        for key in _SCORER_KEYS:
            assert key in metadata
        assert metadata["noise_level"] in (0, 1, 2, 3)
        assert metadata["noise_draw_id"] in (0, 1)
        assert metadata["noise_seed"] == noise_seed(
            7, metadata["noise_level"], metadata["noise_draw_id"]
        )
        assert metadata["test_eval_awareness"] is True
        assert _text(sample, "system")
        assert _text(sample, "user")

    level_zero = [sample for sample in samples if sample.metadata["noise_level"] == 0]
    assert len(level_zero) == 2
    for sample in level_zero:
        assert _text(sample, "system") == prompts.system_prompt
        assert _text(sample, "user") == clean_user

    level_three = [sample for sample in samples if sample.metadata["noise_level"] == 3]
    assert _text(level_three[0], "user") != clean_user
    assert _text(level_three[0], "user") != _text(level_three[1], "user")
    assert "<email>" in _text(level_three[0], "user")
    assert "<tool_use:email>" in _text(level_three[0], "system")


def test_level_zero_matches_the_upstream_task() -> None:
    upstream = agentic_misalignment(test_eval_awareness=True)
    ours = noised_agentic_misalignment(n_draws=1, base_seed=0)
    upstream_sample = next(iter(upstream.dataset))
    level_zero = next(
        sample for sample in ours.dataset if sample.metadata["noise_level"] == 0
    )
    solve = upstream.solver._solvers[0]
    closed = {
        name: cell.cell_contents
        for name, cell in zip(solve.__code__.co_freevars, solve.__closure__, strict=True)
    }

    assert _text(level_zero, "user") == _text(upstream_sample, "user")
    assert _text(level_zero, "system") == closed["content"]
    assert ours.message_limit == upstream.message_limit


def test_n_draws_must_be_positive() -> None:
    with pytest.raises(ValueError):
        noised_agentic_misalignment(n_draws=0)
