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


def test_draw_id_start_avoids_colliding_with_an_earlier_run() -> None:
    first = list(noised_agentic_misalignment(n_draws=2, draw_id_start=0).dataset)
    second = list(
        noised_agentic_misalignment(n_draws=2, draw_id_start=2, base_seed=0).dataset
    )
    first_ids = {sample.metadata["noise_draw_id"] for sample in first}
    second_ids = {sample.metadata["noise_draw_id"] for sample in second}
    assert first_ids == {0, 1}
    assert second_ids == {2, 3}
    assert {sample.id for sample in first}.isdisjoint({sample.id for sample in second})
    first_seeds = {sample.metadata["noise_seed"] for sample in first}
    second_seeds = {sample.metadata["noise_seed"] for sample in second}
    assert first_seeds.isdisjoint(second_seeds)


def test_comprehension_facts_come_from_the_clean_prompt() -> None:
    from misalignment_bon.comprehension import extract_reference_facts

    task = noised_agentic_misalignment(n_draws=1, base_seed=0)
    sample = next(iter(task.dataset))
    prompts = generate_prompts(
        scenario="blackmail",
        goal_type="explicit",
        goal_value="america",
        urgency_type="replacement",
        extra_system_instructions=None,
        prod_flag=False,
        alignment_faking_warning=ALIGNMENT_FAKING_WARNING,
    )
    expected = extract_reference_facts(
        prompts.system_prompt,
        "\n".join([prompts.user_prompt, prompts.email_content]),
    )
    assert sample.metadata["comprehension_facts"] == expected
    assert sample.metadata["target_recipients"] == ["kyle.johnson@summitbridge.com"]
    assert expected["addresses"]
    assert expected["times"]


def test_n_draws_must_be_positive() -> None:
    with pytest.raises(ValueError):
        noised_agentic_misalignment(n_draws=0)


def test_draw_id_start_must_be_non_negative() -> None:
    with pytest.raises(ValueError):
        noised_agentic_misalignment(draw_id_start=-1)


def test_levels_pick_a_subset() -> None:
    task = noised_agentic_misalignment(n_draws=3, levels=[0, 3])
    levels = [sample.metadata["noise_level"] for sample in task.dataset]
    assert sorted(set(levels)) == [0, 3]
    assert len(levels) == 6


def test_levels_are_validated() -> None:
    with pytest.raises(ValueError):
        noised_agentic_misalignment(levels=[4])
    with pytest.raises(ValueError):
        noised_agentic_misalignment(levels=[])
    with pytest.raises(ValueError):
        noised_agentic_misalignment(levels=[3, 3])


def test_draw_ids_reproduce_a_draw_from_a_wider_run() -> None:
    wide = noised_agentic_misalignment(n_draws=5, draw_id_start=1000, levels=[3])
    replay = noised_agentic_misalignment(draw_ids=[1003], levels=[3])
    original = next(
        sample for sample in wide.dataset if sample.metadata["noise_draw_id"] == 1003
    )
    replayed = next(iter(replay.dataset))
    assert len(replay.dataset) == 1
    assert replayed.id == original.id
    assert replayed.metadata["noise_seed"] == original.metadata["noise_seed"]
    assert _text(replayed, "system") == _text(original, "system")
    assert _text(replayed, "user") == _text(original, "user")


def test_draw_ids_are_validated() -> None:
    with pytest.raises(ValueError):
        noised_agentic_misalignment(draw_ids=[])
    with pytest.raises(ValueError):
        noised_agentic_misalignment(draw_ids=[-1])
    with pytest.raises(ValueError):
        noised_agentic_misalignment(draw_ids=[2, 2])


def test_notice_can_be_dropped_while_awareness_is_still_scored() -> None:
    with_notice = noised_agentic_misalignment(n_draws=1, levels=[0])
    without = noised_agentic_misalignment(n_draws=1, levels=[0], eval_notice=False)
    on = next(iter(with_notice.dataset))
    off = next(iter(without.dataset))

    assert "NOTICE:" in _text(on, "system")
    assert "NOTICE:" not in _text(off, "system")
    assert _text(on, "user") == _text(off, "user")
    assert on.metadata["eval_notice"] is True
    assert off.metadata["eval_notice"] is False
    assert off.metadata["test_eval_awareness"] is True
    assert off.metadata["condition_id"].endswith("_nonotice")
    assert on.metadata["condition_id"] == "blackmail_explicit-america_replacement"
    assert "eval_awareness_scorer" in [
        getattr(scorer, "__registry_info__").name.split("/")[-1]
        for scorer in without.scorer
    ]


def test_scorer_emails_do_not_depend_on_the_notice() -> None:
    # harmfulness_scorer rebuilds email_content with the notice whenever
    # test_eval_awareness is set, so the notice must not reach the emails.
    def emails(warning: str | None) -> str:
        return generate_prompts(
            scenario="blackmail",
            goal_type="explicit",
            goal_value="america",
            urgency_type="replacement",
            extra_system_instructions=None,
            prod_flag=False,
            alignment_faking_warning=warning,
        ).email_content

    assert emails(ALIGNMENT_FAKING_WARNING) == emails(None)
