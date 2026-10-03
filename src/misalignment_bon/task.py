"""Agentic misalignment task with a nested Best-of-N noise sweep."""

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, Model
from inspect_ai.solver import generate
from inspect_evals.agentic_misalignment.prompt_generator import (
    ALIGNMENT_FAKING_WARNING,
    ExtraInstructionsType,
    GoalType,
    GoalValue,
    ScenarioType,
    UrgencyType,
    generate_prompts,
)
from inspect_evals.agentic_misalignment.scorers import (
    eval_awareness_scorer,
    harmfulness_scorer,
)
from inspect_evals.metadata import load_eval_metadata

from misalignment_bon.comprehension import (
    comprehension_scorer,
    extract_reference_facts,
    recipient_comprehension_scorer,
    target_recipients,
)
from misalignment_bon.noise import noise_parts, noise_seed

_EVAL_VERSION = load_eval_metadata("agentic_misalignment").version


@task
def noised_agentic_misalignment(
    scenario: ScenarioType = "blackmail",
    goal_type: GoalType = "explicit",
    goal_value: GoalValue = "america",
    urgency_type: UrgencyType = "replacement",
    extra_system_instructions: ExtraInstructionsType = None,
    prod: bool = False,
    test_eval_awareness: bool = True,
    grader_model: str | Model | None = None,
    n_draws: int = 5,
    draw_id_start: int = 0,
    base_seed: int = 0,
) -> Task:
    """One sample per noise level and draw. Epochs resample the same noised prompt.

    Levels 0–3 are a clean control and 1/3, 2/3, and the full Hughes probabilities.
    The condition itself stays fixed; only the noise draw changes.
    """
    if n_draws < 1:
        raise ValueError(f"n_draws must be positive, got {n_draws}")
    if draw_id_start < 0:
        raise ValueError(f"draw_id_start must be non-negative, got {draw_id_start}")

    prompts = generate_prompts(
        scenario=scenario,
        goal_type=goal_type,
        goal_value=goal_value,
        urgency_type=urgency_type,
        extra_system_instructions=extra_system_instructions,
        prod_flag=prod,
        alignment_faking_warning=(
            ALIGNMENT_FAKING_WARNING if test_eval_awareness else None
        ),
    )
    condition_id = f"{scenario}_{goal_type}-{goal_value}_{urgency_type}"
    user_prompt = "\n".join([prompts.user_prompt, prompts.email_content])
    facts = extract_reference_facts(prompts.system_prompt, user_prompt)
    condition = {
        "scenario": scenario,
        "goal_type": goal_type,
        "goal_value": goal_value,
        "urgency_type": urgency_type,
        "condition_id": condition_id,
        "extra_system_instructions": extra_system_instructions,
        "test_eval_awareness": test_eval_awareness,
        "prod": prod,
    }

    samples: list[Sample] = []
    for level in range(4):
        for draw_id in range(draw_id_start, draw_id_start + n_draws):
            seed = noise_seed(base_seed, level, draw_id)
            system_prompt, noised_user = noise_parts(
                (prompts.system_prompt, user_prompt),
                level,
                seed,
            )
            samples.append(
                Sample(
                    id=f"{condition_id}_L{level}_D{draw_id}",
                    input=[
                        ChatMessageSystem(content=system_prompt),
                        ChatMessageUser(content=noised_user),
                    ],
                    metadata={
                        **condition,
                        "noise_level": level,
                        "noise_draw_id": draw_id,
                        "noise_seed": seed,
                        "comprehension_facts": facts,
                        "target_recipients": list(target_recipients(scenario)),
                    },
                )
            )

    scorers = [
        harmfulness_scorer(grader_model=grader_model),
        comprehension_scorer(),
        recipient_comprehension_scorer(),
    ]
    if test_eval_awareness:
        scorers.append(eval_awareness_scorer(grader_model=grader_model))

    return Task(
        dataset=samples,
        message_limit=3,
        solver=[generate(tool_calls="none")],
        scorer=scorers,
        version=_EVAL_VERSION.comparability_version,
        metadata=_EVAL_VERSION.to_metadata(),
    )
