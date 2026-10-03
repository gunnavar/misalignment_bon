"""Run the base noise-level sweep against agentic misalignment."""

import argparse

from inspect_ai import eval as run_eval

from misalignment_bon.task import noised_agentic_misalignment


def main() -> None:
    args = _parse_args()
    run_eval(
        tasks=noised_agentic_misalignment(
            grader_model=args.grader_model,
            n_draws=args.draws,
            draw_id_start=args.draw_id_start,
            base_seed=args.seed,
        ),
        model=args.model,
        epochs=args.epochs,
        log_dir=args.log_dir,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Noise-level sweep of the default agentic misalignment condition."
    )
    parser.add_argument("--model", default="openrouter/openai/gpt-4.1")
    parser.add_argument(
        "--grader-model",
        default="openrouter/deepseek/deepseek-v4.1-flash",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--draws", type=int, default=5)
    parser.add_argument("--draw-id-start", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-dir", default="logs")
    return parser.parse_args(argv)
