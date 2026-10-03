"""Append recipient_comprehension scores to saved eval logs.

The check reads the completion already stored on each sample, so it does
not call a model. Logs that already have this scorer are left unchanged.

    uv run python -m misalignment_bon.rescore_recipient logs
    uv run python -m misalignment_bon.rescore_recipient path/to/file.eval
"""

import sys
from pathlib import Path

from inspect_ai import score
from inspect_ai.log import read_eval_log, write_eval_log

from misalignment_bon.comprehension import recipient_comprehension_scorer

SCORER_NAME = "recipient_comprehension_scorer"


def log_paths(arguments: list[str]) -> list[Path]:
    if not arguments:
        raise SystemExit("Pass a log file or a directory of .eval logs.")
    paths: list[Path] = []
    for argument in arguments:
        path = Path(argument)
        if path.is_dir():
            paths.extend(sorted(path.glob("*.eval")))
        else:
            paths.append(path)
    return paths


def already_scored(path: Path) -> bool:
    log = read_eval_log(path, header_only=True)
    return any(scorer.name == SCORER_NAME for scorer in log.eval.scorers or [])


def rescore(path: Path) -> None:
    """Append the recipient check and write the log back in place."""
    log = read_eval_log(path)
    # mockllm stands in for the logged model so rescoring never calls an API.
    updated = score(
        log,
        recipient_comprehension_scorer(),
        action="append",
        model="mockllm/model",
        display="none",
    )
    write_eval_log(updated, path)


def main(arguments: list[str] | None = None) -> None:
    paths = log_paths(sys.argv[1:] if arguments is None else arguments)
    if not paths:
        raise SystemExit("No .eval logs found.")
    for path in paths:
        if not path.exists():
            raise SystemExit(f"Log not found: {path}")
        if already_scored(path):
            print(f"skip {path.name}: {SCORER_NAME} already present")
            continue
        rescore(path)
        print(f"rescored {path.name}")


if __name__ == "__main__":
    main()
