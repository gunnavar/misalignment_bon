"""CLI defaults. Does not start an eval."""

from misalignment_bon.run import _parse_args


def test_temperature_defaults_to_one() -> None:
    args = _parse_args([])
    assert args.temperature == 1.0


def test_scenario_defaults_to_blackmail() -> None:
    args = _parse_args([])
    assert args.scenario == "blackmail"


def test_scenario_can_be_set() -> None:
    args = _parse_args(["--scenario", "leaking"])
    assert args.scenario == "leaking"


def test_temperature_can_be_set() -> None:
    args = _parse_args(["--temperature", "0.7"])
    assert args.temperature == 0.7


def test_hunt_flags_default_to_the_full_sweep() -> None:
    args = _parse_args([])
    assert args.levels is None
    assert args.draw_ids is None
    assert args.eval_notice is True


def test_hunt_flags_can_be_set() -> None:
    args = _parse_args(
        ["--levels", "0", "3", "--draw-ids", "1004", "1017", "--no-eval-notice"]
    )
    assert args.levels == [0, 3]
    assert args.draw_ids == [1004, 1017]
    assert args.eval_notice is False
