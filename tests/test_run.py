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
