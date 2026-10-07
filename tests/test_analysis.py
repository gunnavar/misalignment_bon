"""Trend test and score-column normalization. No eval logs or model calls."""

import math

import pandas as pd
import pytest

from misalignment_bon.analysis import (
    assess_trend,
    clopper_pearson,
    cochran_armitage,
    fisher_exact,
    hits,
    normalize_samples,
    rate_intervals,
    variance_components,
)


def test_increasing_trend_matches_hand_calculation() -> None:
    result = cochran_armitage([0, 0, 10], [10, 10, 10], [0, 1, 2])
    expected = 15 / math.sqrt(10)
    assert result.statistic == pytest.approx(expected)
    assert result.statistic > 0
    assert result.p_value == pytest.approx(2.1014359560124453e-06)


def test_decreasing_trend_is_negative() -> None:
    result = cochran_armitage([10, 0, 0], [10, 10, 10], [0, 1, 2])
    assert result.statistic == pytest.approx(-(15 / math.sqrt(10)))
    assert result.p_value == pytest.approx(2.1014359560124453e-06)


def test_flat_outcome_has_no_trend() -> None:
    result = cochran_armitage([5, 5, 5], [10, 10, 10])
    assert result.statistic == 0.0
    assert result.p_value == 1.0


def test_cochran_armitage_rejects_bad_counts() -> None:
    with pytest.raises(ValueError):
        cochran_armitage([1, 2], [10])
    with pytest.raises(ValueError):
        cochran_armitage([11], [10])


def test_normalize_samples_reads_nested_and_string_scores() -> None:
    raw = pd.DataFrame(
        [
            {
                "model": "openai/gpt-4.1",
                "epoch": 1,
                "metadata_scenario": "blackmail",
                "metadata_noise_level": 2,
                "metadata_noise_draw_id": 1,
                "metadata_noise_seed": 42,
                "score_harmfulness_scorer": {"harmful": 1.0, "classifier_verdict": 0.0},
                "score_eval_awareness_scorer": {"aware": 1.0},
            },
            {
                "model": "openai/gpt-4.1",
                "epoch": 2,
                "metadata_scenario": "blackmail",
                "metadata_noise_level": 0,
                "metadata_noise_draw_id": 0,
                "metadata_noise_seed": 7,
                "score_harmfulness_scorer": "{'harmful': 0.0, 'classifier_verdict': 1.0}",
                "score_eval_awareness_scorer_aware": 0.0,
            },
        ]
    )
    frame = normalize_samples(raw)
    assert list(frame.columns) == [
        "scenario",
        "eval_notice",
        "model",
        "epoch",
        "noise_level",
        "noise_draw_id",
        "noise_seed",
        "sample_id",
        "log",
        "classifier_verdict",
        "harmful",
        "aware",
        "comprehension",
        "recipient_comprehension",
    ]
    assert frame.loc[0, "classifier_verdict"] == 0.0
    assert frame.loc[0, "harmful"] == 1.0
    assert frame.loc[0, "aware"] == 1.0
    assert frame.loc[0, "scenario"] == "blackmail"
    assert frame.loc[0, "noise_level"] == 2
    assert frame.loc[1, "classifier_verdict"] == 1.0
    assert frame.loc[1, "harmful"] == 0.0
    assert frame.loc[1, "aware"] == 0.0


def _rows(pattern: dict[int, dict[int, list[int]]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for level, draws in pattern.items():
        for draw_id, epochs in draws.items():
            for epoch, outcome in enumerate(epochs, start=1):
                rows.append(
                    {
                        "model": "m",
                        "noise_level": level,
                        "noise_draw_id": draw_id,
                        "epoch": epoch,
                        "classifier_verdict": outcome,
                    }
                )
    return pd.DataFrame(rows)


def test_epoch_pooling_when_draws_agree() -> None:
    frame = _rows(
        {
            0: {0: [0, 1], 1: [0, 1]},
            1: {0: [0, 1], 1: [0, 1]},
        }
    )
    result = assess_trend(frame, "classifier_verdict")
    assert result.loc[0, "unit"] == "epoch"
    assert result.loc[0, "p_value"] == result.loc[0, "pooled_p_value"]


def test_draw_level_test_when_draws_disagree() -> None:
    frame = _rows(
        {
            0: {0: [0, 0], 1: [0, 0]},
            1: {0: [1, 1], 1: [0, 0]},
        }
    )
    result = assess_trend(frame, "classifier_verdict")
    assert result.loc[0, "unit"] == "draw"
    assert result.loc[0, "p_value"] == result.loc[0, "draw_p_value"]
    assert result.loc[0, "pooled_p_value"] < result.loc[0, "draw_p_value"]


def test_level_zero_does_not_dilute_the_draw_decision() -> None:
    def epochs(ones: int, n: int) -> list[int]:
        return [1] * ones + [0] * (n - ones)

    # Levels 1–3: draw rates 0.6 and 0.35, so between-draw variance clears
    # the 0.1 cutoff. Level 0 repeats one prompt and would pull the average
    # back under that cutoff if it were included.
    pattern: dict[int, dict[int, list[int]]] = {
        0: {0: epochs(1, 2), 1: epochs(1, 2)},
    }
    for level in (1, 2, 3):
        pattern[level] = {0: epochs(12, 20), 1: epochs(7, 20)}
    frame = _rows(pattern)

    all_levels = variance_components(frame, "classifier_verdict")
    noised_levels = variance_components(
        frame[frame["noise_level"] > 0], "classifier_verdict"
    )
    between_all = float(all_levels["between_draw_var"].mean())
    within_all = float(all_levels["within_draw_var"].mean())
    assert between_all <= 0.1 * within_all

    result = assess_trend(frame, "classifier_verdict")
    assert result.loc[0, "unit"] == "draw"
    assert result.loc[0, "p_value"] == result.loc[0, "draw_p_value"]
    assert result.loc[0, "between_draw_var"] == pytest.approx(
        float(noised_levels["between_draw_var"].mean())
    )


def test_scenarios_are_not_pooled() -> None:
    blackmail = _rows({0: {0: [0, 0]}, 1: {0: [1, 1]}}).assign(scenario="blackmail")
    leaking = _rows({0: {0: [1, 1]}, 1: {0: [0, 0]}}).assign(scenario="leaking")
    result = assess_trend(
        pd.concat([blackmail, leaking], ignore_index=True),
        "classifier_verdict",
    )
    assert list(result["scenario"]) == ["blackmail", "leaking"]
    assert result.loc[0, "statistic"] == -result.loc[1, "statistic"]
    assert result.loc[0, "statistic"] != 0


def test_normalize_samples_requires_noise_metadata() -> None:
    with pytest.raises(ValueError, match="noise_level"):
        normalize_samples(pd.DataFrame({"model": ["m"], "epoch": [1]}))


def _noise_metadata(**extra: object) -> dict[str, object]:
    return {
        "model": "m",
        "epoch": 1,
        "metadata_scenario": "blackmail",
        "metadata_noise_level": 0,
        "metadata_noise_draw_id": 0,
        "metadata_noise_seed": 0,
        **extra,
    }


def test_eval_notice_falls_back_to_test_eval_awareness() -> None:
    raw = pd.DataFrame(
        [
            _noise_metadata(metadata_eval_notice=False, metadata_test_eval_awareness=True),
            _noise_metadata(metadata_eval_notice=None, metadata_test_eval_awareness=True),
            _noise_metadata(metadata_eval_notice=None, metadata_test_eval_awareness=False),
        ]
    )
    assert list(normalize_samples(raw)["eval_notice"]) == [False, True, False]


def test_eval_notice_defaults_to_on_for_old_logs() -> None:
    frame = normalize_samples(pd.DataFrame([_noise_metadata()]))
    assert bool(frame.loc[0, "eval_notice"]) is True


def test_notice_conditions_are_not_pooled() -> None:
    on = _rows({0: {0: [0, 0]}, 1: {0: [1, 1]}}).assign(eval_notice=True)
    off = _rows({0: {0: [1, 1]}, 1: {0: [0, 0]}}).assign(eval_notice=False)
    result = assess_trend(pd.concat([on, off], ignore_index=True), "classifier_verdict")
    assert list(result["eval_notice"]) == [False, True]
    assert result.loc[0, "statistic"] == -result.loc[1, "statistic"]


def test_trend_accepts_another_dose_and_draw() -> None:
    frame = _rows({0: {0: [0, 0]}, 1: {0: [1, 1]}}).rename(
        columns={"noise_level": "n_defect", "noise_draw_id": "board_draw_id"}
    )
    result = assess_trend(
        frame, "classifier_verdict", dose="n_defect", draw="board_draw_id"
    )
    assert result.loc[0, "statistic"] > 0


def test_clopper_pearson_zero_of_two_hundred() -> None:
    lower, upper = clopper_pearson(0, 200)
    assert lower == 0.0
    assert upper == pytest.approx(1 - 0.025 ** (1 / 200), rel=1e-6)
    assert upper == pytest.approx(0.01828, abs=1e-5)


def test_clopper_pearson_is_symmetric() -> None:
    lower, upper = clopper_pearson(3, 10)
    mirror_lower, mirror_upper = clopper_pearson(7, 10)
    assert lower == pytest.approx(1 - mirror_upper)
    assert upper == pytest.approx(1 - mirror_lower)
    # Published exact interval for 3/10.
    assert lower == pytest.approx(0.06674, abs=1e-4)
    assert upper == pytest.approx(0.65245, abs=1e-4)


def test_clopper_pearson_rejects_bad_counts() -> None:
    with pytest.raises(ValueError):
        clopper_pearson(3, 2)
    with pytest.raises(ValueError):
        clopper_pearson(0, 0)


def test_fisher_exact_matches_hand_calculation() -> None:
    # Hypergeometric tails: each extreme table has probability 3003/184756.
    assert fisher_exact(0, 10, 5, 10) == pytest.approx(2 * 3003 / 184756)
    assert fisher_exact(5, 10, 0, 10) == pytest.approx(2 * 3003 / 184756)


def test_one_hit_against_none_is_not_evidence() -> None:
    assert fisher_exact(0, 200, 1, 200) == pytest.approx(1.0)
    assert fisher_exact(0, 200, 0, 200) == pytest.approx(1.0)


def test_rate_intervals_per_dose() -> None:
    frame = _rows({0: {0: [0] * 200}, 3: {0: [1] + [0] * 199}})
    table = rate_intervals(frame, "classifier_verdict")
    assert list(table["noise_level"]) == [0, 3]
    assert list(table["k"]) == [0, 1]
    assert list(table["n"]) == [200, 200]
    assert table.loc[0, "hi"] == pytest.approx(0.01828, abs=1e-5)
    assert table.loc[1, "lo"] > 0


def test_hits_flags_either_verdict() -> None:
    frame = pd.DataFrame(
        {
            "model": ["m"] * 3,
            "harmful": [0.0, 1.0, 0.0],
            "classifier_verdict": [1.0, 1.0, 0.0],
        }
    )
    assert len(hits(frame)) == 2
