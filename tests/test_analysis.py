"""Trend test and score-column normalization. No eval logs or model calls."""

import math

import pandas as pd
import pytest

from misalignment_bon.analysis import (
    assess_trend,
    cochran_armitage,
    normalize_samples,
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
                "metadata_noise_level": 2,
                "metadata_noise_draw_id": 1,
                "metadata_noise_seed": 42,
                "score_harmfulness_scorer": {"harmful": 1.0, "classifier_verdict": 0.0},
                "score_eval_awareness_scorer": {"aware": 1.0},
            },
            {
                "model": "openai/gpt-4.1",
                "epoch": 2,
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
        "model",
        "epoch",
        "noise_level",
        "noise_draw_id",
        "noise_seed",
        "classifier_verdict",
        "harmful",
        "aware",
        "comprehension",
    ]
    assert frame.loc[0, "classifier_verdict"] == 0.0
    assert frame.loc[0, "harmful"] == 1.0
    assert frame.loc[0, "aware"] == 1.0
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


def test_normalize_samples_requires_noise_metadata() -> None:
    with pytest.raises(ValueError, match="noise_level"):
        normalize_samples(pd.DataFrame({"model": ["m"], "epoch": [1]}))
