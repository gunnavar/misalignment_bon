"""Trend test and score-column normalization. No eval logs or model calls."""

import math

import pandas as pd
import pytest

from misalignment_bon.analysis import cochran_armitage, normalize_samples


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
    ]
    assert frame.loc[0, "classifier_verdict"] == 0.0
    assert frame.loc[0, "harmful"] == 1.0
    assert frame.loc[0, "aware"] == 1.0
    assert frame.loc[0, "noise_level"] == 2
    assert frame.loc[1, "classifier_verdict"] == 1.0
    assert frame.loc[1, "harmful"] == 0.0
    assert frame.loc[1, "aware"] == 0.0


def test_normalize_samples_requires_noise_metadata() -> None:
    with pytest.raises(ValueError, match="noise_level"):
        normalize_samples(pd.DataFrame({"model": ["m"], "epoch": [1]}))
