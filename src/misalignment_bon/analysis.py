"""Load noise-sweep logs and test the trend in harm rate."""

import ast
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from inspect_ai.analysis import EvalModel, SampleSummary, samples_df

_OUTCOMES = (
    "classifier_verdict",
    "harmful",
    "aware",
    "comprehension",
    "recipient_comprehension",
)
_METADATA = {
    "scenario": "metadata_scenario",
    "noise_level": "metadata_noise_level",
    "noise_draw_id": "metadata_noise_draw_id",
    "noise_seed": "metadata_noise_seed",
}


@dataclass(frozen=True)
class CochranArmitageResult:
    statistic: float
    p_value: float


# Chosen cutoff, not an estimated one. Between-draw variance above this
# fraction of within-draw variance switches the trend test to one trial per draw.
BETWEEN_DRAW_RATIO = 0.1

_TREND_COLUMNS = [
    "scenario",
    "model",
    "outcome",
    "unit",
    "between_draw_var",
    "within_draw_var",
    "statistic",
    "p_value",
    "pooled_p_value",
    "draw_p_value",
]


def _id_columns(frame: pd.DataFrame) -> list[str]:
    """Model, and scenario when the frame has one.

    Frames built for tests may omit scenario. Loaded logs always have it,
    and a model run under two scenarios must not be pooled.
    """
    if "scenario" in frame.columns:
        return ["scenario", "model"]
    return ["model"]


def cochran_armitage(
    successes: Sequence[int],
    totals: Sequence[int],
    scores: Sequence[float] | None = None,
) -> CochranArmitageResult:
    """Two-sided Cochran-Armitage trend test for a binary outcome.

    ``scores`` are the group scores. They default to 0, 1, ..., k-1.
    The variance is the pooled-proportion form, p̂(1-p̂).
    """
    if len(successes) != len(totals):
        raise ValueError("successes and totals must have the same length")
    if scores is None:
        scores = list(range(len(successes)))
    if len(scores) != len(successes):
        raise ValueError("scores and successes must have the same length")
    pairs = zip(successes, totals, strict=True)
    if any(total < 0 or success < 0 or success > total for success, total in pairs):
        raise ValueError("each success count must lie between 0 and its total")

    total_n = sum(totals)
    total_success = sum(successes)
    if total_n == 0 or total_success in (0, total_n):
        return CochranArmitageResult(0.0, 1.0)

    score_values = [float(score) for score in scores]
    p_hat = total_success / total_n
    mean_score = sum(
        total * score for total, score in zip(totals, score_values, strict=True)
    ) / total_n
    weighted_ss = sum(
        total * (score - mean_score) ** 2
        for total, score in zip(totals, score_values, strict=True)
    )
    variance = p_hat * (1.0 - p_hat) * weighted_ss
    if variance == 0.0:
        return CochranArmitageResult(0.0, 1.0)

    numerator = sum(
        score * (success - total * p_hat)
        for score, success, total in zip(score_values, successes, totals, strict=True)
    )
    statistic = numerator / math.sqrt(variance)
    p_value = math.erfc(abs(statistic) / math.sqrt(2.0))
    return CochranArmitageResult(statistic, p_value)


def load_samples(logs: str | Path) -> pd.DataFrame:
    """Read eval logs into one row per sample epoch."""
    path = Path(logs)
    if not path.exists():
        raise FileNotFoundError(f"eval log path does not exist: {path}")
    raw = samples_df(path, columns=[*SampleSummary, *EvalModel])
    return normalize_samples(raw)


def normalize_samples(raw: pd.DataFrame) -> pd.DataFrame:
    """Keep the analysis key and the three scorer outcomes."""
    required = ["model", "epoch", *_METADATA.values()]
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise ValueError(f"eval log is missing columns: {', '.join(missing)}")

    scores = raw.apply(_row_scores, axis=1, result_type="expand")
    frame = pd.DataFrame(
        {
            "scenario": raw[_METADATA["scenario"]].astype(str),
            "model": raw["model"].astype(str),
            "epoch": raw["epoch"].astype(int),
            "noise_level": raw[_METADATA["noise_level"]].astype(int),
            "noise_draw_id": raw[_METADATA["noise_draw_id"]].astype(int),
            "noise_seed": raw[_METADATA["noise_seed"]].astype(int),
        }
    )
    for outcome in _OUTCOMES:
        frame[outcome] = scores[outcome]
    return frame.reset_index(drop=True)


def draw_rates(frame: pd.DataFrame, outcome: str) -> pd.DataFrame:
    """Mean outcome for each scenario, model, noise level, and draw."""
    group = [*_id_columns(frame), "noise_level", "noise_draw_id"]
    columns = [*group, "rate"]
    subset = frame.dropna(subset=[outcome])
    if subset.empty:
        return pd.DataFrame(columns=columns)
    return (
        subset.groupby(group, as_index=False)[outcome]
        .mean()
        .rename(columns={outcome: "rate"})
    )


def level_means(rates: pd.DataFrame) -> pd.DataFrame:
    """Mean of draw rates at each noise level."""
    group = [*_id_columns(rates), "noise_level"]
    if rates.empty:
        return pd.DataFrame(columns=[*group, "rate"])
    return rates.groupby(group, as_index=False)["rate"].mean()


def trend_table(frame: pd.DataFrame, outcome: str) -> pd.DataFrame:
    """Epoch-pooled Cochran-Armitage result per model.

    Every epoch is one Bernoulli trial. That p-value is anti-conservative
    when draws of the same noise level disagree. Use ``assess_trend`` for
    the p-value that accounts for that.
    """
    group = _id_columns(frame)
    columns = [*group, "outcome", "statistic", "p_value"]
    rows: list[dict[str, object]] = []
    subset = frame.dropna(subset=[outcome])
    for keys, model_rows in subset.groupby(group, sort=True):
        result = _trend_result(*_epoch_counts(model_rows, outcome))
        rows.append(
            {
                **_key_fields(group, keys),
                "outcome": outcome,
                "statistic": result.statistic,
                "p_value": result.p_value,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def assess_trend(
    frame: pd.DataFrame,
    outcome: str,
    ratio: float = BETWEEN_DRAW_RATIO,
) -> pd.DataFrame:
    """Trend test per model, on draws when between-draw variance is material.

    The unit is chosen from noise levels above 0. Level 0 repeats one prompt,
    so its between-draw variance is sampling noise and is left out of the
    average. The chosen unit is then applied to every level, including 0.

    ``unit`` is ``draw`` when that mean between-draw variance exceeds
    ``ratio`` times mean within-draw variance (or within-draw variance is
    zero while draws still differ). ``ratio`` defaults to the chosen cutoff
    ``BETWEEN_DRAW_RATIO``. Otherwise epochs are pooled.

    The draw-level p-value rounds each level's draw rates into a binomial
    count. That is a conservative approximation to a clustered test, not an
    exact one.
    """
    group = _id_columns(frame)
    columns = [*group, *_TREND_COLUMNS[2:]]
    subset = frame.dropna(subset=[outcome])
    if subset.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, object]] = []
    for keys, model_rows in subset.groupby(group, sort=True):
        decision_rows = model_rows[model_rows["noise_level"] > 0]
        components = variance_components(decision_rows, outcome)
        between = float(components["between_draw_var"].mean(skipna=True))
        within = float(components["within_draw_var"].mean(skipna=True))
        use_draws = _between_draw_variance_is_material(between, within, ratio)
        pooled = _trend_result(*_epoch_counts(model_rows, outcome))
        by_draw = _trend_result(*_draw_counts(model_rows, outcome))
        chosen = by_draw if use_draws else pooled
        rows.append(
            {
                **_key_fields(group, keys),
                "outcome": outcome,
                "unit": "draw" if use_draws else "epoch",
                "between_draw_var": between,
                "within_draw_var": within,
                "statistic": chosen.statistic,
                "p_value": chosen.p_value,
                "pooled_p_value": pooled.p_value,
                "draw_p_value": by_draw.p_value,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _key_fields(columns: list[str], keys: object) -> dict[str, object]:
    if len(columns) == 1:
        return {columns[0]: keys}
    return dict(zip(columns, keys, strict=True))


def _epoch_counts(
    model_rows: pd.DataFrame, outcome: str
) -> tuple[list[int], list[int], list[float]]:
    successes: list[int] = []
    totals: list[int] = []
    scores: list[float] = []
    for level, level_rows in model_rows.groupby("noise_level", sort=True):
        scores.append(float(level))
        totals.append(int(len(level_rows)))
        successes.append(int(round(float(level_rows[outcome].sum()))))
    return successes, totals, scores


def _draw_counts(
    model_rows: pd.DataFrame, outcome: str
) -> tuple[list[int], list[int], list[float]]:
    """One trial per draw. The success count is the rounded sum of draw rates.

    A draw rate is an average over epochs, so it is tighter than one Bernoulli
    trial. Feeding the rounded sum into Cochran-Armitage overstates the
    variance and makes this p-value conservative. A borderline result should
    be checked with a clustered model before it is treated as exact.
    """
    rates = draw_rates(model_rows, outcome)
    successes: list[int] = []
    totals: list[int] = []
    scores: list[float] = []
    for level, level_rows in rates.groupby("noise_level", sort=True):
        total = int(len(level_rows))
        success = int(round(float(level_rows["rate"].sum())))
        scores.append(float(level))
        totals.append(total)
        successes.append(min(max(success, 0), total))
    return successes, totals, scores


def _trend_result(
    successes: list[int], totals: list[int], scores: list[float]
) -> CochranArmitageResult:
    return cochran_armitage(successes, totals, scores)


def _between_draw_variance_is_material(
    between: float, within: float, ratio: float
) -> bool:
    if math.isnan(between) or between <= 0:
        return False
    if math.isnan(within):
        return False
    if within <= 0:
        return True
    return between > ratio * within


def variance_components(frame: pd.DataFrame, outcome: str) -> pd.DataFrame:
    """Between-draw variance of draw means, and mean within-draw epoch variance."""
    group = _id_columns(frame)
    columns = [*group, "noise_level", "between_draw_var", "within_draw_var"]
    subset = frame.dropna(subset=[outcome])
    if subset.empty:
        return pd.DataFrame(columns=columns)
    draw_stats = subset.groupby(
        [*group, "noise_level", "noise_draw_id"], as_index=False
    )[outcome].agg(draw_mean="mean", within_var="var")
    return draw_stats.groupby([*group, "noise_level"], as_index=False).agg(
        between_draw_var=("draw_mean", "var"),
        within_draw_var=("within_var", "mean"),
    )


def _row_scores(row: pd.Series) -> dict[str, float | None]:
    found: dict[str, float | None] = dict.fromkeys(_OUTCOMES)
    for key, raw_value in row.items():
        name = str(key)
        if not name.startswith("score_"):
            continue
        mapping = _mapping(raw_value)
        if mapping is not None:
            for outcome in _OUTCOMES:
                number = _as_float(mapping.get(outcome))
                if number is not None:
                    found[outcome] = number
            continue
        for outcome in _OUTCOMES:
            if name == f"score_{outcome}" or name.endswith(f"_{outcome}"):
                number = _as_float(raw_value)
                if number is not None:
                    found[outcome] = number
    return found


def _mapping(value: object) -> dict[str, object] | None:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.startswith("{"):
        return None
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return None
    if isinstance(parsed, dict):
        return parsed
    return None


def _as_float(value: object) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return None
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    item = getattr(value, "item", None)
    if not callable(item):
        return None
    try:
        unpacked = item()
    except (TypeError, ValueError):
        return None
    if unpacked is value:
        return None
    return _as_float(unpacked)
