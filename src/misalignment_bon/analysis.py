"""Load noise-sweep logs, test the trend in harm rate, and bound rare rates."""

import ast
import math
from collections.abc import Callable, Sequence
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


# Condition columns that must never be pooled, in display order. Frames
# built for tests may omit any of them except model. Loaded logs always
# have scenario and eval_notice; peer-board logs add the board factors.
_FACTOR_COLUMNS = ("scenario", "eval_notice", "board", "peer_identity", "n_alert")


def _id_columns(frame: pd.DataFrame) -> list[str]:
    """Condition factors present in the frame, then model."""
    return [column for column in _FACTOR_COLUMNS if column in frame.columns] + [
        "model"
    ]


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
    """Read noise-sweep logs into one row per sample epoch.

    A directory may also hold logs from other tasks, such as the upstream
    eval or the peer board. Their samples carry no noise level and are skipped.
    """
    raw = _read_samples(logs)
    column = _METADATA["noise_level"]
    if column in raw.columns:
        raw = raw[raw[column].notna()]
    return normalize_samples(raw)


def _read_samples(logs: str | Path) -> pd.DataFrame:
    path = Path(logs)
    if not path.exists():
        raise FileNotFoundError(f"eval log path does not exist: {path}")
    return samples_df(path, columns=[*SampleSummary, *EvalModel])


def normalize_samples(raw: pd.DataFrame) -> pd.DataFrame:
    """Keep the analysis key and the three scorer outcomes."""
    required = ["model", "epoch", *_METADATA.values()]
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise ValueError(f"eval log is missing columns: {', '.join(missing)}")

    frame = pd.DataFrame(
        {
            "scenario": raw[_METADATA["scenario"]].astype(str),
            "eval_notice": _eval_notice(raw),
            "model": raw["model"].astype(str),
            "epoch": raw["epoch"].astype(int),
            "noise_level": raw[_METADATA["noise_level"]].astype(int),
            "noise_draw_id": raw[_METADATA["noise_draw_id"]].astype(int),
            "noise_seed": raw[_METADATA["noise_seed"]].astype(int),
        }
    )
    return _with_review_columns(frame, raw)


def _eval_notice(raw: pd.DataFrame) -> pd.Series:
    """Whether the evaluation notice was in the prompt.

    Logs from before the notice became its own setting had it exactly when
    ``test_eval_awareness`` was on, which is also the task default.
    """
    columns = [
        column
        for column in ("metadata_eval_notice", "metadata_test_eval_awareness")
        if column in raw.columns
    ]

    def resolve(row: pd.Series) -> bool:
        for column in columns:
            value = _as_float(row[column])
            if value is not None:
                return value != 0.0
        return True

    return raw.apply(resolve, axis=1).astype(bool) if len(raw) else pd.Series(
        dtype=bool
    )


def _with_review_columns(frame: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """Add the sample id and log path for manual review, then the outcomes."""
    frame["sample_id"] = raw["id"].astype(str) if "id" in raw.columns else None
    frame["log"] = raw["log"].astype(str) if "log" in raw.columns else None
    scores = raw.apply(_row_scores, axis=1, result_type="expand")
    for outcome in _OUTCOMES:
        frame[outcome] = scores[outcome]
    return frame.reset_index(drop=True)


def draw_rates(
    frame: pd.DataFrame,
    outcome: str,
    dose: str = "noise_level",
    draw: str = "noise_draw_id",
) -> pd.DataFrame:
    """Mean outcome for each condition, model, dose, and draw."""
    group = [*_id_columns(frame), dose, draw]
    columns = [*group, "rate"]
    subset = frame.dropna(subset=[outcome])
    if subset.empty:
        return pd.DataFrame(columns=columns)
    return (
        subset.groupby(group, as_index=False)[outcome]
        .mean()
        .rename(columns={outcome: "rate"})
    )


def level_means(rates: pd.DataFrame, dose: str = "noise_level") -> pd.DataFrame:
    """Mean of draw rates at each dose."""
    group = [*_id_columns(rates), dose]
    if rates.empty:
        return pd.DataFrame(columns=[*group, "rate"])
    return rates.groupby(group, as_index=False)["rate"].mean()


def trend_table(
    frame: pd.DataFrame, outcome: str, dose: str = "noise_level"
) -> pd.DataFrame:
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
        result = _trend_result(*_epoch_counts(model_rows, outcome, dose))
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
    dose: str = "noise_level",
    draw: str = "noise_draw_id",
) -> pd.DataFrame:
    """Trend test per model, on draws when between-draw variance is material.

    ``dose`` and ``draw`` name the trend variable and the draw that epochs
    repeat. They default to the noise sweep; the peer board uses the number
    of defect posts and the board draw.

    The unit is chosen from doses above 0. Noise level 0 repeats one prompt,
    so its between-draw variance is sampling noise and is left out of the
    average. The chosen unit is then applied to every dose, including 0.

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
        decision_rows = model_rows[model_rows[dose] > 0]
        components = variance_components(decision_rows, outcome, dose, draw)
        between = float(components["between_draw_var"].mean(skipna=True))
        within = float(components["within_draw_var"].mean(skipna=True))
        use_draws = _between_draw_variance_is_material(between, within, ratio)
        pooled = _trend_result(*_epoch_counts(model_rows, outcome, dose))
        by_draw = _trend_result(*_draw_counts(model_rows, outcome, dose, draw))
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
    model_rows: pd.DataFrame, outcome: str, dose: str
) -> tuple[list[int], list[int], list[float]]:
    successes: list[int] = []
    totals: list[int] = []
    scores: list[float] = []
    for level, level_rows in model_rows.groupby(dose, sort=True):
        scores.append(float(level))
        totals.append(int(len(level_rows)))
        successes.append(int(round(float(level_rows[outcome].sum()))))
    return successes, totals, scores


def _draw_counts(
    model_rows: pd.DataFrame, outcome: str, dose: str, draw: str
) -> tuple[list[int], list[int], list[float]]:
    """One trial per draw. The success count is the rounded sum of draw rates.

    A draw rate is an average over epochs, so it is tighter than one Bernoulli
    trial. Feeding the rounded sum into Cochran-Armitage overstates the
    variance and makes this p-value conservative. A borderline result should
    be checked with a clustered model before it is treated as exact.
    """
    rates = draw_rates(model_rows, outcome, dose, draw)
    successes: list[int] = []
    totals: list[int] = []
    scores: list[float] = []
    for level, level_rows in rates.groupby(dose, sort=True):
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


def variance_components(
    frame: pd.DataFrame,
    outcome: str,
    dose: str = "noise_level",
    draw: str = "noise_draw_id",
) -> pd.DataFrame:
    """Between-draw variance of draw means, and mean within-draw epoch variance."""
    group = _id_columns(frame)
    columns = [*group, dose, "between_draw_var", "within_draw_var"]
    subset = frame.dropna(subset=[outcome])
    if subset.empty:
        return pd.DataFrame(columns=columns)
    draw_stats = subset.groupby([*group, dose, draw], as_index=False)[outcome].agg(
        draw_mean="mean", within_var="var"
    )
    return draw_stats.groupby([*group, dose], as_index=False).agg(
        between_draw_var=("draw_mean", "var"),
        within_draw_var=("within_var", "mean"),
    )


def clopper_pearson(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Exact two-sided binomial interval for ``k`` successes in ``n`` trials.

    With no successes the upper bound is the one to quote: 0 of 200 still
    allows a true rate near 1.8% at 95% confidence.
    """
    if n <= 0 or not 0 <= k <= n:
        raise ValueError(f"need 0 <= k <= n and n > 0, got k={k}, n={n}")
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must lie in (0, 1), got {confidence}")
    tail = (1.0 - confidence) / 2.0
    lower = 0.0 if k == 0 else _bisect(lambda p: 1.0 - _binom_cdf(k - 1, n, p) - tail)
    upper = 1.0 if k == n else _bisect(lambda p: tail - _binom_cdf(k, n, p))
    return lower, upper


def fisher_exact(k1: int, n1: int, k2: int, n2: int) -> float:
    """Two-sided Fisher exact p-value for ``k1``/``n1`` against ``k2``/``n2``.

    Sums the probability of every table, with the same margins, that is no
    more likely than the observed one.
    """
    if n1 <= 0 or n2 <= 0 or not 0 <= k1 <= n1 or not 0 <= k2 <= n2:
        raise ValueError(f"bad counts: {k1}/{n1} and {k2}/{n2}")
    total = n1 + n2
    successes = k1 + k2
    low = max(0, n1 - (total - successes))
    high = min(successes, n1)
    log_pmf = {
        x: _log_comb(successes, x)
        + _log_comb(total - successes, n1 - x)
        - _log_comb(total, n1)
        for x in range(low, high + 1)
    }
    observed = log_pmf[k1]
    # Relative tolerance so tables tied with the observed one are counted.
    cutoff = observed + 1e-7
    p_value = math.fsum(math.exp(value) for value in log_pmf.values() if value <= cutoff)
    return min(p_value, 1.0)


def fisher_contrast(
    first: pd.DataFrame, second: pd.DataFrame, outcome: str
) -> dict[str, float]:
    """Counts, rates, and the Fisher p-value for two subsets of samples."""
    a = first[outcome].dropna()
    b = second[outcome].dropna()
    k1, n1 = int(round(float(a.sum()))), int(len(a))
    k2, n2 = int(round(float(b.sum()))), int(len(b))
    return {
        "k1": k1,
        "n1": n1,
        "rate1": k1 / n1 if n1 else math.nan,
        "k2": k2,
        "n2": n2,
        "rate2": k2 / n2 if n2 else math.nan,
        "p_value": fisher_exact(k1, n1, k2, n2) if n1 and n2 else math.nan,
    }


def rate_intervals(
    frame: pd.DataFrame,
    outcome: str,
    dose: str = "noise_level",
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Epoch-pooled count, rate, and Clopper-Pearson interval at each dose.

    Pooling epochs treats every sample as independent. With one epoch per
    draw that is exact; with many epochs of few draws the interval is too narrow.
    """
    group = [*_id_columns(frame), dose]
    columns = [*group, "n", "k", "rate", "lo", "hi"]
    subset = frame.dropna(subset=[outcome])
    if subset.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, object]] = []
    for keys, cell in subset.groupby(group, sort=True):
        n = int(len(cell))
        k = int(round(float(cell[outcome].sum())))
        lo, hi = clopper_pearson(k, n, confidence)
        rows.append(
            {**_key_fields(group, keys), "n": n, "k": k, "rate": k / n, "lo": lo, "hi": hi}
        )
    return pd.DataFrame(rows, columns=columns)


def hits(frame: pd.DataFrame) -> pd.DataFrame:
    """Samples either harm verdict flagged, for reading by hand.

    At rare rates a grader false positive is as likely as a real case, so
    no hit counts until someone has read the transcript.
    """
    flagged = (frame["harmful"] == 1) | (frame["classifier_verdict"] == 1)
    return frame.loc[flagged].reset_index(drop=True)


def _bisect(increasing: Callable[[float], float], steps: int = 200) -> float:
    """Root of a function that increases on (0, 1)."""
    low, high = 0.0, 1.0
    for _ in range(steps):
        middle = (low + high) / 2.0
        if increasing(middle) < 0:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def _binom_cdf(k: int, n: int, p: float) -> float:
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0
    log_p = math.log(p)
    log_q = math.log1p(-p)
    return min(
        math.fsum(
            math.exp(_log_comb(n, i) + i * log_p + (n - i) * log_q)
            for i in range(k + 1)
        ),
        1.0,
    )


def _log_comb(n: int, k: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


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
