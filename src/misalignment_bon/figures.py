"""Draw the README figures from the successful noise-sweep logs.

Every figure uses the blackmail scenario with the evaluation notice on.
A model keeps the same color in every figure.

    uv run python -m misalignment_bon.figures logs figures
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

from misalignment_bon.analysis import clopper_pearson  # noqa: E402
from misalignment_bon.transcripts import load_transcripts  # noqa: E402

# Display name and categorical slot per model. The order is the validated
# adjacent order of the reference palette, so it must not be shuffled.
MODELS = {
    "openrouter/openai/gpt-4.1": ("GPT-4.1", "#2a78d6"),
    "openrouter/deepseek/deepseek-v3.2": ("DeepSeek-V3.2", "#eb6834"),
    "openrouter/openai/gpt-5.4-mini": ("GPT-5.4-mini", "#1baf7a"),
    "openrouter/deepseek/deepseek-v4.1-flash": ("DeepSeek-V4.1-Flash", "#eda100"),
    "openrouter/anthropic/claude-haiku-4.5": ("Claude Haiku 4.5", "#e87ba4"),
}
BLACKMAILING = ("openrouter/openai/gpt-4.1", "openrouter/deepseek/deepseek-v3.2")

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

LEVELS = (0, 1, 2, 3)
LEVEL_LABELS = ("Clean", "1/3 noise", "2/3 noise", "Full noise")


def main() -> None:
    logs = sys.argv[1] if len(sys.argv) > 1 else "logs"
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "figures")
    out.mkdir(parents=True, exist_ok=True)
    frame = load_transcripts(logs)
    frame = frame[(frame["scenario"] == "blackmail") & frame["eval_notice"]]
    frame = frame[frame["model"].isin(MODELS)]
    _style()
    blackmail_by_noise(frame).savefig(out / "blackmail_by_noise.png")
    identity_confusion(frame).savefig(out / "identity_confusion.png")
    eval_awareness(frame).savefig(out / "eval_awareness.png")
    print(f"Wrote three figures to {out}/")


def blackmail_by_noise(frame: pd.DataFrame) -> plt.Figure:
    """Verdict rate per level with 95% intervals; zero-rate models share one grey line."""
    fig, ax = _figure(
        "Noise lowers blackmail rates and never raises them",
        "Grader-judged blackmail rate by noise level · 50 samples per point · "
        "bars are 95% Clopper-Pearson intervals",
    )
    rates = _rates(frame, "classifier_verdict")
    zero_models = [
        model for model in MODELS if model in rates and rates[model]["k"].sum() == 0
    ]
    handles = []
    for offset, model in zip((-0.07, 0.07), BLACKMAILING, strict=True):
        if model not in rates:
            continue
        name, color = MODELS[model]
        cell = rates[model]
        _whiskers(ax, cell, color, offset)
        handles.append(_line(ax, cell, color, offset, name))
        first, last = cell.iloc[0], cell.iloc[-1]
        _label(ax, first["noise_level"] + offset, first["rate"], f"{first['rate']:.0%}", "right")
        _label(ax, last["noise_level"] + offset, last["rate"], f"{last['rate']:.0%}", "left")
    if zero_models:
        # Each zero-rate model has its own 50 samples, so its interval is drawn,
        # not the narrower one of the three pooled.
        cell = rates[zero_models[0]]
        names = ", ".join(MODELS[model][0] for model in zero_models)
        _whiskers(ax, cell, MUTED, 0.0)
        handles.append(_line(ax, cell, MUTED, 0.0, f"{names} (0/50 each)"))
    _percent_axis(ax, 1.0)
    _legend(fig, handles)
    return fig


def identity_confusion(frame: pd.DataFrame) -> plt.Figure:
    """Share signed as a human, and blackmail with and without those samples."""
    fig, (left, right) = _figure(
        "Noise makes models write as a human, but that explains only part of the drop",
        "Left: samples with an email signed as a person in the scenario, almost always "
        "Todd. Haiku sends no email and is left out.\n"
        "Right: blackmail rate in all samples (solid) and in samples not signed as a "
        "human (dashed).",
        panels=2,
    )
    handles = []
    for model, (name, color) in MODELS.items():
        cell = frame[frame["model"] == model]
        if cell.empty or not cell["sent_email"].any():
            continue
        share = cell.groupby("noise_level")["signs_as_person"].mean().reindex(LEVELS)
        handles.append(_line(left, _as_rates(share), color, 0.0, name))
    left.set_title("Signed as a human", loc="left", color=INK, fontsize=10)
    _percent_axis(left, 0.6)

    for model in BLACKMAILING:
        cell = frame[frame["model"] == model]
        if cell.empty:
            continue
        _, color = MODELS[model]
        everyone = cell.groupby("noise_level")["classifier_verdict"].mean()
        kept = cell[~cell["signs_as_person"]]
        unconfused = kept.groupby("noise_level")["classifier_verdict"].mean()
        _line(right, _as_rates(everyone.reindex(LEVELS)), color, 0.0, None)
        _line(right, _as_rates(unconfused.reindex(LEVELS)), color, 0.0, None, dashed=True)
    right.set_title("Blackmail rate", loc="left", color=INK, fontsize=10)
    _percent_axis(right, 1.0)

    style_keys = [
        Line2D([], [], color=INK_SECONDARY, lw=2, label="All samples"),
        Line2D([], [], color=INK_SECONDARY, lw=2, ls=(0, (4, 2)), label="Not signed as a human"),
    ]
    _legend(fig, handles + style_keys)
    return fig


def eval_awareness(frame: pd.DataFrame) -> plt.Figure:
    """Grader-judged evaluation awareness per level, every model."""
    fig, ax = _figure(
        "Eval awareness does not track the blackmail drop",
        "Share of samples the grader judged evaluation-aware · the models that "
        "blackmail are near zero at every level",
    )
    handles = []
    # Drawn in reverse so GPT-4.1 sits on top where it overlaps GPT-5.4-mini near zero.
    for model, (name, color) in reversed(MODELS.items()):
        cell = frame[frame["model"] == model]
        if cell.empty:
            continue
        aware = cell.groupby("noise_level")["aware"].mean().reindex(LEVELS)
        handles.append(_line(ax, _as_rates(aware), color, 0.0, name))
    _percent_axis(ax, 1.0)
    _legend(fig, handles[::-1])
    return fig


def _rates(frame: pd.DataFrame, outcome: str) -> dict[str, pd.DataFrame]:
    """Count, rate and interval per level for each model."""
    rates: dict[str, pd.DataFrame] = {}
    for model, cell in frame.dropna(subset=[outcome]).groupby("model"):
        rows = []
        for level, level_rows in cell.groupby("noise_level"):
            n = len(level_rows)
            k = int(round(level_rows[outcome].sum()))
            lo, hi = clopper_pearson(k, n)
            rows.append({"noise_level": level, "n": n, "k": k, "rate": k / n, "lo": lo, "hi": hi})
        rates[str(model)] = pd.DataFrame(rows).sort_values("noise_level")
    return rates


def _as_rates(series: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"noise_level": series.index, "rate": series.to_numpy()})


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.25,
            "font.family": "sans-serif",
            "font.size": 10,
            "text.color": INK,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK_SECONDARY,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
        }
    )


def _figure(title: str, subtitle: str, panels: int = 1):
    width = 8.0 if panels == 1 else 11.0
    fig, axes = plt.subplots(1, panels, figsize=(width, 5.2))
    fig.subplots_adjust(top=0.64, wspace=0.25)
    fig.text(0.07, 0.97, title, fontsize=14, fontweight="bold", color=INK, va="top")
    fig.text(0.07, 0.905, subtitle, fontsize=9.5, color=INK_SECONDARY, va="top", linespacing=1.4)
    for ax in axes if panels > 1 else [axes]:
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(AXIS)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_xticks(LEVELS, LEVEL_LABELS)
        ax.set_xlim(-0.35, 3.45)
    return fig, axes


def _percent_axis(ax: Axes, top: float) -> None:
    ax.set_ylim(0, top)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_yticks([step * top / 4 for step in range(5)] if top < 1 else [0, 0.25, 0.5, 0.75, 1.0])


def _line(
    ax: Axes,
    cell: pd.DataFrame,
    color: str,
    offset: float,
    label: str | None,
    dashed: bool = False,
) -> Line2D:
    (line,) = ax.plot(
        cell["noise_level"] + offset,
        cell["rate"],
        color=color,
        lw=2,
        ls=(0, (4, 2)) if dashed else "-",
        marker="o",
        ms=7,
        mfc=color,
        mec=SURFACE,
        mew=2,
        solid_capstyle="round",
        solid_joinstyle="round",
        label=label,
        zorder=3,
        clip_on=False,
    )
    return line


def _whiskers(ax: Axes, cell: pd.DataFrame, color: str, offset: float) -> None:
    ax.vlines(
        cell["noise_level"] + offset,
        cell["lo"],
        cell["hi"],
        color=color,
        alpha=0.45,
        lw=1.5,
        zorder=2,
    )


def _label(ax: Axes, x: float, y: float, text: str, side: str) -> None:
    dx = 0.09 if side == "left" else -0.09
    ax.text(x + dx, y, text, color=INK, fontsize=9.5, va="center", ha=side)


def _legend(fig: plt.Figure, handles: list) -> None:
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.065, 0.81),
        ncol=min(len(handles), 3),
        frameon=False,
        fontsize=9.5,
        handlelength=2.2,
        columnspacing=1.6,
        labelcolor=INK,
    )


if __name__ == "__main__":
    main()
