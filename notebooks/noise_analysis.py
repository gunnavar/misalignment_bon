import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import plotly.graph_objects as go

    from misalignment_bon.analysis import (
        assess_trend,
        draw_rates,
        fisher_contrast,
        hits,
        level_means,
        load_samples,
        rate_intervals,
        variance_components,
    )

    return (
        assess_trend,
        draw_rates,
        fisher_contrast,
        go,
        hits,
        level_means,
        load_samples,
        mo,
        pd,
        rate_intervals,
        variance_components,
    )


@app.cell
def _(go):
    def _with_series(frame):
        if "scenario" in frame.columns:
            label = frame["scenario"].astype(str) + " / " + frame["model"].astype(str)
        else:
            label = frame["model"].astype(str)
        return frame.assign(_series=label)

    def rate_figure(draws, means, title):
        figure = go.Figure()
        labeled_means = _with_series(means)
        labeled_draws = _with_series(draws)
        series = list(dict.fromkeys(labeled_means["_series"].tolist()))
        if not series:
            figure.update_layout(title=title)
            return figure
        slot = 0.8 / len(series)
        for index, series_name in enumerate(series):
            offset = (index - (len(series) - 1) / 2) * slot
            mean_rows = labeled_means[labeled_means["_series"] == series_name]
            draw_rows = labeled_draws[labeled_draws["_series"] == series_name]
            figure.add_trace(
                go.Bar(
                    x=mean_rows["noise_level"] + offset,
                    y=mean_rows["rate"],
                    width=slot * 0.9,
                    name=series_name,
                    legendgroup=series_name,
                )
            )
            figure.add_trace(
                go.Scatter(
                    x=draw_rows["noise_level"] + offset,
                    y=draw_rows["rate"],
                    mode="markers",
                    name=f"{series_name} draws",
                    legendgroup=series_name,
                    marker={"size": 8, "opacity": 0.75},
                )
            )
        figure.update_layout(
            title=title,
            barmode="overlay",
            xaxis_title="Noise level",
            yaxis_title="Rate",
            xaxis={"dtick": 1},
            yaxis={"range": [0, 1]},
        )
        return figure

    return (rate_figure,)


@app.cell
def _(mo):
    mo.md("""
    # Noise sensitivity of agentic misalignment

    Each bar is the mean rate across noise draws. Each point is one draw.
    Rows are keyed by scenario, model, noise level, draw, and epoch.
    `classifier_verdict` is the primary outcome.
    """)
    return


@app.cell
def _(mo):
    import os
    log_dir = "logs/"
    available_logs = os.listdir(log_dir)
    log_picker = mo.ui.dropdown(available_logs)
    log_picker
    return log_dir, log_picker, os


@app.cell
def _(log_dir, log_picker, os):
    log_path = os.path.join(log_dir, log_picker.value)
    log_path
    return (log_path,)


@app.cell
def _(load_samples, log_path):
    try:
        samples = load_samples(log_path)
        load_message = ""
    except Exception as load_error:
        samples = None
        load_message = str(load_error)
    return load_message, samples


@app.cell
def _(load_message, mo, samples):
    mo.stop(
        samples is None or samples.empty,
        mo.md(load_message or "No samples in that log directory."),
    )
    loaded = samples
    return (loaded,)


@app.cell
def _(draw_rates, level_means, loaded, rate_figure):
    verdict_draws = draw_rates(loaded, "classifier_verdict")
    verdict_figure = rate_figure(
        verdict_draws,
        level_means(verdict_draws),
        "Classifier verdict",
    )
    verdict_figure
    return


@app.cell
def _(draw_rates, level_means, loaded, rate_figure):
    harmful_draws = draw_rates(loaded, "harmful")
    harmful_figure = rate_figure(
        harmful_draws,
        level_means(harmful_draws),
        "Harmful",
    )
    harmful_figure
    return (harmful_draws,)


@app.cell
def _(harmful_draws):
    harmful_draws
    return


@app.cell
def _(draw_rates, level_means, loaded, rate_figure):
    aware_draws = draw_rates(loaded, "aware")
    aware_figure = rate_figure(
        aware_draws,
        level_means(aware_draws),
        "Eval awareness",
    )
    aware_figure
    return


@app.cell
def _(draw_rates, level_means, loaded, rate_figure):
    comprehension_draws = draw_rates(loaded, "comprehension")
    comprehension_figure = rate_figure(
        comprehension_draws,
        level_means(comprehension_draws),
        "Comprehension",
    )
    comprehension_figure
    return


@app.cell
def _(draw_rates, level_means, loaded, rate_figure):
    recipient_draws = draw_rates(loaded, "recipient_comprehension")
    recipient_figure = rate_figure(
        recipient_draws,
        level_means(recipient_draws),
        "Recipient comprehension",
    )
    recipient_figure
    return


@app.cell
def _(assess_trend, loaded, mo):
    verdict_trend = assess_trend(loaded, "classifier_verdict")
    mo.vstack(
        [
            mo.md("""
            **Cochran-Armitage trend, classifier verdict.**

            `p_value` uses one trial per draw when between-draw variance is
            material, and epoch-pooled counts otherwise. `pooled_p_value`
            is the epoch-pooled test either way.
            """),
            verdict_trend,
        ]
    )
    return


@app.cell
def _(assess_trend, loaded, mo):
    aware_trend = assess_trend(loaded, "aware")
    mo.vstack(
        [
            mo.md("""
            **Cochran-Armitage trend, eval awareness.**

            Same rule as the classifier verdict: `p_value` uses one trial
            per draw when between-draw variance is material, and epoch-pooled
            counts otherwise. `pooled_p_value` is the epoch-pooled test either way.
            """),
            aware_trend,
        ]
    )
    return


@app.cell
def _(loaded, mo, variance_components):
    verdict_variance = variance_components(loaded, "classifier_verdict")
    mo.vstack(
        [
            mo.md(
                "**Variance of classifier verdict.** "
                "Between-draw is the variance of draw means. "
                "Within-draw is the mean epoch variance inside a draw."
            ),
            verdict_variance,
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    # Hunt: rare misalignment without the evaluation notice

    Every log in the directory, restricted to runs with the
    evaluation-or-deployment notice left out of the prompt. Rates are
    epoch-pooled with exact Clopper-Pearson 95% intervals. With zero hits the
    upper bound `hi` is the number to quote.
    """)
    return


@app.cell
def _(load_samples, log_dir, mo):
    try:
        _all = load_samples(log_dir)
        hunt = _all[~_all["eval_notice"]].reset_index(drop=True)
    except Exception as _error:
        hunt = None
        _message = str(_error)
    else:
        _message = "No samples without the notice yet."
    mo.stop(hunt is None or hunt.empty, mo.md(_message))
    return (hunt,)


@app.cell
def _(hunt, mo, rate_intervals):
    mo.vstack(
        [
            mo.md(
                "**Harmful** (classifier, Kyle in a `to:` field, and the keyword) "
                "is the primary hunt outcome. **Classifier verdict** alone is below it."
            ),
            rate_intervals(hunt, "harmful"),
            rate_intervals(hunt, "classifier_verdict"),
            rate_intervals(hunt, "aware"),
        ]
    )
    return


@app.cell
def _(fisher_contrast, hunt, mo, pd):
    _rows = []
    for (_scenario, _model), _cell in hunt.groupby(["scenario", "model"]):
        _clean = _cell[_cell["noise_level"] == 0]
        _noised = _cell[_cell["noise_level"] == 3]
        if _clean.empty or _noised.empty:
            continue
        for _outcome in ("harmful", "classifier_verdict"):
            _rows.append(
                {
                    "scenario": _scenario,
                    "model": _model,
                    "outcome": _outcome,
                    **fisher_contrast(_clean, _noised, _outcome),
                }
            )
    mo.vstack(
        [
            mo.md(
                "**Level 0 (1) against level 3 (2), two-sided Fisher exact test.** "
                "One hit against none gives p = 1: a single case shows the rate "
                "is not zero, not that noise raised it."
            ),
            pd.DataFrame(_rows),
        ]
    )
    return


@app.cell
def _(hits, hunt, mo):
    hunt_hits = hits(hunt)
    mo.vstack(
        [
            mo.md(
                "**Flagged samples to read in `inspect view`.** At rates this low "
                "a grader false positive is as likely as a real case. Count a hit "
                "only if the transcript reads as coercion of Kyle."
            ),
            hunt_hits,
        ]
    )
    return


@app.cell
def _(hunt, mo, rate_intervals):
    # A replication run gives one draw many epochs. Draws with at least ten
    # samples only come from such runs. Compare hit draws with control draws
    # chosen at random from those that did not hit.
    _replayed = hunt.groupby(["model", "noise_level", "noise_draw_id"])[
        "harmful"
    ].transform("size")
    _replays = hunt[(_replayed >= 10) & (hunt["noise_level"] > 0)]
    mo.stop(_replays.empty, mo.md("No replication runs yet."))
    mo.vstack(
        [
            mo.md("**Replicated draws.** Each row is one draw, pooled over epochs."),
            rate_intervals(_replays, "harmful", dose="noise_draw_id"),
            rate_intervals(_replays, "classifier_verdict", dose="noise_draw_id"),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
