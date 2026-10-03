import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import plotly.graph_objects as go

    from misalignment_bon.analysis import (
        assess_trend,
        draw_rates,
        level_means,
        load_samples,
        variance_components,
    )

    return (
        assess_trend,
        draw_rates,
        go,
        level_means,
        load_samples,
        mo,
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


if __name__ == "__main__":
    app.run()
