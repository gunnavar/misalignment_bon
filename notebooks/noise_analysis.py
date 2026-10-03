import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import plotly.graph_objects as go

    from misalignment_bon.analysis import (
        draw_rates,
        level_means,
        load_samples,
        trend_table,
        variance_components,
    )

    return (
        draw_rates,
        go,
        level_means,
        load_samples,
        mo,
        trend_table,
        variance_components,
    )


@app.cell
def _(go):
    def rate_figure(draws, means, title):
        figure = go.Figure()
        models = list(dict.fromkeys(means["model"].tolist()))
        if not models:
            figure.update_layout(title=title)
            return figure
        slot = 0.8 / len(models)
        for index, model_name in enumerate(models):
            offset = (index - (len(models) - 1) / 2) * slot
            mean_rows = means[means["model"] == model_name]
            draw_rows = draws[draws["model"] == model_name]
            figure.add_trace(
                go.Bar(
                    x=mean_rows["noise_level"] + offset,
                    y=mean_rows["rate"],
                    width=slot * 0.9,
                    name=str(model_name),
                    legendgroup=str(model_name),
                )
            )
            figure.add_trace(
                go.Scatter(
                    x=draw_rows["noise_level"] + offset,
                    y=draw_rows["rate"],
                    mode="markers",
                    name=f"{model_name} draws",
                    legendgroup=str(model_name),
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
    Rows are keyed by model, noise level, draw, and epoch.
    `classifier_verdict` is the primary outcome.
    """)
    return


@app.cell
def _(mo):
    log_dir = mo.ui.text(value="logs", label="Eval log directory")
    log_dir
    return (log_dir,)


@app.cell
def _(load_samples, log_dir):
    try:
        samples = load_samples(log_dir.value)
        load_message = ""
    except Exception as load_error:
        samples = None
        load_message = str(load_error)
    return (load_message, samples)


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
def _(loaded, mo, trend_table):
    cochran_armitage_results = trend_table(loaded, "classifier_verdict")
    mo.vstack(
        [
            mo.md("**Cochran-Armitage trend, classifier verdict**"),
            cochran_armitage_results,
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
