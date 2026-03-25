import numpy as np
import matplotlib
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Theme configuration (mirrors classification visualization.py)
# ---------------------------------------------------------------------------

_THEME_DARK = {
    "bg": "#0e1117",
    "panel_bg": "#161b27",
    "text": "#e8eaf0",
    "subtext": "#9da5b4",
    "grid": "#252d3d",
    "spine": "#3a4358",
    "scatter_train": "#4c9be8",
    "scatter_test": "#f0824a",
    "pred_line": "#69f0ae",
    "residual_pos": "#ff5252",
    "residual_neg": "#40c8f4",
    "diagonal": "#ffd740",
    "scatter_alpha": 0.75,
    "scatter_edge": "#ffffff",
    "scatter_edge_width": 0.5,
}

_THEME_LIGHT = {
    "bg": "#ffffff",
    "panel_bg": "#f4f6f9",
    "text": "#1a1d23",
    "subtext": "#5a6270",
    "grid": "#dde1e9",
    "spine": "#b0b8c8",
    "scatter_train": "#2176ae",
    "scatter_test": "#e06c1a",
    "pred_line": "#1b5e20",
    "residual_pos": "#c62828",
    "residual_neg": "#1565c0",
    "diagonal": "#b45309",
    "scatter_alpha": 0.70,
    "scatter_edge": "#1a1d23",
    "scatter_edge_width": 0.4,
}


def _get_theme(theme: str) -> dict:
    return _THEME_DARK if theme == "dark" else _THEME_LIGHT


def _reset_matplotlib_style() -> None:
    """Reset matplotlib to a clean state before each figure."""
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    plt.rcParams["figure.dpi"] = 110


def _apply_figure_style(fig: plt.Figure, axes, theme: dict) -> None:
    fig.patch.set_facecolor(theme["bg"])
    ax_list = list(axes) if hasattr(axes, "__iter__") else [axes]
    for ax in ax_list:
        ax.set_facecolor(theme["panel_bg"])
        ax.tick_params(colors=theme["text"], labelsize=9, which="both")
        ax.xaxis.label.set_color(theme["text"])
        ax.yaxis.label.set_color(theme["text"])
        ax.title.set_color(theme["text"])
        for spine in ax.spines.values():
            spine.set_edgecolor(theme["spine"])
            spine.set_linewidth(0.8)
        ax.grid(True, color=theme["grid"], linewidth=0.5, linestyle="--", alpha=0.7)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_color(theme["text"])


# ---------------------------------------------------------------------------
# Prediction Curve Plot (scatter + model prediction line)
# ---------------------------------------------------------------------------

def plot_prediction_curve(
    model,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    show_train: bool = True,
    show_test: bool = True,
    theme: str = "dark",
) -> plt.Figure:
    """
    Plot scatter of real data and the model's prediction curve.

    Parameters
    ----------
    model : fitted sklearn regressor
    X_train, X_test : feature arrays of shape (n, 1)
    y_train, y_test : target arrays
    show_train : bool — overlay training points
    show_test  : bool — overlay test points
    theme : "dark" | "light"

    Returns
    -------
    matplotlib Figure
    """
    _reset_matplotlib_style()
    t = _get_theme(theme)

    fig, ax = plt.subplots(figsize=(6, 5))

    # Smooth prediction curve
    X_all = np.vstack([X_train, X_test])
    x_min, x_max = X_all[:, 0].min(), X_all[:, 0].max()
    margin = (x_max - x_min) * 0.05
    X_curve = np.linspace(x_min - margin, x_max + margin, 300).reshape(-1, 1)
    y_curve = model.predict(X_curve)

    # Plot prediction curve
    ax.plot(
        X_curve[:, 0], y_curve,
        color=t["pred_line"],
        linewidth=2.5,
        label="Predicción",
        zorder=3,
    )

    # Scatter: train
    if show_train:
        ax.scatter(
            X_train[:, 0], y_train,
            c=t["scatter_train"],
            marker="o",
            s=40,
            edgecolors=t["scatter_edge"],
            linewidths=t["scatter_edge_width"],
            alpha=t["scatter_alpha"],
            label="Train",
            zorder=2,
        )

    # Scatter: test
    if show_test:
        ax.scatter(
            X_test[:, 0], y_test,
            c=t["scatter_test"],
            marker="X",
            s=50,
            edgecolors=t["scatter_edge"],
            linewidths=t["scatter_edge_width"],
            alpha=t["scatter_alpha"],
            label="Test",
            zorder=2,
        )

    if show_train or show_test:
        leg = ax.legend(
            fontsize=8,
            framealpha=0.45,
            facecolor=t["panel_bg"],
            edgecolor=t["spine"],
            labelcolor=t["text"],
            loc="best",
        )
        leg.get_frame().set_linewidth(0.8)

    ax.set_xlabel("X", fontsize=10, color=t["text"])
    ax.set_ylabel("y", fontsize=10, color=t["text"])
    ax.set_title("Curva de Predicción", fontsize=12, fontweight="bold", color=t["text"])

    _apply_figure_style(fig, [ax], t)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Residuals Plot
# ---------------------------------------------------------------------------

def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    theme: str = "dark",
    title_suffix: str = "",
) -> plt.Figure:
    """
    Plot residuals (y_true - y_pred) vs predicted values.

    Parameters
    ----------
    y_true, y_pred : arrays of shape (n,)
    theme : "dark" | "light"
    title_suffix : str — appended to the title (e.g. "— Test")

    Returns
    -------
    matplotlib Figure
    """
    _reset_matplotlib_style()
    t = _get_theme(theme)

    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(6, 4))

    colors = np.where(residuals >= 0, t["residual_pos"], t["residual_neg"])
    ax.scatter(
        y_pred, residuals,
        c=colors,
        s=35,
        alpha=t["scatter_alpha"],
        edgecolors=t["scatter_edge"],
        linewidths=t["scatter_edge_width"],
        zorder=2,
    )
    ax.axhline(0, color=t["diagonal"], linewidth=1.5, linestyle="--", alpha=0.8, zorder=1)

    ax.set_xlabel("Predicción (ŷ)", fontsize=10, color=t["text"])
    ax.set_ylabel("Residuo (y − ŷ)", fontsize=10, color=t["text"])
    title = "Residuos"
    if title_suffix:
        title += f" {title_suffix}"
    ax.set_title(title, fontsize=12, fontweight="bold", color=t["text"])

    _apply_figure_style(fig, [ax], t)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Prediction Error Plot (y_true vs y_pred with perfect diagonal)
# ---------------------------------------------------------------------------

def plot_prediction_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    theme: str = "dark",
    title_suffix: str = "",
) -> plt.Figure:
    """
    Plot y_true vs y_pred with a perfect-prediction diagonal.

    Parameters
    ----------
    y_true, y_pred : arrays of shape (n,)
    theme : "dark" | "light"
    title_suffix : str — appended to the title

    Returns
    -------
    matplotlib Figure
    """
    _reset_matplotlib_style()
    t = _get_theme(theme)

    fig, ax = plt.subplots(figsize=(5, 5))

    ax.scatter(
        y_true, y_pred,
        c=t["scatter_train"],
        s=35,
        alpha=t["scatter_alpha"],
        edgecolors=t["scatter_edge"],
        linewidths=t["scatter_edge_width"],
        zorder=2,
    )

    # Perfect prediction diagonal
    all_vals = np.concatenate([y_true, y_pred])
    lo, hi = all_vals.min(), all_vals.max()
    margin = (hi - lo) * 0.05
    ax.plot(
        [lo - margin, hi + margin], [lo - margin, hi + margin],
        color=t["diagonal"],
        linewidth=1.5,
        linestyle="--",
        alpha=0.8,
        label="Predicción perfecta",
        zorder=1,
    )

    leg = ax.legend(
        fontsize=8,
        framealpha=0.45,
        facecolor=t["panel_bg"],
        edgecolor=t["spine"],
        labelcolor=t["text"],
    )
    leg.get_frame().set_linewidth(0.8)

    ax.set_xlabel("Valor Real (y)", fontsize=10, color=t["text"])
    ax.set_ylabel("Predicción (ŷ)", fontsize=10, color=t["text"])
    title = "Error de Predicción"
    if title_suffix:
        title += f" {title_suffix}"
    ax.set_title(title, fontsize=12, fontweight="bold", color=t["text"])

    _apply_figure_style(fig, [ax], t)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Metrics Comparison Bar Chart (regression)
# ---------------------------------------------------------------------------

def plot_regression_metrics_comparison(
    results_list: list[dict],
    theme: str = "dark",
) -> plt.Figure:
    """
    Plot grouped bar chart comparing R², RMSE, MAE across multiple regression models.

    Parameters
    ----------
    results_list : list of dicts, each with keys:
        - "name": str
        - "train": metrics dict
        - "test": metrics dict
    theme : "dark" | "light"

    Returns
    -------
    matplotlib Figure
    """
    _reset_matplotlib_style()
    t = _get_theme(theme)
    n_models = len(results_list)

    metrics = [
        ("r2", "R²"),
        ("rmse", "RMSE"),
        ("mae", "MAE"),
    ]

    fig, axes = plt.subplots(
        1, len(metrics),
        figsize=(max(5, n_models * 1.8 * len(metrics)), 4.5),
        squeeze=False,
    )
    axes = axes[0]

    x = np.arange(n_models)
    width = 0.35
    names = [r["name"] for r in results_list]

    for ax, (key, label) in zip(axes, metrics):
        train_vals = [r["train"].get(key, 0.0) for r in results_list]
        test_vals = [r["test"].get(key, 0.0) for r in results_list]

        ax.bar(
            x - width / 2, train_vals, width, label="Train",
            color=t["scatter_train"], alpha=0.85, edgecolor=t["spine"],
        )
        ax.bar(
            x + width / 2, test_vals, width, label="Test",
            color=t["scatter_test"], alpha=0.85, edgecolor=t["spine"],
        )

        # Annotate bars
        for bars in [ax.containers[0], ax.containers[1]]:
            for bar in bars:
                h = bar.get_height()
                fmt = f"{h:.3f}" if abs(h) < 10 else f"{h:.1f}"
                ax.text(
                    bar.get_x() + bar.get_width() / 2, h + 0.01, fmt,
                    ha="center", va="bottom", fontsize=7.5, color=t["text"],
                )

        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8.5)
        ax.set_ylabel(label, fontsize=10)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.legend(
            fontsize=9, framealpha=0.3, facecolor=t["panel_bg"],
            edgecolor=t["spine"], labelcolor=t["text"],
        )

    _apply_figure_style(fig, axes, t)
    fig.suptitle(
        "Comparación de Modelos", fontsize=13, fontweight="bold",
        color=t["text"], y=1.02,
    )
    fig.tight_layout()
    return fig
