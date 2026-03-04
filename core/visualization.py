import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from sklearn.inspection import DecisionBoundaryDisplay


# ---------------------------------------------------------------------------
# Theme configuration
# ---------------------------------------------------------------------------

_THEME_DARK = {
    "bg": "#0e1117",
    "panel_bg": "#161b27",
    "text": "#e8eaf0",
    "subtext": "#9da5b4",
    "grid": "#252d3d",
    "spine": "#3a4358",
    # Decision boundary: bright colormap that pops on dark background
    "cmap_boundary": "coolwarm",
    "boundary_alpha": 0.55,
    # CM: diverging — diagonal (correct) goes blue, off-diagonal (errors) goes red
    "cm_cmap": "RdBu",
    "cm_annot_thresh": 0.5,  # normalised threshold: above → use dark text, below → light
    "bar_train": "#4c9be8",
    "bar_test": "#f0824a",
    "scatter_alpha": 0.92,
    "scatter_edge": "#ffffff",
    "scatter_edge_width": 0.6,
}

_THEME_LIGHT = {
    "bg": "#ffffff",
    "panel_bg": "#f4f6f9",
    "text": "#1a1d23",
    "subtext": "#5a6270",
    "grid": "#dde1e9",
    "spine": "#b0b8c8",
    "cmap_boundary": "RdBu",
    "boundary_alpha": 0.38,
    "cm_cmap": "RdBu",
    "cm_annot_thresh": 0.5,
    "bar_train": "#2176ae",
    "bar_test": "#e06c1a",
    "scatter_alpha": 0.88,
    "scatter_edge": "#1a1d23",
    "scatter_edge_width": 0.5,
}

# High-contrast discrete colors for scatter classes (up to 5 classes)
_CLASS_COLORS_DARK = ["#ff5252", "#40c8f4", "#ffd740", "#b388ff", "#69f0ae"]
_CLASS_COLORS_LIGHT = ["#c62828", "#1565c0", "#e65100", "#6a1b9a", "#1b5e20"]


def _get_theme(theme: str) -> dict:
    return _THEME_DARK if theme == "dark" else _THEME_LIGHT


def _reset_matplotlib_style() -> None:
    """Reset matplotlib to a clean state before each figure to prevent bleed-through."""
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
        # Ensure tick labels are also themed
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_color(theme["text"])


# ---------------------------------------------------------------------------
# Decision Boundary Plot
# ---------------------------------------------------------------------------

def plot_decision_boundary(
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
    Plot the decision boundary of a fitted sklearn classifier.

    Parameters
    ----------
    model : fitted sklearn estimator with predict method
    X_train, X_test : feature arrays of shape (n, 2)
    y_train, y_test : label arrays
    show_train : bool — overlay training points
    show_test  : bool — overlay test points
    theme : "dark" | "light"

    Returns
    -------
    matplotlib Figure
    """
    _reset_matplotlib_style()
    t = _get_theme(theme)
    class_colors = _CLASS_COLORS_DARK if theme == "dark" else _CLASS_COLORS_LIGHT

    X_all = np.vstack([X_train, X_test])
    n_classes = len(np.unique(np.concatenate([y_train, y_test])))

    fig, ax = plt.subplots(figsize=(6, 5))

    # --- Decision boundary background ---
    try:
        DecisionBoundaryDisplay.from_estimator(
            model,
            X_all,
            ax=ax,
            response_method="predict",
            cmap=t["cmap_boundary"],
            alpha=t["boundary_alpha"],
            plot_method="pcolormesh",
        )
    except Exception:
        _plot_boundary_fallback(model, X_all, ax, t["cmap_boundary"], n_classes, t["boundary_alpha"])

    # --- Scatter: train points ---
    if show_train:
        for cls_idx in range(n_classes):
            mask = y_train == cls_idx
            if mask.sum() == 0:
                continue
            color = class_colors[cls_idx % len(class_colors)]
            ax.scatter(
                X_train[mask, 0], X_train[mask, 1],
                c=color,
                marker="o",
                s=52,
                edgecolors=t["scatter_edge"],
                linewidths=t["scatter_edge_width"],
                alpha=t["scatter_alpha"],
                label=f"Train · clase {cls_idx}",
                zorder=3,
            )

    # --- Scatter: test points ---
    if show_test:
        for cls_idx in range(n_classes):
            mask = y_test == cls_idx
            if mask.sum() == 0:
                continue
            color = class_colors[cls_idx % len(class_colors)]
            ax.scatter(
                X_test[mask, 0], X_test[mask, 1],
                c=color,
                marker="X",
                s=72,
                edgecolors=t["scatter_edge"],
                linewidths=t["scatter_edge_width"],
                alpha=t["scatter_alpha"],
                label=f"Test · clase {cls_idx}",
                zorder=4,
            )

    if show_train or show_test:
        leg = ax.legend(
            fontsize=7.5,
            framealpha=0.45,
            facecolor=t["panel_bg"],
            edgecolor=t["spine"],
            labelcolor=t["text"],
            loc="best",
            ncol=2 if n_classes > 2 else 1,
        )
        leg.get_frame().set_linewidth(0.8)

    ax.set_xlabel("Feature 1", fontsize=10, color=t["text"])
    ax.set_ylabel("Feature 2", fontsize=10, color=t["text"])
    ax.set_title("Frontera de Decisión", fontsize=12, fontweight="bold", color=t["text"])

    _apply_figure_style(fig, [ax], t)
    fig.tight_layout()
    return fig


def _plot_boundary_fallback(model, X, ax, cmap_name, n_classes, alpha=0.45):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200),
    )
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.pcolormesh(xx, yy, Z, cmap=cmap_name, alpha=alpha)


# ---------------------------------------------------------------------------
# Confusion Matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    cm: np.ndarray,
    labels: list[str],
    theme: str = "dark",
    normalize: bool = False,
) -> plt.Figure:
    """
    Plot a styled confusion matrix heatmap.

    Each cell is colored independently using two dedicated colormaps:
    - Blues  → diagonal cells (correct predictions)
    - Reds   → off-diagonal cells (errors)
    Color intensity is based on per-row recall normalization so minority classes
    are never washed out regardless of absolute counts.

    Parameters
    ----------
    cm : np.ndarray of shape (n_classes, n_classes)
    labels : list of class label strings
    theme : "dark" | "light"
    normalize : bool — if True, annotate with row-% instead of raw counts

    Returns
    -------
    matplotlib Figure
    """
    _reset_matplotlib_style()
    t = _get_theme(theme)
    n = cm.shape[0]
    fig_size = max(4.0, n * 1.6)

    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    # Per-row normalization (recall / row-sum) — used for coloring
    row_sums = cm.sum(axis=1, keepdims=True).astype(float)
    cm_row_norm = cm.astype(float) / np.where(row_sums == 0, 1.0, row_sums)

    # Build RGBA image using independent colormaps per cell type
    cmap_correct = plt.get_cmap("Blues")
    cmap_error = plt.get_cmap("Reds")

    # Clamp normalised value to [0.25, 0.95] so cells are never fully white or black
    def _clamp(v: float) -> float:
        return float(np.clip(v, 0.20, 0.95))

    rgba_image = np.zeros((n, n, 4), dtype=float)
    for i in range(n):
        for j in range(n):
            intensity = _clamp(cm_row_norm[i, j])
            if i == j:
                rgba_image[i, j] = cmap_correct(intensity)
            else:
                rgba_image[i, j] = cmap_error(intensity)

    ax.imshow(rgba_image, aspect="auto", interpolation="nearest")

    # Annotation: determine text color per cell for maximum readability
    def _text_color(rgba) -> str:
        # Luminance from sRGB
        r, g, b = rgba[:3]
        lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
        return "#1a1d23" if lum > 0.45 else "#f5f5f5"

    # Draw cell text: primary value + secondary small value
    for i in range(n):
        for j in range(n):
            tc = _text_color(rgba_image[i, j])
            pct = cm_row_norm[i, j] * 100.0
            count = cm[i, j]

            if normalize:
                primary = f"{pct:.1f}%"
                secondary = f"n={count}"
            else:
                primary = str(count)
                secondary = f"{pct:.1f}%"

            # Primary label (large, bold)
            ax.text(
                j, i, primary,
                ha="center", va="center",
                fontsize=max(10, 16 - n * 2),
                fontweight="bold",
                color=tc,
            )
            # Secondary label (small, below primary)
            ax.text(
                j, i + 0.28, secondary,
                ha="center", va="center",
                fontsize=max(7, 10 - n),
                fontweight="normal",
                color=tc,
                alpha=0.80,
            )

    # Draw grid lines between cells
    for pos in np.arange(0.5, n, 1.0):
        ax.axhline(pos, color=t["spine"], linewidth=0.8)
        ax.axvline(pos, color=t["spine"], linewidth=0.8)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color(t["text"])

    ax.set_xlabel("Predicción", fontsize=10, color=t["text"])
    ax.set_ylabel("Valor Real", fontsize=10, color=t["text"])

    mode_label = "(%  por fila)" if normalize else "(conteos  |  % por fila)"
    ax.set_title(f"Matriz de Confusión  {mode_label}", fontsize=10,
                 fontweight="bold", color=t["text"])

    ax.tick_params(colors=t["text"], labelsize=10, length=0)
    for spine in ax.spines.values():
        spine.set_edgecolor(t["spine"])
        spine.set_linewidth(0.8)

    fig.patch.set_facecolor(t["bg"])
    ax.set_facecolor(t["panel_bg"])
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Metrics Comparison Bar Chart
# ---------------------------------------------------------------------------

def plot_metrics_comparison(
    results_list: list[dict],
    theme: str = "dark",
    metrics: tuple[str, ...] = ("accuracy", "f1"),
) -> plt.Figure:
    """
    Plot grouped bar chart comparing train/test metrics across multiple models.

    Parameters
    ----------
    results_list : list of dicts, each with keys:
        - "name": str (model name)
        - "train": metrics dict
        - "test": metrics dict
    theme : "dark" | "light"
    metrics : tuple of metric names to display

    Returns
    -------
    matplotlib Figure
    """
    _reset_matplotlib_style()
    t = _get_theme(theme)
    n_models = len(results_list)
    n_metrics = len(metrics)

    metric_labels = {"accuracy": "Accuracy", "f1": "F1-score", "precision": "Precisión", "recall": "Recall"}

    fig, axes = plt.subplots(1, n_metrics, figsize=(max(5, n_models * 1.8 * n_metrics), 4.5), squeeze=False)
    axes = axes[0]

    x = np.arange(n_models)
    width = 0.35
    names = [r["name"] for r in results_list]

    for ax, metric in zip(axes, metrics):
        train_vals = [r["train"].get(metric, 0.0) for r in results_list]
        test_vals = [r["test"].get(metric, 0.0) for r in results_list]

        bars_train = ax.bar(x - width / 2, train_vals, width, label="Train",
                            color=t["bar_train"], alpha=0.85, edgecolor=t["spine"])
        bars_test = ax.bar(x + width / 2, test_vals, width, label="Test",
                           color=t["bar_test"], alpha=0.85, edgecolor=t["spine"])

        # Annotate bars
        for bar in bars_train:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.2f}",
                    ha="center", va="bottom", fontsize=7.5, color=t["text"])
        for bar in bars_test:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.2f}",
                    ha="center", va="bottom", fontsize=7.5, color=t["text"])

        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8.5)
        ax.set_ylim(0, 1.12)
        ax.set_ylabel(metric_labels.get(metric, metric), fontsize=10)
        ax.set_title(metric_labels.get(metric, metric), fontsize=11, fontweight="bold")
        legend = ax.legend(fontsize=9, framealpha=0.3, facecolor=t["panel_bg"],
                           edgecolor=t["spine"], labelcolor=t["text"])

    _apply_figure_style(fig, axes, t)
    fig.suptitle("Comparación de Modelos", fontsize=13, fontweight="bold", color=t["text"], y=1.02)
    fig.tight_layout()
    return fig
