import streamlit as st
import numpy as np
from core.train_regression import fit_and_evaluate_regression, compute_regression_pedagogical_signals
from core.visualization_regression import (
    plot_prediction_curve,
    plot_regression_metrics_comparison,
)
from core.models_regression import get_regression_model_instance, MODEL_REGISTRY_REGRESSION


def render_compare_view_regression(cfg: dict, X_train, X_test, y_train, y_test) -> None:
    """
    Render the 'Comparar' tab for regression: side-by-side prediction curves + bar chart.

    Parameters
    ----------
    cfg : dict from sidebar.render_sidebar()
    X_train, X_test, y_train, y_test : data splits
    """
    compare_models = cfg.get("compare_models", [])
    theme = cfg["theme"]

    if not compare_models:
        st.info("Selecciona al menos un modelo en la sección **🆚 Comparación** del sidebar para comenzar.")
        return

    # ── Train all selected models ───────────────────────────────────────
    results_list = []
    with st.spinner(f"Entrenando {len(compare_models)} modelo(s)..."):
        for model_name in compare_models:
            cfg_model = MODEL_REGISTRY_REGRESSION[model_name]
            model = get_regression_model_instance(model_name, cfg_model["default_params"])
            result = fit_and_evaluate_regression(model, X_train, y_train, X_test, y_test)
            results_list.append({
                "name": model_name,
                "model": result["model"],
                "train": result["train"],
                "test": result["test"],
            })

    # ── Prediction curve grid ────────────────────────────────────────────
    st.markdown("#### Curvas de Predicción")
    n_models = len(results_list)
    n_cols = min(n_models, 3)
    rows = (n_models + n_cols - 1) // n_cols

    model_idx = 0
    for _ in range(rows):
        cols = st.columns(n_cols, gap="small")
        for col_idx in range(n_cols):
            if model_idx >= n_models:
                break
            res = results_list[model_idx]
            with cols[col_idx]:
                st.markdown(f"**{res['name']}**")
                r2_train = res["train"]["r2"]
                r2_test = res["test"]["r2"]
                gap = r2_train - r2_test
                badge = "🔴" if gap > 0.15 else ("🟡" if gap > 0.08 else "🟢")
                st.caption(
                    f"{badge} R² Train: `{r2_train:.3f}` · Test: `{r2_test:.3f}`"
                )
                fig = plot_prediction_curve(
                    res["model"],
                    X_train, X_test,
                    y_train, y_test,
                    show_train=cfg.get("show_train", True),
                    show_test=cfg.get("show_test", True),
                    theme=theme,
                )
                st.pyplot(fig, width='stretch')
            model_idx += 1

    st.markdown("---")

    # ── Metrics comparison bar chart ─────────────────────────────────────
    st.markdown("#### Comparación de Métricas")
    fig_metrics = plot_regression_metrics_comparison(results_list, theme=theme)
    st.pyplot(fig_metrics, width='stretch')

    st.markdown("---")

    # ── Summary table ────────────────────────────────────────────────────
    st.markdown("#### Tabla Resumen")
    _render_regression_comparison_table(results_list)

    # ── Pedagogical signals per model ────────────────────────────────────
    any_signal = False
    for res in results_list:
        signals = compute_regression_pedagogical_signals(res["train"], res["test"])
        if signals:
            if not any_signal:
                st.markdown("#### Señales Pedagógicas")
                any_signal = True
            st.markdown(f"**{res['name']}**")
            for sig in signals:
                if sig["level"] == "warning":
                    st.warning(sig["message"])
                elif sig["level"] == "info":
                    st.info(sig["message"])
                elif sig["level"] == "success":
                    st.success(sig["message"])


def _render_regression_comparison_table(results_list: list[dict]) -> None:
    """Render a clean summary table of all regression model metrics."""
    header_cols = st.columns([1.6, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
    headers = ["Modelo", "R² Train", "R² Test", "RMSE Train", "RMSE Test", "MAE Train", "MAE Test"]
    for col, h in zip(header_cols, headers):
        col.markdown(f"**{h}**")

    st.markdown("<hr style='margin:4px 0'>", unsafe_allow_html=True)

    for res in results_list:
        tr = res["train"]
        te = res["test"]
        gap = tr["r2"] - te["r2"]
        badge = "🔴" if gap > 0.15 else ("🟡" if gap > 0.08 else "🟢")
        row = st.columns([1.6, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
        row[0].markdown(f"{badge} {res['name']}")
        row[1].markdown(f"`{tr['r2']:.3f}`")
        row[2].markdown(f"`{te['r2']:.3f}`")
        row[3].markdown(f"`{tr['rmse']:.3f}`")
        row[4].markdown(f"`{te['rmse']:.3f}`")
        row[5].markdown(f"`{tr['mae']:.3f}`")
        row[6].markdown(f"`{te['mae']:.3f}`")
