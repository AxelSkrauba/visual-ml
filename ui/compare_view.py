import streamlit as st
import numpy as np
from core.train import fit_and_evaluate, compute_pedagogical_signals
from core.visualization import plot_decision_boundary, plot_metrics_comparison
from core.models import get_model_instance, MODEL_REGISTRY


def render_compare_view(cfg: dict, X_train, X_test, y_train, y_test) -> None:
    """
    Render the 'Comparar' tab: side-by-side decision boundaries + bar chart comparison.

    Parameters
    ----------
    cfg : dict from sidebar.render_sidebar()
    X_train, X_test, y_train, y_test : data splits
    """
    compare_models = cfg.get("compare_models", [])
    theme = cfg["theme"]
    n_classes = len(np.unique(np.concatenate([y_train, y_test])))
    labels = [str(i) for i in range(n_classes)]

    if not compare_models:
        st.info("Selecciona al menos un modelo en la sección **🆚 Comparación** del sidebar para comenzar.")
        return

    # ── Train all selected models ───────────────────────────────────────
    results_list = []
    with st.spinner(f"Entrenando {len(compare_models)} modelo(s)..."):
        for model_name in compare_models:
            cfg_model = MODEL_REGISTRY[model_name]
            model = get_model_instance(model_name, cfg_model["default_params"])
            result = fit_and_evaluate(model, X_train, y_train, X_test, y_test)
            results_list.append({
                "name": model_name,
                "model": result["model"],
                "train": result["train"],
                "test": result["test"],
            })

    # ── Decision boundary grid ──────────────────────────────────────────
    st.markdown("#### Fronteras de Decisión")
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
                acc_train = res["train"]["accuracy"]
                acc_test = res["test"]["accuracy"]
                gap = acc_train - acc_test
                badge = "🔴" if gap > 0.15 else ("🟡" if gap > 0.08 else "🟢")
                st.caption(
                    f"{badge} Train: `{acc_train:.2%}` · Test: `{acc_test:.2%}`"
                )
                fig = plot_decision_boundary(
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

    # ── Metrics comparison bar chart ────────────────────────────────────
    st.markdown("#### Comparación de Métricas")
    fig_metrics = plot_metrics_comparison(results_list, theme=theme, metrics=("accuracy", "f1"))
    st.pyplot(fig_metrics, width='stretch')

    st.markdown("---")

    # ── Summary table ────────────────────────────────────────────────────
    st.markdown("#### Tabla Resumen")
    _render_comparison_table(results_list)

    # ── Pedagogical signals per model ────────────────────────────────────
    any_signal = False
    for res in results_list:
        signals = compute_pedagogical_signals(res["train"], res["test"])
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


def _render_comparison_table(results_list: list[dict]) -> None:
    """Render a clean summary table of all model metrics."""
    header_cols = st.columns([1.6, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
    headers = ["Modelo", "Acc Train", "Acc Test", "F1 Train", "F1 Test", "Prec Test", "Rec Test"]
    for col, h in zip(header_cols, headers):
        col.markdown(f"**{h}**")

    st.markdown("<hr style='margin:4px 0'>", unsafe_allow_html=True)

    for res in results_list:
        tr = res["train"]
        te = res["test"]
        gap = tr["accuracy"] - te["accuracy"]
        badge = "🔴" if gap > 0.15 else ("🟡" if gap > 0.08 else "🟢")
        row = st.columns([1.6, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
        row[0].markdown(f"{badge} {res['name']}")
        row[1].markdown(f"`{tr['accuracy']:.3f}`")
        row[2].markdown(f"`{te['accuracy']:.3f}`")
        row[3].markdown(f"`{tr['f1']:.3f}`")
        row[4].markdown(f"`{te['f1']:.3f}`")
        row[5].markdown(f"`{te['precision']:.3f}`")
        row[6].markdown(f"`{te['recall']:.3f}`")
