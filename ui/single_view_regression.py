import streamlit as st
import numpy as np
from core.train_regression import fit_and_evaluate_regression, compute_regression_pedagogical_signals
from core.visualization_regression import plot_prediction_curve, plot_residuals, plot_prediction_error
from core.models_regression import get_regression_model_instance


def render_single_view_regression(cfg: dict, X_train, X_test, y_train, y_test) -> None:
    """
    Render the 'Explorar' tab for regression: prediction curve + metrics.

    Parameters
    ----------
    cfg : dict from sidebar.render_sidebar()
    X_train, X_test, y_train, y_test : data splits
    """
    model_name = cfg["model_name"]
    theme = cfg["theme"]
    hyperparams = cfg["hyperparams"]

    model = get_regression_model_instance(model_name, hyperparams)

    with st.spinner("Entrenando modelo..."):
        result = fit_and_evaluate_regression(model, X_train, y_train, X_test, y_test)

    # ── Pedagogical signals ─────────────────────────────────────────────
    signals = compute_regression_pedagogical_signals(result["train"], result["test"])
    for sig in signals:
        if sig["level"] == "warning":
            st.warning(sig["message"])
        elif sig["level"] == "info":
            st.info(sig["message"])
        elif sig["level"] == "success":
            st.success(sig["message"])
        elif sig["level"] == "tip":
            st.info(f"💡 {sig['message']}")

    # ── Main layout: 2 columns ──────────────────────────────────────────
    col_curve, col_metrics = st.columns([1.1, 0.9], gap="medium")

    with col_curve:
        st.markdown("#### Curva de Predicción")
        fig_curve = plot_prediction_curve(
            result["model"],
            X_train, X_test,
            y_train, y_test,
            show_train=cfg.get("show_train", True),
            show_test=cfg.get("show_test", True),
            theme=theme,
        )
        st.pyplot(fig_curve, width='stretch')

    with col_metrics:
        st.markdown("#### Métricas de Rendimiento")
        _render_regression_metrics_table(result["train"], result["test"], theme)

        st.markdown("**Error de Predicción — Test**")
        fig_pe = plot_prediction_error(
            result["test"]["y_true"],
            result["test"]["y_pred"],
            theme=theme,
            title_suffix="— Test",
        )
        st.pyplot(fig_pe, width='stretch')

    # ── Residuals plots ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Análisis de Residuos")
    col_res_train, col_res_test = st.columns(2, gap="medium")

    with col_res_train:
        fig_res_tr = plot_residuals(
            result["train"]["y_true"],
            result["train"]["y_pred"],
            theme=theme,
            title_suffix="— Train",
        )
        st.pyplot(fig_res_tr, width='stretch')

    with col_res_test:
        fig_res_te = plot_residuals(
            result["test"]["y_true"],
            result["test"]["y_pred"],
            theme=theme,
            title_suffix="— Test",
        )
        st.pyplot(fig_res_te, width='stretch')


def _render_regression_metrics_table(
    train_metrics: dict, test_metrics: dict, theme: str
) -> None:
    """Render a clean side-by-side regression metrics comparison."""
    is_dark = theme == "dark"
    bg = "#161b27" if is_dark else "#f4f6f9"
    text = "#e8eaf0" if is_dark else "#1a1d23"
    subtext = "#9da5b4" if is_dark else "#5a6270"
    accent = "#4c9be8" if is_dark else "#2176ae"
    warn_color = "#ffd740" if is_dark else "#b45309"
    good_color = "#69f0ae" if is_dark else "#1b5e20"
    border = "#252d3d" if is_dark else "#dde1e9"

    metrics_display = [
        ("R²", "r2", True),
        ("RMSE", "rmse", False),
        ("MAE", "mae", False),
    ]

    header_style = (
        f"background:{bg}; padding:6px 10px; font-size:0.8rem; "
        f"color:{subtext}; font-weight:600; border-bottom:2px solid {border};"
    )
    cell_style = (
        f"padding:6px 10px; font-size:0.9rem; color:{text}; "
        f"border-bottom:1px solid {border};"
    )

    rows_html = ""
    for label, key, higher_is_better in metrics_display:
        train_val = train_metrics.get(key, 0.0)
        test_val = test_metrics.get(key, 0.0)

        train_str = f"<code style='background:transparent;color:{accent};'>{train_val:.4f}</code>"

        # Color coding for test value
        if higher_is_better:
            gap = train_val - test_val
            if gap > 0.15:
                test_color = warn_color
                test_icon = "⚠️ "
            elif test_val >= 0.85 and gap <= 0.05:
                test_color = good_color
                test_icon = "✓ "
            elif test_val < 0.30:
                test_color = warn_color
                test_icon = "↓ "
            else:
                test_color = text
                test_icon = ""
        else:
            # Lower is better (RMSE, MAE)
            ratio = test_val / max(train_val, 1e-10)
            if ratio > 2.0:
                test_color = warn_color
                test_icon = "⚠️ "
            elif ratio <= 1.1 and test_val < 0.5:
                test_color = good_color
                test_icon = "✓ "
            else:
                test_color = text
                test_icon = ""

        test_str = (
            f"<span style='color:{test_color};font-weight:600;'>"
            f"{test_icon}{test_val:.4f}</span>"
        )
        rows_html += f"""
        <tr>
          <td style='{cell_style} color:{subtext};'>{label}</td>
          <td style='{cell_style} text-align:center;'>{train_str}</td>
          <td style='{cell_style} text-align:center;'>{test_str}</td>
        </tr>"""

    table_html = f"""
    <table style='width:100%; border-collapse:collapse; background:{bg};
           border-radius:8px; overflow:hidden; margin-bottom:12px;'>
      <thead>
        <tr>
          <th style='{header_style} text-align:left;'>Métrica</th>
          <th style='{header_style} text-align:center;'>Train</th>
          <th style='{header_style} text-align:center;'>Test</th>
        </tr>
      </thead>
      <tbody>{rows_html}
      </tbody>
    </table>"""
    st.markdown(table_html, unsafe_allow_html=True)
