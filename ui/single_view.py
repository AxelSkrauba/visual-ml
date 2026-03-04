import streamlit as st
import numpy as np
import re
from core.train import fit_and_evaluate, compute_pedagogical_signals
from core.visualization import plot_decision_boundary, plot_confusion_matrix
from core.models import get_model_instance


def render_single_view(cfg: dict, X_train, X_test, y_train, y_test) -> None:
    """
    Render the 'Explorar' tab: decision boundary + metrics for a single model.

    Parameters
    ----------
    cfg : dict from sidebar.render_sidebar()
    X_train, X_test, y_train, y_test : data splits
    """
    model_name = cfg["model_name"]
    theme = cfg["theme"]
    hyperparams = cfg["hyperparams"]

    model = get_model_instance(model_name, hyperparams)

    with st.spinner("Entrenando modelo..."):
        result = fit_and_evaluate(model, X_train, y_train, X_test, y_test)

    n_classes = len(np.unique(np.concatenate([y_train, y_test])))

    # ── Pedagogical signals ─────────────────────────────────────────────
    signals = compute_pedagogical_signals(result["train"], result["test"])
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
    col_boundary, col_metrics = st.columns([1.1, 0.9], gap="medium")

    with col_boundary:
        st.markdown("#### Frontera de Decisión")
        fig_boundary = plot_decision_boundary(
            result["model"],
            X_train, X_test,
            y_train, y_test,
            show_train=cfg["show_train"],
            show_test=cfg["show_test"],
            theme=theme,
        )
        st.pyplot(fig_boundary, width='stretch')

    with col_metrics:
        st.markdown("#### Métricas de Rendimiento")
        _render_metrics_table(result["train"], result["test"], theme)

        labels = [str(i) for i in range(n_classes)]

        st.markdown("**Matriz de Confusión — Test**")
        fig_cm = plot_confusion_matrix(
            result["test"]["confusion_matrix"],
            labels=labels,
            theme=theme,
            normalize=cfg.get("cm_normalize", False),
        )
        st.pyplot(fig_cm, width='stretch')

        with st.expander("📋 Classification Report", expanded=False):
            st.markdown("**Train**")
            _render_classification_report(result["train"]["report"], theme)
            st.markdown("**Test**")
            _render_classification_report(result["test"]["report"], theme)


def _render_metrics_table(train_metrics: dict, test_metrics: dict, theme: str) -> None:
    """Render a clean side-by-side metrics comparison with color-coded gaps."""
    is_dark = theme == "dark"
    bg = "#161b27" if is_dark else "#f4f6f9"
    text = "#e8eaf0" if is_dark else "#1a1d23"
    subtext = "#9da5b4" if is_dark else "#5a6270"
    accent = "#4c9be8" if is_dark else "#2176ae"
    warn_color = "#ffd740" if is_dark else "#b45309"
    good_color = "#69f0ae" if is_dark else "#1b5e20"
    border = "#252d3d" if is_dark else "#dde1e9"

    metrics_display = [
        ("Accuracy", "accuracy"),
        ("Precisión", "precision"),
        ("Recall", "recall"),
        ("F1-score", "f1"),
    ]

    header_style = f"background:{bg}; padding:6px 10px; font-size:0.8rem; color:{subtext}; font-weight:600; border-bottom:2px solid {border};"
    cell_style = f"padding:6px 10px; font-size:0.9rem; color:{text}; border-bottom:1px solid {border};"

    rows_html = ""
    for label, key in metrics_display:
        train_val = train_metrics.get(key, 0.0)
        test_val = test_metrics.get(key, 0.0)
        gap = train_val - test_val

        train_str = f"<code style='background:transparent;color:{accent};'>{train_val:.3f}</code>"

        if gap > 0.15:
            test_color = warn_color
            test_icon = "⚠️ "
        elif test_val >= 0.85 and gap <= 0.05:
            test_color = good_color
            test_icon = "✓ "
        elif test_val < 0.60:
            test_color = warn_color
            test_icon = "↓ "
        else:
            test_color = text
            test_icon = ""

        test_str = f"<span style='color:{test_color};font-weight:600;'>{test_icon}{test_val:.3f}</span>"
        rows_html += f"""
        <tr>
          <td style='{cell_style} color:{subtext};'>{label}</td>
          <td style='{cell_style} text-align:center;'>{train_str}</td>
          <td style='{cell_style} text-align:center;'>{test_str}</td>
        </tr>"""

    table_html = f"""
    <table style='width:100%; border-collapse:collapse; background:{bg}; border-radius:8px; overflow:hidden; margin-bottom:12px;'>
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


def _render_classification_report(report_str: str, theme: str) -> None:
    """Parse sklearn classification report string into a styled HTML table."""
    is_dark = theme == "dark"
    bg = "#161b27" if is_dark else "#f4f6f9"
    bg2 = "#1e2538" if is_dark else "#eaecf0"
    text = "#e8eaf0" if is_dark else "#1a1d23"
    subtext = "#9da5b4" if is_dark else "#5a6270"
    accent = "#4c9be8" if is_dark else "#2176ae"
    warn_color = "#ff5252" if is_dark else "#c62828"
    good_color = "#69f0ae" if is_dark else "#2e7d32"
    border = "#252d3d" if is_dark else "#dde1e9"

    lines = [l for l in report_str.strip().split("\n") if l.strip()]
    if len(lines) < 2:
        st.code(report_str, language=None)
        return

    # Header row
    header_line = lines[0]
    headers = ["Clase"] + re.split(r"\s{2,}", header_line.strip())

    rows = []
    for line in lines[1:]:
        parts = re.split(r"\s{2,}", line.strip())
        if len(parts) >= 4:
            rows.append(parts)

    header_style = (
        f"padding:6px 10px; font-size:0.78rem; color:{subtext}; "
        f"font-weight:700; background:{bg2}; border-bottom:2px solid {border}; text-align:center;"
    )
    th0_style = header_style.replace("text-align:center", "text-align:left")

    def _cell_color(val_str: str) -> str:
        try:
            v = float(val_str)
            if v < 0.60:
                return warn_color
            elif v >= 0.90:
                return good_color
            return text
        except ValueError:
            return text

    def _is_summary(label: str) -> bool:
        return any(k in label.lower() for k in ("accuracy", "avg", "weighted", "macro"))

    thead = f"<tr><th style='{th0_style}'>{headers[0]}</th>"
    for h in headers[1:]:
        thead += f"<th style='{header_style}'>{h}</th>"
    thead += "</tr>"

    tbody = ""
    for row in rows:
        label = row[0]
        values = row[1:]
        is_sum = _is_summary(label)
        row_bg = bg2 if is_sum else bg
        label_style = (
            f"padding:5px 10px; font-size:0.82rem; color:{subtext if is_sum else text}; "
            f"font-style:{'italic' if is_sum else 'normal'}; font-weight:{'600' if is_sum else '400'}; "
            f"background:{row_bg}; border-bottom:1px solid {border};"
        )
        tbody += f"<tr><td style='{label_style}'>{label}</td>"
        for i, v in enumerate(values):
            c = _cell_color(v) if not _is_summary(label) else text
            if _is_summary(label) and i == len(values) - 1:
                c = subtext
            val_style = (
                f"padding:5px 10px; font-size:0.82rem; color:{c}; font-weight:600; "
                f"text-align:center; background:{row_bg}; border-bottom:1px solid {border};"
            )
            tbody += f"<td style='{val_style}'>{v}</td>"
        tbody += "</tr>"

    table_html = f"""
    <div style='overflow-x:auto; margin-bottom:8px;'>
    <table style='width:100%; border-collapse:collapse; background:{bg}; border-radius:8px; overflow:hidden;'>
      <thead>{thead}</thead>
      <tbody>{tbody}</tbody>
    </table>
    </div>"""
    st.markdown(table_html, unsafe_allow_html=True)
