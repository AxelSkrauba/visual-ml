import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split

from core.datasets import generate_dataset
from ui.sidebar import render_sidebar
from ui.single_view import render_single_view
from ui.compare_view import render_compare_view


# ── Page configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Visual ML",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Theme CSS injection ───────────────────────────────────────────────────────
def _inject_theme_css(theme: str) -> None:
    if theme == "dark":
        bg = "#0e1117"
        surface = "#161b27"
        surface2 = "#1e2538"
        text = "#e8eaf0"
        subtext = "#9da5b4"
        accent = "#4c9be8"
        accent2 = "#3a7bc8"
        border = "#252d3d"
        border2 = "#3a4358"
        input_bg = "#1e2538"
        warning_bg = "#2d2208"
        warning_text = "#ffd740"
        info_bg = "#0d1f35"
        info_text = "#4fc3f7"
        success_bg = "#0d2b1a"
        success_text = "#69f0ae"
    else:
        bg = "#ffffff"
        surface = "#f4f6f9"
        surface2 = "#eaecf0"
        text = "#1a1d23"
        subtext = "#5a6270"
        accent = "#2176ae"
        accent2 = "#1a5e8a"
        border = "#dde1e9"
        border2 = "#b0b8c8"
        input_bg = "#ffffff"
        warning_bg = "#fff8e1"
        warning_text = "#6d4c00"
        info_bg = "#e3f2fd"
        info_text = "#0d47a1"
        success_bg = "#e8f5e9"
        success_text = "#1b5e20"

    # Toolbar bg/text for the Streamlit chrome (rendered at body level, not inside .stApp)
    toolbar_bg = "#161b27" if theme == "dark" else "#ffffff"
    toolbar_text = "#e8eaf0" if theme == "dark" else "#1a1d23"
    toolbar_border = "#252d3d" if theme == "dark" else "#dde1e9"
    # Image overlay (fullscreen button) colors
    overlay_bg = "#1e2538" if theme == "dark" else "#f0f2f6"
    overlay_text = "#e8eaf0" if theme == "dark" else "#1a1d23"

    css = f"""
    <style>
        /* ══════════════════════════════════════════════════════════════
           BODY-LEVEL: covers portals rendered OUTSIDE .stApp
           (tooltips, dropdowns, number_input step buttons, toolbar,
            image fullscreen overlay — all injected directly into <body>)
           ══════════════════════════════════════════════════════════════ */

        /* Base body */
        body {{
            background-color: {bg} !important;
            color: {text} !important;
        }}

        /* ── Streamlit toolbar (top bar with Deploy, ☰ menu, status) ── */
        [data-testid="stToolbar"],
        header[data-testid="stHeader"],
        header[data-testid="stHeader"] * {{
            background-color: {toolbar_bg} !important;
            color: {toolbar_text} !important;
            border-bottom-color: {toolbar_border} !important;
        }}
        /* Status indicator (running/stopped) text */
        [data-testid="stStatusWidget"] span,
        [data-testid="stStatusWidget"] label {{
            color: {toolbar_text} !important;
        }}
        /* Hamburger / Deploy button in toolbar */
        [data-testid="stToolbarActions"] button,
        [data-testid="stToolbarActions"] button svg,
        header button, header button svg {{
            color: {toolbar_text} !important;
            fill: {toolbar_text} !important;
        }}
        header a, header a span {{
            color: {toolbar_text} !important;
        }}

        /* ── Tooltips (rendered as body-level portal by baseweb) ── */
        [data-baseweb="tooltip"] [role="tooltip"],
        div[role="tooltip"] {{
            background-color: {surface2} !important;
            color: {text} !important;
            border: 1px solid {border2} !important;
            border-radius: 6px !important;
            padding: 6px 10px !important;
        }}
        [data-baseweb="tooltip"] [role="tooltip"] *,
        div[role="tooltip"] * {{
            color: {text} !important;
            background-color: {surface2} !important;
        }}
        /* Arrow of tooltip */
        [data-baseweb="tooltip"] [data-popper-arrow]::before,
        [data-baseweb="tooltip"] [data-popper-arrow] {{
            background-color: {surface2} !important;
            border-color: {border2} !important;
        }}

        /* ── Dropdown / selectbox list (body portal) ── */
        [data-baseweb="popover"] [data-baseweb="menu"],
        [data-baseweb="popover"] ul[role="listbox"] {{
            background-color: {surface2} !important;
            border: 1px solid {border2} !important;
            border-radius: 6px !important;
            max-height: 320px !important;
        }}
        [data-baseweb="popover"] li[role="option"],
        [data-baseweb="popover"] li {{
            background-color: {surface2} !important;
            color: {text} !important;
        }}
        [data-baseweb="popover"] li[role="option"]:hover,
        [data-baseweb="popover"] li[aria-selected="true"] {{
            background-color: {border2} !important;
            color: {text} !important;
        }}

        /* ── Number input step buttons (+/-) — only the stepper buttons, NOT the help icon ── */
        [data-testid="stNumberInput"] [data-testid="stNumberInputStepDown"],
        [data-testid="stNumberInput"] [data-testid="stNumberInputStepUp"],
        [data-baseweb="input"] > div > button {{
            background-color: {surface2} !important;
            color: {text} !important;
            border-color: {border2} !important;
        }}
        [data-testid="stNumberInput"] [data-testid="stNumberInputStepDown"]:hover,
        [data-testid="stNumberInput"] [data-testid="stNumberInputStepUp"]:hover {{
            background-color: {border2} !important;
        }}
        [data-testid="stNumberInput"] [data-testid="stNumberInputStepDown"] svg,
        [data-testid="stNumberInput"] [data-testid="stNumberInputStepUp"] svg {{
            fill: {text} !important;
        }}

        /* ── Help/tooltip trigger icon (?) — SVG fill for visibility ── */
        [data-testid="stWidgetHelp"] button,
        [data-testid="stTooltipIcon"] button,
        button[data-testid="tooltipHoverTarget"] {{
            background: transparent !important;
            border: none !important;
        }}
        [data-testid="stWidgetHelp"] svg,
        [data-testid="stTooltipIcon"] svg,
        button[data-testid="tooltipHoverTarget"] svg,
        [data-testid="stWidgetHelp"] path,
        [data-testid="stTooltipIcon"] path,
        button[data-testid="tooltipHoverTarget"] path {{
            fill: {subtext} !important;
            color: {subtext} !important;
        }}

        /* ── Image fullscreen overlay (body portal) ── */
        [data-testid="stImageContainer"] button,
        button[title="View fullscreen"],
        button[aria-label="Fullscreen"] {{
            background-color: {overlay_bg} !important;
            color: {overlay_text} !important;
            border: 1px solid {border2} !important;
            border-radius: 4px !important;
            opacity: 1 !important;
        }}
        button[title="View fullscreen"] svg,
        button[aria-label="Fullscreen"] svg {{
            fill: {overlay_text} !important;
        }}
        /* Plot toolbar buttons (matplotlib figure toolbar) */
        [data-testid="stImage"] ~ div button,
        .element-container button {{
            color: {overlay_text} !important;
        }}

        /* ══════════════════════════════════════════════════════════════
           .stApp-LEVEL RULES
           ══════════════════════════════════════════════════════════════ */

        /* ── Global app background + default text ── */
        .stApp {{
            background-color: {bg} !important;
            background-image: none !important;
        }}
        .stApp * {{
            color: {text};
        }}
        /* Remove diagonal-line artifact */
        .stApp::before, .stApp::after {{
            display: none !important;
        }}

        /* ── Main content area ── */
        .main .block-container {{
            background-color: {bg};
            padding-top: 1.5rem;
        }}

        /* ── Sidebar ── */
        section[data-testid="stSidebar"] {{
            background-color: {surface};
            border-right: 1px solid {border2};
        }}
        section[data-testid="stSidebar"] * {{
            color: {text} !important;
        }}

        /* ── Headings and paragraphs ── */
        h1, h2, h3, h4, h5, h6 {{
            color: {text} !important;
        }}
        p, li {{
            color: {text};
        }}

        /* ── Metrics widget ── */
        [data-testid="stMetric"] {{
            background-color: {surface};
            border: 1px solid {border};
            border-radius: 8px;
            padding: 10px 14px;
        }}
        [data-testid="stMetricLabel"] > div,
        [data-testid="stMetricLabel"] span {{
            color: {subtext} !important;
            font-size: 0.82rem;
        }}
        [data-testid="stMetricValue"] > div,
        [data-testid="stMetricValue"] span {{
            color: {text} !important;
            font-size: 1.3rem;
            font-weight: 600;
        }}

        /* ── Markdown / Text ── */
        .stMarkdown, .stMarkdown p, .stMarkdown li {{
            color: {text};
        }}
        .stCaption, [data-testid="stCaptionContainer"] p {{
            color: {subtext} !important;
        }}

        /* ── Input widgets ── */
        [data-baseweb="select"] > div {{
            background-color: {input_bg} !important;
            border-color: {border2} !important;
        }}
        [data-baseweb="select"] span,
        [data-baseweb="select"] div {{
            color: {text} !important;
        }}
        .stSlider [data-testid="stWidgetLabel"] p,
        .stSelectbox [data-testid="stWidgetLabel"] p,
        .stNumberInput [data-testid="stWidgetLabel"] p,
        .stCheckbox [data-testid="stWidgetLabel"] p,
        .stMultiSelect [data-testid="stWidgetLabel"] p {{
            color: {text} !important;
        }}
        .stSlider [data-testid="stTickBarMin"],
        .stSlider [data-testid="stTickBarMax"] {{
            color: {subtext};
        }}

        /* ── Number input field ── */
        input[type="number"] {{
            background-color: {input_bg} !important;
            color: {text} !important;
            border-color: {border2} !important;
        }}

        /* ── Multiselect tags ── */
        [data-baseweb="tag"] {{
            background-color: {accent2} !important;
            color: #ffffff !important;
        }}

        /* ── Tabs ── */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 4px;
            background-color: {surface};
            border-radius: 8px;
            padding: 4px;
            border: 1px solid {border};
        }}
        .stTabs [data-baseweb="tab"] {{
            color: {text} !important;
            border-radius: 6px;
            padding: 6px 18px;
            font-weight: 500;
        }}
        .stTabs [aria-selected="true"] {{
            background-color: {accent} !important;
            color: #ffffff !important;
        }}

        /* ── Expander ── */
        [data-testid="stExpander"] {{
            background-color: {surface};
            border: 1px solid {border};
            border-radius: 8px;
        }}
        [data-testid="stExpander"] summary span,
        [data-testid="stExpander"] div {{
            color: {text} !important;
        }}

        /* ── Code blocks ── */
        code, pre {{
            background-color: {surface2} !important;
            color: {accent} !important;
            border-radius: 4px;
        }}
        pre code {{
            color: {text} !important;
        }}

        /* ── Alert / notification boxes ── */
        div[data-testid="stNotification"],
        div[role="alert"] {{
            color: {text} !important;
            background-color: {surface2} !important;
            border: 1px solid {border2} !important;
        }}
        div[data-testid="stNotification"] p,
        div[role="alert"] p,
        div[role="alert"] span,
        .stAlert [data-testid="stMarkdownContainer"] p {{
            color: {text} !important;
        }}

        /* ── Dividers ── */
        hr {{
            border-color: {border2} !important;
            border-width: 1px 0 0 0 !important;
            border-style: solid !important;
            border-image: none !important;
            background: none !important;
        }}

        /* ── Spinner ── */
        [data-testid="stSpinner"] p {{
            color: {subtext};
        }}

        /* ── Checkbox label ── */
        [data-testid="stCheckbox"] label span {{
            color: {text} !important;
        }}

        /* ── Primary button ── */
        .stButton > button {{
            background-color: {accent};
            color: #ffffff !important;
            border: none;
            border-radius: 6px;
            font-weight: 500;
        }}
        .stButton > button:hover {{
            background-color: {accent2};
            color: #ffffff !important;
        }}

        /* ── Footer ── */
        footer, footer * {{
            color: {subtext} !important;
        }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# ── Cached data generation ───────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _get_dataset(dataset_name, n_samples, noise, n_classes, factor, cluster_std, imbalance_ratio, random_seed):
    return generate_dataset(
        name=dataset_name,
        n_samples=n_samples,
        random_state=random_seed,
        noise=noise,
        n_classes=n_classes,
        factor=factor,
        cluster_std=cluster_std,
        imbalance_ratio=imbalance_ratio,
    )


# ── Main app ─────────────────────────────────────────────────────────────────
def main():
    cfg = render_sidebar()

    theme = cfg["theme"]
    _inject_theme_css(theme)

    # ── Header ───────────────────────────────────────────────────────────
    text_color = "#e8eaf0" if theme == "dark" else "#1a1d23"
    subtext_color = "#9da5b4" if theme == "dark" else "#5a6270"
    st.markdown(
        f"<h1 style='margin-bottom:0; color:{text_color};'>🧠 Visual ML</h1>"
        f"<p style='margin-top:4px; color:{subtext_color}; font-size:1rem;'>"
        f"Exploración interactiva de fronteras de decisión · Modelos clásicos de Machine Learning</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ── Data generation ───────────────────────────────────────────────────
    X, y = _get_dataset(
        cfg["dataset_name"],
        cfg["n_samples"],
        cfg["noise"],
        cfg["n_classes"],
        cfg["factor"],
        cfg["cluster_std"],
        cfg["imbalance_ratio"],
        cfg["random_seed"],
    )

    test_size = 1.0 - cfg["train_test_split"] / 100.0
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=cfg["random_seed"],
        stratify=y if len(np.unique(y)) > 1 else None,
    )

    # ── Dataset info bar ─────────────────────────────────────────────────
    from core.datasets import DATASET_CONFIGS
    ds_cfg = DATASET_CONFIGS[cfg["dataset_name"]]
    n_train, n_test = len(y_train), len(y_test)
    n_classes_actual = len(np.unique(y))

    info_cols = st.columns(5)
    info_cols[0].metric("Dataset", ds_cfg["label"])
    info_cols[1].metric("Muestras", cfg["n_samples"])
    info_cols[2].metric("Train / Test", f"{n_train} / {n_test}")
    info_cols[3].metric("Clases", n_classes_actual)
    if cfg["dataset_name"] == "imbalanced":
        minority_pct = cfg["imbalance_ratio"] * 100
        info_cols[4].metric("Minoría", f"{minority_pct:.0f}%")
    else:
        info_cols[4].metric("Ruido", f"{cfg['noise']:.2f}")

    st.markdown("---")

    # ── Tabs ─────────────────────────────────────────────────────────────
    tab_explore, tab_compare = st.tabs(["🔬 Explorar Modelo", "🆚 Comparar Modelos"])

    with tab_explore:
        if cfg["auto_update"]:
            render_single_view(cfg, X_train, X_test, y_train, y_test)
        else:
            if st.button("▶️ Entrenar y visualizar", type="primary", use_container_width=False):
                render_single_view(cfg, X_train, X_test, y_train, y_test)
            else:
                st.info("Presiona **▶️ Entrenar y visualizar** para ver los resultados (modo manual activo).")

    with tab_compare:
        if cfg["auto_update"]:
            render_compare_view(cfg, X_train, X_test, y_train, y_test)
        else:
            if st.button("▶️ Comparar modelos", type="primary", use_container_width=False, key="btn_compare"):
                render_compare_view(cfg, X_train, X_test, y_train, y_test)
            else:
                st.info("Presiona **▶️ Comparar modelos** para ver los resultados (modo manual activo).")


if __name__ == "__main__":
    main()
