import math
import streamlit as st
from core.datasets import DATASET_CONFIGS
from core.models import (
    MODEL_REGISTRY,
    get_hyperparameter_controls,
)
from core.datasets_regression import REGRESSION_DATASET_CONFIGS
from core.models_regression import (
    MODEL_REGISTRY_REGRESSION,
    get_regression_hyperparameter_controls,
)


def render_sidebar() -> dict:
    """
    Render all sidebar controls and return the current configuration as a dict.

    Returns
    -------
    dict with keys:
        paradigm, theme, auto_update,
        dataset_name, n_samples, noise, n_classes, factor, cluster_std, train_test_split, random_seed,
        model_name, hyperparams,
        show_train, show_test,
        compare_models
        (regression adds: degree, frequency, n_steps)
    """
    with st.sidebar:
        st.markdown("## ⚙️ Visual ML")
        st.markdown("---")

        # ── Paradigm selector ────────────────────────────────────────────
        paradigm = st.radio(
            "Paradigma",
            options=["Clasificación", "Regresión"],
            index=0,
            key="paradigm",
            horizontal=True,
            help="Clasificación: fronteras de decisión. Regresión: curvas de predicción.",
        )
        is_regression = paradigm == "Regresión"

        # ── Global settings ──────────────────────────────────────────────
        col_theme, col_auto = st.columns(2)
        with col_theme:
            theme = st.selectbox(
                "🎨 Tema",
                options=["dark", "light"],
                index=0,
                key="theme",
                help="dark: mejor en pantallas. light: mejor para proyección en aula.",
            )
        with col_auto:
            auto_update = st.checkbox(
                "⚡ Auto",
                value=True,
                key="auto_update",
                help="Auto-actualizar al mover sliders. Desactivar en hardware lento.",
            )

        st.markdown("---")

        # ── Dataset ───────────────────────────────────────────────────────
        st.markdown("### 📊 Dataset")

        if is_regression:
            cfg = _render_regression_dataset_section()
        else:
            cfg = _render_classification_dataset_section()

        col_split, col_seed = st.columns(2)
        with col_split:
            train_test_split = st.slider(
                "Train %",
                min_value=50,
                max_value=90,
                value=75,
                step=5,
                key="train_test_split",
                help="Porcentaje de datos usado para entrenamiento.",
            )
        with col_seed:
            random_seed = st.number_input(
                "Seed",
                min_value=0,
                max_value=9999,
                value=42,
                step=1,
                key="random_seed",
                help="Semilla aleatoria para reproducibilidad.",
            )

        st.markdown("---")

        # ── Single Model ──────────────────────────────────────────────────
        st.markdown("### 🤖 Modelo (Explorar)")

        if is_regression:
            all_model_names = list(MODEL_REGISTRY_REGRESSION.keys())
            default_model = "Decision Tree Regressor"
            registry = MODEL_REGISTRY_REGRESSION
            hp_func = get_regression_hyperparameter_controls
        else:
            all_model_names = list(MODEL_REGISTRY.keys())
            default_model = "Decision Tree"
            registry = MODEL_REGISTRY
            hp_func = get_hyperparameter_controls

        model_name = st.selectbox(
            "Modelo",
            options=all_model_names,
            index=all_model_names.index(default_model) if default_model in all_model_names else 0,
            key="model_name",
            help=registry[all_model_names[0]]["description"],
        )

        # Model description
        st.caption(f"*{registry[model_name]['description']}*")

        hyperparams = _render_hyperparameter_controls(model_name, hp_func)

        st.markdown("---")

        # ── Visualization toggles ────────────────────────────────────────
        st.markdown("### 🔍 Visualización")
        col_tr, col_te = st.columns(2)
        with col_tr:
            show_train = st.checkbox(
                "Mostrar Train", value=True, key="show_train",
                help="Superponer los puntos del conjunto de entrenamiento."
            )
        with col_te:
            show_test = st.checkbox(
                "Mostrar Test", value=True, key="show_test",
                help="Superponer los puntos del conjunto de test."
            )

        cm_normalize = False
        if not is_regression:
            cm_normalize = st.checkbox(
                "CM: % por fila", value=False, key="cm_normalize",
                help="Matriz de Confusión: muestra porcentaje por fila (recall por clase) como valor principal."
            )

        st.markdown("---")

        # ── Comparison ───────────────────────────────────────────────────
        st.markdown("### 🆚 Comparación")
        if is_regression:
            default_compare = ["Linear Regression", "Decision Tree Regressor", "KNN Regressor"]
            compare_help = "Selecciona hasta 6 modelos de regresión para comparar."
        else:
            default_compare = ["Logistic Regression", "Decision Tree", "KNN"]
            compare_help = "Selecciona hasta 6 modelos para comparar sus fronteras de decisión."

        compare_models = st.multiselect(
            "Modelos a comparar",
            options=all_model_names,
            default=default_compare,
            key="compare_models",
            help=compare_help,
            max_selections=6,
        )

        st.markdown("---")
        st.markdown(
            "<div style='text-align:center; font-size:0.75rem; opacity:0.5;'>"
            "Visual ML · <a href='https://github.com/AxelSkrauba/visual-ml' target='_blank'>@AxelSkrauba</a>"
            "</div>",
            unsafe_allow_html=True,
        )

    result = {
        "paradigm": paradigm,
        "theme": theme,
        "auto_update": auto_update,
        "train_test_split": int(train_test_split),
        "random_seed": int(random_seed),
        "model_name": model_name,
        "hyperparams": hyperparams,
        "show_train": show_train,
        "show_test": show_test,
        "cm_normalize": cm_normalize,
        "compare_models": compare_models,
    }
    result.update(cfg)
    return result


def _render_classification_dataset_section() -> dict:
    """Render classification dataset selector and params. Returns partial config."""
    dataset_options = {cfg["label"]: key for key, cfg in DATASET_CONFIGS.items()}
    dataset_label = st.selectbox(
        "Tipo",
        options=list(dataset_options.keys()),
        index=0,
        key="dataset_label",
        help="Selecciona el tipo de problema de clasificación.",
    )
    dataset_name = dataset_options[dataset_label]
    ds_cfg = DATASET_CONFIGS[dataset_name]

    n_samples = st.slider(
        "Muestras (n)",
        min_value=100,
        max_value=500,
        value=300,
        step=50,
        key="n_samples",
        help="Número total de puntos en el dataset.",
    )

    # Dataset-specific parameters driven by ds_cfg["params"]
    ds_params = ds_cfg.get("params", [])

    noise = ds_cfg.get("default_noise", 0.2)
    n_classes = 2
    factor = 0.5
    cluster_std = 1.0
    imbalance_ratio = 0.1

    if "noise" in ds_params:
        noise = st.slider(
            "Ruido",
            min_value=0.0,
            max_value=0.5,
            value=0.20,
            step=0.01,
            key="noise",
            help="Nivel de ruido añadido al dataset. Más ruido → mayor solapamiento entre clases.",
        )
    if "n_classes" in ds_params:
        n_classes = st.slider(
            "Clases (n)",
            min_value=2,
            max_value=5,
            value=2,
            step=1,
            key="n_classes",
            help="Número de clases. Datasets binarios (Moons, Circles, XOR) son fijos en 2.",
        )
    if "factor" in ds_params:
        factor = st.slider(
            "Factor (radio interno/externo)",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.05,
            key="factor",
            help="Proporción del radio del círculo interno vs externo.",
        )
    if "cluster_std" in ds_params:
        cluster_std = st.slider(
            "Dispersión de clusters (std)",
            min_value=0.3,
            max_value=3.5,
            value=1.0,
            step=0.1,
            key="cluster_std",
            help="Desviación estándar de cada cluster gaussiano. Valores altos → solapamiento.",
        )
    if "imbalance_ratio" in ds_params:
        imbalance_ratio = st.slider(
            "Ratio de minoría (%)",
            min_value=5,
            max_value=50,
            value=10,
            step=5,
            key="imbalance_ratio",
            help="Fracción (%) de la clase minoritaria. 10% = 90/10 de distribución.",
        ) / 100.0

    return {
        "dataset_name": dataset_name,
        "n_samples": int(n_samples),
        "noise": float(noise),
        "n_classes": int(n_classes),
        "factor": float(factor),
        "cluster_std": float(cluster_std),
        "imbalance_ratio": float(imbalance_ratio),
    }


def _render_regression_dataset_section() -> dict:
    """Render regression dataset selector and params. Returns partial config."""
    dataset_options = {cfg["label"]: key for key, cfg in REGRESSION_DATASET_CONFIGS.items()}
    dataset_label = st.selectbox(
        "Tipo",
        options=list(dataset_options.keys()),
        index=0,
        key="reg_dataset_label",
        help="Selecciona el tipo de problema de regresión.",
    )
    dataset_name = dataset_options[dataset_label]
    ds_cfg = REGRESSION_DATASET_CONFIGS[dataset_name]

    n_samples = st.slider(
        "Muestras (n)",
        min_value=50,
        max_value=500,
        value=200,
        step=25,
        key="reg_n_samples",
        help="Número total de puntos en el dataset.",
    )

    ds_params = ds_cfg.get("params", [])

    noise = 0.2
    degree = 3
    frequency = 1.5
    n_steps = 3

    if "noise" in ds_params:
        noise = st.slider(
            "Ruido (σ)",
            min_value=0.0,
            max_value=1.0,
            value=0.20,
            step=0.05,
            key="reg_noise",
            help="Desviación estándar del ruido gaussiano añadido al target.",
        )
    if "degree" in ds_params:
        degree = st.slider(
            "Grado del polinomio",
            min_value=1,
            max_value=6,
            value=3,
            step=1,
            key="reg_degree",
            help="Grado del polinomio. Grado 1 = lineal. Mayor grado → curva más compleja.",
        )
    if "frequency" in ds_params:
        frequency = st.slider(
            "Frecuencia",
            min_value=0.5,
            max_value=4.0,
            value=1.5,
            step=0.25,
            key="reg_frequency",
            help="Multiplicador de frecuencia de la onda sinusoidal.",
        )
    if "n_steps" in ds_params:
        n_steps = st.slider(
            "Escalones",
            min_value=2,
            max_value=6,
            value=3,
            step=1,
            key="reg_n_steps",
            help="Número de escalones en la función.",
        )

    return {
        "dataset_name": dataset_name,
        "n_samples": int(n_samples),
        "noise": float(noise),
        "degree": int(degree),
        "frequency": float(frequency),
        "n_steps": int(n_steps),
    }


def _render_hyperparameter_controls(model_name: str, hp_func=None) -> dict:
    """Render dynamic hyperparameter sliders/selects for the given model."""
    if hp_func is None:
        hp_func = get_hyperparameter_controls
    controls = hp_func(model_name)
    params = {}

    # We need to track kernel for conditional rendering of SVC params
    # First pass: collect non-conditional params and identify conditionals
    kernel_value = None
    for ctrl in controls:
        if ctrl["name"] == "kernel":
            kernel_value = st.selectbox(
                ctrl["label"],
                options=ctrl["options"],
                index=ctrl["options"].index(ctrl["default"]),
                key=f"hp_{model_name}_{ctrl['name']}",
                help=ctrl.get("tooltip", ""),
            )
            params["kernel"] = kernel_value

    for ctrl in controls:
        name = ctrl["name"]
        if name == "kernel":
            continue  # already rendered

        # Check conditional visibility
        cond = ctrl.get("conditional")
        if cond:
            current_val = params.get(cond["param"], kernel_value)
            if current_val not in cond["values"]:
                continue

        hp_type = ctrl["type"]
        key = f"hp_{model_name}_{name}"
        tooltip = ctrl.get("tooltip", "")

        if hp_type == "slider_int":
            val = st.slider(
                ctrl["label"],
                min_value=int(ctrl["min"]),
                max_value=int(ctrl["max"]),
                value=int(ctrl["default"]),
                step=1,
                key=key,
                help=tooltip,
            )
            params[name] = int(val)

        elif hp_type == "slider_float":
            step = (ctrl["max"] - ctrl["min"]) / 100
            val = st.slider(
                ctrl["label"],
                min_value=float(ctrl["min"]),
                max_value=float(ctrl["max"]),
                value=float(ctrl["default"]),
                step=round(step, 4),
                key=key,
                help=tooltip,
            )
            params[name] = float(val)

        elif hp_type == "slider_log":
            log_min = math.log10(ctrl["min"])
            log_max = math.log10(ctrl["max"])
            log_default = math.log10(ctrl["default"])
            log_val = st.slider(
                f"{ctrl['label']} (log₁₀)",
                min_value=float(log_min),
                max_value=float(log_max),
                value=float(log_default),
                step=0.1,
                key=key,
                help=tooltip,
            )
            val = 10 ** log_val
            st.caption(f"Valor: `{val:.4g}`")
            params[name] = float(val)

        elif hp_type == "select":
            options = ctrl["options"]
            val = st.selectbox(
                ctrl["label"],
                options=options,
                index=options.index(ctrl["default"]) if ctrl["default"] in options else 0,
                key=key,
                help=tooltip,
            )
            params[name] = val

        elif hp_type == "checkbox":
            val = st.checkbox(
                ctrl["label"],
                value=bool(ctrl["default"]),
                key=key,
                help=tooltip,
            )
            params[name] = bool(val)

    return params
