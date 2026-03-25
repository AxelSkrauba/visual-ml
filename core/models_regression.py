import inspect

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor


MODEL_REGISTRY_REGRESSION: dict = {
    "Linear Regression": {
        "label": "Linear Regression",
        "group": "Lineales",
        "description": "Regresión lineal por mínimos cuadrados. Ajusta una recta (o hiperplano). Referencia base.",
        "class": LinearRegression,
        "default_params": {},
        "hyperparameters": [],
    },
    "Ridge": {
        "label": "Ridge",
        "group": "Lineales",
        "description": "Regresión lineal con regularización L2. Penaliza coeficientes grandes → menor sobreajuste.",
        "class": Ridge,
        "default_params": {"alpha": 1.0, "random_state": 42},
        "hyperparameters": [
            {
                "name": "alpha",
                "label": "Alpha (regularización L2)",
                "type": "slider_log",
                "min": 0.001,
                "max": 100.0,
                "default": 1.0,
                "tooltip": "Alpha alto → mayor regularización → curva más suave. Alpha bajo → se acerca a OLS.",
            },
        ],
    },
    "Lasso": {
        "label": "Lasso",
        "group": "Lineales",
        "description": "Regresión lineal con regularización L1. Puede forzar coeficientes a cero (selección de features).",
        "class": Lasso,
        "default_params": {"alpha": 1.0, "random_state": 42, "max_iter": 5000},
        "hyperparameters": [
            {
                "name": "alpha",
                "label": "Alpha (regularización L1)",
                "type": "slider_log",
                "min": 0.001,
                "max": 100.0,
                "default": 1.0,
                "tooltip": "Alpha alto → más coeficientes se anulan. Alpha bajo → menos regularización.",
            },
        ],
    },
    "SVR": {
        "label": "SVR",
        "group": "Kernel / Margen",
        "description": "Support Vector Regressor. Usa kernels para capturar no-linealidad. Epsilon define la banda de tolerancia.",
        "class": SVR,
        "default_params": {"C": 1.0, "kernel": "rbf", "epsilon": 0.1, "gamma": "scale"},
        "hyperparameters": [
            {
                "name": "C",
                "label": "C (regularización inversa)",
                "type": "slider_log",
                "min": 0.01,
                "max": 100.0,
                "default": 1.0,
                "tooltip": "C bajo → más tolerancia a errores. C alto → ajuste más estricto.",
            },
            {
                "name": "kernel",
                "label": "Kernel",
                "type": "select",
                "options": ["linear", "rbf", "poly"],
                "default": "rbf",
                "tooltip": "linear → recta. rbf → curva suave. poly → polinómica.",
            },
            {
                "name": "epsilon",
                "label": "Epsilon (banda de tolerancia)",
                "type": "slider_float",
                "min": 0.01,
                "max": 1.0,
                "default": 0.1,
                "tooltip": "Ancho del tubo epsilon. Errores dentro del tubo no se penalizan.",
            },
            {
                "name": "gamma",
                "label": "Gamma (solo RBF/poly)",
                "type": "slider_log",
                "min": 0.001,
                "max": 10.0,
                "default": 0.1,
                "tooltip": "Radio de influencia de cada punto. Gamma alto → curva muy ajustada.",
                "conditional": {"param": "kernel", "values": ["rbf", "poly"]},
            },
        ],
    },
    "KNN Regressor": {
        "label": "KNN Regressor",
        "group": "Vecindad",
        "description": "K-Vecinos para regresión. Predice promediando los K vecinos más cercanos.",
        "class": KNeighborsRegressor,
        "default_params": {"n_neighbors": 5, "weights": "uniform"},
        "hyperparameters": [
            {
                "name": "n_neighbors",
                "label": "K (vecinos)",
                "type": "slider_int",
                "min": 1,
                "max": 30,
                "default": 5,
                "tooltip": "K=1 → curva muy irregular. K grande → curva muy suave.",
            },
            {
                "name": "weights",
                "label": "Pesos",
                "type": "select",
                "options": ["uniform", "distance"],
                "default": "uniform",
                "tooltip": "uniform → promedio simple. distance → los más cercanos pesan más.",
            },
        ],
    },
    "Decision Tree Regressor": {
        "label": "Decision Tree Regressor",
        "group": "Árbol",
        "description": "Árbol de decisión para regresión. Genera predicción escalonada (constante por hoja).",
        "class": DecisionTreeRegressor,
        "default_params": {"max_depth": 4, "min_samples_split": 2, "random_state": 42},
        "hyperparameters": [
            {
                "name": "max_depth",
                "label": "Profundidad máxima",
                "type": "slider_int",
                "min": 1,
                "max": 20,
                "default": 4,
                "tooltip": "Profundidad 1 → predicción casi constante. Mayor profundidad → más escalones.",
            },
            {
                "name": "min_samples_split",
                "label": "Mín. muestras para dividir",
                "type": "slider_int",
                "min": 2,
                "max": 30,
                "default": 2,
                "tooltip": "Valores altos regularizan el árbol (menos divisiones).",
            },
        ],
    },
    "Random Forest Regressor": {
        "label": "Random Forest Regressor",
        "group": "Ensemble",
        "description": "Ensemble de árboles por bagging. Promedia predicciones → curva más suave que un solo árbol.",
        "class": RandomForestRegressor,
        "default_params": {"n_estimators": 100, "max_depth": 5, "random_state": 42},
        "hyperparameters": [
            {
                "name": "n_estimators",
                "label": "Número de árboles",
                "type": "slider_int",
                "min": 10,
                "max": 200,
                "default": 100,
                "tooltip": "Más árboles → predicción más estable. Aumenta el tiempo de cómputo.",
            },
            {
                "name": "max_depth",
                "label": "Profundidad máxima",
                "type": "slider_int",
                "min": 1,
                "max": 20,
                "default": 5,
                "tooltip": "Profundidad de cada árbol individual.",
            },
        ],
    },
    "MLP Regressor": {
        "label": "MLP Regressor",
        "group": "Red Neuronal",
        "description": "Red neuronal densa para regresión. Puede aproximar cualquier función continua.",
        "class": MLPRegressor,
        "default_params": {
            "hidden_layer_sizes": (64, 32),
            "activation": "relu",
            "alpha": 0.0001,
            "max_iter": 500,
            "random_state": 42,
        },
        "hyperparameters": [
            {
                "name": "hidden_layer_sizes",
                "label": "Neuronas por capa oculta",
                "type": "select",
                "options": ["(16,)", "(32,)", "(64,)", "(32, 16)", "(64, 32)", "(128, 64)", "(64, 32, 16)"],
                "default": "(64, 32)",
                "tooltip": "Más neuronas/capas → mayor capacidad → más riesgo de sobreajuste.",
            },
            {
                "name": "activation",
                "label": "Función de activación",
                "type": "select",
                "options": ["relu", "tanh", "logistic"],
                "default": "relu",
                "tooltip": "relu: rápido y estable. tanh/logistic: saturan para valores grandes.",
            },
            {
                "name": "alpha",
                "label": "Alpha (regularización L2)",
                "type": "slider_log",
                "min": 0.00001,
                "max": 1.0,
                "default": 0.0001,
                "tooltip": "Penalidad L2 sobre los pesos. Alpha alto → curva más suave.",
            },
            {
                "name": "max_iter",
                "label": "Max iteraciones",
                "type": "slider_int",
                "min": 100,
                "max": 2000,
                "default": 500,
                "tooltip": "Número máximo de épocas de entrenamiento.",
            },
        ],
    },
}


def get_regression_model_instance(model_name: str, params: dict):
    """
    Instantiate an sklearn regressor by registry name with given parameters.

    Parameters
    ----------
    model_name : str
        Key in MODEL_REGISTRY_REGRESSION.
    params : dict
        Hyperparameter values. Missing keys fall back to default_params.

    Returns
    -------
    Unfitted sklearn estimator.
    """
    if model_name not in MODEL_REGISTRY_REGRESSION:
        raise ValueError(
            f"Unknown regression model: '{model_name}'. "
            f"Choose from {list(MODEL_REGISTRY_REGRESSION.keys())}"
        )

    cfg = MODEL_REGISTRY_REGRESSION[model_name]
    merged = {**cfg["default_params"], **params}

    # Handle special cases
    if model_name == "SVR":
        if "gamma" in merged and merged.get("kernel") == "linear":
            merged.pop("gamma", None)

    if model_name == "MLP Regressor":
        hls = merged.get("hidden_layer_sizes")
        if isinstance(hls, str):
            import ast as _ast
            merged["hidden_layer_sizes"] = _ast.literal_eval(hls)

    cls = cfg["class"]
    valid_keys = set(inspect.signature(cls.__init__).parameters.keys()) - {"self"}
    filtered = {k: v for k, v in merged.items() if k in valid_keys}

    return cls(**filtered)


def get_regression_hyperparameter_controls(model_name: str) -> list[dict]:
    """Return the list of hyperparameter control definitions for a regression model."""
    if model_name not in MODEL_REGISTRY_REGRESSION:
        raise ValueError(f"Unknown regression model: '{model_name}'")
    return MODEL_REGISTRY_REGRESSION[model_name]["hyperparameters"]


def get_regression_models_by_group() -> dict[str, list[str]]:
    """Return regression model names grouped by their group label."""
    groups: dict[str, list[str]] = {}
    for name, cfg in MODEL_REGISTRY_REGRESSION.items():
        g = cfg["group"]
        groups.setdefault(g, []).append(name)
    return groups
