from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


MODEL_REGISTRY: dict = {
    "Logistic Regression": {
        "label": "Logistic Regression",
        "group": "Lineales",
        "description": "Clasificador lineal probabilístico. Frontera siempre lineal (hiperplano).",
        "class": LogisticRegression,
        "default_params": {"C": 1.0, "max_iter": 500, "random_state": 42},
        "hyperparameters": [
            {
                "name": "C",
                "label": "C (Regularización inversa)",
                "type": "slider_log",
                "min": 0.01,
                "max": 100.0,
                "default": 1.0,
                "tooltip": "Valores bajos → mayor regularización → frontera más suave. Valores altos → menos regularización → puede sobreajustar.",
            },
            {
                "name": "max_iter",
                "label": "Max iteraciones",
                "type": "slider_int",
                "min": 100,
                "max": 2000,
                "default": 500,
                "tooltip": "Número máximo de iteraciones del solver.",
            },
        ],
    },
    "Perceptron": {
        "label": "Perceptron",
        "group": "Lineales",
        "description": "Clasificador lineal simple, el más básico. Solo converge si los datos son linealmente separables.",
        "class": Perceptron,
        "default_params": {"max_iter": 1000, "random_state": 42},
        "hyperparameters": [
            {
                "name": "max_iter",
                "label": "Max iteraciones",
                "type": "slider_int",
                "min": 100,
                "max": 3000,
                "default": 1000,
                "tooltip": "Número de épocas de entrenamiento.",
            },
            {
                "name": "eta0",
                "label": "Tasa de aprendizaje (η)",
                "type": "slider_float",
                "min": 0.0001,
                "max": 1.0,
                "default": 1.0,
                "tooltip": "Tasa de aprendizaje inicial.",
            },
        ],
    },
    "MLP": {
        "label": "MLP",
        "group": "Red Neuronal",
        "description": "Multi-Layer Perceptron. Red neuronal densa con capas ocultas. Puede aprender fronteras arbitrariamente complejas a diferencia del Perceptron simple.",
        "class": MLPClassifier,
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
                "tooltip": "Arquitectura de capas ocultas. Más neuronas/capas → mayor capacidad → más riesgo de sobreajuste.",
            },
            {
                "name": "activation",
                "label": "Función de activación",
                "type": "select",
                "options": ["relu", "tanh", "logistic"],
                "default": "relu",
                "tooltip": "relu: más rápido y estable. tanh/logistic: saturan para valores grandes (vanishing gradient).",
            },
            {
                "name": "alpha",
                "label": "Alpha (regularización L2)",
                "type": "slider_log",
                "min": 0.00001,
                "max": 1.0,
                "default": 0.0001,
                "tooltip": "Penalidad L2 sobre los pesos. Alpha alto → mayor regularización → frontera más suave.",
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
    "SVC": {
        "label": "SVC",
        "group": "Kernel / Margen",
        "description": "Support Vector Classifier. Maximiza el margen entre clases. Con kernel RBF/poly puede trazar fronteras no lineales.",
        "class": SVC,
        "default_params": {"C": 1.0, "kernel": "rbf", "gamma": "scale", "degree": 3, "random_state": 42},
        "hyperparameters": [
            {
                "name": "C",
                "label": "C (Parámetro de margen)",
                "type": "slider_log",
                "min": 0.01,
                "max": 100.0,
                "default": 1.0,
                "tooltip": "C bajo → margen amplio (más regularización). C alto → margen estrecho (puede sobreajustar).",
            },
            {
                "name": "kernel",
                "label": "Kernel",
                "type": "select",
                "options": ["linear", "rbf", "poly", "sigmoid"],
                "default": "rbf",
                "tooltip": "linear → frontera recta. rbf → frontera circular/suave. poly → frontera polinómica. sigmoid → similar a red neuronal.",
            },
            {
                "name": "gamma",
                "label": "Gamma (solo RBF/poly/sigmoid)",
                "type": "slider_log",
                "min": 0.001,
                "max": 10.0,
                "default": 0.1,
                "tooltip": "Controla el radio de influencia de cada punto. Gamma alto → frontera muy ajustada (sobreajuste). Gamma bajo → frontera suave.",
                "conditional": {"param": "kernel", "values": ["rbf", "poly", "sigmoid"]},
            },
            {
                "name": "degree",
                "label": "Grado (solo poly)",
                "type": "slider_int",
                "min": 2,
                "max": 6,
                "default": 3,
                "tooltip": "Grado del polinomio para kernel poly.",
                "conditional": {"param": "kernel", "values": ["poly"]},
            },
        ],
    },
    "KNN": {
        "label": "KNN",
        "group": "Vecindad",
        "description": "K-Vecinos Más Próximos. Frontera determinada por vecindad local. Muy flexible con K pequeño.",
        "class": KNeighborsClassifier,
        "default_params": {"n_neighbors": 5, "weights": "uniform", "metric": "euclidean"},
        "hyperparameters": [
            {
                "name": "n_neighbors",
                "label": "K (vecinos)",
                "type": "slider_int",
                "min": 1,
                "max": 30,
                "default": 5,
                "tooltip": "K=1 → frontera muy irregular (sobreajuste). K grande → frontera suave (subajuste).",
            },
            {
                "name": "weights",
                "label": "Pesos",
                "type": "select",
                "options": ["uniform", "distance"],
                "default": "uniform",
                "tooltip": "uniform → todos los vecinos pesan igual. distance → los más cercanos pesan más.",
            },
            {
                "name": "metric",
                "label": "Métrica de distancia",
                "type": "select",
                "options": ["euclidean", "manhattan", "chebyshev"],
                "default": "euclidean",
                "tooltip": "Métrica para calcular distancia entre puntos.",
            },
        ],
    },
    "Decision Tree": {
        "label": "Decision Tree",
        "group": "Árbol",
        "description": "Árbol de decisión. Fronteras rectangulares (ejes alineados). Fácil de sobreajustar aumentando la profundidad.",
        "class": DecisionTreeClassifier,
        "default_params": {"max_depth": 3, "criterion": "gini", "min_samples_split": 2, "random_state": 42},
        "hyperparameters": [
            {
                "name": "max_depth",
                "label": "Profundidad máxima",
                "type": "slider_int",
                "min": 1,
                "max": 20,
                "default": 3,
                "tooltip": "Profundidad 1 → stump (línea recta). Mayor profundidad → frontera más compleja → sobreajuste.",
            },
            {
                "name": "criterion",
                "label": "Criterio de división",
                "type": "select",
                "options": ["gini", "entropy", "log_loss"],
                "default": "gini",
                "tooltip": "Gini e impureza de información. En la práctica producen resultados similares.",
            },
            {
                "name": "min_samples_split",
                "label": "Mín. muestras para dividir",
                "type": "slider_int",
                "min": 2,
                "max": 30,
                "default": 2,
                "tooltip": "Mínimo de muestras en un nodo para seguir dividiendo. Valores altos regularizan el árbol.",
            },
        ],
    },
    "Random Forest": {
        "label": "Random Forest",
        "group": "Ensemble Bagging",
        "description": "Ensemble de árboles de decisión por bagging. Frontera suavizada por votación. Más robusto que un árbol solo.",
        "class": RandomForestClassifier,
        "default_params": {"n_estimators": 100, "max_depth": 5, "max_features": "sqrt", "random_state": 42},
        "hyperparameters": [
            {
                "name": "n_estimators",
                "label": "Número de árboles",
                "type": "slider_int",
                "min": 10,
                "max": 200,
                "default": 100,
                "tooltip": "Más árboles → frontera más suave y estable. Aumenta el tiempo de cómputo.",
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
            {
                "name": "max_features",
                "label": "Features por split",
                "type": "select",
                "options": ["sqrt", "log2", "None"],
                "default": "sqrt",
                "tooltip": "Número de features consideradas en cada split. sqrt es el default recomendado para clasificación.",
            },
        ],
    },
    "Gradient Boosting": {
        "label": "Gradient Boosting",
        "group": "Ensemble Boosting",
        "description": "Boosting secuencial. Construye árboles para corregir errores del anterior. Muy potente, puede sobreajustar.",
        "class": GradientBoostingClassifier,
        "default_params": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3, "random_state": 42},
        "hyperparameters": [
            {
                "name": "n_estimators",
                "label": "Número de árboles",
                "type": "slider_int",
                "min": 10,
                "max": 200,
                "default": 100,
                "tooltip": "Más árboles → mejor ajuste a entrenamiento. Puede sobreajustar si es muy alto.",
            },
            {
                "name": "learning_rate",
                "label": "Tasa de aprendizaje",
                "type": "slider_log",
                "min": 0.01,
                "max": 1.0,
                "default": 0.1,
                "tooltip": "Shrinkage: tasa baja + muchos árboles = mejor generalización.",
            },
            {
                "name": "max_depth",
                "label": "Profundidad máxima",
                "type": "slider_int",
                "min": 1,
                "max": 10,
                "default": 3,
                "tooltip": "Profundidad de cada árbol base. Valores bajos (1-3) son habituales en boosting.",
            },
        ],
    },
    "AdaBoost": {
        "label": "AdaBoost",
        "group": "Ensemble Boosting",
        "description": "Adaptive Boosting. Pesa los errores anteriores para mejorar iterativamente. Usa stumps por defecto.",
        "class": AdaBoostClassifier,
        "default_params": {"n_estimators": 50, "learning_rate": 1.0, "random_state": 42},
        "hyperparameters": [
            {
                "name": "n_estimators",
                "label": "Número de estimadores",
                "type": "slider_int",
                "min": 10,
                "max": 200,
                "default": 50,
                "tooltip": "Número de clasificadores débiles a combinar.",
            },
            {
                "name": "learning_rate",
                "label": "Tasa de aprendizaje",
                "type": "slider_log",
                "min": 0.01,
                "max": 2.0,
                "default": 1.0,
                "tooltip": "Pondera la contribución de cada clasificador.",
            },
        ],
    },
    "Gaussian NB": {
        "label": "Gaussian NB",
        "group": "Probabilístico",
        "description": "Naive Bayes Gaussiano. Asume independencia condicional y distribución normal. Frontera parabólica.",
        "class": GaussianNB,
        "default_params": {"var_smoothing": 1e-9},
        "hyperparameters": [
            {
                "name": "var_smoothing",
                "label": "Suavizado de varianza",
                "type": "slider_log",
                "min": 1e-12,
                "max": 1.0,
                "default": 1e-9,
                "tooltip": "Fracción de la mayor varianza añadida a todas las varianzas (estabilidad numérica).",
            },
        ],
    },
    "QDA": {
        "label": "QDA",
        "group": "Probabilístico",
        "description": "Análisis Discriminante Cuadrático. Estima una gaussiana por clase. Frontera cuadrática (elipsoidal).",
        "class": QuadraticDiscriminantAnalysis,
        "default_params": {"reg_param": 0.0},
        "hyperparameters": [
            {
                "name": "reg_param",
                "label": "Parámetro de regularización",
                "type": "slider_float",
                "min": 0.0,
                "max": 1.0,
                "default": 0.0,
                "tooltip": "Regularización de la matriz de covarianza. 0 = sin regularización, 1 = máxima.",
            },
        ],
    },
}


def get_model_instance(model_name: str, params: dict):
    """
    Instantiate an sklearn model by registry name with given parameters.

    Parameters
    ----------
    model_name : str
        Key in MODEL_REGISTRY.
    params : dict
        Hyperparameter values. Missing keys fall back to default_params.

    Returns
    -------
    Unfitted sklearn estimator.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: '{model_name}'. Choose from {list(MODEL_REGISTRY.keys())}")

    cfg = MODEL_REGISTRY[model_name]
    merged = {**cfg["default_params"], **params}

    # Handle special cases
    if model_name == "SVC":
        # gamma: if using numeric value, pass it; if "scale" or "auto", pass as-is
        if "gamma" in merged and merged.get("kernel") == "linear":
            merged.pop("gamma", None)

    if model_name == "Random Forest":
        # "None" string → None
        if merged.get("max_features") == "None":
            merged["max_features"] = None

    if model_name == "MLP":
        # hidden_layer_sizes may arrive as a string like "(64, 32)" from the selectbox
        hls = merged.get("hidden_layer_sizes")
        if isinstance(hls, str):
            import ast as _ast
            merged["hidden_layer_sizes"] = _ast.literal_eval(hls)

    cls = cfg["class"]
    # Filter only valid constructor kwargs for this class
    import inspect
    valid_keys = set(inspect.signature(cls.__init__).parameters.keys()) - {"self"}
    filtered = {k: v for k, v in merged.items() if k in valid_keys}

    return cls(**filtered)


def get_hyperparameter_controls(model_name: str) -> list[dict]:
    """
    Return the list of hyperparameter control definitions for the given model.

    Parameters
    ----------
    model_name : str
        Key in MODEL_REGISTRY.

    Returns
    -------
    List of hyperparameter dicts as defined in MODEL_REGISTRY.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: '{model_name}'")
    return MODEL_REGISTRY[model_name]["hyperparameters"]


def get_models_by_group() -> dict[str, list[str]]:
    """Return model names grouped by their group label."""
    groups: dict[str, list[str]] = {}
    for name, cfg in MODEL_REGISTRY.items():
        g = cfg["group"]
        groups.setdefault(g, []).append(name)
    return groups
