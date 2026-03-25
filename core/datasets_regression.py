import numpy as np


REGRESSION_DATASET_CONFIGS = {
    "reg_linear": {
        "label": "Lineal",
        "description": "Relación lineal simple con ruido gaussiano. Ideal para verificar modelos lineales.",
        "params": ["noise", "n_samples"],
    },
    "reg_polynomial": {
        "label": "Polinomial",
        "description": "Curva polinómica de grado configurable. Los modelos lineales subajustan si el grado > 1.",
        "params": ["noise", "degree", "n_samples"],
    },
    "reg_sinusoidal": {
        "label": "Sinusoidal",
        "description": "Onda sinusoidal. Altamente no lineal, requiere modelos flexibles.",
        "params": ["noise", "frequency", "n_samples"],
    },
    "reg_step": {
        "label": "Escalón",
        "description": "Función escalonada con saltos abruptos. Los árboles la capturan bien, los lineales no.",
        "params": ["noise", "n_steps", "n_samples"],
    },
    "reg_exponential": {
        "label": "Exponencial",
        "description": "Crecimiento exponencial con ruido. Desafío para modelos que asumen varianza constante.",
        "params": ["noise", "n_samples"],
    },
}


def generate_regression_dataset(
    name: str,
    n_samples: int = 200,
    random_state: int = 42,
    noise: float = 0.2,
    degree: int = 3,
    frequency: float = 1.5,
    n_steps: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a 1D regression dataset by name.

    Parameters
    ----------
    name : str
        One of the keys in REGRESSION_DATASET_CONFIGS.
    n_samples : int
        Total number of samples.
    random_state : int
        Random seed for reproducibility.
    noise : float
        Noise standard deviation added to the target.
    degree : int
        Polynomial degree (only for reg_polynomial).
    frequency : float
        Sinusoidal frequency multiplier (only for reg_sinusoidal).
    n_steps : int
        Number of steps (only for reg_step).

    Returns
    -------
    X : np.ndarray of shape (n_samples, 1)
    y : np.ndarray of shape (n_samples,)
    """
    if name not in REGRESSION_DATASET_CONFIGS:
        raise ValueError(
            f"Unknown regression dataset: '{name}'. "
            f"Choose from {list(REGRESSION_DATASET_CONFIGS.keys())}"
        )

    rng = np.random.default_rng(random_state)
    X = np.sort(rng.uniform(-3, 3, size=n_samples))
    noise_vec = rng.normal(0, noise, size=n_samples)

    if name == "reg_linear":
        y = 2.0 * X + 1.0 + noise_vec

    elif name == "reg_polynomial":
        degree = int(np.clip(degree, 1, 6))
        # Normalised polynomial so values stay in a reasonable range
        y = np.polyval([1.0 / max(1, degree - 1)] * degree + [0.0], X) + noise_vec

    elif name == "reg_sinusoidal":
        frequency = float(np.clip(frequency, 0.5, 4.0))
        y = np.sin(frequency * np.pi * X) + noise_vec

    elif name == "reg_step":
        n_steps = int(np.clip(n_steps, 2, 6))
        thresholds = np.linspace(X.min(), X.max(), n_steps + 1)[1:-1]
        y = np.zeros_like(X)
        for i, thr in enumerate(thresholds):
            y += (X >= thr).astype(float)
        y = y + noise_vec

    elif name == "reg_exponential":
        # Scaled exponential to keep values manageable
        y = np.exp(0.5 * X) + noise_vec

    X = X.reshape(-1, 1).astype(np.float64)
    y = y.astype(np.float64)
    return X, y
