import numpy as np
from sklearn.datasets import make_moons, make_circles, make_blobs, make_classification


DATASET_CONFIGS = {
    "moons": {
        "label": "Moons",
        "description": "Dos lunas entrelazadas. Frontera curva, no lineal clásico.",
        "fixed_classes": True,
        "n_classes": 2,
        "default_noise": 0.2,
        "supports_multiclass": False,
        "params": ["noise", "n_samples"],
    },
    "circles": {
        "label": "Circles",
        "description": "Círculos concéntricos. Separación radial, los modelos lineales fallan.",
        "fixed_classes": True,
        "n_classes": 2,
        "default_noise": 0.1,
        "supports_multiclass": False,
        "params": ["noise", "factor", "n_samples"],
    },
    "blobs": {
        "label": "Blobs",
        "description": "Clusters gaussianos. Lineal/multi-clase, controla solapamiento.",
        "fixed_classes": False,
        "n_classes": 2,
        "default_noise": 1.0,
        "supports_multiclass": True,
        "params": ["cluster_std", "n_classes", "n_samples"],
    },
    "xor": {
        "label": "XOR",
        "description": "Patrón XOR (cuadrantes). No lineal, no separable linealmente.",
        "fixed_classes": True,
        "n_classes": 2,
        "default_noise": 0.2,
        "supports_multiclass": False,
        "params": ["noise", "n_samples"],
    },
    "linear": {
        "label": "Linear",
        "description": "Clases linealmente separables con margen configurable.",
        "fixed_classes": False,
        "n_classes": 2,
        "default_noise": 0.1,
        "supports_multiclass": True,
        "params": ["noise", "n_classes", "n_samples"],
    },
    "spirals": {
        "label": "Spirals",
        "description": "Espirales entrelazadas. Alta no-linealidad, desafío extremo.",
        "fixed_classes": False,
        "n_classes": 2,
        "default_noise": 0.1,
        "supports_multiclass": True,
        "params": ["noise", "n_classes", "n_samples"],
    },
    "imbalanced": {
        "label": "Desbalanceado",
        "description": "Clases gaussianas con desbalance configurable. Ideal para estudiar el impacto del desbalance en métricas y fronteras.",
        "fixed_classes": True,
        "n_classes": 2,
        "default_noise": 1.0,
        "supports_multiclass": False,
        "params": ["imbalance_ratio", "cluster_std", "n_samples"],
    },
}


def generate_dataset(
    name: str,
    n_samples: int = 300,
    random_state: int = 42,
    noise: float = 0.2,
    n_classes: int = 2,
    factor: float = 0.5,
    cluster_std: float = 1.0,
    imbalance_ratio: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a 2D toy dataset by name.

    Parameters
    ----------
    name : str
        One of the keys in DATASET_CONFIGS.
    n_samples : int
        Total number of samples.
    random_state : int
        Random seed for reproducibility.
    noise : float
        Noise level (dataset-dependent).
    n_classes : int
        Number of classes (only for datasets that support it).
    factor : float
        Inner/outer radius ratio for circles dataset [0.1–0.9].
    cluster_std : float
        Cluster standard deviation for blobs dataset.

    Returns
    -------
    X : np.ndarray of shape (n_samples, 2)
    y : np.ndarray of shape (n_samples,)

    Notes
    -----
    imbalance_ratio : float in [0.05, 0.5]
        Fraction of minority class in 'imbalanced' dataset.
        0.1 means 10% minority vs 90% majority.
    """
    if name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: '{name}'. Choose from {list(DATASET_CONFIGS.keys())}")

    rng = np.random.default_rng(random_state)

    if name == "moons":
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)

    elif name == "circles":
        factor = float(np.clip(factor, 0.05, 0.95))
        X, y = make_circles(n_samples=n_samples, noise=noise, factor=factor, random_state=random_state)

    elif name == "blobs":
        n_classes = int(np.clip(n_classes, 2, 5))
        X, y = make_blobs(
            n_samples=n_samples,
            centers=n_classes,
            cluster_std=cluster_std,
            random_state=random_state,
        )

    elif name == "xor":
        X = rng.uniform(-1.5, 1.5, size=(n_samples, 2))
        y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
        X += rng.normal(0, noise, size=X.shape)

    elif name == "linear":
        n_classes = int(np.clip(n_classes, 2, 5))
        # n_informative must satisfy: n_classes <= 2^n_informative
        # For n_classes=5 we need n_informative>=3; we use max(2, ceil(log2(n_classes)))
        import math as _math
        n_informative = max(2, _math.ceil(_math.log2(n_classes)) + (1 if n_classes > 2 else 0))
        n_features = n_informative  # keep 2D by projecting below
        X_raw, y = make_classification(
            n_samples=n_samples,
            n_features=n_informative,
            n_informative=n_informative,
            n_redundant=0,
            n_clusters_per_class=1,
            n_classes=n_classes,
            flip_y=noise,
            random_state=random_state,
            class_sep=1.5,
        )
        # Project to 2D via PCA for visualization
        from sklearn.decomposition import PCA as _PCA
        X = _PCA(n_components=2, random_state=random_state).fit_transform(X_raw)

    elif name == "spirals":
        n_classes = int(np.clip(n_classes, 2, 5))
        X, y = _make_spirals(n_samples=n_samples, n_classes=n_classes, noise=noise, random_state=random_state)

    elif name == "imbalanced":
        ratio = float(np.clip(imbalance_ratio, 0.05, 0.50))
        n_minority = max(5, int(n_samples * ratio))
        n_majority = n_samples - n_minority
        std = max(0.3, float(cluster_std))
        X_maj, y_maj = make_blobs(
            n_samples=n_majority, centers=[[0.0, 0.0]], cluster_std=std, random_state=random_state
        )
        X_min, y_min = make_blobs(
            n_samples=n_minority, centers=[[3.0, 3.0]], cluster_std=std * 0.7, random_state=random_state + 1
        )
        y_maj[:] = 0
        y_min[:] = 1
        X = np.vstack([X_maj, X_min])
        y = np.concatenate([y_maj, y_min])
        # Shuffle
        rng2 = np.random.default_rng(random_state)
        idx = rng2.permutation(len(y))
        X, y = X[idx], y[idx]

    return X.astype(np.float64), y.astype(int)


def _make_spirals(
    n_samples: int,
    n_classes: int,
    noise: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate interleaved spiral dataset with `n_classes` arms."""
    rng = np.random.default_rng(random_state)
    n_per_class = n_samples // n_classes
    remainder = n_samples - n_per_class * n_classes

    X_list, y_list = [], []
    for k in range(n_classes):
        n_k = n_per_class + (1 if k < remainder else 0)
        t = np.linspace(0, 1, n_k)
        angle = t * 3 * np.pi + (2 * np.pi * k / n_classes)
        r = t
        x = r * np.cos(angle)
        y_coord = r * np.sin(angle)
        X_k = np.column_stack([x, y_coord])
        X_k += rng.normal(0, noise, size=X_k.shape)
        X_list.append(X_k)
        y_list.append(np.full(n_k, k, dtype=int))

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    # Shuffle
    idx = rng.permutation(len(y))
    return X[idx], y[idx]
