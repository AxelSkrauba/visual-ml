import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def evaluate_regression_model(model, X: np.ndarray, y: np.ndarray) -> dict:
    """
    Evaluate a fitted regression model on (X, y) and return a metrics dictionary.

    Parameters
    ----------
    model : fitted sklearn estimator with predict method
    X : np.ndarray of shape (n_samples, 1)
    y : np.ndarray of shape (n_samples,)

    Returns
    -------
    dict with keys: r2, rmse, mae, y_true, y_pred
    """
    y_pred = model.predict(X)
    return {
        "r2": float(r2_score(y, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
        "mae": float(mean_absolute_error(y, y_pred)),
        "y_true": y,
        "y_pred": y_pred,
    }


def fit_and_evaluate_regression(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """
    Fit a regression model on training data and evaluate on both splits.

    Parameters
    ----------
    model : unfitted sklearn estimator
    X_train, y_train : training split
    X_test, y_test : test split

    Returns
    -------
    dict with keys:
        - "model": fitted estimator
        - "train": metrics dict on training data
        - "test": metrics dict on test data
    """
    model.fit(X_train, y_train)
    return {
        "model": model,
        "train": evaluate_regression_model(model, X_train, y_train),
        "test": evaluate_regression_model(model, X_test, y_test),
    }


def compute_regression_pedagogical_signals(
    train_metrics: dict, test_metrics: dict
) -> list[dict]:
    """
    Derive pedagogical warning/info/success/tip signals from train vs test
    regression metrics.

    Returns a list of dicts with keys:
        level : 'warning' | 'info' | 'success' | 'tip'
        message : str
    """
    signals = []
    r2_train = train_metrics["r2"]
    r2_test = test_metrics["r2"]
    rmse_train = train_metrics["rmse"]
    rmse_test = test_metrics["rmse"]
    gap = r2_train - r2_test

    # ── Overfitting ──────────────────────────────────────────────────────
    if gap > 0.30:
        signals.append({
            "level": "warning",
            "message": (
                f"Sobreajuste severo: R² train={r2_train:.3f} vs test={r2_test:.3f} "
                f"(diferencia={gap:.3f}). El modelo memoriza el ruido en lugar de aprender la tendencia. "
                "→ Prueba reducir la complejidad (profundidad, neuronas) o aumentar la regularización."
            ),
        })
    elif gap > 0.12:
        signals.append({
            "level": "info",
            "message": (
                f"Posible sobreajuste moderado: brecha R² train/test = {gap:.3f}. "
                "→ Considera reducir la complejidad del modelo o aplicar regularización."
            ),
        })

    # ── Underfitting ─────────────────────────────────────────────────────
    if r2_train < 0.30:
        signals.append({
            "level": "warning",
            "message": (
                f"Subajuste (underfitting): R² en train = {r2_train:.3f}. "
                "El modelo es demasiado simple para capturar la tendencia de los datos. "
                "→ Prueba un modelo más flexible (mayor profundidad, kernel no lineal, más neuronas)."
            ),
        })
    elif r2_train < 0.60 and gap < 0.05:
        signals.append({
            "level": "info",
            "message": (
                f"Rendimiento bajo tanto en train (R²={r2_train:.3f}) como en test (R²={r2_test:.3f}). "
                "Este dataset puede requerir un modelo más flexible. "
                "→ Prueba un modelo no lineal o revisa el nivel de ruido."
            ),
        })

    # ── Near-perfect train (memorization risk) ───────────────────────────
    if r2_train >= 0.999 and r2_test < 0.80:
        signals.append({
            "level": "warning",
            "message": (
                f"R² train = {r2_train:.3f} (perfecto), pero test = {r2_test:.3f}. "
                "El modelo se ajusta exactamente a los datos de entrenamiento (memorización). "
                "→ Aplica regularización o limita la complejidad del modelo."
            ),
        })

    # ── Heteroscedasticity hint ──────────────────────────────────────────
    rmse_ratio = rmse_test / max(rmse_train, 1e-10)
    if rmse_ratio > 2.0 and r2_train > 0.70:
        signals.append({
            "level": "info",
            "message": (
                f"RMSE test ({rmse_test:.3f}) es {rmse_ratio:.1f}× mayor que RMSE train ({rmse_train:.3f}). "
                "Esto puede indicar heterocedasticidad (varianza del error no constante) o "
                "que el modelo no generaliza bien en ciertas regiones del espacio. "
                "→ Revisa el gráfico de residuos para detectar patrones."
            ),
        })

    # ── Negative R² on test ──────────────────────────────────────────────
    if r2_test < 0.0:
        signals.append({
            "level": "warning",
            "message": (
                f"R² test = {r2_test:.3f} (negativo). El modelo es peor que predecir la media. "
                "→ El modelo es inadecuado para estos datos. Cambia de modelo o ajusta hiperparámetros."
            ),
        })

    # ── Good generalization ───────────────────────────────────────────────
    if gap <= 0.05 and r2_test >= 0.85:
        signals.append({
            "level": "success",
            "message": (
                f"Buena generalización: R² train={r2_train:.3f}, test={r2_test:.3f}. "
                "El modelo captura bien la tendencia sin memorizar."
            ),
        })
    elif gap <= 0.05 and r2_test >= 0.60:
        signals.append({
            "level": "success",
            "message": (
                f"Modelo estable: la brecha R² train/test es pequeña ({gap:.3f}). "
                "No hay señales claras de sobreajuste."
            ),
        })

    # ── Tip when no signals ───────────────────────────────────────────────
    if not signals:
        signals.append({
            "level": "tip",
            "message": (
                "Mueve los sliders de hiperparámetros y observa cómo cambia la curva de predicción. "
                "¿La curva se vuelve más compleja con más parámetros? ¿Mejora R² en train pero no en test?"
            ),
        })

    return signals
