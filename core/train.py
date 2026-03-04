import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def evaluate_model(model, X: np.ndarray, y: np.ndarray) -> dict:
    """
    Evaluate a fitted model on (X, y) and return a metrics dictionary.

    Parameters
    ----------
    model : fitted sklearn estimator
    X : np.ndarray of shape (n_samples, 2)
    y : np.ndarray of shape (n_samples,)

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1, confusion_matrix, report, y_true, y_pred
    """
    y_pred = model.predict(X)
    n_classes = len(np.unique(y))
    avg = "binary" if n_classes == 2 else "macro"
    zero_div = 0

    return {
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, average=avg, zero_division=zero_div)),
        "recall": float(recall_score(y, y_pred, average=avg, zero_division=zero_div)),
        "f1": float(f1_score(y, y_pred, average=avg, zero_division=zero_div)),
        "confusion_matrix": confusion_matrix(y, y_pred),
        "report": classification_report(y, y_pred, zero_division=zero_div),
        "y_true": y,
        "y_pred": y_pred,
    }


def fit_and_evaluate(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """
    Fit the model on training data and evaluate on both train and test sets.

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
        "train": evaluate_model(model, X_train, y_train),
        "test": evaluate_model(model, X_test, y_test),
    }


def compute_pedagogical_signals(train_metrics: dict, test_metrics: dict) -> list[dict]:
    """
    Derive pedagogical warning/info/success/tip signals from train vs test metrics.

    Returns a list of dicts with keys:
        level : 'warning' | 'info' | 'success' | 'tip'
        message : str
    """
    signals = []
    acc_train = train_metrics["accuracy"]
    acc_test = test_metrics["accuracy"]
    f1_train = train_metrics.get("f1", 0.0)
    f1_test = test_metrics.get("f1", 0.0)
    gap = acc_train - acc_test

    # ── Overfitting ──────────────────────────────────────────────────────
    if gap > 0.30:
        signals.append({
            "level": "warning",
            "message": (
                f"Sobreajuste severo (overfitting): train={acc_train:.2%} vs test={acc_test:.2%} "
                f"(diferencia={gap:.2%}). El modelo memoriza el ruido en lugar de aprender patrones generales. "
                "→ Prueba reducir la profundidad máxima, aumentar la regularización (C↓, lambda↑) o usar más datos."
            ),
        })
    elif gap > 0.12:
        signals.append({
            "level": "info",
            "message": (
                f"Posible sobreajuste moderado: brecha train/test = {gap:.2%}. "
                "El modelo funciona mejor en los datos que ya vio. "
                "→ Considera reducir la complejidad del modelo o aplicar regularización."
            ),
        })

    # ── Underfitting ─────────────────────────────────────────────────────
    if acc_train < 0.60:
        signals.append({
            "level": "warning",
            "message": (
                f"Subajuste (underfitting): accuracy en train = {acc_train:.2%}. "
                "El modelo es demasiado simple para capturar la estructura de los datos. "
                "→ Aumenta la complejidad (más profundidad, más vecinos, kernel no lineal)."
            ),
        })
    elif acc_train < 0.72 and gap < 0.05:
        signals.append({
            "level": "info",
            "message": (
                f"Rendimiento bajo tanto en train ({acc_train:.2%}) como en test ({acc_test:.2%}). "
                "Este dataset puede ser intrínsecamente difícil para este tipo de frontera. "
                "→ Prueba un modelo más flexible o revisa el nivel de ruido."
            ),
        })

    # ── Perfect train score (memorization risk) ──────────────────────────
    if acc_train >= 0.999 and acc_test < 0.90:
        signals.append({
            "level": "warning",
            "message": (
                f"Train accuracy = {acc_train:.2%} (perfecto), pero test = {acc_test:.2%}. "
                "Una accuracy perfecta en entrenamiento casi siempre indica memorización. "
                "→ En producción, lo que importa es el test. Aplica regularización o poda."
            ),
        })

    # ── Imbalance signal: accuracy vs F1 gap ─────────────────────────────
    acc_f1_gap = abs(acc_test - f1_test)
    if acc_f1_gap > 0.10 and acc_test > 0.75:
        signals.append({
            "level": "warning",
            "message": (
                f"Posible impacto de desbalance de clases: accuracy test = {acc_test:.2%} pero "
                f"F1 test = {f1_test:.2%} (diferencia={acc_f1_gap:.2%}). "
                "Con clases desbalanceadas, la accuracy puede ser engañosamente alta. "
                "→ Fíjate en el F1, la Matriz de Confusión y el Classification Report por clase."
            ),
        })

    # ── Low F1 despite acceptable accuracy ───────────────────────────────
    if f1_test < 0.60 and acc_test >= 0.65 and acc_f1_gap <= 0.10:
        signals.append({
            "level": "info",
            "message": (
                f"F1-score test = {f1_test:.2%} es bajo. Esto puede indicar que el modelo falla "
                "en alguna clase específica. Revisa el Classification Report para identificar "
                "qué clase tiene peor precision/recall."
            ),
        })

    # ── Good generalization ───────────────────────────────────────────────
    if gap <= 0.05 and acc_test >= 0.85 and f1_test >= 0.80:
        signals.append({
            "level": "success",
            "message": (
                f"Buena generalización: train={acc_train:.2%}, test={acc_test:.2%}, "
                f"F1 test={f1_test:.2%}. El modelo aprende sin memorizar. "
                "→ Buen punto de partida; puedes explorar si más datos mejoran aún más el resultado."
            ),
        })
    elif gap <= 0.05 and acc_test >= 0.72:
        signals.append({
            "level": "success",
            "message": (
                f"Modelo estable: la brecha train/test es pequeña ({gap:.2%}). "
                "No hay señales claras de sobreajuste ni subajuste."
            ),
        })

    # ── Pedagogical tip ───────────────────────────────────────────────────
    if not signals:
        signals.append({
            "level": "tip",
            "message": (
                "Mueve los sliders de hiperparámetros y observa cómo cambia la frontera de decisión. "
                "¿La frontera se vuelve más compleja con más parámetros? ¿Mejora en train pero no en test?"
            ),
        })

    return signals
