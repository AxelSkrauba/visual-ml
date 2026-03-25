import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor

from core.train_regression import (
    evaluate_regression_model,
    fit_and_evaluate_regression,
    compute_regression_pedagogical_signals,
)


@pytest.fixture
def simple_regression_data():
    """Simple 1D regression dataset for fast testing."""
    rng = np.random.default_rng(42)
    X = np.sort(rng.uniform(-2, 2, 60)).reshape(-1, 1)
    y = 2 * X[:, 0] + 1 + rng.normal(0, 0.1, 60)
    return X, y


@pytest.fixture
def fitted_linear(simple_regression_data):
    X, y = simple_regression_data
    model = LinearRegression()
    model.fit(X, y)
    return model, X, y


class TestEvaluateRegressionModel:
    """Test evaluate_regression_model."""

    def test_returns_expected_keys(self, fitted_linear):
        model, X, y = fitted_linear
        result = evaluate_regression_model(model, X, y)
        assert "r2" in result
        assert "rmse" in result
        assert "mae" in result
        assert "y_true" in result
        assert "y_pred" in result

    def test_r2_range(self, fitted_linear):
        model, X, y = fitted_linear
        result = evaluate_regression_model(model, X, y)
        assert result["r2"] > 0.9  # linear model on linear data

    def test_rmse_positive(self, fitted_linear):
        model, X, y = fitted_linear
        result = evaluate_regression_model(model, X, y)
        assert result["rmse"] >= 0

    def test_mae_positive(self, fitted_linear):
        model, X, y = fitted_linear
        result = evaluate_regression_model(model, X, y)
        assert result["mae"] >= 0

    def test_y_pred_shape(self, fitted_linear):
        model, X, y = fitted_linear
        result = evaluate_regression_model(model, X, y)
        assert result["y_pred"].shape == y.shape

    def test_perfect_model(self):
        X = np.array([[1], [2], [3]], dtype=float)
        y = np.array([1.0, 2.0, 3.0])
        model = LinearRegression()
        model.fit(X, y)
        result = evaluate_regression_model(model, X, y)
        assert result["r2"] > 0.999
        assert result["rmse"] < 0.001
        assert result["mae"] < 0.001


class TestFitAndEvaluateRegression:
    """Test fit_and_evaluate_regression."""

    def test_returns_expected_keys(self, simple_regression_data):
        X, y = simple_regression_data
        X_train, X_test = X[:40], X[40:]
        y_train, y_test = y[:40], y[40:]
        model = LinearRegression()
        result = fit_and_evaluate_regression(model, X_train, y_train, X_test, y_test)
        assert "model" in result
        assert "train" in result
        assert "test" in result

    def test_model_is_fitted(self, simple_regression_data):
        X, y = simple_regression_data
        model = LinearRegression()
        result = fit_and_evaluate_regression(model, X[:40], y[:40], X[40:], y[40:])
        preds = result["model"].predict(X[:5])
        assert preds.shape == (5,)

    def test_train_metrics_complete(self, simple_regression_data):
        X, y = simple_regression_data
        model = Ridge(alpha=0.1)
        result = fit_and_evaluate_regression(model, X[:40], y[:40], X[40:], y[40:])
        for key in ("r2", "rmse", "mae", "y_true", "y_pred"):
            assert key in result["train"]
            assert key in result["test"]


class TestComputeRegressionPedagogicalSignals:
    """Test pedagogical signal generation for regression."""

    def _make_metrics(self, r2, rmse):
        return {"r2": r2, "rmse": rmse, "mae": rmse * 0.8}

    def test_severe_overfitting(self):
        train = self._make_metrics(r2=0.99, rmse=0.05)
        test = self._make_metrics(r2=0.50, rmse=0.5)
        signals = compute_regression_pedagogical_signals(train, test)
        assert any("Sobreajuste severo" in s["message"] for s in signals)

    def test_moderate_overfitting(self):
        train = self._make_metrics(r2=0.90, rmse=0.1)
        test = self._make_metrics(r2=0.75, rmse=0.2)
        signals = compute_regression_pedagogical_signals(train, test)
        assert any("sobreajuste moderado" in s["message"].lower() for s in signals)

    def test_underfitting(self):
        train = self._make_metrics(r2=0.20, rmse=0.8)
        test = self._make_metrics(r2=0.18, rmse=0.85)
        signals = compute_regression_pedagogical_signals(train, test)
        assert any("Subajuste" in s["message"] for s in signals)

    def test_negative_r2(self):
        train = self._make_metrics(r2=0.10, rmse=0.9)
        test = self._make_metrics(r2=-0.5, rmse=1.5)
        signals = compute_regression_pedagogical_signals(train, test)
        assert any("negativo" in s["message"] for s in signals)

    def test_good_generalization(self):
        train = self._make_metrics(r2=0.92, rmse=0.1)
        test = self._make_metrics(r2=0.90, rmse=0.12)
        signals = compute_regression_pedagogical_signals(train, test)
        assert any(s["level"] == "success" for s in signals)

    def test_memorization_risk(self):
        train = self._make_metrics(r2=1.0, rmse=0.0)
        test = self._make_metrics(r2=0.50, rmse=0.5)
        signals = compute_regression_pedagogical_signals(train, test)
        levels = [s["level"] for s in signals]
        assert "warning" in levels

    def test_tip_when_no_signals(self):
        train = self._make_metrics(r2=0.75, rmse=0.2)
        test = self._make_metrics(r2=0.70, rmse=0.25)
        signals = compute_regression_pedagogical_signals(train, test)
        assert len(signals) >= 1

    def test_heteroscedasticity_hint(self):
        train = self._make_metrics(r2=0.85, rmse=0.1)
        test = self._make_metrics(r2=0.80, rmse=0.25)
        signals = compute_regression_pedagogical_signals(train, test)
        assert any("RMSE" in s["message"] for s in signals)

    def test_always_returns_list(self):
        train = self._make_metrics(r2=0.50, rmse=0.5)
        test = self._make_metrics(r2=0.48, rmse=0.52)
        signals = compute_regression_pedagogical_signals(train, test)
        assert isinstance(signals, list)
        assert len(signals) >= 1
        for s in signals:
            assert "level" in s
            assert "message" in s
