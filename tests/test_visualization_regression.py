import numpy as np
import pytest
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from core.visualization_regression import (
    plot_prediction_curve,
    plot_residuals,
    plot_prediction_error,
    plot_regression_metrics_comparison,
)

matplotlib.use("Agg")


@pytest.fixture
def regression_data_and_model():
    """Fitted linear model on simple 1D data."""
    rng = np.random.default_rng(42)
    X_train = np.sort(rng.uniform(-2, 2, 40)).reshape(-1, 1)
    y_train = 2 * X_train[:, 0] + 1 + rng.normal(0, 0.1, 40)
    X_test = np.sort(rng.uniform(-2, 2, 20)).reshape(-1, 1)
    y_test = 2 * X_test[:, 0] + 1 + rng.normal(0, 0.1, 20)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test


@pytest.fixture
def tree_data_and_model():
    """Fitted decision tree regressor."""
    rng = np.random.default_rng(7)
    X_train = np.sort(rng.uniform(-3, 3, 50)).reshape(-1, 1)
    y_train = np.sin(X_train[:, 0]) + rng.normal(0, 0.1, 50)
    X_test = np.sort(rng.uniform(-3, 3, 20)).reshape(-1, 1)
    y_test = np.sin(X_test[:, 0]) + rng.normal(0, 0.1, 20)
    model = DecisionTreeRegressor(max_depth=4, random_state=7)
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test


class TestPlotPredictionCurve:
    """Test plot_prediction_curve."""

    @pytest.mark.parametrize("theme", ["dark", "light"])
    def test_returns_figure(self, regression_data_and_model, theme):
        model, X_tr, X_te, y_tr, y_te = regression_data_and_model
        fig = plot_prediction_curve(model, X_tr, X_te, y_tr, y_te, theme=theme)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_show_train_only(self, regression_data_and_model):
        model, X_tr, X_te, y_tr, y_te = regression_data_and_model
        fig = plot_prediction_curve(model, X_tr, X_te, y_tr, y_te, show_train=True, show_test=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_show_test_only(self, regression_data_and_model):
        model, X_tr, X_te, y_tr, y_te = regression_data_and_model
        fig = plot_prediction_curve(model, X_tr, X_te, y_tr, y_te, show_train=False, show_test=True)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_show_neither(self, regression_data_and_model):
        model, X_tr, X_te, y_tr, y_te = regression_data_and_model
        fig = plot_prediction_curve(model, X_tr, X_te, y_tr, y_te, show_train=False, show_test=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_tree_model(self, tree_data_and_model):
        model, X_tr, X_te, y_tr, y_te = tree_data_and_model
        fig = plot_prediction_curve(model, X_tr, X_te, y_tr, y_te, theme="dark")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotResiduals:
    """Test plot_residuals."""

    @pytest.mark.parametrize("theme", ["dark", "light"])
    def test_returns_figure(self, theme):
        y_true = np.array([1, 2, 3, 4, 5], dtype=float)
        y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1], dtype=float)
        fig = plot_residuals(y_true, y_pred, theme=theme)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_title_suffix(self):
        y_true = np.array([1, 2, 3], dtype=float)
        y_pred = np.array([1.1, 2.1, 2.9], dtype=float)
        fig = plot_residuals(y_true, y_pred, title_suffix="— Test")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_perfect_predictions(self):
        y = np.array([1, 2, 3, 4], dtype=float)
        fig = plot_residuals(y, y, theme="light")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_large_residuals(self):
        y_true = np.array([0, 0, 0], dtype=float)
        y_pred = np.array([10, -10, 5], dtype=float)
        fig = plot_residuals(y_true, y_pred)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotPredictionError:
    """Test plot_prediction_error."""

    @pytest.mark.parametrize("theme", ["dark", "light"])
    def test_returns_figure(self, theme):
        y_true = np.array([1, 2, 3, 4, 5], dtype=float)
        y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1], dtype=float)
        fig = plot_prediction_error(y_true, y_pred, theme=theme)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_title_suffix(self):
        y_true = np.array([1, 2, 3], dtype=float)
        y_pred = np.array([1.1, 2.1, 2.9], dtype=float)
        fig = plot_prediction_error(y_true, y_pred, title_suffix="— Train")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_perfect_predictions(self):
        y = np.array([1, 2, 3, 4], dtype=float)
        fig = plot_prediction_error(y, y, theme="dark")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotRegressionMetricsComparison:
    """Test plot_regression_metrics_comparison."""

    def _make_results(self, n=3):
        results = []
        for i in range(n):
            results.append({
                "name": f"Model_{i}",
                "train": {"r2": 0.9 - i * 0.1, "rmse": 0.1 + i * 0.05, "mae": 0.08 + i * 0.04},
                "test": {"r2": 0.85 - i * 0.1, "rmse": 0.15 + i * 0.05, "mae": 0.1 + i * 0.04},
            })
        return results

    @pytest.mark.parametrize("theme", ["dark", "light"])
    def test_returns_figure(self, theme):
        results = self._make_results(3)
        fig = plot_regression_metrics_comparison(results, theme=theme)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_single_model(self):
        results = self._make_results(1)
        fig = plot_regression_metrics_comparison(results)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_six_models(self):
        results = self._make_results(6)
        fig = plot_regression_metrics_comparison(results)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
