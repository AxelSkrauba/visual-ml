import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for tests
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from core.models import MODEL_REGISTRY, get_model_instance
from core.train import fit_and_evaluate
from core.visualization import (
    plot_decision_boundary,
    plot_confusion_matrix,
    plot_metrics_comparison,
)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture
def binary_result(small_binary_dataset):
    X, y = small_binary_dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = get_model_instance("Decision Tree", {"max_depth": 3})
    return fit_and_evaluate(model, X_train, y_train, X_test, y_test), X_train, X_test, y_train, y_test


@pytest.fixture
def multiclass_result(small_multiclass_dataset):
    X, y = small_multiclass_dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = get_model_instance("KNN", {"n_neighbors": 3})
    return fit_and_evaluate(model, X_train, y_train, X_test, y_test), X_train, X_test, y_train, y_test


class TestPlotDecisionBoundary:
    def test_returns_figure(self, binary_result):
        result, X_train, X_test, y_train, y_test = binary_result
        fig = plot_decision_boundary(
            result["model"], X_train, X_test, y_train, y_test,
            show_train=True, show_test=True, theme="dark"
        )
        assert isinstance(fig, plt.Figure)

    def test_returns_figure_light_theme(self, binary_result):
        result, X_train, X_test, y_train, y_test = binary_result
        fig = plot_decision_boundary(
            result["model"], X_train, X_test, y_train, y_test,
            show_train=True, show_test=True, theme="light"
        )
        assert isinstance(fig, plt.Figure)

    def test_show_train_only(self, binary_result):
        result, X_train, X_test, y_train, y_test = binary_result
        fig = plot_decision_boundary(
            result["model"], X_train, X_test, y_train, y_test,
            show_train=True, show_test=False, theme="dark"
        )
        assert isinstance(fig, plt.Figure)

    def test_show_test_only(self, binary_result):
        result, X_train, X_test, y_train, y_test = binary_result
        fig = plot_decision_boundary(
            result["model"], X_train, X_test, y_train, y_test,
            show_train=False, show_test=True, theme="dark"
        )
        assert isinstance(fig, plt.Figure)

    def test_show_neither_does_not_crash(self, binary_result):
        result, X_train, X_test, y_train, y_test = binary_result
        fig = plot_decision_boundary(
            result["model"], X_train, X_test, y_train, y_test,
            show_train=False, show_test=False, theme="dark"
        )
        assert isinstance(fig, plt.Figure)

    def test_multiclass_does_not_crash(self, multiclass_result):
        result, X_train, X_test, y_train, y_test = multiclass_result
        fig = plot_decision_boundary(
            result["model"], X_train, X_test, y_train, y_test,
            show_train=True, show_test=True, theme="dark"
        )
        assert isinstance(fig, plt.Figure)

    @pytest.mark.parametrize("model_name", ["Logistic Regression", "SVC", "KNN",
                                             "Decision Tree", "Random Forest",
                                             "Gaussian NB", "QDA"])
    def test_various_models_do_not_crash(self, model_name, small_binary_dataset):
        X, y = small_binary_dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        cfg = MODEL_REGISTRY[model_name]
        model = get_model_instance(model_name, cfg["default_params"])
        result = fit_and_evaluate(model, X_train, y_train, X_test, y_test)
        fig = plot_decision_boundary(
            result["model"], X_train, X_test, y_train, y_test,
            show_train=True, show_test=True, theme="dark"
        )
        assert isinstance(fig, plt.Figure)


class TestPlotConfusionMatrix:
    def test_returns_figure_binary(self, binary_result):
        result, *_ = binary_result
        cm = result["test"]["confusion_matrix"]
        fig = plot_confusion_matrix(cm, labels=["0", "1"], theme="dark")
        assert isinstance(fig, plt.Figure)

    def test_returns_figure_multiclass(self, multiclass_result):
        result, *_ = multiclass_result
        cm = result["test"]["confusion_matrix"]
        labels = [str(i) for i in range(cm.shape[0])]
        fig = plot_confusion_matrix(cm, labels=labels, theme="dark")
        assert isinstance(fig, plt.Figure)

    def test_returns_figure_light_theme(self, binary_result):
        result, *_ = binary_result
        cm = result["test"]["confusion_matrix"]
        fig = plot_confusion_matrix(cm, labels=["0", "1"], theme="light")
        assert isinstance(fig, plt.Figure)

    def test_cm_is_square(self, binary_result):
        result, *_ = binary_result
        cm = result["test"]["confusion_matrix"]
        assert cm.shape[0] == cm.shape[1]

    def test_normalize_true_returns_figure(self, binary_result):
        result, *_ = binary_result
        cm = result["test"]["confusion_matrix"]
        fig = plot_confusion_matrix(cm, labels=["0", "1"], theme="dark", normalize=True)
        assert isinstance(fig, plt.Figure)

    def test_normalize_false_returns_figure(self, binary_result):
        result, *_ = binary_result
        cm = result["test"]["confusion_matrix"]
        fig = plot_confusion_matrix(cm, labels=["0", "1"], theme="dark", normalize=False)
        assert isinstance(fig, plt.Figure)

    def test_normalize_light_theme(self, binary_result):
        result, *_ = binary_result
        cm = result["test"]["confusion_matrix"]
        fig = plot_confusion_matrix(cm, labels=["0", "1"], theme="light", normalize=True)
        assert isinstance(fig, plt.Figure)

    def test_perfect_cm_does_not_crash(self):
        """Diagonal-only CM (perfect model) — no zero off-diagonal rows."""
        cm = np.array([[30, 0], [0, 25]])
        fig = plot_confusion_matrix(cm, labels=["0", "1"], theme="dark")
        assert isinstance(fig, plt.Figure)

    def test_zero_row_cm_does_not_crash(self):
        """Row with all zeros (class never predicted) — should not divide by zero."""
        cm = np.array([[20, 5], [0, 0]])
        fig = plot_confusion_matrix(cm, labels=["0", "1"], theme="dark")
        assert isinstance(fig, plt.Figure)

    def test_imbalanced_cm_does_not_crash(self):
        """Strongly imbalanced: majority class 90%, minority 10%."""
        cm = np.array([[85, 5], [3, 7]])
        fig = plot_confusion_matrix(cm, labels=["0", "1"], theme="light", normalize=True)
        assert isinstance(fig, plt.Figure)

    def test_4class_cm_does_not_crash(self):
        cm = np.array([[18, 0, 0, 1], [0, 17, 1, 0], [0, 1, 18, 0], [7, 0, 0, 12]])
        labels = ["0", "1", "2", "3"]
        fig = plot_confusion_matrix(cm, labels=labels, theme="dark")
        assert isinstance(fig, plt.Figure)


class TestPlotMetricsComparison:
    def test_returns_figure_single_model(self, binary_result):
        result, *_ = binary_result
        results_list = [{"name": "Decision Tree", "train": result["train"], "test": result["test"]}]
        fig = plot_metrics_comparison(results_list, theme="dark")
        assert isinstance(fig, plt.Figure)

    def test_returns_figure_multiple_models(self, small_binary_dataset):
        X, y = small_binary_dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        results_list = []
        for name in ["Logistic Regression", "Decision Tree", "KNN"]:
            cfg = MODEL_REGISTRY[name]
            model = get_model_instance(name, cfg["default_params"])
            res = fit_and_evaluate(model, X_train, y_train, X_test, y_test)
            results_list.append({"name": name, "train": res["train"], "test": res["test"]})
        fig = plot_metrics_comparison(results_list, theme="dark")
        assert isinstance(fig, plt.Figure)

    def test_returns_figure_light_theme(self, binary_result):
        result, *_ = binary_result
        results_list = [{"name": "Decision Tree", "train": result["train"], "test": result["test"]}]
        fig = plot_metrics_comparison(results_list, theme="light")
        assert isinstance(fig, plt.Figure)
