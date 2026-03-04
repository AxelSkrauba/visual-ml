import numpy as np
import pytest
from sklearn.model_selection import train_test_split
from core.train import fit_and_evaluate, evaluate_model, compute_pedagogical_signals
from core.models import MODEL_REGISTRY, get_model_instance


class TestFitAndEvaluate:
    def test_returns_dict_with_required_keys(self, small_binary_dataset):
        X, y = small_binary_dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = get_model_instance("Logistic Regression", {})
        result = fit_and_evaluate(model, X_train, y_train, X_test, y_test)
        required = {"model", "train", "test"}
        assert required.issubset(result.keys())

    def test_metrics_structure(self, small_binary_dataset):
        X, y = small_binary_dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = get_model_instance("Logistic Regression", {})
        result = fit_and_evaluate(model, X_train, y_train, X_test, y_test)
        for split in ("train", "test"):
            metrics = result[split]
            assert "accuracy" in metrics
            assert "precision" in metrics
            assert "recall" in metrics
            assert "f1" in metrics
            assert "confusion_matrix" in metrics
            assert "report" in metrics
            assert "y_true" in metrics
            assert "y_pred" in metrics

    def test_accuracy_in_valid_range(self, small_binary_dataset):
        X, y = small_binary_dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = get_model_instance("Decision Tree", {"max_depth": 3})
        result = fit_and_evaluate(model, X_train, y_train, X_test, y_test)
        assert 0.0 <= result["train"]["accuracy"] <= 1.0
        assert 0.0 <= result["test"]["accuracy"] <= 1.0

    def test_f1_in_valid_range(self, small_binary_dataset):
        X, y = small_binary_dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = get_model_instance("KNN", {"n_neighbors": 5})
        result = fit_and_evaluate(model, X_train, y_train, X_test, y_test)
        assert 0.0 <= result["train"]["f1"] <= 1.0
        assert 0.0 <= result["test"]["f1"] <= 1.0

    def test_confusion_matrix_shape_binary(self, small_binary_dataset):
        X, y = small_binary_dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = get_model_instance("Logistic Regression", {})
        result = fit_and_evaluate(model, X_train, y_train, X_test, y_test)
        cm = result["test"]["confusion_matrix"]
        assert cm.shape == (2, 2)

    def test_confusion_matrix_shape_multiclass(self, small_multiclass_dataset):
        X, y = small_multiclass_dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = get_model_instance("KNN", {"n_neighbors": 3})
        result = fit_and_evaluate(model, X_train, y_train, X_test, y_test)
        n_classes = len(np.unique(y))
        assert result["test"]["confusion_matrix"].shape == (n_classes, n_classes)

    def test_report_is_string(self, small_binary_dataset):
        X, y = small_binary_dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = get_model_instance("Logistic Regression", {})
        result = fit_and_evaluate(model, X_train, y_train, X_test, y_test)
        assert isinstance(result["test"]["report"], str)
        assert isinstance(result["train"]["report"], str)

    def test_model_is_fitted(self, small_binary_dataset):
        from sklearn.exceptions import NotFittedError
        X, y = small_binary_dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = get_model_instance("Decision Tree", {})
        result = fit_and_evaluate(model, X_train, y_train, X_test, y_test)
        # Should not raise NotFittedError
        preds = result["model"].predict(X_test)
        assert len(preds) == len(y_test)

    def test_y_pred_length_matches_y_true(self, small_binary_dataset):
        X, y = small_binary_dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = get_model_instance("Random Forest", {"n_estimators": 10})
        result = fit_and_evaluate(model, X_train, y_train, X_test, y_test)
        assert len(result["train"]["y_pred"]) == len(y_train)
        assert len(result["test"]["y_pred"]) == len(y_test)

    @pytest.mark.parametrize("model_name", list(MODEL_REGISTRY.keys()))
    def test_all_models_complete_evaluation(self, model_name, small_binary_dataset):
        X, y = small_binary_dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        cfg = MODEL_REGISTRY[model_name]
        model = get_model_instance(model_name, cfg["default_params"])
        result = fit_and_evaluate(model, X_train, y_train, X_test, y_test)
        assert "train" in result and "test" in result
        assert 0.0 <= result["test"]["accuracy"] <= 1.0

    def test_train_accuracy_gte_test_for_overfit_model(self, small_binary_dataset):
        """A deep tree on small data should have train_acc >= test_acc."""
        X, y = small_binary_dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
        model = get_model_instance("Decision Tree", {"max_depth": 20})
        result = fit_and_evaluate(model, X_train, y_train, X_test, y_test)
        assert result["train"]["accuracy"] >= result["test"]["accuracy"]


class TestEvaluateModel:
    def test_evaluate_returns_metrics(self, fitted_decision_tree):
        model, X, y = fitted_decision_tree
        metrics = evaluate_model(model, X, y)
        assert "accuracy" in metrics
        assert "confusion_matrix" in metrics

    def test_evaluate_accuracy_perfect_on_train(self, fitted_decision_tree):
        """A fully grown tree should get near-perfect train accuracy."""
        from sklearn.tree import DecisionTreeClassifier
        model, X, y = fitted_decision_tree
        clf = DecisionTreeClassifier(random_state=42)
        clf.fit(X, y)
        metrics = evaluate_model(clf, X, y)
        assert metrics["accuracy"] > 0.95


class TestPedagogicalSignals:
    def _make_metrics(self, acc, f1=None, precision=None, recall=None):
        f1 = f1 if f1 is not None else acc
        return {
            "accuracy": acc,
            "f1": f1,
            "precision": precision if precision is not None else acc,
            "recall": recall if recall is not None else acc,
        }

    def test_returns_list(self):
        signals = compute_pedagogical_signals(
            self._make_metrics(0.9), self._make_metrics(0.85)
        )
        assert isinstance(signals, list)
        assert len(signals) >= 1

    def test_each_signal_has_level_and_message(self):
        signals = compute_pedagogical_signals(
            self._make_metrics(0.9), self._make_metrics(0.85)
        )
        for s in signals:
            assert "level" in s
            assert "message" in s
            assert s["level"] in ("warning", "info", "success", "tip")
            assert isinstance(s["message"], str) and len(s["message"]) > 10

    def test_severe_overfit_triggers_warning(self):
        signals = compute_pedagogical_signals(
            self._make_metrics(0.99), self._make_metrics(0.60)
        )
        levels = [s["level"] for s in signals]
        assert "warning" in levels

    def test_moderate_overfit_triggers_info(self):
        signals = compute_pedagogical_signals(
            self._make_metrics(0.90), self._make_metrics(0.75)
        )
        levels = [s["level"] for s in signals]
        assert "info" in levels

    def test_underfitting_triggers_warning(self):
        signals = compute_pedagogical_signals(
            self._make_metrics(0.55), self._make_metrics(0.53)
        )
        levels = [s["level"] for s in signals]
        assert "warning" in levels

    def test_good_generalization_triggers_success(self):
        signals = compute_pedagogical_signals(
            self._make_metrics(0.92, f1=0.91), self._make_metrics(0.90, f1=0.89)
        )
        levels = [s["level"] for s in signals]
        assert "success" in levels

    def test_imbalance_signal_when_acc_f1_gap_large(self):
        """High accuracy but low F1 suggests class imbalance impact."""
        signals = compute_pedagogical_signals(
            self._make_metrics(0.95, f1=0.70), self._make_metrics(0.92, f1=0.55)
        )
        levels = [s["level"] for s in signals]
        assert "warning" in levels
        messages = " ".join(s["message"] for s in signals)
        assert "desbalance" in messages.lower() or "f1" in messages.lower()

    def test_perfect_train_score_triggers_warning(self):
        signals = compute_pedagogical_signals(
            self._make_metrics(1.0), self._make_metrics(0.72)
        )
        levels = [s["level"] for s in signals]
        assert "warning" in levels

    def test_tip_returned_when_no_other_signals(self):
        """Edge case: mediocre but symmetric — should fall through to tip."""
        signals = compute_pedagogical_signals(
            self._make_metrics(0.75, f1=0.74), self._make_metrics(0.74, f1=0.73)
        )
        assert len(signals) >= 1
        # At minimum a success or tip should appear
        levels = [s["level"] for s in signals]
        assert any(l in ("success", "tip") for l in levels)
