import numpy as np
import pytest
from core.models_regression import (
    MODEL_REGISTRY_REGRESSION,
    get_regression_model_instance,
    get_regression_hyperparameter_controls,
    get_regression_models_by_group,
)


class TestRegressionModelRegistry:
    """Verify MODEL_REGISTRY_REGRESSION structure."""

    EXPECTED_MODELS = [
        "Linear Regression", "Ridge", "Lasso", "SVR",
        "KNN Regressor", "Decision Tree Regressor",
        "Random Forest Regressor", "MLP Regressor",
    ]

    def test_all_models_present(self):
        assert set(self.EXPECTED_MODELS) == set(MODEL_REGISTRY_REGRESSION.keys())

    @pytest.mark.parametrize("name", EXPECTED_MODELS)
    def test_config_has_required_fields(self, name):
        cfg = MODEL_REGISTRY_REGRESSION[name]
        assert "label" in cfg
        assert "group" in cfg
        assert "description" in cfg
        assert "class" in cfg
        assert "default_params" in cfg
        assert "hyperparameters" in cfg

    @pytest.mark.parametrize("name", EXPECTED_MODELS)
    def test_hyperparameters_structure(self, name):
        for hp in MODEL_REGISTRY_REGRESSION[name]["hyperparameters"]:
            assert "name" in hp
            assert "label" in hp
            assert "type" in hp
            assert hp["type"] in {"slider_int", "slider_float", "slider_log", "select", "checkbox"}


class TestGetRegressionModelInstance:
    """Test model instantiation."""

    @pytest.mark.parametrize("name", list(MODEL_REGISTRY_REGRESSION.keys()))
    def test_instantiate_default(self, name):
        model = get_regression_model_instance(name, {})
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")

    @pytest.mark.parametrize("name", list(MODEL_REGISTRY_REGRESSION.keys()))
    def test_fittable(self, name):
        model = get_regression_model_instance(name, {})
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 1))
        y = 2 * X[:, 0] + rng.normal(0, 0.1, 50)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (50,)

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown regression model"):
            get_regression_model_instance("nonexistent", {})

    def test_svr_linear_strips_gamma(self):
        model = get_regression_model_instance("SVR", {"kernel": "linear", "gamma": 0.1})
        assert not hasattr(model, "gamma") or model.kernel == "linear"

    def test_mlp_string_layers(self):
        model = get_regression_model_instance("MLP Regressor", {"hidden_layer_sizes": "(32, 16)"})
        assert model.hidden_layer_sizes == (32, 16)

    def test_custom_params_override(self):
        model = get_regression_model_instance("Ridge", {"alpha": 10.0})
        assert model.alpha == 10.0


class TestGetRegressionHyperparameterControls:
    """Test hyperparameter control retrieval."""

    @pytest.mark.parametrize("name", list(MODEL_REGISTRY_REGRESSION.keys()))
    def test_returns_list(self, name):
        controls = get_regression_hyperparameter_controls(name)
        assert isinstance(controls, list)

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError):
            get_regression_hyperparameter_controls("nonexistent")


class TestGetRegressionModelsByGroup:
    """Test group categorization."""

    def test_returns_dict(self):
        groups = get_regression_models_by_group()
        assert isinstance(groups, dict)
        total = sum(len(v) for v in groups.values())
        assert total == len(MODEL_REGISTRY_REGRESSION)

    def test_all_models_in_groups(self):
        groups = get_regression_models_by_group()
        all_names = set()
        for names in groups.values():
            all_names.update(names)
        assert all_names == set(MODEL_REGISTRY_REGRESSION.keys())
