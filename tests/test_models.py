import pytest
from core.models import MODEL_REGISTRY, get_model_instance, get_hyperparameter_controls


class TestModelRegistry:
    def test_all_expected_models_present(self):
        expected = {
            "Logistic Regression", "Perceptron", "MLP",
            "SVC", "KNN",
            "Decision Tree",
            "Random Forest", "Gradient Boosting", "AdaBoost",
            "Gaussian NB", "QDA",
        }
        assert expected.issubset(set(MODEL_REGISTRY.keys()))

    def test_each_model_has_required_keys(self):
        required = {"label", "group", "description", "class", "default_params", "hyperparameters"}
        for name, cfg in MODEL_REGISTRY.items():
            for key in required:
                assert key in cfg, f"Model '{name}' missing key '{key}'"

    def test_hyperparameters_have_required_fields(self):
        required_fields = {"name", "label", "type", "default"}
        for model_name, cfg in MODEL_REGISTRY.items():
            for hp in cfg["hyperparameters"]:
                for field in required_fields:
                    assert field in hp, (
                        f"Model '{model_name}', hyperparameter '{hp.get('name', '?')}' missing field '{field}'"
                    )

    def test_hyperparameter_types_are_valid(self):
        valid_types = {"slider_int", "slider_float", "slider_log", "select", "checkbox"}
        for model_name, cfg in MODEL_REGISTRY.items():
            for hp in cfg["hyperparameters"]:
                assert hp["type"] in valid_types, (
                    f"Model '{model_name}', param '{hp['name']}': invalid type '{hp['type']}'"
                )

    def test_slider_params_have_min_max(self):
        slider_types = {"slider_int", "slider_float", "slider_log"}
        for model_name, cfg in MODEL_REGISTRY.items():
            for hp in cfg["hyperparameters"]:
                if hp["type"] in slider_types:
                    assert "min" in hp and "max" in hp, (
                        f"Model '{model_name}', param '{hp['name']}' missing min/max"
                    )
                    assert hp["min"] < hp["max"], (
                        f"Model '{model_name}', param '{hp['name']}': min >= max"
                    )

    def test_select_params_have_options(self):
        for model_name, cfg in MODEL_REGISTRY.items():
            for hp in cfg["hyperparameters"]:
                if hp["type"] == "select":
                    assert "options" in hp, (
                        f"Model '{model_name}', param '{hp['name']}' select missing options"
                    )
                    assert len(hp["options"]) >= 2

    def test_default_is_within_range_for_sliders(self):
        slider_types = {"slider_int", "slider_float", "slider_log"}
        for model_name, cfg in MODEL_REGISTRY.items():
            for hp in cfg["hyperparameters"]:
                if hp["type"] in slider_types:
                    assert hp["min"] <= hp["default"] <= hp["max"], (
                        f"Model '{model_name}', param '{hp['name']}': default {hp['default']} "
                        f"outside [{hp['min']}, {hp['max']}]"
                    )

    def test_default_in_options_for_select(self):
        for model_name, cfg in MODEL_REGISTRY.items():
            for hp in cfg["hyperparameters"]:
                if hp["type"] == "select":
                    assert hp["default"] in hp["options"], (
                        f"Model '{model_name}', param '{hp['name']}': default '{hp['default']}' "
                        f"not in options {hp['options']}"
                    )

    def test_groups_are_valid(self):
        valid_groups = {
            "Lineales", "Red Neuronal", "Kernel / Margen", "Vecindad",
            "Árbol", "Ensemble Bagging", "Ensemble Boosting", "Probabilístico"
        }
        for name, cfg in MODEL_REGISTRY.items():
            assert cfg["group"] in valid_groups, (
                f"Model '{name}' has invalid group '{cfg['group']}'"
            )


class TestGetModelInstance:
    def test_returns_unfitted_sklearn_model(self):
        from sklearn.base import BaseEstimator
        model = get_model_instance("Logistic Regression", {})
        assert isinstance(model, BaseEstimator)

    def test_custom_params_applied(self):
        model = get_model_instance("Decision Tree", {"max_depth": 5, "criterion": "entropy"})
        assert model.max_depth == 5
        assert model.criterion == "entropy"

    def test_default_params_used_when_empty(self):
        model = get_model_instance("KNN", {})
        cfg = MODEL_REGISTRY["KNN"]
        expected_k = cfg["default_params"].get("n_neighbors", 5)
        assert model.n_neighbors == expected_k

    @pytest.mark.parametrize("model_name", list(MODEL_REGISTRY.keys()))
    def test_all_models_instantiable(self, model_name):
        cfg = MODEL_REGISTRY[model_name]
        model = get_model_instance(model_name, cfg["default_params"])
        assert model is not None

    @pytest.mark.parametrize("model_name", list(MODEL_REGISTRY.keys()))
    def test_all_models_are_fittable(self, model_name, small_binary_dataset):
        X, y = small_binary_dataset
        cfg = MODEL_REGISTRY[model_name]
        model = get_model_instance(model_name, cfg["default_params"])
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(y)


class TestMLPModel:
    def test_mlp_instantiable_with_default_params(self):
        cfg = MODEL_REGISTRY["MLP"]
        model = get_model_instance("MLP", cfg["default_params"])
        assert model is not None

    def test_mlp_hidden_layer_sizes_string_parsed(self):
        """Selectbox returns string like '(64, 32)' — must be converted to tuple."""
        model = get_model_instance("MLP", {"hidden_layer_sizes": "(32, 16)"})
        assert model.hidden_layer_sizes == (32, 16)

    def test_mlp_single_layer_string_parsed(self):
        model = get_model_instance("MLP", {"hidden_layer_sizes": "(64,)"})
        assert model.hidden_layer_sizes == (64,)

    def test_mlp_tuple_passthrough(self):
        """Tuple default_params pass through unchanged."""
        model = get_model_instance("MLP", {"hidden_layer_sizes": (64, 32)})
        assert model.hidden_layer_sizes == (64, 32)

    def test_mlp_fittable(self, small_binary_dataset):
        X, y = small_binary_dataset
        model = get_model_instance("MLP", {"hidden_layer_sizes": "(32,)", "max_iter": 200})
        model.fit(X, y)
        assert len(model.predict(X)) == len(y)


class TestGetHyperparameterControls:
    def test_returns_list(self):
        controls = get_hyperparameter_controls("SVC")
        assert isinstance(controls, list)

    def test_svc_kernel_conditional_gamma(self):
        controls = get_hyperparameter_controls("SVC")
        names = [c["name"] for c in controls]
        assert "kernel" in names
        assert "C" in names

    def test_decision_tree_has_max_depth(self):
        controls = get_hyperparameter_controls("Decision Tree")
        names = [c["name"] for c in controls]
        assert "max_depth" in names

    def test_knn_has_n_neighbors(self):
        controls = get_hyperparameter_controls("KNN")
        names = [c["name"] for c in controls]
        assert "n_neighbors" in names
