import numpy as np
import pytest
from core.datasets_regression import REGRESSION_DATASET_CONFIGS, generate_regression_dataset


class TestRegressionDatasetConfigs:
    """Verify REGRESSION_DATASET_CONFIGS structure."""

    def test_all_keys_present(self):
        expected = {"reg_linear", "reg_polynomial", "reg_sinusoidal", "reg_step", "reg_exponential"}
        assert set(REGRESSION_DATASET_CONFIGS.keys()) == expected

    @pytest.mark.parametrize("name", list(REGRESSION_DATASET_CONFIGS.keys()))
    def test_config_has_required_fields(self, name):
        cfg = REGRESSION_DATASET_CONFIGS[name]
        assert "label" in cfg
        assert "description" in cfg
        assert "params" in cfg
        assert isinstance(cfg["params"], list)


class TestGenerateRegressionDataset:
    """Test generate_regression_dataset for all dataset types."""

    @pytest.mark.parametrize("name", list(REGRESSION_DATASET_CONFIGS.keys()))
    def test_output_shape(self, name):
        X, y = generate_regression_dataset(name, n_samples=100)
        assert X.shape == (100, 1)
        assert y.shape == (100,)

    @pytest.mark.parametrize("name", list(REGRESSION_DATASET_CONFIGS.keys()))
    def test_output_dtype(self, name):
        X, y = generate_regression_dataset(name, n_samples=50)
        assert X.dtype == np.float64
        assert y.dtype == np.float64

    @pytest.mark.parametrize("name", list(REGRESSION_DATASET_CONFIGS.keys()))
    def test_reproducibility(self, name):
        X1, y1 = generate_regression_dataset(name, n_samples=80, random_state=42)
        X2, y2 = generate_regression_dataset(name, n_samples=80, random_state=42)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    @pytest.mark.parametrize("name", list(REGRESSION_DATASET_CONFIGS.keys()))
    def test_different_seeds(self, name):
        X1, y1 = generate_regression_dataset(name, n_samples=80, random_state=1)
        X2, y2 = generate_regression_dataset(name, n_samples=80, random_state=99)
        assert not np.array_equal(y1, y2)

    def test_unknown_dataset_raises(self):
        with pytest.raises(ValueError, match="Unknown regression dataset"):
            generate_regression_dataset("nonexistent")

    @pytest.mark.parametrize("n", [50, 100, 300])
    def test_various_sample_sizes(self, n):
        X, y = generate_regression_dataset("reg_linear", n_samples=n)
        assert X.shape[0] == n
        assert y.shape[0] == n

    def test_polynomial_degree_clamp(self):
        X, y = generate_regression_dataset("reg_polynomial", degree=10)
        assert X.shape == (200, 1)

    def test_sinusoidal_frequency_clamp(self):
        X, y = generate_regression_dataset("reg_sinusoidal", frequency=10.0)
        assert X.shape == (200, 1)

    def test_step_n_steps_clamp(self):
        X, y = generate_regression_dataset("reg_step", n_steps=10)
        assert X.shape == (200, 1)

    def test_noise_zero(self):
        X, y = generate_regression_dataset("reg_linear", noise=0.0, n_samples=50)
        # With zero noise, y should be exactly 2*X + 1
        expected = 2.0 * X[:, 0] + 1.0
        np.testing.assert_allclose(y, expected, atol=1e-10)

    def test_noise_effect(self):
        _, y_low = generate_regression_dataset("reg_linear", noise=0.01, random_state=42)
        _, y_high = generate_regression_dataset("reg_linear", noise=1.0, random_state=42)
        # Higher noise → higher variance
        assert np.std(y_high) > np.std(y_low)

    def test_x_is_sorted(self):
        X, _ = generate_regression_dataset("reg_linear", n_samples=100)
        assert np.all(X[:-1, 0] <= X[1:, 0])
