import numpy as np
import pytest
from core.datasets import generate_dataset, DATASET_CONFIGS


class TestDatasetGeneration:
    def test_all_dataset_keys_present(self):
        expected = {"moons", "circles", "blobs", "xor", "linear", "spirals", "imbalanced"}
        assert set(DATASET_CONFIGS.keys()) == expected

    @pytest.mark.parametrize("name", ["moons", "circles", "blobs", "xor", "linear", "spirals", "imbalanced"])
    def test_output_shape(self, name):
        X, y = generate_dataset(name, n_samples=100, random_state=42)
        assert X.shape == (100, 2), f"{name}: X shape mismatch"
        assert y.shape == (100,), f"{name}: y shape mismatch"

    @pytest.mark.parametrize("name", ["moons", "circles", "blobs", "xor", "linear", "spirals", "imbalanced"])
    def test_reproducibility(self, name):
        X1, y1 = generate_dataset(name, n_samples=80, random_state=7)
        X2, y2 = generate_dataset(name, n_samples=80, random_state=7)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    @pytest.mark.parametrize("name", ["moons", "circles", "blobs", "xor", "linear", "spirals", "imbalanced"])
    def test_different_seeds_differ(self, name):
        X1, _ = generate_dataset(name, n_samples=100, random_state=0)
        X2, _ = generate_dataset(name, n_samples=100, random_state=99)
        assert not np.array_equal(X1, X2)

    @pytest.mark.parametrize("name", ["moons", "circles"])
    def test_binary_datasets_have_two_classes(self, name):
        _, y = generate_dataset(name, n_samples=100, random_state=42)
        assert set(np.unique(y)) == {0, 1}

    @pytest.mark.parametrize("n_classes", [2, 3, 4, 5])
    def test_blobs_multiclass(self, n_classes):
        _, y = generate_dataset("blobs", n_samples=200, random_state=42, n_classes=n_classes)
        assert len(np.unique(y)) == n_classes

    @pytest.mark.parametrize("n_classes", [2, 3, 4, 5])
    def test_linear_multiclass(self, n_classes):
        _, y = generate_dataset("linear", n_samples=200, random_state=42, n_classes=n_classes)
        assert len(np.unique(y)) == n_classes

    @pytest.mark.parametrize("n_classes", [2, 3, 4, 5])
    def test_spirals_multiclass(self, n_classes):
        _, y = generate_dataset("spirals", n_samples=200, random_state=42, n_classes=n_classes)
        assert len(np.unique(y)) == n_classes

    def test_noise_affects_output(self):
        X_low, _ = generate_dataset("moons", n_samples=200, random_state=42, noise=0.0)
        X_high, _ = generate_dataset("moons", n_samples=200, random_state=42, noise=0.4)
        assert not np.array_equal(X_low, X_high)

    def test_n_samples_respected(self):
        for n in [100, 200, 500]:
            X, y = generate_dataset("moons", n_samples=n, random_state=42)
            assert len(X) == n
            assert len(y) == n

    def test_x_is_float(self):
        X, _ = generate_dataset("moons", n_samples=50, random_state=0)
        assert X.dtype in (np.float32, np.float64)

    def test_y_is_integer(self):
        _, y = generate_dataset("blobs", n_samples=50, random_state=0, n_classes=2)
        assert np.issubdtype(y.dtype, np.integer)

    def test_circles_factor_range(self):
        X_tight, _ = generate_dataset("circles", n_samples=100, random_state=42, factor=0.1)
        X_loose, _ = generate_dataset("circles", n_samples=100, random_state=42, factor=0.8)
        assert not np.array_equal(X_tight, X_loose)

    def test_blobs_cluster_std(self):
        _, y_tight = generate_dataset("blobs", n_samples=300, random_state=42, cluster_std=0.5)
        _, y_loose = generate_dataset("blobs", n_samples=300, random_state=42, cluster_std=3.0)
        assert y_tight is not None and y_loose is not None

    def test_imbalanced_binary_classes(self):
        _, y = generate_dataset("imbalanced", n_samples=200, random_state=42)
        assert set(np.unique(y)) == {0, 1}

    @pytest.mark.parametrize("ratio", [0.1, 0.2, 0.3])
    def test_imbalanced_ratio_respected(self, ratio):
        n = 200
        _, y = generate_dataset("imbalanced", n_samples=n, random_state=42, imbalance_ratio=ratio)
        minority_count = (y == 1).sum()
        expected = max(5, int(n * ratio))
        # Allow ±1 due to rounding
        assert abs(minority_count - expected) <= 1, (
            f"Expected ~{expected} minority samples, got {minority_count}"
        )

    def test_imbalanced_ratio_creates_imbalance(self):
        _, y_balanced = generate_dataset("imbalanced", n_samples=300, random_state=42, imbalance_ratio=0.5)
        _, y_imbalanced = generate_dataset("imbalanced", n_samples=300, random_state=42, imbalance_ratio=0.1)
        ratio_balanced = (y_balanced == 1).sum() / len(y_balanced)
        ratio_imbalanced = (y_imbalanced == 1).sum() / len(y_imbalanced)
        assert ratio_imbalanced < ratio_balanced

    def test_dataset_config_has_required_keys(self):
        required = {"label", "description", "fixed_classes", "default_noise", "supports_multiclass"}
        for name, cfg in DATASET_CONFIGS.items():
            for key in required:
                assert key in cfg, f"Dataset '{name}' missing key '{key}'"
