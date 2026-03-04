import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression


@pytest.fixture
def small_binary_dataset():
    """50-sample, 2-class dataset for fast testing."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((50, 2))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


@pytest.fixture
def small_multiclass_dataset():
    """60-sample, 3-class dataset for fast testing."""
    from sklearn.datasets import make_blobs
    X, y = make_blobs(n_samples=60, centers=3, random_state=42)
    return X, y


@pytest.fixture
def fitted_decision_tree(small_binary_dataset):
    X, y = small_binary_dataset
    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(X, y)
    return clf, X, y


@pytest.fixture
def fitted_logreg(small_binary_dataset):
    X, y = small_binary_dataset
    clf = LogisticRegression(random_state=42, max_iter=200)
    clf.fit(X, y)
    return clf, X, y
