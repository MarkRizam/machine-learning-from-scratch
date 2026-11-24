"""AIR-001 V1.1 regression checks from the scientific audit."""
import numpy as np
import pytest
from ml_from_scratch import models as m


@pytest.mark.parametrize('factory', [lambda: m.KNN(1), m.GaussianNB, m.DecisionTree, lambda: m.RandomForest(2)])
@pytest.mark.parametrize('bad_label', [np.nan, np.inf, -np.inf, None])
def test_invalid_class_labels(factory, bad_label):
    with pytest.raises(ValueError):
        factory().fit([[0], [1], [2]], [0, 1, bad_label])


@pytest.mark.parametrize('factory', [lambda: m.KNN(1), m.GaussianNB, m.DecisionTree, lambda: m.RandomForest(5)])
def test_string_labels_supported(factory):
    model = factory().fit([[0], [0.1], [0.2], [10], [10.1], [10.2]], ['low']*3 + ['high']*3)
    np.testing.assert_array_equal(model.predict([[0], [10]]), ['low', 'high'])


@pytest.mark.parametrize('factory', [
    lambda: m.DecisionTree(max_depth=np.nan),
    lambda: m.DecisionTree(max_depth=1.5),
    lambda: m.DecisionTree(min_leaf=1.5),
    lambda: m.DecisionTree(max_features=1.5),
    lambda: m.RandomForest(max_depth=np.nan),
    lambda: m.RandomForest(n_trees=True),
    lambda: m.RandomForest(max_features=1.5),
    lambda: m.KMeans(tol=np.nan),
    lambda: m.KMeans(tol=np.inf),
    lambda: m.KMeans(steps=1.5),
    lambda: m.KMeans(k=True),
    lambda: m.KNN(k=True),
    lambda: m.LogisticRegression(steps=True),
    lambda: m.PCA(True).fit([[0, 1], [1, 2]]),
    lambda: m.gradient_descent([[0], [1]], [0, 1], epochs=1.5),
    lambda: m.gradient_descent([[0], [1]], [0, 1], batch_size=True),
])
def test_invalid_hyperparameters(factory):
    with pytest.raises(ValueError):
        factory()
