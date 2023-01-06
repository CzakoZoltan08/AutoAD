import pytest

import numpy as np
from autoad.algorithms.deep_autoencoding_gaussian_mixture_model \
    import DeepAutoencodingGaussianMixtureModel
from autoad.algorithms.deep_sad.deepsad import DeepSAD
from autoad.algorithms.deeplog_model import DeepLogModel

from autoad.algorithms.isolation_forest import IsolationForest
from autoad.data_generators.anomaly_data_generator import AnomalyDataGenerator

from sklearn.preprocessing import StandardScaler

from pyod.utils.utility import precision_n_scores
from sklearn.metrics import roc_auc_score


@pytest.fixture
def dataset():
    anomaly_data_generator = AnomalyDataGenerator()
    return anomaly_data_generator.generate(scaler=StandardScaler())


@pytest.fixture
def X_train(dataset):
    return dataset['X_train']


@pytest.fixture
def X_test(dataset):
    return dataset['X_test']


@pytest.fixture
def y_test(dataset):
    return dataset['y_test'].ravel()


@pytest.fixture
def y_train(dataset):
    return dataset['y_train'].ravel()


@pytest.fixture
def outliers_fraction(y_train, y_test):
    return (np.count_nonzero(y_train)+np.count_nonzero(y_test)) \
        / (len(y_train) + len(y_test))


@pytest.fixture
def random_state():
    return np.random.RandomState(42)


@pytest.fixture
def isolation_forest_classifier(outliers_fraction, random_state):
    return IsolationForest(contamination=outliers_fraction,
                           random_state=random_state)


@pytest.fixture
def deep_autoencoding_gaussian_mixture_model():
    return DeepAutoencodingGaussianMixtureModel()


@pytest.fixture
def deep_log_model():
    return DeepLogModel()


@pytest.fixture
def deep_sad_model():
    return DeepSAD()


def test_isolation_forest(X_train,
                          X_test,
                          y_test,
                          isolation_forest_classifier):
    isolation_forest_classifier.fit(X_train)

    y_pred = isolation_forest_classifier.predict(X_test)

    roc = round(roc_auc_score(y_test, y_pred), ndigits=4)
    prn = round(precision_n_scores(y_test, y_pred), ndigits=4)

    assert roc > 0.5
    assert prn > 0.5


def test_deep_autoencoding_gaussian_mixture_model(
        X_train,
        X_test,
        y_test,
        deep_autoencoding_gaussian_mixture_model):
    deep_autoencoding_gaussian_mixture_model.fit(X_train)

    energy = deep_autoencoding_gaussian_mixture_model.predict(X_test)

    ano_index = np.arange(len(energy))[energy > np.percentile(energy, 75)]

    y_pred = y_test.copy()
    for index in ano_index:
        y_pred[index] = 1

    roc = round(roc_auc_score(y_test, y_pred), ndigits=4)

    assert roc > 0.5


def test_deep_sad_model(X_train,
                        X_test,
                        y_train,
                        y_test,
                        deep_sad_model):
    deep_sad_model.fit(X_train, y_train)

    y_pred = deep_sad_model.predict(X_test)

    roc = round(roc_auc_score(y_test, y_pred), ndigits=4)
    prn = round(precision_n_scores(y_test, y_pred), ndigits=4)

    assert roc > 0.5
    assert prn > 0.5
