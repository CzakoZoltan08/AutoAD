import pytest

from autoad.data_generators.anomaly_data_generator import AnomalyDataGenerator
from autoad.data_generators.noise_type import NoiseType

from sklearn.preprocessing import StandardScaler, MinMaxScaler


@pytest.fixture
def anomaly_data_generator():
    return AnomalyDataGenerator()


def test_anomaly_data_generator(anomaly_data_generator):
    data = anomaly_data_generator.generate()

    assert len(data['X_train']) == 1316
    assert len(data['y_train']) == 1316
    assert len(data['X_test']) == 439
    assert len(data['y_test']) == 439


def test_anomaly_data_generator_with_standard_scaler(anomaly_data_generator):
    data = anomaly_data_generator.generate(
        apply_data_scaling=True, scaler=StandardScaler())

    assert any(i > 1.0 for i in data['X_train'][:, 0]) is True
    assert any(i > 1.0 for i in data['X_test'][:, 0]) is True


def test_anomaly_data_generator_with_minmax_scaler(anomaly_data_generator):
    data = anomaly_data_generator.generate(
        apply_data_scaling=True, scaler=MinMaxScaler())

    assert max(data['X_train'][0]) <= 1
    assert max(data['X_test'][0]) <= 1
    assert min(data['X_train'][0]) >= 0
    assert min(data['X_test'][0]) >= 0


def test_anomaly_data_generator_with_duplicates(anomaly_data_generator):
    data = anomaly_data_generator.generate(
        noise_type=NoiseType.DUPLICATES, noise_ratio=2)

    assert len(data['X_train']) == 1391
    assert len(data['y_train']) == 1391
    assert len(data['X_test']) == 464
    assert len(data['y_test']) == 464


def test_anomaly_data_generator_with_irrelevand_features(
        anomaly_data_generator):
    data = anomaly_data_generator.generate(
        noise_type=NoiseType.IRRELEVANT_FEATURES, noise_ratio=0.5)

    assert data['X_train'].shape[1] == 42
    assert data['X_test'].shape[1] == 42


def test_anomaly_data_generator_with_error_labels(anomaly_data_generator):
    data = anomaly_data_generator.generate(
        noise_type=NoiseType.LABEL_ERROR, noise_ratio=0.5)

    assert len([x for x in data['y_train'] if x == 0]) < 700


def test_anomaly_data_generator_with_label_ratio(anomaly_data_generator):
    data = anomaly_data_generator.generate(labeled_anomaly_ratio=0.5)

    assert len([x for x in data['y_train'] if x == 0]) == 1279


def test_anomaly_data_generator_with_threshold(anomaly_data_generator):
    data = anomaly_data_generator.generate(threshold=1000)

    assert len(data['y_train'])+len(data['y_test']) <= 1010
