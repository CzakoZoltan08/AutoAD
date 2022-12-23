import numpy as np
import os
import sys

from autoad.data_generators.anomaly_type import AnomalyType
from autoad.data_generators.dataset import Dataset
from autoad.data_generators.gaussian_mixture_data_generator import \
    GaussianMixtureDataGenerator
from autoad.data_generators.noise_type import NoiseType
from autoad.data_generators.vine_copula_data_generator import \
    VineCopulaDataGenerator

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import TransformerMixin


class AnomalyDataGenerator():
    def __init__(self) -> None:
        pass

    def _generate_data_by_anomaly_type(self, X, y, anomaly_type: AnomalyType):
        data_count = len(np.where(y == 0)[0])

        X = X[y == 0]
        y = y[y == 0]

        if anomaly_type in [AnomalyType.LOCAL,
                            AnomalyType.GLOBAL,
                            AnomalyType.CLUSTER]:
            X_normal, X_anomalies = GaussianMixtureDataGenerator(
                anomaly_type).generate(X, data_count)
        else:
            X_normal, X_anomalies = VineCopulaDataGenerator.generate(
                X, data_count)

        X = np.concatenate((X_normal, X_anomalies), axis=0)
        y = np.append(np.repeat(0, X_normal.shape[0]),
                      np.repeat(1, X_anomalies.shape[0]))

        return X, y

    def _add_redundant_anomalies(self, X, y, redundancy_level: int):
        if redundancy_level <= 1:
            pass
        else:
            normal_data = np.where(y == 0)[0]
            anomaly = np.where(y == 1)[0]

            anomaly = np.random.choice(
                np.where(y == 1)[0], int(len(anomaly) * redundancy_level))

            x = np.append(normal_data, anomaly)
            np.random.shuffle(x)
            X = X[x]
            y = y[x]

        return X, y

    def _add_irrelevant_features(self, X, irrelevant_feature_ratio: float):
        if irrelevant_feature_ratio == 0.0:
            pass
        else:
            noise_dim = int(irrelevant_feature_ratio /
                            (1 - irrelevant_feature_ratio) * X.shape[1])
            if noise_dim > 0:
                X_noise = []
                for i in range(noise_dim):
                    idx = np.random.choice(np.arange(X.shape[1]), 1)
                    X_min = np.min(X[:, idx])
                    X_max = np.max(X[:, idx])

                    X_noise.append(np.random.uniform(
                        X_min, X_max, size=(X.shape[0], 1)))

                X_noise = np.hstack(X_noise)
                X = np.concatenate((X, X_noise), axis=1)
                idx = np.random.choice(
                    np.arange(X.shape[1]), X.shape[1], replace=False)
                X = X[:, idx]

        return X

    def _add_wrong_label(self, y, error_ratio: float):
        if error_ratio == 0.0:
            pass
        else:
            idx_flips = np.random.choice(np.arange(len(y)), int(
                len(y) * error_ratio), replace=False)
            y[idx_flips] = 1 - y[idx_flips]

        return y

    def _remove_labels(self, y_train, labeled_anomaly_ratio):
        ids_normal = np.where(y_train == 0)[0]
        ids_anomaly = np.where(y_train == 1)[0]

        ids_labeled_anomaly = np.random.choice(ids_anomaly, int(
            labeled_anomaly_ratio * len(ids_anomaly)), replace=False)
        ids_unlabeled_anomaly = np.setdiff1d(ids_anomaly, ids_labeled_anomaly)
        ids_unlabeled = np.append(ids_normal, ids_unlabeled_anomaly)
        del ids_anomaly, ids_unlabeled_anomaly

        y_train[ids_unlabeled] = 0
        y_train[ids_labeled_anomaly] = 1

        return y_train

    def _add_noise(self,
                   X,
                   y,
                   noise_type,
                   noise_ratio):
        if noise_type == NoiseType.DUPLICATES:
            X, y = self._add_redundant_anomalies(
                X, y, redundancy_level=noise_ratio)
        elif noise_type == NoiseType.IRRELEVANT_FEATURES:
            X = self._add_irrelevant_features(
                X, irrelevant_feature_ratio=noise_ratio)
        elif noise_type == NoiseType.LABEL_ERROR:
            y = self._add_wrong_label(y, error_ratio=noise_ratio)

        return X, y

    def _sub_sample(self, threshold, X, y):
        if len(y) > threshold:
            idx_sample = np.random.choice(
                np.arange(len(y)-1), threshold, replace=False)
            X = X[idx_sample]
            y = y[idx_sample]

        return X, y

    def generate(self,
                 dataset: Dataset = Dataset.CARDIO,
                 anomaly_type: AnomalyType = AnomalyType.LOCAL,
                 noise_type: NoiseType = NoiseType.NONE,
                 labeled_anomaly_ratio: float = 1.0,
                 noise_ratio=0.1,
                 test_size: float = 0.25,
                 threshold: int = sys.maxsize,
                 apply_data_scaling: bool = False,
                 scaler: TransformerMixin = MinMaxScaler()):
        data = np.load(os.path.join(os.path.dirname(__file__), 'datasets',
                       dataset + '.npz'), allow_pickle=True)
        X = data['X']
        y = data['y']

        X, y = self._generate_data_by_anomaly_type(X, y, anomaly_type)
        X, y = self._add_noise(X, y, noise_type, noise_ratio)
        X, y = self._sub_sample(threshold, X, y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=True, stratify=y)

        if apply_data_scaling is True:
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)

        y_train = self._remove_labels(y_train, labeled_anomaly_ratio)

        return {'X_train': X_train,
                'y_train': y_train,
                'X_test': X_test,
                'y_test': y_test}
