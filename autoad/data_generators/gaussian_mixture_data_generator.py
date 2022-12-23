import numpy as np
from autoad.data_generators.anomaly_type import AnomalyType

from autoad.data_generators.data_generator import DataGenerator
from sklearn.mixture import GaussianMixture


class GaussianMixtureDataGenerator(DataGenerator):
    def __init__(self, anomaly_type: AnomalyType) -> None:
        self.anomaly_type = anomaly_type

    def generate(self,
                 X,
                 normal_count: int = 1000,
                 anomaly_count: int = 100):
        metric_list = []
        n_components_list = list(np.arange(1, 10))

        for n_components in n_components_list:
            gm = GaussianMixture(n_components=n_components).fit(X)
            metric_list.append(gm.bic(X))

        best_n_components = n_components_list[np.argmin(metric_list)]

        gm = GaussianMixture(n_components=best_n_components).fit(X)

        X_synthetic_normal = gm.sample(normal_count)[0]

        if self.anomaly_type is AnomalyType.LOCAL:
            gm.covariances_ = 5 * gm.covariances_
            X_synthetic_anomalies = gm.sample(anomaly_count)[0]
        elif self.anomaly_type is AnomalyType.CLUSTER:
            gm.means_ = 5 * gm.means_
            X_synthetic_anomalies = gm.sample(anomaly_count)[0]
        else:
            X_synthetic_anomalies = []

            for i in range(X_synthetic_normal.shape[1]):
                low = np.min(X_synthetic_normal[:, i]) * (1 + 0.1)
                high = np.max(X_synthetic_normal[:, i]) * (1 + 0.1)

                X_synthetic_anomalies.append(
                    np.random.uniform(low=low, high=high, size=anomaly_count))

            X_synthetic_anomalies = np.array(X_synthetic_anomalies).T

        return X_synthetic_normal, X_synthetic_anomalies
