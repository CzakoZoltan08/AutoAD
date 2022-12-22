import numpy as np
import pandas as pd

from autoad.data_generators.data_generator import DataGenerator
from copulas.multivariate import VineCopula
from copulas.univariate import GaussianUnivariate
from numpy import ndarray


class VineCopulaDataGenerator(DataGenerator):
    def __init__(self) -> None:
        pass

    def generate(X,
                 normal_count: int = 1000,
                 anomaly_count: int = 100) -> tuple[ndarray, ndarray]:

        if X.shape[1] > 50:
            idx = np.random.choice(np.arange(X.shape[1]), 50, replace=False)
            X = X[:, idx]

        copula = VineCopula('center')  # default is the C-vine copula
        copula.fit(pd.DataFrame(X))

        X_synthetic_normal = copula.sample(normal_count).values

        X_synthetic_anomalies = np.zeros((anomaly_count, X.shape[1]))

        for i in range(X.shape[1]):
            kde = GaussianUnivariate()
            kde.fit(X[:, i])
            X_synthetic_anomalies[:, i] = kde.sample(anomaly_count)

        return X_synthetic_normal, X_synthetic_anomalies
