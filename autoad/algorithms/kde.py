from autoad.algorithms.base_detector import BaseDetector

from pyod.models.kde import KDE


class KDEAnomalyDetector(BaseDetector):
    def __init__(self,
                 bandwidth=1.0,
                 algorithm='auto',
                 metric='minkowski',
                 leaf_size=30,
                 contamination=0.1):
        super(KDEAnomalyDetector, self).__init__(
            contamination=contamination)
        self.bandwidth = bandwidth
        self.algorithm = algorithm
        self.contamination = contamination
        self.leaf_size = leaf_size
        self.metric = metric

    def fit(self, X, y=None):
        self.detector_ = KDE(contamination=self.contamination,
                             bandwidth=self.bandwidth,
                             algorithm=self.algorithm,
                             leaf_size=self.leaf_size,
                             metric=self.metric
                             )

        self.detector_.fit(X=X)

        return self

    def predict(self, X):
        return self.detector_.predict(X)
