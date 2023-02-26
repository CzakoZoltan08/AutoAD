from autoad.algorithms.base_detector import BaseDetector

from pyod.models.abod import ABOD


class AngleBaseOutlierDetection(BaseDetector):
    def __init__(self,
                 n_neighbors=5,
                 method='fast',
                 contamination=0.1):
        super(AngleBaseOutlierDetection, self).__init__(
            contamination=contamination)
        self.method = method
        self.n_neighbors = n_neighbors
        self.contamination = contamination

    def fit(self, X, y=None):
        self.detector_ = ABOD(contamination=self.contamination,
                              n_neighbors=self.n_neighbors,
                              method=self.method,
                              )

        self.detector_.fit(X=X)

        return self

    def predict(self, X):
        return self.detector_.predict(X)
