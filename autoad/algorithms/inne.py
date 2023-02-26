from autoad.algorithms.base_detector import BaseDetector

from pyod.models.inne import INNE


class InneAnomalyDetector(BaseDetector):
    def __init__(self,
                 n_estimators=200,
                 contamination=0.1):
        super(InneAnomalyDetector, self).__init__(
            contamination=contamination)
        self.n_estimators = n_estimators
        self.contamination = contamination

    def fit(self, X, y=None):
        self.detector_ = INNE(contamination=self.contamination,
                              n_estimators=self.n_estimators,
                              )

        self.detector_.fit(X=X)

        return self

    def predict(self, X):
        return self.detector_.predict(X)
