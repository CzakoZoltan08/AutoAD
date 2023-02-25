from autoad.algorithms.base_detector import BaseDetector

from pyod.models.feature_bagging import FeatureBagging


class FeatureBaggingOutlierDetection(BaseDetector):
    def __init__(self,
                 n_estimators=10,
                 max_features=1.0,
                 contamination=0.1):
        super(FeatureBaggingOutlierDetection, self).__init__(
            contamination=contamination)
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.contamination = contamination

    def fit(self, X, y=None):
        self.detector_ = FeatureBagging(contamination=self.contamination,
                                        n_estimators=self.n_estimators,
                                        max_features=self.max_features
                                        )

        self.detector_.fit(X=X)

        return self

    def predict(self, X):
        return self.detector_.decision_function(X)
