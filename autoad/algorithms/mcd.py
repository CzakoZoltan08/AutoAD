from autoad.algorithms.base_detector import BaseDetector

from pyod.models.mcd import MCD


class MinimumCovarianceDeterminant(BaseDetector):
    def __init__(self,
                 assume_centered=False,
                 support_fraction=True,
                 contamination=0.1):
        super(MinimumCovarianceDeterminant, self).__init__(
            contamination=contamination)
        self.assume_centered = assume_centered
        self.support_fraction = support_fraction
        self.contamination = contamination

    def fit(self, X, y=None):
        self.detector_ = MCD(contamination=self.contamination,
                             assume_centered=self.assume_centered,
                             support_fraction=self.support_fraction
                             )

        self.detector_.fit(X=X)

        return self

    def predict(self, X):
        return self.detector_.decision_function(X)
