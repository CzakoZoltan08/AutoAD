from autoad.algorithms.base_detector import BaseDetector

from pyod.models.hbos import HBOS


class HistogramBasedOutlierDetection(BaseDetector):
    def __init__(self,
                 n_bins=10,
                 alpha=0.1,
                 tol=0.5,
                 contamination=0.1):
        super(HistogramBasedOutlierDetection, self).__init__(contamination=contamination)
        self.n_bins = n_bins
        self.alpha = alpha
        self.tol = tol
        self.contamination = contamination

    def fit(self, X, y=None):
        self.detector_ = HBOS(contamination=self.contamination,
                              n_bins=self.n_bins,
                              alpha=self.alpha,
                              tol=self.tol,
                              )

        self.detector_.fit(X=X)

        return self

    def predict(self, X):
        return self.detector_.decision_function(X)
