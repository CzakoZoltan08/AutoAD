from autoad.algorithms.base_detector import BaseDetector

from pyod.models.loda import LODA


class LightweightOnlineDetector(BaseDetector):
    def __init__(self,
                 n_bins=10,
                 n_random_cuts=100,
                 contamination=0.1):
        super(LightweightOnlineDetector, self).__init__(contamination=contamination)
        self.n_bins = n_bins
        self.n_random_cuts = n_random_cuts
        self.contamination = contamination

    def fit(self, X, y=None):
        self.detector_ = LODA(contamination=self.contamination,
                              n_bins=self.n_bins,
                              n_random_cuts=self.n_random_cuts,
                              )

        self.detector_.fit(X=X)

        return self

    def predict(self, X):
        return self.detector_.decision_function(X)
