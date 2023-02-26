from autoad.algorithms.base_detector import BaseDetector

from pyod.models.lmdd import LMDD


class LMDDAnomalyDetector(BaseDetector):
    def __init__(self,
                 dis_measure='aad',
                 n_iter=200,
                 contamination=0.1):
        super(LMDDAnomalyDetector, self).__init__(
            contamination=contamination)
        self.dis_measure = dis_measure
        self.n_iter = n_iter
        self.contamination = contamination

    def fit(self, X, y=None):
        self.detector_ = LMDD(contamination=self.contamination,
                              dis_measure=self.dis_measure,
                              n_iter=self.n_iter
                              )

        self.detector_.fit(X=X)

        return self

    def predict(self, X):
        return self.detector_.predict(X)
