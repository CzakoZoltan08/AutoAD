from autoad.algorithms.base_detector import BaseDetector

from pyod.models.sod import SOD


class SubspaceOutlierDetection(BaseDetector):
    def __init__(self,
                 n_neighbors=20,
                 ref_set=10,
                 alpha=0.8,
                 contamination=0.1):
        super(SubspaceOutlierDetection, self).__init__(
            contamination=contamination)
        self.n_neighbors = n_neighbors
        self.ref_set = ref_set
        self.alpha = alpha
        self.contamination = contamination

    def fit(self, X, y=None):
        self.detector_ = SOD(contamination=self.contamination,
                             n_neighbors=self.n_neighbors,
                             ref_set=self.ref_set,
                             alpha=self.alpha,
                             )

        self.detector_.fit(X=X)

        return self

    def predict(self, X):
        return self.detector_.predict(X)
