from autoad.algorithms.base_detector import BaseDetector

from pyod.models.lof import LOF


class LocalOutlierFactor(BaseDetector):
    def __init__(self,
                 n_neighbors=20,
                 algorithm='auto',
                 leaf_size=30,
                 metric='minkowski',
                 p=2,
                 metric_params=None,
                 contamination=0.1,
                 n_jobs=1,
                 novelty=True):
        super(LocalOutlierFactor, self).__init__(contamination=contamination)
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.metric = metric
        self.p = p
        self.metric_params = metric_params
        self.n_jobs = n_jobs
        self.novelty = novelty
        self.contamination = contamination

    def fit(self, X, y=None):
        self.detector_ = LOF(contamination=self.contamination,
                             n_neighbors=self.n_neighbors,
                             algorithm=self.algorithm,
                             leaf_size=self.leaf_size,
                             metric=self.metric,
                             p=self.p,
                             metric_params=self.metric_params,
                             )

        self.detector_.fit(X=X)

        return self

    def predict(self, X):
        return self.detector_.predict(X)
