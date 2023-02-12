from autoad.algorithms.base_detector import BaseDetector

from pyod.models.cblof import CBLOF


class ClusterBasedLocalOutlierFactor(BaseDetector):
    def __init__(self,
                 n_clusters=8,
                 clustering_estimator=None,
                 alpha=0.9,
                 beta=5,
                 use_weights=False,
                 check_estimator=False,
                 random_state=None,
                 contamination=0.1):
        super(ClusterBasedLocalOutlierFactor, self).__init__(contamination=contamination)
        self.n_clusters = n_clusters
        self.clustering_estimator = clustering_estimator
        self.alpha = alpha
        self.beta = beta
        self.use_weights = use_weights
        self.check_estimator = check_estimator
        self.random_state = random_state

    def fit(self, X, y=None):
        self.detector_ = CBLOF(contamination=self.contamination,
                               n_clusters=self.n_clusters,
                               alpha=self.alpha,
                               beta=self.beta,
                               use_weights=self.use_weights,
                               check_estimator=self.check_estimator,
                               random_state=self.random_state,
                               )

        self.detector_.fit(X=X)

        return self

    def predict(self, X):
        return self.detector_.decision_function(X)
