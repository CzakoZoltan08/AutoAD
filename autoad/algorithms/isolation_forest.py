from autoad.algorithms.base_detector import BaseDetector

from pyod.models.iforest import IForest


class IsolationForest(BaseDetector):
    def __init__(self,
                 n_estimators=100,
                 max_samples="auto",
                 contamination=0.1,
                 max_features=1.,
                 bootstrap=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0):
        super(IsolationForest, self).__init__(contamination=contamination)
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y=None):
        self.detector_ = IForest(n_estimators=self.n_estimators,
                                 max_samples=self.max_samples,
                                 contamination=self.contamination,
                                 max_features=self.max_features,
                                 bootstrap=self.bootstrap,
                                 n_jobs=self.n_jobs,
                                 random_state=self.random_state,
                                 verbose=self.verbose)

        self.detector_.fit(X=X)

        return self

    def predict(self, X):
        return self.detector_.decision_function(X)
