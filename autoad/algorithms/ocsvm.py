from autoad.algorithms.base_detector import BaseDetector

from pyod.models.ocsvm import OCSVM


class OneClassSVM(BaseDetector):
    def __init__(self,
                 kernel='rbf',
                 degree=3,
                 gamma='auto',
                 coef0=0.0,
                 tol=1e-3,
                 nu=0.5,
                 shrinking=True,
                 cache_size=200,
                 verbose=False,
                 max_iter=-1,
                 contamination=0.1):
        super(OneClassSVM, self).__init__(contamination=contamination)
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.nu = nu
        self.shrinking = shrinking
        self.cache_size = cache_size
        self.verbose = verbose
        self.max_iter = max_iter
        self.contamination = contamination

    def fit(self, X, y=None):
        self.detector_ = OCSVM(contamination=self.contamination,
                               kernel=self.kernel,
                               nu=self.nu,
                               degree=self.degree,
                               gamma=self.gamma,
                               coef0=self.coef0,
                               tol=self.tol,
                               shrinking=self.shrinking,
                               cache_size=self.cache_size,
                               verbose=self.verbose,
                               max_iter=self.max_iter)

        self.detector_.fit(X=X)

        return self

    def predict(self, X):
        return self.detector_.predict(X)
