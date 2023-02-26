from autoad.algorithms.base_detector import BaseDetector

from pyod.models.gmm import GMM


class GMMAnomalyDetector(BaseDetector):
    def __init__(self,
                 n_components=1,
                 covariance_type='full',
                 tol=1e-3,
                 reg_covar=1e-6,
                 contamination=0.1):
        super(GMMAnomalyDetector, self).__init__(
            contamination=contamination)
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.tol = tol
        self.reg_covar = reg_covar
        self.contamination = contamination

    def fit(self, X, y=None):
        self.detector_ = GMM(contamination=self.contamination,
                             n_components=self.n_components,
                             covariance_type=self.covariance_type,
                             tol=self.tol,
                             reg_covar=self.reg_covar
                             )

        self.detector_.fit(X=X)

        return self

    def predict(self, X):
        return self.detector_.predict(X)
