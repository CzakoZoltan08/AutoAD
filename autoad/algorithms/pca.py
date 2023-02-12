from autoad.algorithms.base_detector import BaseDetector

from pyod.models.pca import PCA


class PCAAnomalyDetector(BaseDetector):
    def __init__(self,
                 n_components=None,
                 n_selected_components=None,
                 copy=True,
                 whiten=False,
                 svd_solver='auto',
                 tol=0.0,
                 iterated_power='auto',
                 random_state=None,
                 weighted=True,
                 standardization=True,
                 contamination=0.1):
        super(PCAAnomalyDetector, self).__init__(contamination=contamination)
        self.n_components = n_components
        self.n_selected_components = n_selected_components
        self.copy = copy
        self.whiten = whiten
        self.svd_solver = svd_solver
        self.tol = tol
        self.iterated_power = iterated_power
        self.random_state = random_state
        self.weighted = weighted
        self.standardization = standardization

    def fit(self, X, y=None):
        self.detector_ = PCA(contamination=self.contamination,
                             n_components=self.n_components,
                             n_selected_components=self.n_selected_components,
                             whiten=self.whiten,
                             svd_solver=self.svd_solver,
                             tol=self.tol,
                             iterated_power=self.iterated_power,
                             random_state=self.random_state,
                             standardization=self.standardization
                             )
        self.detector_.fit(X=X, y=y)

        return self

    def predict(self, X):
        return self.detector_.decision_function(X)
