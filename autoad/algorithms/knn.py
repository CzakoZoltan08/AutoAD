from autoad.algorithms.base_detector import BaseDetector

from pyod.models.knn import KNN


class KNearestNeighbors(BaseDetector):
    def __init__(self,
                 n_neighbors=100,
                 method='largest',  # {'largest', 'mean', 'median'}
                 radius=1.0,
                 algorithm='auto',  # {'auto', 'ball_tree', 'kd_tree', 'brute'}
                 leaf_size=30,
                 metric='minkowski',
                 p=2,
                 contamination=0.1):
        super(KNearestNeighbors, self).__init__(contamination=contamination)
        self.n_neighbors = n_neighbors
        self.method = method
        self.radius = radius
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.metric = metric
        self.p = p

    def fit(self, X, y=None):
        self.detector_ = KNN(contamination=self.contamination,
                             n_neighbors=self.n_neighbors,
                             method=self.method,
                             radius=self.radius,
                             algorithm=self.algorithm,
                             leaf_size=self.leaf_size,
                             metric=self.metric,
                             metric_params=None,
                             p=self.p)

        self.detector_.fit(X=X)

        return self

    def predict(self, X):
        return self.detector_.predict(X)
