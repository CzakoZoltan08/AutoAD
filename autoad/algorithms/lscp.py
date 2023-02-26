from autoad.algorithms.base_detector import BaseDetector

from pyod.models.lof import LOF
from pyod.models.lscp import LSCP


class LocallySelectiveCombination(BaseDetector):
    def __init__(self,
                 n_bins=10,
                 local_region_size=30,
                 local_max_features=0.5,
                 contamination=0.1):
        super(LocallySelectiveCombination, self).__init__(
            contamination=contamination)
        self.n_bins = n_bins
        self.local_region_size = local_region_size
        self.local_max_features = local_max_features
        self.contamination = contamination

    def fit(self, X, y=None):
        detector_list = [LOF(n_neighbors=5), LOF(n_neighbors=10), LOF(n_neighbors=15),
                         LOF(n_neighbors=20), LOF(
                             n_neighbors=25), LOF(n_neighbors=30),
                         LOF(n_neighbors=35), LOF(
                             n_neighbors=40), LOF(n_neighbors=45),
                         LOF(n_neighbors=50)]

        self.detector_ = LSCP(contamination=self.contamination,
                              detector_list=detector_list,
                              local_region_size=self.local_region_size,
                              local_max_features=self.local_max_features,
                              n_bins=self.n_bins
                              )

        self.detector_.fit(X=X)

        return self

    def predict(self, X):
        return self.detector_.predict(X)
