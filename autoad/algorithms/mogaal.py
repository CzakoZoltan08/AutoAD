from autoad.algorithms.base_detector import BaseDetector
from pyod.models.mo_gaal import MO_GAAL


class MultiObjectiveGenerativeAdversarialActiveLearning(BaseDetector):
    def __init__(self,
                 k=10,
                 stop_epochs=20,
                 lr_d=0.01,
                 lr_g=0.0001,
                 momentum=0.9,
                 contamination=0.1):
        super(MultiObjectiveGenerativeAdversarialActiveLearning,
              self).__init__(contamination=contamination)
        self.k = k
        self.stop_epochs = stop_epochs
        self.lr_d = lr_d
        self.lr_g = lr_g
        self.momentum = momentum
        self.contamination = contamination

    def fit(self, X, y=None):
        self.detector_ = MO_GAAL(stop_epochs=self.stop_epochs,
                                 k=self.k,
                                 lr_d=self.lr_d,
                                 lr_g=self.lr_g,
                                 momentum=self.momentum,
                                 contamination=self.contamination,
                                 )

        self.detector_.fit(X=X)

        return self

    def predict(self, X):
        return self.detector_.decision_function(X)
