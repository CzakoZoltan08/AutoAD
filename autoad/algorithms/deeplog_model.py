from autoad.algorithms.base_detector import BaseDetector

from autoad.algorithms.deeplog import DeepLog


class DeepLogModel(BaseDetector):
    def __init__(self):
        self.detector_ = DeepLog(
            input_size=300,  # Number of different events to expect
            hidden_size=64,  # Hidden dimension, we suggest 64
            output_size=300)

    def fit(self, X, y=None):
        self.detector_.fit(X)

    def predict(self, X):
        return self.detector_.predict(X)
