import tensorflow as tf

from autoad.algorithms.base_detector import BaseDetector

from dagmm import DAGMM


class DeepAutoencodingGaussianMixtureModel(BaseDetector):
    def __init__(self):
        self.detector_ = DAGMM(
            comp_hiddens=[16, 8, 1],
            comp_activation=tf.nn.tanh,
            est_hiddens=[8, 4],
            est_activation=tf.nn.tanh,
            est_dropout_ratio=0.25,
            epoch_size=1000,
            minibatch_size=128)

    def fit(self, X, y=None):
        self.detector_.fit(X)

    def predict(self, X):
        return self.detector_.predict(X)
