from autoad.algorithms.base_detector import BaseDetector
from pyod.models.auto_encoder import AutoEncoder as PyODAutoEncoder

from tensorflow.keras.losses import mean_squared_error


class AutoEncoder(BaseDetector):
    def __init__(self,
                 hidden_neurons=None,
                 hidden_activation='relu',
                 output_activation='sigmoid',
                 loss=mean_squared_error,
                 optimizer='adam',
                 epochs=100,
                 batch_size=32,
                 dropout_rate=0.2,
                 l2_regularizer=0.1,
                 validation_size=0.1,
                 preprocessing=True,
                 verbose=1,
                 random_state=None,
                 contamination=0.1):
        super(AutoEncoder, self).__init__(contamination=contamination)
        self.hidden_neurons = hidden_neurons
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.loss = loss
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.l2_regularizer = l2_regularizer
        self.validation_size = validation_size
        self.preprocessing = preprocessing
        self.verbose = verbose
        self.random_state = random_state

        # default values
        if self.hidden_neurons is None:
            self.hidden_neurons = [32, 16, 16, 32]

        # Verify the network design is valid
        if not self.hidden_neurons == self.hidden_neurons[::-1]:
            print(self.hidden_neurons)
            raise ValueError("Hidden units should be symmetric")

        self.hidden_neurons_ = self.hidden_neurons

    def fit(self, X, y=None):
        self.detector_ = PyODAutoEncoder(contamination=self.contamination,
                                         hidden_neurons=self.hidden_neurons,
                                         hidden_activation=self.hidden_activation,
                                         output_activation=self.output_activation,
                                         loss=self.loss,
                                         optimizer=self.optimizer,
                                         epochs=self.epochs,
                                         batch_size=self.batch_size,
                                         dropout_rate=self.dropout_rate,
                                         l2_regularizer=self.l2_regularizer,
                                         validation_size=self.validation_size,
                                         preprocessing=self.preprocessing,
                                         verbose=self.verbose,
                                         random_state=self.random_state)

        self.detector_.fit(X=X)

        return self

    def predict(self, X):
        return self.detector_.predict(X)
