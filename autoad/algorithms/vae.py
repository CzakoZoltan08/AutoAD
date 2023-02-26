from autoad.algorithms.base_detector import BaseDetector

from pyod.models.vae import VAE

from tensorflow.keras.losses import mse


class VariationalAutoEncoder(BaseDetector):
    def __init__(self,
                 encoder_neurons=None,
                 decoder_neurons=None,
                 latent_dim=2,
                 hidden_activation='relu',
                 output_activation='sigmoid',
                 loss=mse,
                 optimizer='adam',
                 epochs=100,
                 batch_size=32,
                 dropout_rate=0.2,
                 l2_regularizer=0.1,
                 validation_size=0.1,
                 preprocessing=True,
                 verbose=1,
                 random_state=None,
                 contamination=0.1,
                 gamma=1.0,
                 capacity=0.0):
        super(VariationalAutoEncoder, self).__init__(
            contamination=contamination)
        self.encoder_neurons = encoder_neurons
        self.decoder_neurons = decoder_neurons
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
        self.latent_dim = latent_dim
        self.gamma = gamma
        self.capacity = capacity
        self.contamination = contamination

    def fit(self, X, y=None):
        self.detector_ = VAE(contamination=self.contamination,
                             encoder_neurons=self.encoder_neurons,
                             decoder_neurons=self.decoder_neurons,
                             hidden_activation=self.hidden_activation,
                             output_activation=self.output_activation,
                             loss=self.loss,
                             gamma=self.gamma,
                             capacity=self.capacity,
                             optimizer=self.optimizer,
                             epochs=self.epochs,
                             batch_size=self.batch_size,
                             dropout_rate=self.dropout_rate,
                             l2_regularizer=self.l2_regularizer,
                             validation_size=self.validation_size,
                             preprocessing=self.preprocessing,
                             verbose=self.verbose,
                             random_state=self.random_state,
                             )

        self.detector_.fit(X=X)

        return self

    def predict(self, X):
        return self.detector_.predict(X)
