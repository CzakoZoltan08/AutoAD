import random
import numpy as np
import os

import torch
import tensorflow as tf
from autoad.algorithms.base_detector import BaseDetector

from autoad.algorithms.repen.model import RepenModel


class REPEN(BaseDetector):
    def __init__(self,
                 seed: int = 42,
                 save_suffix='test',
                 mode: str = 'supervised',
                 hidden_dim: int = 20,
                 batch_size: int = 256,
                 nb_batch: int = 50,
                 n_epochs: int = 200):
        self.device = self._get_device()  # get device
        self.seed = seed

        self.MAX_INT = np.iinfo(np.int32).max
        self.MAX_FLOAT = np.finfo(np.float32).max

        # self.sess = tf.Session()
        # K.set_session(self.sess)

        # hyper-parameters
        self.mode = mode
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.nb_batch = nb_batch
        self.n_epochs = n_epochs

        self.save_suffix = save_suffix
        if not os.path.exists('autoad/algorithms/repen/model'):
            os.makedirs('autoad/algorithms/repen/model')

    def _set_seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)

        try:
            tf.random.set_seed(seed)  # for tf >= 2.0
        except Exception:
            tf.set_random_seed(seed)
            tf.random.set_random_seed(seed)

        # pytorch seed
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _get_device(self, gpu_specific=False):
        if gpu_specific:
            if torch.cuda.is_available():
                n_gpu = torch.cuda.device_count()
                print(f'number of gpu: {n_gpu}')
                print(f'cuda name: {torch.cuda.get_device_name(0)}')
                print('GPU is on')
            else:
                print('GPU is off')

            device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cpu")
        return device

    def fit(self, X, y):
        # initialization the network
        self._set_seed(self.seed)

        # change the model type when no label information is available
        if sum(y) == 0:
            self.mode = 'unsupervised'

        # model initialization
        self.model = RepenModel(
            mode=self.mode,
            hidden_dim=self.hidden_dim,
            batch_size=self.batch_size,
            nb_batch=self.nb_batch,
            n_epochs=self.n_epochs,
            known_outliers=1000000,
            save_suffix=self.save_suffix)

        # fitting
        self.model.fit(X, y)

        return self

    def predict(self, X):
        score = self.model.decision_function(X)
        return score
