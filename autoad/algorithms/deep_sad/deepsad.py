from autoad.algorithms.base_detector import BaseDetector
from autoad.algorithms.deep_sad.deepsad_model import DeepSADModel
from autoad.algorithms.deep_sad.odds import ODDSADDataset

import torch
import numpy as np


class DeepSAD(BaseDetector):
    def __init__(self,
                 lr=0.001,
                 n_epochs=50,
                 optimizer_name='adam',
                 weight_decay=1e-6,
                 num_threads=0):
        self.seed = 42
        self.net_name = 'dense'
        self.xp_path = None
        self.load_config = None
        self.load_model = None
        self.eta = 1.0  # eta in the loss function
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.n_epochs = n_epochs
        self.lr_milestone = [0]
        self.batch_size = 128
        self.weight_decay = weight_decay
        self.pretrain = True  # whether to use auto-encoder for pretraining
        self.ae_optimizer_name = optimizer_name
        self.ae_lr = lr
        self.ae_n_epochs = n_epochs
        self.ae_lr_milestone = [0]
        self.ae_batch_size = 128
        self.ae_weight_decay = weight_decay
        self.num_threads = num_threads
        self.n_jobs_dataloader = 0

    def fit(self, X_train, y_train):
        torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Load data
        data = {'X_train': X_train, 'y_train': y_train}
        dataset = ODDSADDataset(data=data, train=True)
        input_size = dataset.train_set.data.size(1)  # input size

        self.deepSAD = DeepSADModel(input_size=input_size)

        if self.pretrain:
            self.deepSAD.pretrain(dataset, input_size=input_size)

        self.net = self.deepSAD.train(dataset)

        return self

    def predict(self, X):
        data = {'X_test': X, 'y_test': np.random.choice([0, 1], X.shape[0])}
        dataset = ODDSADDataset(data=data, train=False)

        score = self.deepSAD.test(dataset)

        return score
