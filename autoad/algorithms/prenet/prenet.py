import random
import torch
from torch import nn
import numpy as np

import tensorflow as tf
from torch.autograd import Variable
from autoad.algorithms.base_detector import BaseDetector
from autoad.algorithms.prenet.model import PreNetModel

'''
The unofficial implement (with PyTorch)
of the PReNet model in the paper
"Deep Weakly-supervised Anomaly Detection"
The default hyper-parameter is the
same as in the original paper
'''


class PReNet(BaseDetector):
    def __init__(self,
                 seed: int = 42,
                 lr: float = 1e-3,
                 s_a_a=8,
                 s_a_u=4,
                 s_u_u=0,
                 weight_decay: float = 1e-2,
                 model_name='PReNet',
                 epochs: int = 50,
                 batch_num: int = 20,
                 batch_size: int = 512,
                 act_fun=nn.ReLU()):

        self.seed = seed
        self.device = self._get_device()

        # hyper-parameters
        self.epochs = epochs
        self.batch_num = batch_num
        self.batch_size = batch_size
        self.act_fun = act_fun
        self.lr = lr
        self.weight_decay = weight_decay

        self.s_a_a = s_a_a
        self.s_a_u = s_a_u
        self.s_u_u = s_u_u

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

    def _fit_model(self,
                   X_train_tensor,
                   y_train, model,
                   optimizer,
                   epochs,
                   batch_num,
                   batch_size,
                   s_a_a,
                   s_a_u,
                   s_u_u,
                   device=None):
        # epochs
        for epoch in range(epochs):
            # generate the batch samples
            X_train_loader, y_train_loader = self._sampler_pairs(
                X_train_tensor,
                y_train,
                epoch,
                batch_num,
                batch_size,
                s_a_a=s_a_a,
                s_a_u=s_a_u,
                s_u_u=s_u_u)
            for i in range(len(X_train_loader)):
                X_left, X_right = X_train_loader[i][0], X_train_loader[i][1]
                y = y_train_loader[i]

                # to device
                X_left = X_left.to(device)
                X_right = X_right.to(device)
                y = y.to(device)
                # to variable
                X_left = Variable(X_left)
                X_right = Variable(X_right)
                y = Variable(y)

                # clear gradient
                model.zero_grad()

                # loss forward
                score = model(X_left, X_right)
                loss = torch.mean(torch.abs(y - score))

                # loss backward
                loss.backward()
                # update model parameters
                optimizer.step()

    def _unique(self, a, b):
        u = 0.5 * (a + b) * (a + b + 1) + b
        return int(u)

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

    def _sampler_pairs(self,
                       X_train_tensor,
                       y_train,
                       epoch,
                       batch_num,
                       batch_size,
                       s_a_a,
                       s_a_u,
                       s_u_u):
        '''
        X_train_tensor: the input X in the torch.tensor form
        y_train: label in the numpy.array form

        batch_num: generate how many batches in one epoch
        batch_size: the batch size
        '''
        data_loader_X = []
        data_loader_y = []

        index_a = np.where(y_train == 1)[0]
        index_u = np.where(y_train == 0)[0]

        for i in range(batch_num):
            index = []

            for j in range(6):
                seed = self._unique(epoch, i)
                seed = self._unique(seed, j)
                self._set_seed(seed)

                if j < 3:
                    index_sub = np.random.choice(
                        index_a, batch_size // 4, replace=True)
                    index.append(list(index_sub))

                if j == 3:
                    index_sub = np.random.choice(
                        index_u, batch_size // 4, replace=True)
                    index.append(list(index_sub))

                if j > 3:
                    index_sub = np.random.choice(
                        index_u, batch_size // 2, replace=True)
                    index.append(list(index_sub))

            index_left = index[0] + index[2] + index[4]
            index_right = index[1] + index[3] + index[5]

            X_train_tensor_left = X_train_tensor[index_left]
            X_train_tensor_right = X_train_tensor[index_right]

            # generate label
            y_train_new = np.append(
                np.repeat(s_a_a, batch_size // 4),
                np.repeat(s_a_u, batch_size // 4))
            y_train_new = np.append(
                y_train_new, np.repeat(s_u_u, batch_size // 2))
            y_train_new = torch.from_numpy(y_train_new).float()

            # shuffle
            index_shuffle = np.arange(len(y_train_new))
            index_shuffle = np.random.choice(
                index_shuffle, len(index_shuffle), replace=False)

            X_train_tensor_left = X_train_tensor_left[index_shuffle]
            X_train_tensor_right = X_train_tensor_right[index_shuffle]
            y_train_new = y_train_new[index_shuffle]

            # save
            data_loader_X.append(
                [X_train_tensor_left, X_train_tensor_right])  # 注意left和right顺序
            data_loader_y.append(y_train_new)

        return data_loader_X, data_loader_y

    def fit(self, X, y):

        input_size = X.shape[1]  # input size
        self.X_train_tensor = torch.from_numpy(X).float()  # testing set
        self.y_train = y

        self._set_seed(self.seed)
        self.model = PreNetModel(input_size=input_size, act_fun=self.act_fun)
        optimizer = torch.optim.RMSprop(self.model.parameters(
        ), lr=self.lr, weight_decay=self.weight_decay)  # optimizer

        # training
        self._fit_model(
            X_train_tensor=self.X_train_tensor,
            y_train=y,
            model=self.model,
            optimizer=optimizer,
            epochs=self.epochs,
            batch_num=self.batch_num,
            batch_size=self.batch_size,
            s_a_a=self.s_a_a,
            s_a_u=self.s_a_u,
            s_u_u=self.s_u_u,
            device=self.device)

        return self

    def predict(self, X):
        num = 30
        self.model = self.model.eval()

        if torch.is_tensor(X):
            pass
        else:
            X = torch.from_numpy(X)

        X = X.float()
        X = X.to(self.device)

        score = []
        for i in range(X.size(0)):
            # postive sample in training set
            index_a = np.random.choice(np.where(self.y_train == 1)[
                                       0], num, replace=True)
            # negative sample in training set
            index_u = np.random.choice(np.where(self.y_train == 0)[
                                       0], num, replace=True)

            X_train_a_tensor = self.X_train_tensor[index_a]
            X_train_u_tensor = self.X_train_tensor[index_u]

            with torch.no_grad():
                score_a_x = self.model(
                    X_train_a_tensor, torch.cat(num * [X[i].view(1, -1)]))
                score_x_u = self.model(
                    torch.cat(num * [X[i].view(1, -1)]), X_train_u_tensor)

            score_sub = torch.mean(score_a_x + score_x_u)
            score_sub = score_sub.numpy()[()]

            # entire score
            score.append(score_sub)

        return np.array(score)
