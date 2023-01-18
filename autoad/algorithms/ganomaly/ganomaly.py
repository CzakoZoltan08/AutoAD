import random
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from torch import nn

import numpy as np
import tensorflow as tf

from autoad.algorithms.ganomaly.model import Discriminator, Generator
from autoad.algorithms.base_detector import BaseDetector


class GANomaly(BaseDetector):
    def __init__(self,
                 epochs: int = 100,
                 batch_size: int = 64,
                 act_fun=nn.Tanh(),
                 lr: float = 1e-2,
                 mom: float = 0.7):
        self.device = self._get_device()
        self.seed = 42
        self.epochs = epochs
        self.batch_size = batch_size
        self.act_fun = act_fun
        self.lr = lr
        self.mom = mom

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

    def _fit_model(self,
                   train_loader,
                   net_generator,
                   net_discriminator,
                   optimizer_G,
                   optimizer_D,
                   epochs,
                   batch_size,
                   print_loss,
                   device):

        L1_criterion = nn.L1Loss(reduction='mean')
        L2_criterion = nn.MSELoss(reduction='mean')
        BCE_criterion = nn.BCELoss(reduction='mean')

        for epoch in range(epochs):
            for i, data in enumerate(train_loader):
                X, _ = data
                y_real = torch.FloatTensor(batch_size).fill_(0)
                y_fake = torch.FloatTensor(batch_size).fill_(1)

                X = X.to(device)
                y_real = y_real.to(device)
                y_fake = y_fake.to(device)

                X = Variable(X)
                y_real = Variable(y_real)
                y_fake = Variable(y_fake)

                # zero grad for discriminator
                net_discriminator.zero_grad()
                # training the discriminator with real sample
                _, output = net_discriminator(X)
                loss_D_real = BCE_criterion(output.view(-1), y_real)

                # training the discriminator with fake sample
                _, X_hat, _ = net_generator(X)
                _, output = net_discriminator(X_hat)

                loss_D_fake = BCE_criterion(output.view(-1), y_fake)

                # entire loss in discriminator
                loss_D = (loss_D_real + loss_D_fake) / 2

                loss_D.backward()
                optimizer_D.step()

                # training the generator based on the result
                # from the discriminator
                net_generator.zero_grad()

                z, X_hat, z_hat = net_generator(X)

                # latent loss
                feature_real, _ = net_discriminator(X)
                feature_fake, _ = net_discriminator(X_hat)

                loss_G_latent = L2_criterion(feature_fake, feature_real)

                # contexutal loss
                loss_G_contextual = L1_criterion(X, X_hat)
                # entire loss in generator

                # encoder loss
                loss_G_encoder = L1_criterion(z, z_hat)

                loss_G = (loss_G_latent + loss_G_contextual +
                          loss_G_encoder) / 3

                loss_G.backward()
                optimizer_G.step()

                if print_loss:
                    print('[%d/%d] [%d/%d] Loss D: %.4f / Loss G: %.4f' %
                          (epoch + 1, epochs, i, len(train_loader),
                           loss_D, loss_G))

    def fit(self, X, y):
        X_train = X[y == 0]
        y_train = y[y == 0]

        train_tensor = TensorDataset(torch.from_numpy(
            X_train).float(), torch.tensor(y_train).float())
        train_loader = DataLoader(
            train_tensor,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True)

        input_size = X_train.shape[1]
        if input_size < 8:
            hidden_size = input_size // 2
        else:
            hidden_size = input_size // 4

        # model initialization, there exists randomness
        # because of weight initialization***
        self._set_seed(self.seed)
        self.net_generator = Generator(
            input_size=input_size,
            hidden_size=hidden_size,
            act_fun=self.act_fun)
        self.net_discriminator = Discriminator(
            input_size=input_size, act_fun=self.act_fun)

        self.net_generator = self.net_generator.to(self.device)
        self.net_discriminator = self.net_discriminator.to(self.device)

        optimizer_G = torch.optim.SGD(
            self.net_generator.parameters(), lr=self.lr, momentum=self.mom)
        optimizer_D = torch.optim.SGD(
            self.net_discriminator.parameters(), lr=self.lr, momentum=self.mom)

        # fitting
        self._fit_model(
            train_loader=train_loader,
            net_generator=self.net_generator,
            net_discriminator=self.net_discriminator,
            optimizer_G=optimizer_G,
            optimizer_D=optimizer_D,
            epochs=self.epochs,
            batch_size=self.batch_size,
            print_loss=True,
            device=self.device)

        return self

    def predict(self, X):
        L1_criterion = nn.L1Loss(reduction='none')
        self.net_generator.eval()

        if torch.is_tensor(X):
            pass
        else:
            X = torch.from_numpy(X)

        X = X.float()
        X = X.to(self.device)

        with torch.no_grad():
            z, _, z_hat = self.net_generator(X)
            score = L1_criterion(z, z_hat)
            score = torch.sum(score, dim=1).cpu().detach().numpy()

        return score
