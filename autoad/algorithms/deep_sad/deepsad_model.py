from autoad.algorithms.base_detector import BaseDetector
from autoad.algorithms.deep_sad.mlp import MLP, MLP_Autoencoder
from autoad.algorithms.deep_sad.odds import ODDSADDataset

import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


class DeepSADModel(BaseDetector):
    def __init__(self,
                 input_size,
                 optimizer_name: str = 'adam',
                 eta: float = 1.0):
        self.optimizer_name = optimizer_name
        self.net = MLP(x_dim=input_size, h_dims=[
            100, 20], rep_dim=10, bias=False)
        self.device = self._get_device()
        self.net = self.net.to(self.device)
        self.eta = eta
        self.eps = 1e-6

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

    def _init_center_c(self, train_loader: DataLoader, net: MLP, eps=0.1):
        """Initialize hypersphere center c as the mean
        from an initial forward pass on the data."""
        n_samples = 0
        c = torch.zeros(net.rep_dim, device=self.device)

        net.eval()
        with torch.no_grad():
            for data in train_loader:
                # get the inputs of the batch
                inputs, _, _, _ = data
                inputs = inputs.to(self.device)
                outputs = net(inputs)
                n_samples += outputs.shape[0]
                c += torch.sum(outputs, dim=0)

        c /= n_samples

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit
        # can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        return c

    def _init_network_weights_from_pretraining(self):
        net_dict = self.net.state_dict()
        ae_net_dict = self.ae_net.state_dict()

        # Filter out decoder network keys
        ae_net_dict = {k: v for k, v in ae_net_dict.items() if k in net_dict}
        # Overwrite values in the existing state_dict
        net_dict.update(ae_net_dict)
        # Load the new state_dict
        self.net.load_state_dict(net_dict)

    def train(self,
              dataset: ODDSADDataset,
              lr: float = 0.001,
              n_epochs: int = 50,
              lr_milestones: tuple = (),
              batch_size: int = 128):
        optimizer = optim.Adam(self.net.parameters(), lr=lr)

        train_loader = dataset.loaders(
            batch_size=batch_size, num_workers=0)

        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=lr_milestones, gamma=0.1)

        self.c = self._init_center_c(train_loader, self.net)

        start_time = time.time()
        self.net.train()

        for epoch in range(n_epochs):
            epoch_loss = 0.0
            n_batches = 0
            for data in train_loader:
                inputs, _, semi_targets, _ = data
                inputs, semi_targets = inputs.to(
                    self.device), semi_targets.to(self.device)

                # transfer the label "1" to "-1" for the inverse loss
                semi_targets[semi_targets == 1] = -1

                # Zero the network parameter gradients
                optimizer.zero_grad()

                # Update network parameters via backpropagation: forward
                #  + backward + optimize
                outputs = self.net(inputs)
                dist = torch.sum((outputs - self.c) ** 2, dim=1)
                losses = torch.where(
                    semi_targets == 0, dist, self.eta *
                    ((dist + self.eps) ** semi_targets.float()))
                loss = torch.mean(losses)
                loss.backward()
                optimizer.step()
                scheduler.step()
                # if epoch in self.lr_milestones:
                #     logger.info('  LR scheduler: new learning rate is
                #  %g' % float(scheduler.get_lr()[0]))

                epoch_loss += loss.item()
                n_batches += 1

        self.train_time = time.time() - start_time
        # logger.info('Training Time: {:.3f}s'.format(self.train_time))
        # logger.info('Finished training.')

        return self.net

    def pretrain(self,
                 dataset: ODDSADDataset,
                 input_size,
                 optimizer_name: str = 'adam',
                 lr: float = 0.001,
                 n_epochs: int = 100,
                 lr_milestones: tuple = (),
                 batch_size: int = 128):
        self.ae_net = MLP_Autoencoder(
            x_dim=input_size,
            h_dims=[100, 20],
            rep_dim=10,
            bias=False)

        # Train
        self.ae_optimizer_name = optimizer_name

        train_loader = dataset.loaders(
            batch_size=batch_size, num_workers=0)

        criterion = nn.MSELoss(reduction='none')

        ae_net = self.ae_net.to(self.device)
        criterion = criterion.to(self.device)

        optimizer = optim.Adam(ae_net.parameters(), lr=lr)

        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=lr_milestones, gamma=0.1)

        start_time = time.time()
        ae_net.train()
        for epoch in range(n_epochs):

            epoch_loss = 0.0
            n_batches = 0
            for data in train_loader:
                inputs, _, _, _ = data
                inputs = inputs.to(self.device)

                # Zero the network parameter gradients
                optimizer.zero_grad()

                # Update network parameters via backpropagation:
                # forward + backward + optimize
                rec = ae_net(inputs)
                rec_loss = criterion(rec, inputs)
                loss = torch.mean(rec_loss)
                loss.backward()
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()
                n_batches += 1

        self.train_time = time.time() - start_time

        self.ae_net = ae_net

        # Initialize Deep SAD network weights from pre-trained encoder
        self._init_network_weights_from_pretraining()

    def test(self, dataset: ODDSADDataset, batch_size: int = 128):
        test_loader = dataset.loaders(
            batch_size=batch_size, num_workers=0)

        epoch_loss = 0.0
        n_batches = 0
        start_time = time.time()
        idx_label_score = []
        self.net.eval()
        with torch.no_grad():
            for data in test_loader:
                inputs, labels, semi_targets, idx = data

                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                semi_targets = semi_targets.to(self.device)
                idx = idx.to(self.device)

                outputs = self.net(inputs)
                dist = torch.sum((outputs - self.c) ** 2, dim=1)
                losses = torch.where(
                    semi_targets == 0, dist, self.eta *
                    ((dist + self.eps) ** semi_targets.float()))
                loss = torch.mean(losses)
                scores = dist

                idx_label_score += list(zip(
                    idx.cpu().data.numpy().tolist(),
                    labels.cpu().data.numpy().tolist(),
                    scores.cpu().data.numpy().tolist()))

                epoch_loss += loss.item()
                n_batches += 1

        self.test_time = time.time() - start_time
        self.test_scores = idx_label_score

        _, labels, scores = zip(*idx_label_score)
        scores = np.array(scores)

        return scores
