import random
from autoad.algorithms.base_detector import BaseDetector

import rtdl
import scipy.special
import torch
import torch.nn.functional as F
import delu
import numpy as np
import tensorflow as tf

from sklearn.metrics import roc_auc_score, average_precision_score


class FTTransformer(BaseDetector):
    '''
    The original code: https://yura52.github.io/rtdl/stable/index.html
    The original paper: "Revisiting Deep Learning Models for Tabular Data",
    NIPS 2019
    '''

    def __init__(self, seed: int = 42, n_epochs=50, batch_size=64):

        self.seed = seed

        # device
        self.device = self._get_device(gpu_specific=True)

        # Docs:
        # https://yura52.github.io/zero/0.0.4/reference/api/zero.improve_reproducibility.html
        # zero.improve_reproducibility(seed=self.seed)
        delu.improve_reproducibility(base_seed=int(self.seed))

        # hyper-parameter
        self.n_epochs = n_epochs  # default is 1000
        self.batch_size = batch_size  # default is 256

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

    def _metric(self, y_true, y_score, pos_label=1):
        aucroc = roc_auc_score(y_true=y_true, y_score=y_score)
        aucpr = average_precision_score(
            y_true=y_true, y_score=y_score, pos_label=1)

        return {'aucroc': aucroc, 'aucpr': aucpr}

    @torch.no_grad()
    def _evaluate(self, X, y=None):
        self.model.eval()
        score = []
        # for batch in delu.iter_batches(X[part], 1024):
        for batch in delu.iter_batches(X, self.batch_size):
            score.append(self.model(batch, None))
        score = torch.cat(score).squeeze(1).cpu().numpy()
        score = scipy.special.expit(score)

        # calculate the metric
        if y is not None:
            target = y.cpu().numpy()
            metric = self._metric(y_true=target, y_score=score)
        else:
            metric = {'aucroc': None, 'aucpr': None}

        return score, metric['aucpr']

    def fit(self, X, y):
        # set seed
        self._set_seed(self.seed)

        # training set is used as the validation
        # set in the anomaly detection task
        X_set = {'train': torch.from_numpy(X).float().to(self.device),
                 'val': torch.from_numpy(X).float().to(self.device)}

        y_set = {'train': torch.from_numpy(y).float().to(self.device),
                 'val': torch.from_numpy(y).float().to(self.device)}

        task_type = 'binclass'
        n_classes = None
        d_out = n_classes or 1
        lr = 0.001
        weight_decay = 0.0

        self.model = rtdl.FTTransformer.make_default(
            n_num_features=X.shape[1],
            cat_cardinalities=None,
            last_layer_query_idx=[-1],  # it makes the model faster
            # and does NOT affect its output
            d_out=d_out,
        )

        self.model.to(self.device)
        optimizer = (
            self.model.make_default_optimizer()
            if isinstance(self.model, rtdl.FTTransformer)
            else torch.optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay)
        )
        loss_fn = (
            F.binary_cross_entropy_with_logits
            if task_type == 'binclass'
            else F.cross_entropy
            if task_type == 'multiclass'
            else F.mse_loss
        )

        # Create a dataloader for batches of indices
        # Docs:
        # https://yura52.github.io/zero/reference/api/zero.data.IndexLoader.html
        train_loader = delu.data.IndexLoader(
            len(X_set['train']), self.batch_size, device=self.device)

        # Create a progress tracker for early stopping
        # Docs:
        # https://yura52.github.io/zero/reference/api/zero.ProgressTracker.html
        progress = delu.ProgressTracker(patience=100)

        # training
        # report_frequency = len(X['train']) // self.batch_size // 5

        for epoch in range(1, self.n_epochs + 1):
            for iteration, batch_idx in enumerate(train_loader):
                self.model.train()
                optimizer.zero_grad()
                x_batch = X_set['train'][batch_idx]
                y_batch = y_set['train'][batch_idx]
                loss = loss_fn(self.model(x_batch, None).squeeze(1), y_batch)
                loss.backward()
                optimizer.step()

            _, val_metric = self._evaluate(X=X_set['val'], y=y_set['val'])
            print(
                f'Epoch {epoch:03d} | Validation metric: {val_metric:.4f}')
            progress.update(
                (-1 if task_type == 'regression' else 1) * val_metric)
            if progress.success:
                print(' <<< BEST VALIDATION EPOCH', end='')
            print()
            if progress.fail:
                break

        return self

    def predict(self, X):
        X = torch.from_numpy(X).float().to(self.device)
        score, _ = self._evaluate(X=X, y=None)
        return score
