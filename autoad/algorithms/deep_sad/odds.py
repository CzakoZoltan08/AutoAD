from torch.utils.data import DataLoader

from autoad.algorithms.deep_sad.odds_dataset import ODDSDataset


class ODDSADDataset():

    def __init__(self, data, train):
        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = (0,)
        self.outlier_classes = (1,)

        # training or testing dataset
        self.train = train

        if self.train:
            # Get training set
            self.train_set = ODDSDataset(data=data, train=True)
        else:
            # Get testing set
            self.test_set = ODDSDataset(data=data, train=False)

    def loaders(
            self,
            batch_size: int,
            shuffle_train=True,
            shuffle_test=False,
            num_workers: int = 0):

        if self.train:
            train_loader = DataLoader(
                dataset=self.train_set,
                batch_size=batch_size,
                shuffle=shuffle_train,
                num_workers=num_workers,
                drop_last=True)
            return train_loader
        else:
            test_loader = DataLoader(
                dataset=self.test_set,
                batch_size=batch_size,
                shuffle=shuffle_test,
                num_workers=num_workers,
                drop_last=False)
            return test_loader
