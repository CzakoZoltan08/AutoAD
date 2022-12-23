import abc


class BaseDetector(object):
    @abc.abstractmethod
    def __init__(self, contamination=0.1) -> None:
        if not (0. < contamination <= 0.5):
            raise ValueError("contamination must be in (0, 0.5], "
                             "got: %f" % contamination)

        self.contamination = contamination

    @abc.abstractmethod
    def fit(self, X, y=None):
        pass

    @abc.abstractmethod
    def predict(self, X):
        pass
