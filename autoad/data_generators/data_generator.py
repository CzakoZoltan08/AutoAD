import abc


class DataGenerator():
    @abc.abstractclassmethod
    def generate(X, normal_count: int = 1000, anomaly_count: int = 100):
        pass
