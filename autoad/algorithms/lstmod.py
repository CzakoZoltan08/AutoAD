import warnings
from autoad.algorithms.base_detector import BaseDetector

from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array

from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential

from numpy import percentile
import numpy as np


class LSTMOutlierDetector(BaseDetector):
    def __init__(self,
                 train_contamination=0.0,
                 min_attack_time=5,
                 danger_coefficient_weight=0.5,
                 loss='mean_squared_error',
                 optimizer='adam',
                 epochs=10,
                 batch_size=8,
                 dropout_rate=0.0,
                 feature_dim=9,
                 hidden_dim=1,
                 n_hidden_layer=0,
                 activation=None,
                 diff_group_method='average',
                 contamination=0.1):
        super(LSTMOutlierDetector, self).__init__(contamination=contamination)
        self.train_contamination = train_contamination
        self.min_attack_time = min_attack_time
        self.window_size = min_attack_time
        self.danger_coefficient_weight = danger_coefficient_weight
        self.relative_error_threshold = None
        self.loss = loss
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.n_hidden_layer = n_hidden_layer
        self.diff_group_method = diff_group_method
        self.activation = activation

    def _build_model(self):
        model = Sequential()
        model.add(LSTM(units=self.hidden_dim,
                       input_shape=(self.feature_dim, 1),
                       dropout=self.dropout_rate,
                       activation=self.activation,
                       return_sequences=bool(self.n_hidden_layer > 0)))

        for layer_idx in range(self.n_hidden_layer):
            model.add(LSTM(units=self.hidden_dim,
                           input_shape=(self.hidden_dim, 1),
                           dropout=self.dropout_rate,
                           activation=self.activation,
                           return_sequences=bool(layer_idx <
                                                 self.n_hidden_layer - 1)))

        model.add(Dense(units=self.feature_dim, input_shape=(
            self.hidden_dim, 1), activation=None))

        model.compile(loss=self.loss, optimizer=self.optimizer)
        return model

    def _set_n_classes(self, y):  # pragma: no cover
        """Set the number of classes if `y` is presented, which is not
        expected. It could be useful for multi-class outlier detection.

        Parameters
        ----------
        y : numpy array of shape (n_samples,)
            Ground truth.

        Returns
        -------
        self
        """

        self._classes = 2  # default as binary classification
        if y is not None:
            check_classification_targets(y)
            self._classes = len(np.unique(y))
            warnings.warn(
                "y should not be presented in unsupervised learning.")
        return self

    def _decision_function(self, X: np.array):
        """Predict raw anomaly scores of X using the fitted detector.

        The anomaly score of an input sample is computed based on the fitted
        detector. For consistency, outliers are assigned with
        higher anomaly scores.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples. Sparse matrices are accepted only
            if they are supported by the base estimator.

        Returns
        -------
        anomaly_scores : numpy array of shape (n_samples,)
            The anomaly score of the input samples.
        """
        check_is_fitted(self, ['model_'])

        relative_error = self._relative_error(X)

        error_num_buf = (
            relative_error > self.relative_error_threshold).astype(int)

        if not (self.diff_group_method in ['max', 'min', 'average']):
            raise ValueError(self.diff_group_method, "is not a valid method")

        relative_error_left_inds = np.ones(
            (len(relative_error), )) * len(relative_error)
        relative_error_right_inds = np.zeros((len(relative_error), ))

        if self.diff_group_method == 'average':
            danger_coefficient = np.zeros(relative_error.shape)
            averaged_relative_error = np.zeros(relative_error.shape)
            calculated_times = np.zeros(relative_error.shape)

            for i in range(len(relative_error) - self.min_attack_time + 1):
                dc_tmp = error_num_buf[i:i +
                                       self.min_attack_time].sum() / \
                    self.min_attack_time
                are_tmp = relative_error[i:i +
                                         self.min_attack_time].sum() / \
                    self.min_attack_time

                for j in range(self.min_attack_time):
                    averaged_relative_error[i + j] += are_tmp
                    danger_coefficient[i + j] += dc_tmp
                    calculated_times[i + j] += 1
                    relative_error_left_inds[i + j] = \
                        i if i < relative_error_left_inds[i +
                                                          j] else relative_error_left_inds[i + j]
                    relative_error_right_inds[i + j] = \
                        i+self.min_attack_time if i + \
                        self.min_attack_time > relative_error_right_inds[i + j] else relative_error_left_inds[i + j]

            # print(calculated_times)
            danger_coefficient /= calculated_times
            averaged_relative_error /= calculated_times
            # print(danger_coefficient, averaged_relative_error)

        else:  # pragma: no cover
            danger_coefficient = np.zeros(relative_error.shape)
            averaged_relative_error = np.zeros(relative_error.shape)

            if self.diff_group_method == 'min':
                danger_coefficient += float('inf')
                averaged_relative_error += float('inf')

            for i in range(len(relative_error) - self.min_attack_time + 1):
                dc_tmp = error_num_buf[i:i +
                                       self.min_attack_time].sum() / self.min_attack_time
                are_tmp = relative_error[i:i +
                                         self.min_attack_time].sum() / self.min_attack_time

                if self.diff_group_method == 'max':
                    for j in range(self.min_attack_time):
                        if are_tmp > averaged_relative_error[i + j] or dc_tmp > danger_coefficient[i+j]:
                            relative_error_left_inds[i + j] = i
                            relative_error_right_inds[i + j] = i+self.min_attack_time
                        if are_tmp > averaged_relative_error[i + j]:
                            averaged_relative_error[i + j] = are_tmp
                        if dc_tmp > danger_coefficient[i+j]:
                            danger_coefficient[i + j] = dc_tmp

                else:
                    for j in range(self.min_attack_time):
                        if are_tmp < averaged_relative_error[i + j] or dc_tmp < danger_coefficient[i+j]:
                            relative_error_left_inds[i + j] = i
                            relative_error_right_inds[i +
                                                      j] = i+self.min_attack_time
                        if are_tmp < averaged_relative_error[i + j]:
                            averaged_relative_error[i + j] = are_tmp
                        if dc_tmp < danger_coefficient[i+j]:
                            danger_coefficient[i + j] = dc_tmp

        # print(relative_error_left_inds)
        # print(relative_error_right_inds)
        pred_score = danger_coefficient * self.danger_coefficient_weight + \
            averaged_relative_error * (1 - self.danger_coefficient_weight)

        pred_score = np.concatenate(
            (np.zeros((self.window_size,)), pred_score))
        relative_error_left_inds = np.concatenate(
            (np.arange(self.window_size), relative_error_left_inds+self.window_size))
        relative_error_right_inds = np.concatenate((np.arange(
            self.window_size)+self.window_size, relative_error_right_inds+self.window_size))

        return pred_score, relative_error_left_inds, relative_error_right_inds

    def _process_decision_scores(self):  # pragma: no cover
        """Internal function to calculate key attributes:

        - threshold_: used to decide the binary label
        - labels_: binary labels of training data

        Returns
        -------
        self
        """

        self.threshold_ = percentile(self.decision_scores_,
                                     100 * (1 - self.contamination))
        self.labels_ = (self.decision_scores_ > self.threshold_).astype(
            'int').ravel()

        # calculate for predict_proba()

        self._mu = np.mean(self.decision_scores_)
        self._sigma = np.std(self.decision_scores_)

        return self

    def _get_sub_matrices(self, X: np.array):
        # return X[:-1].reshape(-1, 1, self.feature_dim), X[1:]
        return np.expand_dims(X[:-1], axis=2), X[1:]

    def _relative_error(self, X: np.array):

        X = check_array(X).astype(np.float)
        X_buf, y_buf = self._get_sub_matrices(X)

        y_predict = self.model_.predict(X_buf)

        relative_error = (np.linalg.norm(y_predict - y_buf, axis=1) /
                          np.linalg.norm(y_buf + 1e-6, axis=1)).ravel()

        return relative_error

    def fit(self, X, y=None):
        X = check_array(X).astype(np.float)
        self._set_n_classes(None)
        X_buf, y_buf = self._get_sub_matrices(X)
        self.feature_dim = X_buf.shape[1]
        self.model_ = self._build_model()

        # fit the LSTM model
        self.model_.fit(X_buf, y_buf, epochs=self.epochs,
                        batch_size=self.batch_size)

        relative_error = self._relative_error(X)

        if self.train_contamination < 1e-6:
            self.relative_error_threshold = max(relative_error)
        else:
            self.relative_error_threshold = np.percentile(
                relative_error, 100 * (1 - self.train_contamination))

        self.decision_scores_, self.left_inds_, self.right_inds_ = \
            self._decision_function(X)
        self._process_decision_scores()

        return self

    def predict(self, X):
        check_is_fitted(self, ['decision_scores_', 'threshold_', 'labels_'])

        pred_score, X_left_inds, X_right_inds = self._decision_function(X)

        return (pred_score > self.threshold_).astype(
            'int').ravel(), X_left_inds.ravel(), X_right_inds.ravel()
