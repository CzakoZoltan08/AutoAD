# -*- coding: utf-8 -*-
"""
@author: Xucheng Song
(https://github.com/GuansongPang/deviation-network)
(https://github.com/GuansongPang/deviation-network)
(https://arxiv.org/abs/2105.10500).
"""

import random
from autoad.algorithms.base_detector import BaseDetector
import numpy as np
import os
import sys
from scipy.sparse import csc_matrix

import torch
import tensorflow as tf
import keras
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, Subtract, concatenate, Lambda, Reshape
from keras.callbacks import ModelCheckpoint
from keras.losses import mean_squared_error

try:
    from keras.optimizers import Adam
except Exception:
    from tensorflow.keras.optimizers import Adam

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()


class FEAWAD(BaseDetector):
    def __init__(self,
                 network_depth=2,
                 cont_rate=0.02,
                 known_outliers=15,
                 model_name='FEAWAD',
                 save_suffix='test'):
        self.device = self._get_device()  # get device
        self.seed = 42
        self.MAX_INT = 99999
        self.network_depth = network_depth
        self.batch_size = 15
        self.nb_batch = 20
        self.epochs = 50
        self.runs = 15
        self.known_outliers = known_outliers
        self.cont_rate = cont_rate
        self.input_path = './dataset/'
        self.data_set = 'nslkdd_normalization'
        self.data_format = '0'
        self.data_dim = 122
        self.output = './proposed_devnet_auc_performance.csv'
        self.ramdn_seed = 42
        self.data_format = 0
        self.save_suffix = save_suffix

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

    def auto_encoder(self, input_shape):
        x_input = Input(shape=input_shape)
        length = K.int_shape(x_input)[1]

        input_vector = Dense(
            length,
            kernel_initializer='glorot_normal',
            use_bias=True,
            activation='relu',
            name='ain')(x_input)
        en1 = Dense(128, kernel_initializer='glorot_normal',
                    use_bias=True, activation='relu', name='ae1')(input_vector)
        en2 = Dense(64, kernel_initializer='glorot_normal',
                    use_bias=True, activation='relu', name='ae2')(en1)
        de1 = Dense(128, kernel_initializer='glorot_normal',
                    use_bias=True, activation='relu', name='ad1')(en2)
        de2 = Dense(length, kernel_initializer='glorot_normal',
                    use_bias=True, activation='relu', name='ad2')(de1)

        model = Model(x_input, de2)
        adm = Adam(lr=0.0001)
        model.compile(loss=mean_squared_error, optimizer=adm)

        return model

    def dev_network_d(self, input_shape, modelname, testflag):
        '''
        deeper network architecture with three hidden layers
        '''
        x_input = Input(shape=input_shape)
        length = K.int_shape(x_input)[1]

        input_vector = Dense(
            length,
            kernel_initializer='glorot_normal',
            use_bias=True,
            activation='relu',
            name='ain')(x_input)
        en1 = Dense(128, kernel_initializer='glorot_normal',
                    use_bias=True, activation='relu', name='ae1')(input_vector)
        en2 = Dense(64, kernel_initializer='glorot_normal',
                    use_bias=True, activation='relu', name='ae2')(en1)
        de1 = Dense(128, kernel_initializer='glorot_normal',
                    use_bias=True, activation='relu', name='ad1')(en2)
        de2 = Dense(length, kernel_initializer='glorot_normal',
                    use_bias=True, activation='relu', name='ad2')(de1)

        if testflag == 0:
            AEmodel = Model(x_input, de2)
            AEmodel.load_weights(modelname)
            print('load autoencoder model')

            # reconstruction residual error
            sub_result = Subtract()([x_input, de2])
            cal_norm2 = Lambda(lambda x: tf.norm(x, ord=2, axis=1))
            sub_norm2 = cal_norm2(sub_result)
            sub_norm2 = Reshape((1,))(sub_norm2)
            division = Lambda(lambda x: tf.divide(x[0], x[1]))
            # normalized reconstruction residual error
            sub_result = division([sub_result, sub_norm2])
            # [hidden representation, normalized reconstruction residual error]
            conca_tensor = concatenate([sub_result, en2], axis=1)

            # [hidden representation, normalized reconstruction residual error,
            # residual error]
            conca_tensor = concatenate([conca_tensor, sub_norm2], axis=1)
        else:
            sub_result = Subtract()([x_input, de2])
            cal_norm2 = Lambda(lambda x: tf.norm(x, ord=2, axis=1))
            sub_norm2 = cal_norm2(sub_result)
            sub_norm2 = Reshape((1,))(sub_norm2)
            division = Lambda(lambda x: tf.divide(x[0], x[1]))
            sub_result = division([sub_result, sub_norm2])
            conca_tensor = concatenate([sub_result, en2], axis=1)

            conca_tensor = concatenate([conca_tensor, sub_norm2], axis=1)

        print(keras.__version__)
        print(tf.__version__)

        intermediate = Dense(256,
                             kernel_initializer='glorot_normal',
                             use_bias=True,
                             activation='relu',
                             name='hl2')(conca_tensor)
        # concat the intermediate vector with the residual error
        intermediate = concatenate([intermediate, sub_norm2], axis=1)
        intermediate = Dense(32,
                             kernel_initializer='glorot_normal',
                             use_bias=True,
                             activation='relu',
                             name='hl3')(intermediate)
        # again, concat the intermediate vector with the residual error
        intermediate = concatenate([intermediate, sub_norm2], axis=1)
        output_pre = Dense(
            1,
            kernel_initializer='glorot_normal',
            use_bias=True,
            activation='linear',
            name='score')(intermediate)
        dev_model = Model(x_input, output_pre)

        def multi_loss(y_true, y_pred):
            confidence_margin = 5.

            dev = y_pred
            inlier_loss = K.abs(dev)
            outlier_loss = K.abs(K.maximum(confidence_margin - dev, 0.))

            sub_nor = tf.norm(sub_result, ord=2, axis=1)
            outlier_sub_loss = K.abs(
                K.maximum(confidence_margin - sub_nor, 0.))
            loss1 = (1 - y_true) * (inlier_loss+sub_nor) + \
                y_true * (outlier_loss+outlier_sub_loss)

            return loss1

        adm = Adam(lr=0.0001)
        dev_model.compile(loss=multi_loss, optimizer=adm)

        return dev_model

    def deviation_network(
            self, input_shape, network_depth, modelname, testflag):
        '''
        construct the deviation network-based detection model
        '''
        if network_depth == 4:
            model = self.dev_network_d(input_shape, modelname, testflag)
        elif network_depth == 2:
            model = self.auto_encoder(input_shape)

        else:
            sys.exit("The network depth is not set properly")
        return model

    def auto_encoder_batch_generator_sup(
            self,
            x,
            inlier_indices,
            batch_size,
            nb_batch,
            rng):
        """auto encoder batch generator
        """
        self._set_seed(self.seed)
        # rng = np.random.RandomState(rng.randint(self.MAX_INT, size = 1))
        rng = np.random.RandomState(np.random.randint(self.MAX_INT, size=1))
        counter = 0
        while 1:
            if self.data_format == 0:
                ref, training_labels = self.AE_input_batch_generation_sup(
                    x, inlier_indices, batch_size, rng)
            else:
                ref, training_labels = self.input_batch_generation_sup_sparse(
                    x, inlier_indices, batch_size, rng)
            counter += 1
            yield (ref, training_labels)
            if (counter > nb_batch):
                counter = 0

    def AE_input_batch_generation_sup(
            self, train_x, inlier_indices, batch_size, rng):
        '''
        batchs of samples. This is for csv data.
        Alternates between positive and negative pairs.
        '''
        rng = np.random.RandomState(self.seed)

        dim = train_x.shape[1]
        ref = np.empty((batch_size, dim))
        training_labels = np.empty((batch_size, dim))
        n_inliers = len(inlier_indices)
        for i in range(batch_size):
            sid = rng.choice(n_inliers, 1)
            ref[i] = train_x[inlier_indices[sid]]
            training_labels[i] = train_x[inlier_indices[sid]]
        return np.array(ref), np.array(training_labels, dtype=float)

    def batch_generator_sup(
            self,
            x,
            outlier_indices,
            inlier_indices,
            batch_size,
            nb_batch,
            rng):
        """batch generator
        """
        self._set_seed(self.seed)
        # rng = np.random.RandomState(rng.randint(self.MAX_INT, size = 1))
        rng = np.random.RandomState(np.random.randint(self.MAX_INT, size=1))
        counter = 0
        while 1:
            if self.data_format == 0:
                ref, training_labels = self.input_batch_generation_sup(
                    x, outlier_indices, inlier_indices, batch_size, rng)
            else:
                ref, training_labels = self.input_batch_generation_sup_sparse(
                    x, outlier_indices, inlier_indices, batch_size, rng)
            counter += 1
            yield (ref, training_labels)
            if (counter > nb_batch):
                counter = 0

    def input_batch_generation_sup(
            self,
            train_x,
            outlier_indices,
            inlier_indices,
            batch_size,
            rng):
        '''
        batchs of samples. This is for csv data.
        Alternates between positive and negative pairs.
        '''
        rng = np.random.RandomState(self.seed)
        dim = train_x.shape[1]
        ref = np.empty((batch_size, dim))
        training_labels = []
        n_inliers = len(inlier_indices)
        n_outliers = len(outlier_indices)
        for i in range(batch_size):
            if (i % 2 == 0):
                sid = rng.choice(n_inliers, 1)
                ref[i] = train_x[inlier_indices[sid]]
                training_labels += [0]
            else:
                sid = rng.choice(n_outliers, 1)
                ref[i] = train_x[outlier_indices[sid]]
                training_labels += [1]
        return np.array(ref), np.array(training_labels, dtype=float)

    def input_batch_generation_sup_sparse(
            self,
            train_x,
            outlier_indices,
            inlier_indices,
            batch_size,
            rng):
        '''
        batchs of samples. This is for libsvm stored sparse data.
        Alternates between positive and negative pairs.
        '''
        rng = np.random.RandomState(self.seed)
        ref = np.empty((batch_size))
        training_labels = []
        n_inliers = len(inlier_indices)
        n_outliers = len(outlier_indices)
        for i in range(batch_size):
            if (i % 2 == 0):
                sid = rng.choice(n_inliers, 1)
                ref[i] = inlier_indices[sid]
                training_labels += [0]
            else:
                sid = rng.choice(n_outliers, 1)
                ref[i] = outlier_indices[sid]
                training_labels += [1]
        ref = train_x[ref, :].toarray()
        return ref, np.array(training_labels, dtype=float)

    def load_model_weight_predict(
            self, model_name, input_shape, network_depth, test_x):
        '''
        load the saved weights to make predictions
        '''
        model = self.deviation_network(
            input_shape, network_depth, model_name, 1)
        model.load_weights(model_name)
        scoring_network = Model(inputs=model.input, outputs=model.output)

        if self.data_format == 0:
            scores = scoring_network.predict(test_x)
        else:
            data_size = test_x.shape[0]
            scores = np.zeros([data_size, 1])
            count = 512
            i = 0
            while i < data_size:
                subset = test_x[i:count].toarray()
                scores[i:count] = scoring_network.predict(subset)
                if i % 1024 == 0:
                    print(i)
                i = count
                count += 512
                if count > data_size:
                    count = data_size
            assert count == data_size
        return scores

    def inject_noise_sparse(self, seed, n_out, random_seed):
        '''
        add anomalies to training data to
        replicate anomaly contaminated data sets.
        we randomly swape 5% features of anomalies
        to avoid duplicate contaminated anomalies.
        This is for sparse data.
        '''
        rng = np.random.RandomState(random_seed)
        n_sample, dim = seed.shape
        swap_ratio = 0.05
        n_swap_feat = int(swap_ratio * dim)
        seed = seed.tocsc()
        noise = csc_matrix((n_out, dim))
        print(noise.shape)
        for i in np.arange(n_out):
            outlier_idx = rng.choice(n_sample, 2, replace=False)
            o1 = seed[outlier_idx[0]]
            o2 = seed[outlier_idx[1]]
            swap_feats = rng.choice(dim, n_swap_feat, replace=False)
            noise[i] = o1.copy()
            noise[i, swap_feats] = o2[0, swap_feats]
        return noise.tocsr()

    def inject_noise(self, seed, n_out, random_seed):
        '''
        add anomalies to training data to
        replicate anomaly contaminated data sets.
        we randomly swape 5% features of
        anomalies to avoid duplicate contaminated anomalies.
        this is for dense data
        '''
        rng = np.random.RandomState(random_seed)
        n_sample, dim = seed.shape
        swap_ratio = 0.05
        n_swap_feat = int(swap_ratio * dim)
        noise = np.empty((n_out, dim))
        for i in np.arange(n_out):
            outlier_idx = rng.choice(n_sample, 2, replace=False)
            o1 = seed[outlier_idx[0]]
            o2 = seed[outlier_idx[1]]
            swap_feats = rng.choice(dim, n_swap_feat, replace=False)
            noise[i] = o1.copy()
            noise[i, swap_feats] = o2[swap_feats]
        return noise

    def fit(self, X_train, y_train, ratio=None):
        # network_depth = int(self.network_depth)
        self._set_seed(self.seed)
        rng = np.random.RandomState(self.seed)

        # index
        outlier_indices = np.where(y_train == 1)[0]
        inlier_indices = np.where(y_train == 0)[0]

        # X_train_inlier = np.delete(X_train, outlier_indices, axis=0)
        self.input_shape = X_train.shape[1:]

        # pre-trained autoencoder
        self._set_seed(self.seed)
        AEmodel = self.deviation_network(
            self.input_shape, 2, None, 0)  # pretrain auto-encoder model
        print('autoencoder pre-training start....')
        AEmodel_name = os.path.join(
            os.getcwd(),
            'autoad',
            'algorithms',
            'feawad',
            'model',
            'pretrained_autoencoder_'+self.save_suffix+'.h5')
        ae_checkpointer = ModelCheckpoint(
            AEmodel_name,
            monitor='loss',
            verbose=0,
            save_best_only=True,
            save_weights_only=True)
        AEmodel.fit_generator(
            self.auto_encoder_batch_generator_sup(
                X_train, inlier_indices, self.batch_size, self.nb_batch, rng),
            steps_per_epoch=self.nb_batch, epochs=self.epochs,
            callbacks=[ae_checkpointer], verbose=0)

        # end-to-end devnet model
        print('load pretrained autoencoder model....')
        self._set_seed(self.seed)
        self.dev_model = self.deviation_network(
            self.input_shape, 4, AEmodel_name, 0)
        print('end-to-end training start....')
        self.dev_model_name = os.path.join(
            os.getcwd(),
            'autoad',
            'algorithms',
            'feawad',
            'model',
            'devnet_'+self.save_suffix+'.h5')
        checkpointer = ModelCheckpoint(
            self.dev_model_name, monitor='loss', verbose=0,
            save_best_only=True, save_weights_only=True)
        self.dev_model.fit_generator(self.batch_generator_sup(
            X_train,
            outlier_indices,
            inlier_indices,
            self.batch_size,
            self.nb_batch,
            rng),
            steps_per_epoch=self.nb_batch,
            epochs=self.epochs,
            callbacks=[checkpointer], verbose=0)

        return self

    def predict(self, X):
        score = self.load_model_weight_predict(
            self.dev_model_name, self.input_shape, 4, X)
        return score
