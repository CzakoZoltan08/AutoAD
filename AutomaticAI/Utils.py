# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 12:35:38 2019

@author: czzo
"""

import AutomaticAI.FEAWADAlgorithmFactory as FEAWADaf
import AutomaticAI.PReNetAlgorithmFactory as prenetaf
import AutomaticAI.DeepSADAlgorithmFactory as dsadaf
import AutomaticAI.BayesianRidgeRegressionAlgorithmFactory as brraf
import AutomaticAI.PassiveAgressiveRegressionAlgorithmFactory as paraf
import AutomaticAI.OrthogonalMatchingPursuitRegressionAlgorthmFactory as ompraf
import AutomaticAI.LarsLassoRegressionAlgorithmFactory as llraf
import AutomaticAI.LarsRegressionAlgorithmFactory as larsraf
import AutomaticAI.ElasticNetAlgorithmFactory as enraf
import AutomaticAI.LassoRegressionAlgorithmFactory as lraf
import AutomaticAI.KernelRidgeRegressionAlgorithmFactory as krraf
import AutomaticAI.SGDRegressionAlgorithmFactory as sgdraf
import AutomaticAI.ExtraTreesRegressionAlgorithmFactory as etsraf
import AutomaticAI.ExtraTreeRegressionAlgorithmFactory as etraf
import AutomaticAI.RandomForestRegressionAlgorithmFactory as rfraf
import AutomaticAI.KNNRegressionAlgorithmFactory as knnraf
import AutomaticAI.RidgeRegressionAlgorithmFactory as rraf
import AutomaticAI.DecisionTreeRegressorAlgorithmFactory as dtrf
import AutomaticAI.ExtraTreesClassifierAlgorithmFactory as etscaf
import AutomaticAI.SGDClassifierAlgorithmFactory as sgdcaf
import AutomaticAI.PassiveAgressiveClassifierAlgorithmFactory as pacaf
import AutomaticAI.DecisionTreeClassifierAlgorithmFactory as dtcaf
import AutomaticAI.ExtraTreeClassifierAlgorithmFactory as etcaf
import AutomaticAI.RandomForestAlgorithmFactory as rfaf
import AutomaticAI.KnnAlgorithmFactory as kaf
from scipy.spatial import distance as dc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
import numpy as np
import sys
sys.path.append('C:/University/Git/AutomaticAI_Flask')


# Classification metrics

# Prediction metrics


# Geometrical functions


# Classification algorithm factories
# import AlgorithmFactories.ClassificationAlgorithmFactories.XGBoostClassifierAlgorithmFactory as xgbcaf

# Adaptive classification algorithm factories
# import AutomaticAI.AdaptiveRandomForestClassifierAlgorithmFactory as arfcaf
# import AutomaticAI.AdaptiveKnnClassifierAlgorithmFactory as aknncaf
# import AutomaticAI.AdaptiveMultinomialNBClassifierAlgorithmFactory as amlnbcaf
# import AutomaticAI.AdaptiveGaussianNBClassifierAlgorithmFactory as agnbcaf
# import AutomaticAI.AdaptivePerceptronClassifierAlgorithmFactory as apcaf

# Regression algorithm factories
# import AlgorithmFactories.RegressionAlgorithmFactories.XGBoostRegressionAlgorithmFactory as xgbraf

# Unsupervised algorithm factories


classification_algorithms = [  # lrcaf.get_algorithm(),
    kaf.get_algorithm(),
    rfaf.get_algorithm(),
    etcaf.get_algorithm(),
    dtcaf.get_algorithm(),
    # rcaf.get_algorithm(),
    pacaf.get_algorithm(),
    # abcaf.get_algorithm(),
    # gbcaf.get_algorithm(),
    sgdcaf.get_algorithm(),
    etscaf.get_algorithm(),
    # xgbcaf.get_algorithm(),
]

adaptive_classification_algorithms = [  # arfcaf.get_algorithm(),
    #   aknncaf.get_algorithm(),
    #   #sgdcaf.get_algorithm(),
    #   #pacaf.get_algorithm(),
    #   amlnbcaf.get_algorithm(),
    #   agnbcaf.get_algorithm(),
    #   apcaf.get_algorithm()
]

regression_algorithms = [dtrf.get_algorithm(),
                         lraf.get_algorithm(),
                         rraf.get_algorithm(),
                         knnraf.get_algorithm(),
                         rfraf.get_algorithm(),
                         # abraf.get_algorithm(),
                         etraf.get_algorithm(),
                         etsraf.get_algorithm(),
                         # gbraf.get_algorithm()
                         sgdraf.get_algorithm(),
                         krraf.get_algorithm(),
                         # svraf.get_algorithm(),
                         lraf.get_algorithm(),
                         enraf.get_algorithm(),
                         larsraf.get_algorithm(),
                         llraf.get_algorithm(),
                         ompraf.get_algorithm(),
                         paraf.get_algorithm(),
                         brraf.get_algorithm(),
                         # ardraf.get_algorithm(),
                         # xgbraf.get_algorithm(),
                         ]

adaptive_regression_algorithms = []

anomaly_detection_semisupervised_algorithms = [
    dsadaf.get_algorithm(),
    prenetaf.get_algorithm(),
    # devnetaf.get_algorithm(),
    # ganomalyaf.get_algorithm(),
    FEAWADaf.get_algorithm(),
    # XGBODaf.get_algorithm(),
]

anomaly_detection_supervised_algorithms = []


def space_gen(xmin, xmax, len, min_dist, result=[]):
    if len:
        for x in range(xmin, xmax - (len - 1) * min_dist):
            yield from space_gen(x + min_dist, xmax, len - 1, min_dist, result + [x])
    else:
        yield result


def calculate_min_distance(list_of_vectors, vector):
    minimum = 99999
    if len(list_of_vectors) > 0:
        distances = dc.cdist(list_of_vectors, [vector])
        minimum = np.min(distances)

    return minimum


def generate_initial_parameters(particle_count, bounds, distance_between_initial_particles=0.7):
    initial = []

    for i in range(particle_count):
        hyper_parameter_list = []
        minimum_distance = 0
        while minimum_distance < distance_between_initial_particles:
            hyper_parameter_list = []
            for j in range(len(bounds)):
                min_bound = bounds[j][0]
                max_bound = bounds[j][1]
                initial_parameter_value = np.random.uniform(
                    min_bound, max_bound)
                hyper_parameter_list.append(initial_parameter_value)
            minimum_distance = calculate_min_distance(
                initial, hyper_parameter_list)
        initial.append(hyper_parameter_list)

    return initial


def generate_initial_best_positions(algorithms):
    pos = []

    current_algorithm_name = ""

    for algorithm in algorithms:
        if algorithm.algorithm_name != current_algorithm_name:
            current_algorithm_name = algorithm.algorithm_name
            algorithm_best_positions = []
            for i in range(len(algorithm.bounds)):
                algorithm_best_positions.append(0.0)
            pos.append(algorithm_best_positions)

    return pos


def evaluate_particle(particle,
                      epoch,
                      verbose,
                      cost_function,
                      X_train,
                      X_test,
                      Y_train,
                      Y_test):
    particle.evaluate(
        cost_function,
        X_train,
        X_test,
        Y_train,
        Y_test,
        epoch=epoch,
        verbose=verbose)

    return particle


def train_algorithm(curr_params, param, Xtrain, Xvalid, Ytrain, Yvalid, algorithm, metric=accuracy_score):
    params_copy = param.copy()
    params_copy.update(curr_params)
    model = algorithm(**params_copy)
    model.fit(Xtrain, Ytrain)
    preds = model.predict(Xvalid)
    metric_val = metric(Yvalid, preds)

    return model, metric_val


def train_semisupervised_algorithm(curr_params, param, Xtrain, Xvalid, Ytrain, Yvalid, algorithm, metric=roc_auc_score):
    params_copy = param.copy()
    params_copy.update(curr_params)
    model = algorithm(**params_copy)
    model.fit(Xtrain, Ytrain)
    preds = model.predict(Xvalid)
    metric_val = metric(Yvalid, preds)

    return model, metric_val


def traint_adaptive_algorithm(curr_params, param, Xtrain, Xvalid, Ytrain, Yvalid, algorithm, metric=accuracy_score):
    params_copy = param.copy()
    params_copy.update(curr_params)
    model = algorithm(**params_copy)

    try:
        model.partial_fit(Xtrain, Ytrain.ravel())
    except Exception:
        model.partial_fit(Xtrain, Ytrain.ravel(), classes=np.unique(Ytrain))

    preds = model.predict(Xvalid)
    metric_val = metric(Yvalid, preds)

    return model, metric_val


def train_regression_algorithm(curr_params, param, Xtrain, Xvalid, Ytrain, Yvalid, algorithm, metric=r2_score):
    return train_algorithm(curr_params, param, Xtrain, Xvalid, Ytrain, Yvalid, algorithm, metric=metric)


def train_adaptive_regression_algorithm(curr_params, param, Xtrain, Xvalid, Ytrain, Yvalid, algorithm, metric=r2_score):
    return traint_adaptive_algorithm(curr_params, param, Xtrain, Xvalid, Ytrain, Yvalid, algorithm, metric=metric)


def get_classification_algorithm_mapping():
    algorithm_mapping = {}
    index = 0
    for algorithm in classification_algorithms:
        key = algorithm.algorithm_name
        algorithm_mapping[key] = index
        index += 1

    return algorithm_mapping


def get_adaptive_classification_algorithm_mapping():
    algorithm_mapping = {}
    index = 0
    for algorithm in adaptive_classification_algorithms:
        key = algorithm.algorithm_name
        algorithm_mapping[key] = index
        index += 1

    return algorithm_mapping


def get_regression_algorithm_mapping():
    algorithm_mapping = {}
    index = 0
    for algorithm in regression_algorithms:
        key = algorithm.algorithm_name
        algorithm_mapping[key] = index
        index += 1

    return algorithm_mapping


def get_adaptive_regression_algorithm_mapping():
    algorithm_mapping = {}
    index = 0
    for algorithm in adaptive_regression_algorithms:
        key = algorithm.algorithm_name
        algorithm_mapping[key] = index
        index += 1

    return algorithm_mapping


def get_semisupervised_algorithm_mapping():
    algorithm_mapping = {}
    index = 0
    for algorithm in anomaly_detection_semisupervised_algorithms:
        key = algorithm.algorithm_name
        algorithm_mapping[key] = index
        index += 1

    return algorithm_mapping


def create_all_supported_algorithm_list(particle_count):
    algorithm_type_list = []

    for algorithm in classification_algorithms:
        for i in range(particle_count):
            algorithm_type_list.append(algorithm)

    return algorithm_type_list


def create_all_supported_adaptive_algorithm_list(particle_count):
    algorithm_type_list = []

    for algorithm in adaptive_classification_algorithms:
        for i in range(particle_count):
            algorithm_type_list.append(algorithm)

    return algorithm_type_list


def create_all_supported_regresion_algorithm_list(particle_count):
    algorithm_type_list = []

    for algorithm in regression_algorithms:
        total = particle_count
        if algorithm.algorithm_name == lraf.get_algorithm().algorithm_name:
            total = 3
        for i in range(total):
            algorithm_type_list.append(algorithm)

    return algorithm_type_list


def create_all_supported_adaptive_regresion_algorithm_list(particle_count):
    algorithm_type_list = []

    for algorithm in adaptive_regression_algorithms:
        total = particle_count
        if algorithm.algorithm_name == lraf.get_algorithm().algorithm_name:
            total = 3
        for i in range(total):
            algorithm_type_list.append(algorithm)

    return algorithm_type_list


def create_all_supported_semisupervised_algorithm_list(particle_count):
    algorithm_type_list = []

    for algorithm in anomaly_detection_semisupervised_algorithms:
        total = particle_count
        if algorithm.algorithm_name == lraf.get_algorithm().algorithm_name:
            total = 3
        for i in range(total):
            algorithm_type_list.append(algorithm)

    return algorithm_type_list


def generate_initial_particle_positions(num_particles=10, distance_between_initial_particles=0.7):
    initial = []
    for algorithm in classification_algorithms:
        initial_of_algorithm = generate_initial_parameters(
            num_particles, algorithm.bounds, distance_between_initial_particles)
        initial.extend(initial_of_algorithm)

    return initial


def generate_initial_particle_positions_for_semisupervised_anomaly_detection(
        num_particles=10,
        distance_between_initial_particles=0.7):
    initial = []
    for algorithm in anomaly_detection_semisupervised_algorithms:
        initial_of_algorithm = generate_initial_parameters(
            num_particles, algorithm.bounds, distance_between_initial_particles)
        initial.extend(initial_of_algorithm)

    return initial


def generate_initial_particle_positions_for_adaptive_classification(
        num_particles=10,
        distance_between_initial_particles=0.7):
    initial = []
    for algorithm in adaptive_classification_algorithms:
        initial_of_algorithm = generate_initial_parameters(
            num_particles, algorithm.bounds, distance_between_initial_particles)
        initial.extend(initial_of_algorithm)

    return initial


def generate_initial_particle_positions_for_regression(num_particles=10, distance_between_initial_particles=0.7):
    initial = []
    for algorithm in regression_algorithms:
        if algorithm.algorithm_name == lraf.get_algorithm().algorithm_name:
            initial_of_algorithm = generate_initial_parameters(
                3, algorithm.bounds, distance_between_initial_particles)
        else:
            initial_of_algorithm = generate_initial_parameters(
                num_particles, algorithm.bounds, distance_between_initial_particles)
        initial.extend(initial_of_algorithm)

    return initial


def generate_initial_particle_positions_for_adaptive_regression(
        num_particles=10,
        distance_between_initial_particles=0.7):
    initial = []
    for algorithm in adaptive_regression_algorithms:
        initial_of_algorithm = generate_initial_parameters(
            num_particles, algorithm.bounds, distance_between_initial_particles)
        initial.extend(initial_of_algorithm)

    return initial
