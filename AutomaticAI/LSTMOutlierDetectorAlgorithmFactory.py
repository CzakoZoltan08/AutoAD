# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 18:25:21 2019

@author: Zoltan
"""
from AutomaticAI.Algorithm import Algorithm
from collections import OrderedDict

from autoad.algorithms.lstmod import LSTMOutlierDetector

const_param = {}

dicrete_hyper_parameter_list_of_min_attack_time = range(1, 200)
dicrete_hyper_parameter_list_of_feature_dim = range(1, 200)
dicrete_hyper_parameter_list_of_hidden_dim = range(1, 200)
dicrete_hyper_parameter_list_of_n_hidden_layer = range(1, 200)

continuous_hyper_parameter_mapping_index_key_mapping = [
    "train_contamination", "danger_coefficient_weight", "dropout_rate"]
discrete_hyper_parameter_mapping = []
discrete_hyper_parameter_mapping = [
    "min_attack_time", "feature_dim", "hidden_dim", "n_hidden_layer"]

discrete_parameter_dict = OrderedDict()
discrete_parameter_dict["min_attack_time"] = dicrete_hyper_parameter_list_of_min_attack_time
discrete_parameter_dict["feature_dim"] = dicrete_hyper_parameter_list_of_feature_dim
discrete_parameter_dict["hidden_dim"] = dicrete_hyper_parameter_list_of_hidden_dim
discrete_parameter_dict["n_hidden_layer"] = dicrete_hyper_parameter_list_of_n_hidden_layer
parameter_constraint_dict = OrderedDict()

# dictionary of parameters
param_dict = OrderedDict()
param_dict['train_contamination'] = 0.02
param_dict['danger_coefficient_weight'] = 0.5
param_dict['dropout_rate'] = 0.0

bounds = [(0.001, 0.4999), (0.00001, 0.9999),
          (0.00001, 0.9999), (1.0000001, 99.99), (1, 98), (1, 98), (1, 10)]


def get_algorithm():
    return Algorithm(algorithm_type=LSTMOutlierDetector,
                     algorithm_name="LSTMOutlierDetector",
                     hyper_parameter_dict=param_dict,
                     discrete_hyper_parameter_dict=discrete_parameter_dict,
                     discrete_hyper_parameter_mapping=discrete_hyper_parameter_mapping,
                     continuous_hyper_parameter_mapping=continuous_hyper_parameter_mapping_index_key_mapping,
                     parameter_constraint_dict=parameter_constraint_dict,
                     constant_hyper_parameter_dict=const_param,
                     bounds=bounds)
