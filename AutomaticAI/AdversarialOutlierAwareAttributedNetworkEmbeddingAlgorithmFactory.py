# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 14:03:58 2019

@author: Zoltan
"""

from collections import OrderedDict

from AutomaticAI.Algorithm import Algorithm
from autoad.algorithms.adone import AdversarialOutlierAwareAttributedNetworkEmbedding


const_param = {
    'metric': 'minkowski'
}

dicrete_hyper_parameter_list_of_algorithms = [
    "auto", "ball_tree", "kd_tree", "brute"]
dicrete_hyper_parameter_list_of_powers = range(1, 10)
dicrete_hyper_parameter_list_of_neighbors = range(1, 220)
dicrete_hyper_parameter_list_of_method = [
    'largest', 'mean', 'median']
continuous_hyper_parameter_mapping_index_key_mapping = ["radius"]
discrete_hyper_parameter_mapping = [
    "p", "n_neighbors", "algorithm", "method", "leaf_size"]

discrete_parameter_dict = OrderedDict()
discrete_parameter_dict["p"] = dicrete_hyper_parameter_list_of_powers
discrete_parameter_dict["n_neighbors"] = dicrete_hyper_parameter_list_of_neighbors
discrete_parameter_dict["algorithm"] = dicrete_hyper_parameter_list_of_algorithms
discrete_parameter_dict["method"] = dicrete_hyper_parameter_list_of_algorithms
discrete_parameter_dict["leaf_size"] = range(10, 100)
parameter_constraint_dict = OrderedDict()

# logistic regression
param_dict_logistic_regression = OrderedDict()
param_dict_logistic_regression['p'] = 1
param_dict_logistic_regression['n_neighbors'] = 1
param_dict_logistic_regression['algorithm'] = 'auto'

param_dict = OrderedDict()
param_dict['radius'] = 1.0


bounds = [(0.00001, 10.99), (1.001, 4.99),
          (1.001, 199.99), (0.001, 2.99), (0.001, 1.99), (0.001, 98.99)]


def get_algorithm():
    return Algorithm(algorithm_type=AdversarialOutlierAwareAttributedNetworkEmbedding,
                     algorithm_name="AdversarialOutlierAwareAttributedNetworkEmbedding",
                     hyper_parameter_dict=param_dict_logistic_regression,
                     discrete_hyper_parameter_dict=discrete_parameter_dict,
                     discrete_hyper_parameter_mapping=discrete_hyper_parameter_mapping,
                     continuous_hyper_parameter_mapping=continuous_hyper_parameter_mapping_index_key_mapping,
                     parameter_constraint_dict=parameter_constraint_dict,
                     constant_hyper_parameter_dict=const_param,
                     bounds=bounds)
