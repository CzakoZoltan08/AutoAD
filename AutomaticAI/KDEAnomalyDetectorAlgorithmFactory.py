# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 18:25:21 2019

@author: Zoltan
"""
from AutomaticAI.Algorithm import Algorithm
from collections import OrderedDict

from autoad.algorithms.kde import KDEAnomalyDetector

const_param = {}

dicrete_hyper_parameter_list_of_algorithm = ['auto', 'ball_tree', 'kd_tree']
dicrete_hyper_parameter_list_of_metric = ['cityblock', 'euclidean', 'l1', 'l2',
                                          'manhattan', 'braycurtis', 'canberra', 'chebyshev',
                                          'dice', 'hamming', 'jaccard', 'kulsinski',
                                          'matching', 'minkowski', 'rogerstanimoto',
                                          'russellrao', 'sokalmichener', 'sokalsneath',
                                          'yule']
dicrete_hyper_parameter_list_of_leaf_size = range(10, 200)

continuous_hyper_parameter_mapping_index_key_mapping = [
    "contamination", "bandwidth"]
discrete_hyper_parameter_mapping = ["algorithm", "metric", "leaf_size"]

discrete_parameter_dict = OrderedDict()
discrete_parameter_dict["algorithm"] = dicrete_hyper_parameter_list_of_algorithm
discrete_parameter_dict["metric"] = dicrete_hyper_parameter_list_of_metric
discrete_parameter_dict["leaf_size"] = dicrete_hyper_parameter_list_of_leaf_size
parameter_constraint_dict = OrderedDict()

# dictionary of parameters
param_dict = OrderedDict()
param_dict['contamination'] = 0.1
param_dict['bandwidth'] = 1.0

bounds = [(0.001, 0.4999), (0.0000001, 2.99), (0.0000001, 1.99),
          (0.0000001, 17.99), (10.0000001, 92.99)]


def get_algorithm():
    return Algorithm(algorithm_type=KDEAnomalyDetector,
                     algorithm_name="KDEAnomalyDetector",
                     hyper_parameter_dict=param_dict,
                     discrete_hyper_parameter_dict=discrete_parameter_dict,
                     discrete_hyper_parameter_mapping=discrete_hyper_parameter_mapping,
                     continuous_hyper_parameter_mapping=continuous_hyper_parameter_mapping_index_key_mapping,
                     parameter_constraint_dict=parameter_constraint_dict,
                     constant_hyper_parameter_dict=const_param,
                     bounds=bounds)
