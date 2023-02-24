# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 18:25:21 2019

@author: Zoltan
"""
from AutomaticAI.Algorithm import Algorithm
from collections import OrderedDict

from autoad.algorithms.pca import PCAAnomalyDetector

const_param = {}

dicrete_hyper_parameter_list_of_svd_solver = [
    'auto', 'full', 'arpack', 'randomized']

continuous_hyper_parameter_mapping_index_key_mapping = ["contamination", "tol"]
discrete_hyper_parameter_mapping = []
discrete_hyper_parameter_mapping = ["svd_solver"]

discrete_parameter_dict = OrderedDict()
discrete_parameter_dict["svd_solver"] = dicrete_hyper_parameter_list_of_svd_solver
parameter_constraint_dict = OrderedDict()

# dictionary of parameters
param_dict = OrderedDict()
param_dict['contamination'] = 0.1
param_dict['tol'] = 0.0

bounds = [(0.001, 0.4999), (0.0000001, 2.99), (1, 2.99)]


def get_algorithm():
    return Algorithm(algorithm_type=PCAAnomalyDetector,
                     algorithm_name="PCAAnomalyDetector",
                     hyper_parameter_dict=param_dict,
                     discrete_hyper_parameter_dict=discrete_parameter_dict,
                     discrete_hyper_parameter_mapping=discrete_hyper_parameter_mapping,
                     continuous_hyper_parameter_mapping=continuous_hyper_parameter_mapping_index_key_mapping,
                     parameter_constraint_dict=parameter_constraint_dict,
                     constant_hyper_parameter_dict=const_param,
                     bounds=bounds)
