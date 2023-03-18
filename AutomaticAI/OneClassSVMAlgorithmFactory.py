# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 18:25:21 2019

@author: Zoltan
"""
from AutomaticAI.Algorithm import Algorithm
from collections import OrderedDict

from autoad.algorithms.ocsvm import OneClassSVM

const_param = {}

dicrete_hyper_parameter_list_of_kernel = [
    'linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
dicrete_hyper_parameter_list_of_degree = range(1, 100)

continuous_hyper_parameter_mapping_index_key_mapping = [
    "contamination", "coef0", "tol", "nu"]
discrete_hyper_parameter_mapping = []
discrete_hyper_parameter_mapping = ["kernel", "degree"]

discrete_parameter_dict = OrderedDict()
discrete_parameter_dict["kernel"] = dicrete_hyper_parameter_list_of_kernel
discrete_parameter_dict["degree"] = dicrete_hyper_parameter_list_of_degree
parameter_constraint_dict = OrderedDict()

# dictionary of parameters
param_dict = OrderedDict()
param_dict['contamination'] = 0.1
param_dict['coef0'] = 0.0
param_dict['tol'] = 1e-3
param_dict['nu'] = 0.5

bounds = [(0.001, 0.4999), (0.0000001, 0.99), (0.0000001, 0.99),
          (0.0000001, 0.99), (0.0, 1.999), (1.0, 2.0)]


def get_algorithm():
    return Algorithm(algorithm_type=OneClassSVM,
                     algorithm_name="OneClassSVM",
                     hyper_parameter_dict=param_dict,
                     discrete_hyper_parameter_dict=discrete_parameter_dict,
                     discrete_hyper_parameter_mapping=discrete_hyper_parameter_mapping,
                     continuous_hyper_parameter_mapping=continuous_hyper_parameter_mapping_index_key_mapping,
                     parameter_constraint_dict=parameter_constraint_dict,
                     constant_hyper_parameter_dict=const_param,
                     bounds=bounds)
