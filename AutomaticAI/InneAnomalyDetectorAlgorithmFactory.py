# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 18:25:21 2019

@author: Zoltan
"""
from AutomaticAI.Algorithm import Algorithm
from collections import OrderedDict

from autoad.algorithms.inne import InneAnomalyDetector

const_param = {}

dicrete_hyper_parameter_list_of_n_estimators = range(1, 300)

continuous_hyper_parameter_mapping_index_key_mapping = ["contamination"]
discrete_hyper_parameter_mapping = []
discrete_hyper_parameter_mapping = ["n_estimators"]

discrete_parameter_dict = OrderedDict()
discrete_parameter_dict["n_estimators"] = dicrete_hyper_parameter_list_of_n_estimators
parameter_constraint_dict = OrderedDict()

# dictionary of parameters
param_dict = OrderedDict()
param_dict['contamination'] = 0.1

bounds = [(0.001, 0.4999), (1, 295)]


def get_algorithm():
    return Algorithm(algorithm_type=InneAnomalyDetector,
                     algorithm_name="InneAnomalyDetector",
                     hyper_parameter_dict=param_dict,
                     discrete_hyper_parameter_dict=discrete_parameter_dict,
                     discrete_hyper_parameter_mapping=discrete_hyper_parameter_mapping,
                     continuous_hyper_parameter_mapping=continuous_hyper_parameter_mapping_index_key_mapping,
                     parameter_constraint_dict=parameter_constraint_dict,
                     constant_hyper_parameter_dict=const_param,
                     bounds=bounds)
