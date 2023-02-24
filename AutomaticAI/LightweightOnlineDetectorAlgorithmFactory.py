# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 18:25:21 2019

@author: Zoltan
"""
from AutomaticAI.Algorithm import Algorithm
from collections import OrderedDict

from autoad.algorithms.loda import LightweightOnlineDetector

const_param = {}

dicrete_hyper_parameter_list_of_n_bins = range(3, 200)
dicrete_hyper_parameter_list_of_n_random_cuts = range(1, 200)

continuous_hyper_parameter_mapping_index_key_mapping = ["contamination"]
discrete_hyper_parameter_mapping = []
discrete_hyper_parameter_mapping = ["n_bins", "n_random_cuts"]

discrete_parameter_dict = OrderedDict()
discrete_parameter_dict["n_bins"] = dicrete_hyper_parameter_list_of_n_bins
discrete_parameter_dict["n_random_cuts"] = dicrete_hyper_parameter_list_of_n_random_cuts
parameter_constraint_dict = OrderedDict()

# dictionary of parameters
param_dict = OrderedDict()
param_dict['contamination'] = 0.1

bounds = [(0.001, 0.4999), (3.0, 97.99), (1, 97)]


def get_algorithm():
    return Algorithm(algorithm_type=LightweightOnlineDetector,
                     algorithm_name="LightweightOnlineDetector",
                     hyper_parameter_dict=param_dict,
                     discrete_hyper_parameter_dict=discrete_parameter_dict,
                     discrete_hyper_parameter_mapping=discrete_hyper_parameter_mapping,
                     continuous_hyper_parameter_mapping=continuous_hyper_parameter_mapping_index_key_mapping,
                     parameter_constraint_dict=parameter_constraint_dict,
                     constant_hyper_parameter_dict=const_param,
                     bounds=bounds)
