# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 18:25:21 2019

@author: Zoltan
"""
from AutomaticAI.Algorithm import Algorithm
from collections import OrderedDict

from autoad.algorithms.lscp import LocallySelectiveCombination

const_param = {}

dicrete_hyper_parameter_list_of_n_bins = range(3, 200)
dicrete_hyper_parameter_list_of_local_region_size = range(3, 200)

continuous_hyper_parameter_mapping_index_key_mapping = [
    "contamination", "local_max_features"]
discrete_hyper_parameter_mapping = []
discrete_hyper_parameter_mapping = ["n_bins", "local_region_size"]

discrete_parameter_dict = OrderedDict()
discrete_parameter_dict["n_bins"] = dicrete_hyper_parameter_list_of_n_bins
discrete_parameter_dict["local_region_size"] = dicrete_hyper_parameter_list_of_local_region_size
parameter_constraint_dict = OrderedDict()

# dictionary of parameters
param_dict = OrderedDict()
param_dict['contamination'] = 0.1
param_dict['local_max_features'] = 0.5

bounds = [(0.001, 0.4999), (0.5, 0.9999), (3, 98), (3, 98)]


def get_algorithm():
    return Algorithm(algorithm_type=LocallySelectiveCombination,
                     algorithm_name="LocallySelectiveCombination",
                     hyper_parameter_dict=param_dict,
                     discrete_hyper_parameter_dict=discrete_parameter_dict,
                     discrete_hyper_parameter_mapping=discrete_hyper_parameter_mapping,
                     continuous_hyper_parameter_mapping=continuous_hyper_parameter_mapping_index_key_mapping,
                     parameter_constraint_dict=parameter_constraint_dict,
                     constant_hyper_parameter_dict=const_param,
                     bounds=bounds)
