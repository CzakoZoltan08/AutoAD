# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 18:25:21 2019

@author: Zoltan
"""
from AutomaticAI.Algorithm import Algorithm
from collections import OrderedDict

from autoad.algorithms.sod import SubspaceOutlierDetection

const_param = {}

dicrete_hyper_parameter_list_of_n_neighbors = range(1, 200)
dicrete_hyper_parameter_list_of_ref_set = range(1, 200)

continuous_hyper_parameter_mapping_index_key_mapping = [
    "contamination", "alpha"]
discrete_hyper_parameter_mapping = []
discrete_hyper_parameter_mapping = ["n_neighbors", "ref_set"]

discrete_parameter_dict = OrderedDict()
discrete_parameter_dict["n_neighbors"] = dicrete_hyper_parameter_list_of_n_neighbors
discrete_parameter_dict["ref_set"] = dicrete_hyper_parameter_list_of_ref_set
parameter_constraint_dict = OrderedDict()

# dictionary of parameters
param_dict = OrderedDict()
param_dict['contamination'] = 0.1
param_dict['alpha'] = 0.1

bounds = [(0.001, 0.4999), (0.0000001, 0.99), (1, 98), (1, 2)]


def get_algorithm():
    return Algorithm(algorithm_type=SubspaceOutlierDetection,
                     algorithm_name="SubspaceOutlierDetection",
                     hyper_parameter_dict=param_dict,
                     discrete_hyper_parameter_dict=discrete_parameter_dict,
                     discrete_hyper_parameter_mapping=discrete_hyper_parameter_mapping,
                     continuous_hyper_parameter_mapping=continuous_hyper_parameter_mapping_index_key_mapping,
                     parameter_constraint_dict=parameter_constraint_dict,
                     constant_hyper_parameter_dict=const_param,
                     bounds=bounds)
