# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 18:25:21 2019

@author: Zoltan
"""
from AutomaticAI.Algorithm import Algorithm
from collections import OrderedDict

from autoad.algorithms.cblof import ClusterBasedLocalOutlierFactor

const_param = {}

dicrete_hyper_parameter_list_of_n_clusters = range(5, 100)
dicrete_hyper_parameter_list_of_beta = range(1, 100)

continuous_hyper_parameter_mapping_index_key_mapping = ["alpha"]
discrete_hyper_parameter_mapping = []
discrete_hyper_parameter_mapping = ["n_clusters", "beta"]

discrete_parameter_dict = OrderedDict()
discrete_parameter_dict["n_clusters"] = dicrete_hyper_parameter_list_of_n_clusters
discrete_parameter_dict["beta"] = dicrete_hyper_parameter_list_of_beta
parameter_constraint_dict = OrderedDict()

# dictionary of parameters
param_dict = OrderedDict()
param_dict['alpha'] = 0.02

bounds = [(0.00001, 0.99), (5.0000001, 48.99), (1, 98.99)]


def get_algorithm():
    return Algorithm(algorithm_type=ClusterBasedLocalOutlierFactor,
                     algorithm_name="ClusterBasedLocalOutlierFactor",
                     hyper_parameter_dict=param_dict,
                     discrete_hyper_parameter_dict=discrete_parameter_dict,
                     discrete_hyper_parameter_mapping=discrete_hyper_parameter_mapping,
                     continuous_hyper_parameter_mapping=continuous_hyper_parameter_mapping_index_key_mapping,
                     parameter_constraint_dict=parameter_constraint_dict,
                     constant_hyper_parameter_dict=const_param,
                     bounds=bounds)
