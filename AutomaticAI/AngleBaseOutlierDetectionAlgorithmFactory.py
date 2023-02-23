# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 18:25:21 2019

@author: Zoltan
"""
from AutomaticAI.Algorithm import Algorithm
from collections import OrderedDict

from autoad.algorithms.abod import AngleBaseOutlierDetection


const_param = {}

continuous_hyper_parameter_mapping_index_key_mapping = []
discrete_hyper_parameter_mapping = ["n_neighbors"]

discrete_parameter_dict = OrderedDict()
discrete_parameter_dict["n_neighbors"] = range(1, 30)

parameter_constraint_dict = OrderedDict()

# dictionary of parameters
param_dict = OrderedDict()

bounds = [(0.01, 28.99)]


def get_algorithm():
    return Algorithm(algorithm_type=AngleBaseOutlierDetection,
                     algorithm_name="AngleBaseOutlierDetection",
                     hyper_parameter_dict=param_dict,
                     discrete_hyper_parameter_dict=discrete_parameter_dict,
                     discrete_hyper_parameter_mapping=discrete_hyper_parameter_mapping,
                     continuous_hyper_parameter_mapping=continuous_hyper_parameter_mapping_index_key_mapping,
                     parameter_constraint_dict=parameter_constraint_dict,
                     constant_hyper_parameter_dict=const_param,
                     bounds=bounds)
