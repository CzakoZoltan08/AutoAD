# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 18:25:21 2019

@author: Zoltan
"""
from AutomaticAI.Algorithm import Algorithm
from autoad.algorithms.feawad.feawad import FEAWAD
from collections import OrderedDict

const_param = {}

dicrete_hyper_parameter_list_of_network_depth = [1, 2, 4]
dicrete_hyper_parameter_list_of_known_outliers = range(1, 100)

continuous_hyper_parameter_mapping_index_key_mapping = ["cont_rate"]
discrete_hyper_parameter_mapping = []
discrete_hyper_parameter_mapping = ["network_depth", "known_outliers"]

discrete_parameter_dict = OrderedDict()
discrete_parameter_dict["network_depth"] = dicrete_hyper_parameter_list_of_network_depth
discrete_parameter_dict["known_outliers"] = dicrete_hyper_parameter_list_of_known_outliers
parameter_constraint_dict = OrderedDict()

# dictionary of parameters
param_dict = OrderedDict()
param_dict['cont_rate'] = 0.02

bounds = [(0.001, 0.999), (0.0000001, 2.99), (1, 98)]


def get_algorithm():
    return Algorithm(algorithm_type=FEAWAD,
                     algorithm_name="FEAWAD",
                     hyper_parameter_dict=param_dict,
                     discrete_hyper_parameter_dict=discrete_parameter_dict,
                     discrete_hyper_parameter_mapping=discrete_hyper_parameter_mapping,
                     continuous_hyper_parameter_mapping=continuous_hyper_parameter_mapping_index_key_mapping,
                     parameter_constraint_dict=parameter_constraint_dict,
                     constant_hyper_parameter_dict=const_param,
                     bounds=bounds)
