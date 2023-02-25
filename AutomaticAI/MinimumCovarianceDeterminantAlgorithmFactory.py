# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 18:25:21 2019

@author: Zoltan
"""
from AutomaticAI.Algorithm import Algorithm
from collections import OrderedDict

from autoad.algorithms.mcd import MinimumCovarianceDeterminant

const_param = {
    'support_fraction': True,
}

dicrete_hyper_parameter_list_of_assume_centered = [False, True]

continuous_hyper_parameter_mapping_index_key_mapping = ["contamination"]
discrete_hyper_parameter_mapping = ["assume_centered"]

discrete_parameter_dict = OrderedDict()
discrete_parameter_dict["assume_centered"] = dicrete_hyper_parameter_list_of_assume_centered
parameter_constraint_dict = OrderedDict()

# dictionary of parameters
param_dict = OrderedDict()
param_dict['contamination'] = 0.1

bounds = [(0.001, 0.4999), (0.0, 1.0)]


def get_algorithm():
    return Algorithm(algorithm_type=MinimumCovarianceDeterminant,
                     algorithm_name="MinimumCovarianceDeterminant",
                     hyper_parameter_dict=param_dict,
                     discrete_hyper_parameter_dict=discrete_parameter_dict,
                     discrete_hyper_parameter_mapping=discrete_hyper_parameter_mapping,
                     continuous_hyper_parameter_mapping=continuous_hyper_parameter_mapping_index_key_mapping,
                     parameter_constraint_dict=parameter_constraint_dict,
                     constant_hyper_parameter_dict=const_param,
                     bounds=bounds)
