# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 18:25:21 2019

@author: Zoltan
"""
from AutomaticAI.Algorithm import Algorithm
from collections import OrderedDict

from autoad.algorithms.ganomaly.ganomaly import GANomaly


const_param = {}

continuous_hyper_parameter_mapping_index_key_mapping = ["lr", "mom"]
discrete_hyper_parameter_mapping = []

discrete_parameter_dict = OrderedDict()

parameter_constraint_dict = OrderedDict()

# dictionary of parameters
param_dict = OrderedDict()
param_dict['lr'] = 0.01
param_dict['mom'] = 0.7

bounds = [(0.0000001, 9.99), (0.001, 0.99)]


def get_algorithm():
    return Algorithm(algorithm_type=GANomaly,
                     algorithm_name="GANomaly",
                     hyper_parameter_dict=param_dict,
                     discrete_hyper_parameter_dict=discrete_parameter_dict,
                     discrete_hyper_parameter_mapping=discrete_hyper_parameter_mapping,
                     continuous_hyper_parameter_mapping=continuous_hyper_parameter_mapping_index_key_mapping,
                     parameter_constraint_dict=parameter_constraint_dict,
                     constant_hyper_parameter_dict=const_param,
                     bounds=bounds)
