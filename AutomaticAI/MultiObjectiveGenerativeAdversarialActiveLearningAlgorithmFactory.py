# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 18:25:21 2019

@author: Zoltan
"""
from AutomaticAI.Algorithm import Algorithm
from collections import OrderedDict

from autoad.algorithms.mogaal import MultiObjectiveGenerativeAdversarialActiveLearning

const_param = {}

dicrete_hyper_parameter_list_of_k = range(3, 200)

continuous_hyper_parameter_mapping_index_key_mapping = [
    "contamination", "lr_d", "lr_g", "momentum"]
discrete_hyper_parameter_mapping = ["k"]

discrete_parameter_dict = OrderedDict()
discrete_parameter_dict["k"] = dicrete_hyper_parameter_list_of_k
parameter_constraint_dict = OrderedDict()

# dictionary of parameters
param_dict = OrderedDict()
param_dict['contamination'] = 0.1
param_dict['lr_d'] = 0.01
param_dict['lr_g'] = 0.0001
param_dict['momentum'] = 0.9

bounds = [(0.001, 0.4999), (0.0000001, 0.99),
          (0.0000001, 0.99), (0.0, 1.0), (1.0, 10.0)]


def get_algorithm():
    return Algorithm(algorithm_type=MultiObjectiveGenerativeAdversarialActiveLearning,
                     algorithm_name="MultiObjectiveGenerativeAdversarialActiveLearning",
                     hyper_parameter_dict=param_dict,
                     discrete_hyper_parameter_dict=discrete_parameter_dict,
                     discrete_hyper_parameter_mapping=discrete_hyper_parameter_mapping,
                     continuous_hyper_parameter_mapping=continuous_hyper_parameter_mapping_index_key_mapping,
                     parameter_constraint_dict=parameter_constraint_dict,
                     constant_hyper_parameter_dict=const_param,
                     bounds=bounds)
