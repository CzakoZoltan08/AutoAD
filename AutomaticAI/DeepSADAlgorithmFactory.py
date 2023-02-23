# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 18:25:21 2019

@author: Zoltan
"""
from AutomaticAI.Algorithm import Algorithm
from collections import OrderedDict

from autoad.algorithms.deep_sad.deepsad import DeepSAD

const_param = {}

dicrete_hyper_parameter_list_of_optimizer_name = [
    "adam", "sgd", "adadelta", "adagrad", "adamw", "sparseadam", "adamax", "asgd", "nadam", "radam", "rmsprop", "rprop"]

continuous_hyper_parameter_mapping_index_key_mapping = ["lr", "eta", "eps"]
discrete_hyper_parameter_mapping = ["optimizer_name"]

discrete_parameter_dict = OrderedDict()
discrete_parameter_dict["optimizer_name"] = dicrete_hyper_parameter_list_of_optimizer_name

parameter_constraint_dict = OrderedDict()

# dictionary of parameters
param_dict = OrderedDict()
param_dict['lr'] = 0.000001
param_dict['eta'] = 0.000001
param_dict['eps'] = 0.000001

bounds = [(0.0000001, 9.99), (0.0000001, 9.99),
          (0.0000001, 1.99), (0.001, 0.99)]


def get_algorithm():
    return Algorithm(algorithm_type=DeepSAD,
                     algorithm_name="Deep SAD",
                     hyper_parameter_dict=param_dict,
                     discrete_hyper_parameter_dict=discrete_parameter_dict,
                     discrete_hyper_parameter_mapping=discrete_hyper_parameter_mapping,
                     continuous_hyper_parameter_mapping=continuous_hyper_parameter_mapping_index_key_mapping,
                     parameter_constraint_dict=parameter_constraint_dict,
                     constant_hyper_parameter_dict=const_param,
                     bounds=bounds)
