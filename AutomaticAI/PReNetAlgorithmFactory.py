# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 18:25:21 2019

@author: Zoltan
"""
from AutomaticAI.Algorithm import Algorithm
from collections import OrderedDict
import sys

from autoad.algorithms.prenet.prenet import PReNet
sys.path.append('C:/University/Git/AutomaticAI_Flask/ADBench')


const_param = {}

dicrete_hyper_parameter_list_of_s_a_a = range(0, 200)
dicrete_hyper_parameter_list_of_s_a_u = range(0, 200)
dicrete_hyper_parameter_list_of_s_u_u = range(0, 200)

continuous_hyper_parameter_mapping_index_key_mapping = ["lr"]
discrete_hyper_parameter_mapping = ["s_a_a", "s_a_u", "s_u_u"]

discrete_parameter_dict = OrderedDict()
discrete_parameter_dict["s_a_a"] = dicrete_hyper_parameter_list_of_s_a_a
discrete_parameter_dict["s_a_u"] = dicrete_hyper_parameter_list_of_s_a_u
discrete_parameter_dict["s_u_u"] = dicrete_hyper_parameter_list_of_s_u_u

parameter_constraint_dict = OrderedDict()

# dictionary of parameters
param_dict = OrderedDict()
param_dict['lr'] = 0.000001
param_dict['seed'] = 42

bounds = [(0.0000001, 0.99), (0.001, 10.99), (0.001, 10.99), (0.001, 10.99)]


def get_algorithm():
    return Algorithm(algorithm_type=PReNet,
                     algorithm_name="PReNet",
                     hyper_parameter_dict=param_dict,
                     discrete_hyper_parameter_dict=discrete_parameter_dict,
                     discrete_hyper_parameter_mapping=discrete_hyper_parameter_mapping,
                     continuous_hyper_parameter_mapping=continuous_hyper_parameter_mapping_index_key_mapping,
                     parameter_constraint_dict=parameter_constraint_dict,
                     constant_hyper_parameter_dict=const_param,
                     bounds=bounds)
