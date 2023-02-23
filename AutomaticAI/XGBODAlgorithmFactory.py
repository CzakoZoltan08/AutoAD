# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 18:25:21 2019

@author: Zoltan
"""
import sys
sys.path.append('C:/University/Git/AutomaticAI_Flask/ADBench')

from collections import OrderedDict

from baseline.PyOD import XGBOD

from AutomaticAI.Algorithm import Algorithm


const_param = {}

dicrete_hyper_parameter_list_of_max_depth = range(0,20)
dicrete_hyper_parameter_list_of_n_estimators = range(0,500)

continuous_hyper_parameter_mapping_index_key_mapping = ["learning_rate", "gamma"]
discrete_hyper_parameter_mapping = []

discrete_parameter_dict = OrderedDict()
discrete_parameter_dict["max_depth"] = dicrete_hyper_parameter_list_of_max_depth
discrete_parameter_dict["n_estimators"] = dicrete_hyper_parameter_list_of_n_estimators

parameter_constraint_dict = OrderedDict()

# dictionary of parameters
param_dict = OrderedDict()
param_dict['learning_rate'] = 0.000001
param_dict['gamma'] = 0.000001
param_dict['reg_alpha'] = 0.000001
param_dict['reg_lambda'] = 0.000001
param_dict['seed'] = 42

bounds=[(0.0000001,10.99), (0.000001,10.9999), (0.000001,10.9999), (0.000001,10.9999), (0.1,18.9999), (0.1,498.9999)]


def get_algorithm():
    return Algorithm(algorithm_type=XGBOD,
                     algorithm_name="XGBOD",
                     hyper_parameter_dict=param_dict,
                     discrete_hyper_parameter_dict=discrete_parameter_dict,
                     discrete_hyper_parameter_mapping=discrete_hyper_parameter_mapping,
                     continuous_hyper_parameter_mapping=continuous_hyper_parameter_mapping_index_key_mapping,
                     parameter_constraint_dict=parameter_constraint_dict,
                     constant_hyper_parameter_dict=const_param,
                     bounds=bounds)