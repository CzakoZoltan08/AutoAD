# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 21:54:44 2019

@author: Zoltan
"""

from collections import OrderedDict

from sklearn.naive_bayes import GaussianNB 

from AutomaticAI.Algorithm import Algorithm


const_param = {

}

continuous_hyper_parameter_mapping_index_key_mapping = ["var_smoothing"]
discrete_hyper_parameter_mapping = []

discrete_parameter_dict = OrderedDict()
parameter_constraint_dict = OrderedDict()

# dictionary of parameters
param_dict = OrderedDict()
param_dict['var_smoothing'] = 1e-9


bounds=[(1e-11,1.0)]


def get_algorithm():
    return Algorithm(algorithm_type=GaussianNB,
                     algorithm_name="ADAPTIVE GAUSSIANNB CLASSIFIER",
                     hyper_parameter_dict=param_dict,
                     discrete_hyper_parameter_dict=discrete_parameter_dict,
                     discrete_hyper_parameter_mapping=discrete_hyper_parameter_mapping,
                     continuous_hyper_parameter_mapping=continuous_hyper_parameter_mapping_index_key_mapping,
                     parameter_constraint_dict=parameter_constraint_dict,
                     constant_hyper_parameter_dict=const_param,
                     bounds=bounds)