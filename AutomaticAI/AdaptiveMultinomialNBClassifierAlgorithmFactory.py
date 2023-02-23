# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 21:54:44 2019

@author: Zoltan
"""

from collections import OrderedDict

from sklearn.naive_bayes import MultinomialNB 

from AutomaticAI.Algorithm import Algorithm


const_param = {

}

continuous_hyper_parameter_mapping_index_key_mapping = ["alpha"]
discrete_hyper_parameter_mapping = []

discrete_parameter_dict = OrderedDict()
parameter_constraint_dict = OrderedDict()

# dictionary of parameters
param_dict = OrderedDict()
param_dict['alpha'] = 1.0


bounds=[(0.0001,2.99)]


def get_algorithm():
    return Algorithm(algorithm_type=MultinomialNB,
                     algorithm_name="ADAPTIVE MULTINOMIALNB CLASSIFIER",
                     hyper_parameter_dict=param_dict,
                     discrete_hyper_parameter_dict=discrete_parameter_dict,
                     discrete_hyper_parameter_mapping=discrete_hyper_parameter_mapping,
                     continuous_hyper_parameter_mapping=continuous_hyper_parameter_mapping_index_key_mapping,
                     parameter_constraint_dict=parameter_constraint_dict,
                     constant_hyper_parameter_dict=const_param,
                     bounds=bounds)