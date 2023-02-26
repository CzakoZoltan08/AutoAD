# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 18:25:21 2019

@author: Zoltan
"""
from AutomaticAI.Algorithm import Algorithm
from collections import OrderedDict

from autoad.algorithms.hbos import HistogramBasedOutlierDetection

const_param = {}

dicrete_hyper_parameter_list_of_n_bins = range(3, 200)

continuous_hyper_parameter_mapping_index_key_mapping = [
    "contamination", "alpha", "tol"]
discrete_hyper_parameter_mapping = []
discrete_hyper_parameter_mapping = ["n_bins"]

discrete_parameter_dict = OrderedDict()
discrete_parameter_dict["n_bins"] = dicrete_hyper_parameter_list_of_n_bins
parameter_constraint_dict = OrderedDict()

# dictionary of parameters
param_dict = OrderedDict()
param_dict['contamination'] = 0.1
param_dict['alpha'] = 0.1
param_dict['tol'] = 0.5

bounds = [(0.000001, 0.5), (0.00000001, 0.9999999),
          (0.000001, 0.9999999), (3, 95)]


def get_algorithm():
    return Algorithm(algorithm_type=HistogramBasedOutlierDetection,
                     algorithm_name="HistogramBasedOutlierDetection",
                     hyper_parameter_dict=param_dict,
                     discrete_hyper_parameter_dict=discrete_parameter_dict,
                     discrete_hyper_parameter_mapping=discrete_hyper_parameter_mapping,
                     continuous_hyper_parameter_mapping=continuous_hyper_parameter_mapping_index_key_mapping,
                     parameter_constraint_dict=parameter_constraint_dict,
                     constant_hyper_parameter_dict=const_param,
                     bounds=bounds)
