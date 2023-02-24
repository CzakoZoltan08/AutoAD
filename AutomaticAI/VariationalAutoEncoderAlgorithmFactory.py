# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 18:25:21 2019

@author: Zoltan
"""
from AutomaticAI.Algorithm import Algorithm
from collections import OrderedDict

from autoad.algorithms.vae import VariationalAutoEncoder

const_param = {
    "encoder_neurons": [32, 16],
    "decoder_neurons": [16, 32]
}

dicrete_hyper_parameter_list_of_hidden_activation = [
    "relu", "sigmoid", "softmax", "softplus", "softsign", "tanh", "selu", "elu", "exponential"]
dicrete_hyper_parameter_list_of_output_activation = [
    "relu", "sigmoid", "softmax", "softplus", "softsign", "tanh", "selu", "elu", "exponential"]
dicrete_hyper_parameter_list_of_optimizer = [
    "SGD", "RMSprop", "Adam", "Adadelta", "Adagrad", "Adamax"]

continuous_hyper_parameter_mapping_index_key_mapping = [
    "contamination", "gamma", "capacity", "l2_regularizer", "dropout_rate"]
discrete_hyper_parameter_mapping = []
discrete_hyper_parameter_mapping = [
    "hidden_activation", "output_activation", "optimizer"]

discrete_parameter_dict = OrderedDict()
discrete_parameter_dict["hidden_activation"] = dicrete_hyper_parameter_list_of_hidden_activation
discrete_parameter_dict["output_activation"] = dicrete_hyper_parameter_list_of_output_activation
discrete_parameter_dict["optimizer"] = dicrete_hyper_parameter_list_of_optimizer
parameter_constraint_dict = OrderedDict()

# dictionary of parameters
param_dict = OrderedDict()
param_dict['contamination'] = 0.1
param_dict['gamma'] = 1.0
param_dict['capacity'] = 0.0
param_dict['l2_regularizer'] = 0.1
param_dict['dropout_rate'] = 0.2

bounds = [(0.001, 0.4999), (0.0000001, 9.99), (0.0000001, 9.99),
          (0.0000001, 9.99), (0.0000001, 0.99), (0.00001, 7.99), (0.00001, 7.99), (0.00001, 4.99)]


def get_algorithm():
    return Algorithm(algorithm_type=VariationalAutoEncoder,
                     algorithm_name="VariationalAutoEncoder",
                     hyper_parameter_dict=param_dict,
                     discrete_hyper_parameter_dict=discrete_parameter_dict,
                     discrete_hyper_parameter_mapping=discrete_hyper_parameter_mapping,
                     continuous_hyper_parameter_mapping=continuous_hyper_parameter_mapping_index_key_mapping,
                     parameter_constraint_dict=parameter_constraint_dict,
                     constant_hyper_parameter_dict=const_param,
                     bounds=bounds)
