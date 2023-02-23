from collections import OrderedDict

from sklearn.linear_model import Perceptron

from AutomaticAI.Algorithm import Algorithm



const_param_logistic_regression = {

}

dicrete_hyper_parameter_list_of_penalty = ["l2", "l1", "elasticnet"]

continuous_hyper_parameter_mapping_index_key_mapping = ["alpha", "eta0"]
discrete_hyper_parameter_mapping = ["penalty"]

discrete_parameter_dict_logistic_regression = OrderedDict()
discrete_parameter_dict_logistic_regression["penalty"] = dicrete_hyper_parameter_list_of_penalty

parameter_constraint_dict = OrderedDict()

# logistic regression
param_dict_logistic_regression = OrderedDict()
param_dict_logistic_regression['eta0'] = 1.0
param_dict_logistic_regression['alpha'] = 0.0001
param_dict_logistic_regression['penalty'] = 'l1'

bounds=[(0.001,3),(0.000001,3),(0.001,2)] 


def get_algorithm():
    return Algorithm(algorithm_type=Perceptron,
                       algorithm_name="ADAPTIVE PERCEPTRON CLASSIFIER",
                       hyper_parameter_dict=param_dict_logistic_regression,
                       discrete_hyper_parameter_dict=discrete_parameter_dict_logistic_regression,
                       discrete_hyper_parameter_mapping=discrete_hyper_parameter_mapping,
                       continuous_hyper_parameter_mapping=continuous_hyper_parameter_mapping_index_key_mapping,
                       parameter_constraint_dict=parameter_constraint_dict,
                       constant_hyper_parameter_dict=const_param_logistic_regression,
                       bounds=bounds)