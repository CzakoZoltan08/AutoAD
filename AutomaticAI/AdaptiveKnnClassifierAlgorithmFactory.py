from collections import OrderedDict

from skmultiflow.lazy import KNNClassifier 

from AutomaticAI.Algorithm import Algorithm


const_param = {
    'max_window_size': 1000
}

dicrete_hyper_parameter_list_of_algorithms = ["euclidean", "manhattan", "chebyshev"]
dicrete_hyper_parameter_list_of_leaf_size = range(1,120)
dicrete_hyper_parameter_list_of_neighbors = range(1,120)
continuous_hyper_parameter_mapping_index_key_mapping = []
discrete_hyper_parameter_mapping = ["n_neighbors", "leaf_size", "metric"]

discrete_parameter_dict = OrderedDict()
discrete_parameter_dict["n_neighbors"] = dicrete_hyper_parameter_list_of_neighbors
discrete_parameter_dict["leaf_size"] = dicrete_hyper_parameter_list_of_leaf_size
discrete_parameter_dict["metric"] = dicrete_hyper_parameter_list_of_algorithms
parameter_constraint_dict = OrderedDict()

# logistic regression
param_dict_logistic_regression = OrderedDict()
param_dict_logistic_regression['n_neighbors'] = 1
param_dict_logistic_regression['leaf_size'] = 1
param_dict_logistic_regression['metric'] = 'euclidean'


bounds=[(1.001,99.99),(1.001,99.99),(0.001,2.99)]


def get_algorithm():
    return Algorithm(algorithm_type=KNNClassifier,
                     algorithm_name="ADAPTIVE KNN CLASSIFIER",
                     hyper_parameter_dict=param_dict_logistic_regression,
                     discrete_hyper_parameter_dict=discrete_parameter_dict,
                     discrete_hyper_parameter_mapping=discrete_hyper_parameter_mapping,
                     continuous_hyper_parameter_mapping=continuous_hyper_parameter_mapping_index_key_mapping,
                     parameter_constraint_dict=parameter_constraint_dict,
                     constant_hyper_parameter_dict=const_param,
                     bounds=bounds)