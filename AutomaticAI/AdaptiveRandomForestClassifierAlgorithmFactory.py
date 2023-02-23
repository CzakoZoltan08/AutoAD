from collections import OrderedDict

# from skmultiflow.meta import AdaptiveRandomForestClassifier

from AutomaticAI.Algorithm import Algorithm

const_param = {
    'drift_detection_method': None,
    'warning_detection_method': None
}

dicrete_hyper_parameter_list_of_split_criterion = ["gini", "info_gain"]
dicrete_hyper_parameter_list_of_estimators = range(1,320)
continuous_hyper_parameter_mapping_index_key_mapping = []
discrete_hyper_parameter_mapping = ["n_estimators","split_criterion"]

discrete_parameter_dict = OrderedDict()
discrete_parameter_dict["n_estimators"] = dicrete_hyper_parameter_list_of_estimators
discrete_parameter_dict["split_criterion"] = dicrete_hyper_parameter_list_of_split_criterion
parameter_constraint_dict = OrderedDict()

# logistic regression
param_dict_logistic_regression = OrderedDict()
param_dict_logistic_regression['n_estimators'] = 1
param_dict_logistic_regression['split_criterion'] = 'info_gain'


bounds=[(1.001,299.99),(0.001,1.99)]


def get_algorithm():
    return Algorithm(algorithm_type=AdaptiveRandomForestClassifier,
                     algorithm_name="ADAPTIVE RANDOM FOREST CLASSIFIER",
                     hyper_parameter_dict=param_dict_logistic_regression,
                     discrete_hyper_parameter_dict=discrete_parameter_dict,
                     discrete_hyper_parameter_mapping=discrete_hyper_parameter_mapping,
                     continuous_hyper_parameter_mapping=continuous_hyper_parameter_mapping_index_key_mapping,
                     parameter_constraint_dict=parameter_constraint_dict,
                     constant_hyper_parameter_dict=const_param,
                     bounds=bounds)