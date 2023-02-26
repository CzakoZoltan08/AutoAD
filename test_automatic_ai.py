# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 11:30:29 2019

@author: czzo
"""
import time

from sklearn.preprocessing import StandardScaler
from AutomaticAI import ParticleSwarmOptimization as pso_algorithm
# from sklearn.metrics import roc_auc_score
# from pyod.utils.utility import precision_n_scores
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, f1_score, precision_score
from sklearn.model_selection import train_test_split
# from sklearn import datasets

from autoad.data_generators.anomaly_data_generator import AnomalyDataGenerator
# import sys
# sys.path.append('C:/University/Git/AutomaticAI_Flask')

# import AutomaticAI.ClusterBasedLocalOutlierFactorAlgorithmFactory as coblofaf
# import AutomaticAI.HistogramBasedOutlierDetectionAlgorithmFactory as hbodaf
import AutomaticAI.IsolationForestAlgorithmFactory as ifaf
# import AutomaticAI.SemiSupervisedKNNAlgorithmFactory as ssknnaf

from pprint import pprint


# --- MAIN ---------------------------------------------------------------------+
def main():
    # load the MNIST digits dataset
    # mnist = datasets.load_digits()

    anomaly_data_generator = AnomalyDataGenerator()
    dataset = anomaly_data_generator.generate(scaler=StandardScaler())

    # X = mnist.data
    # y = mnist.target

    X = dataset['X_train']
    y = dataset['y_train'].ravel()

    # Splitting the data into training set, test set and validation set
    x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=42)

    num_particles = 10
    num_iterations = 5

    anomaly_detection_semisupervised_algorithms = [
        # coblofaf.get_algorithm(),
        # hbodaf.get_algorithm(),
        ifaf.get_algorithm(),
        # ssknnaf.get_algorithm(),
    ]

    pso_semi_supervised = pso_algorithm.PSO(particle_count=num_particles,
                                            is_semisupervised=True,
                                            distance_between_initial_particles=0.7,
                                            evaluation_metric=f1_score,
                                            is_custom_algorithm_list=True,
                                            algorithm_list=anomaly_detection_semisupervised_algorithms)

    # pso_supervised = pso_algorithm.PSO(particle_count=num_particles, is_semisupervised=False,
    #                                    distance_between_initial_particles=0.7, evaluation_metric=roc_auc_score)

    start_time = time.time()

    # best_results_supervised = pso_supervised.fit(X_train=x_train,
    #                                              X_test=x_test,
    #                                              Y_train=y_train,
    #                                              Y_test=y_test,
    #                                              maxiter=num_iterations,
    #                                              verbose=True,
    #                                              compare_models=True,
    #                                              max_distance=0.05,
    #                                              agents=1)

    should_compare_models = True

    best_model, best_results_semi_supervised = pso_semi_supervised.fit(X_train=x_train,
                                                                       X_test=x_test,
                                                                       Y_train=y_train,
                                                                       Y_test=y_test,
                                                                       maxiter=num_iterations,
                                                                       verbose=True,
                                                                       compare_models=should_compare_models,
                                                                       max_distance=0.05,
                                                                       agents=1)

    print("--- %s seconds ---" % (time.time() - start_time))

    y_pred = best_model.model_best_i.predict(x_test)
    roc_auc = round(roc_auc_score(y_test, y_pred), ndigits=4)
    accuracy = round(accuracy_score(y_test, y_pred), ndigits=4)
    recall = round(recall_score(y_test, y_pred), ndigits=4)
    f1 = round(f1_score(y_test, y_pred), ndigits=4)
    precision = round(precision_score(y_test, y_pred), ndigits=4)

    if should_compare_models is False:
        print(
            f"Best metric: {best_model.metric_best_i} - Best roc_auc: {roc_auc} - Best recall: {recall}" +
            f"- Best accuracy: {accuracy} - Best F1: {f1} - Precision: {precision}")
        pprint(vars(best_model))
    else:
        best_results = {**best_results_semi_supervised}

        print("################## START ########################")

        print(" !!!!!!!! GLOBAL BEST !!!!!!!!!!")
        print(
            f"Best metric: {best_model.metric_best_i} - Best roc_auc: {roc_auc} - Best recall: {recall}" +
            f"- Best accuracy: {accuracy} - Best F1: {f1} - Precision: {precision}")
        pprint(vars(best_model))
        print(" !!!!!!!! GLOBAL BEST !!!!!!!!!!")

        for key, value in best_results.items():
            print("---------------------------------------------------------------------------------------------------")
            print(
                f'{key} -- {value.metric_i} -- Training time: {value.training_time}')
            pprint(vars(value.model_i))
            print("---------------------------------------------------------------------------------------------------")

        print("################### END #########################")


if __name__ == "__main__":
    main()
