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
import AutomaticAI.SemiSupervisedKNNAlgorithmFactory as ssknnaf

from pprint import pprint

import pandas as pd

from datetime import datetime

from autoad.data_generators.anomaly_type import AnomalyType
from autoad.data_generators.dataset import Dataset
from autoad.data_generators.noise_type import NoiseType


class TestData:
    def __init__(self,
                 dataset_name,
                 anomaly_type,
                 noise_type,
                 labeled_anomaly_ratio,
                 noise_ratio,
                 apply_data_scaling,
                 scaler,
                 algorithm) -> None:
        self.dataset_name = dataset_name
        self.anomaly_type = anomaly_type
        self.noise_type = noise_type
        self.labeled_anomaly_ratio = labeled_anomaly_ratio
        self.noise_ratio = noise_ratio
        self.apply_data_scaling = apply_data_scaling
        self.scaler = scaler,
        self.algorithm = algorithm


# --- MAIN ---------------------------------------------------------------------+
def main():
    # load the MNIST digits dataset
    # mnist = datasets.load_digits()

    test_cases = [
        TestData(
            dataset_name=Dataset.CARDIO,
            anomaly_type=AnomalyType.LOCAL,
            noise_type=NoiseType.NONE,
            labeled_anomaly_ratio=1.0,
            noise_ratio=0.1,
            apply_data_scaling=False,
            scaler=StandardScaler(),
            algorithm=ifaf.get_algorithm()
        ),
        TestData(
            dataset_name=Dataset.CARDIO,
            anomaly_type=AnomalyType.LOCAL,
            noise_type=NoiseType.NONE,
            labeled_anomaly_ratio=1.0,
            noise_ratio=0.5,
            apply_data_scaling=False,
            scaler=StandardScaler(),
            algorithm=ifaf.get_algorithm()
        ),
        TestData(
            dataset_name=Dataset.CARDIO,
            anomaly_type=AnomalyType.LOCAL,
            noise_type=NoiseType.NONE,
            labeled_anomaly_ratio=1.0,
            noise_ratio=0.5,
            apply_data_scaling=False,
            scaler=StandardScaler(),
            algorithm=ssknnaf.get_algorithm()
        )
    ]

    index = 0
    df_for_csv = {}

    for test_case in test_cases:
        anomaly_data_generator = AnomalyDataGenerator()
        dataset = anomaly_data_generator.generate(
            dataset=test_case.dataset_name,
            anomaly_type=test_case.anomaly_type,
            noise_type=test_case.noise_type,
            labeled_anomaly_ratio=test_case.labeled_anomaly_ratio,
            noise_ratio=test_case.noise_ratio,
            apply_data_scaling=test_case.apply_data_scaling,
            scaler=test_case.scaler
        )

        anomaly_detection_semisupervised_algorithms = [
            test_case.algorithm
        ]

        # X = mnist.data
        # y = mnist.target

        X = dataset['X_train']
        y = dataset['y_train'].ravel()

        # Splitting the data into training set, test set and validation set
        x_train, x_test, y_train, y_test = train_test_split(
            X, y, random_state=42)

        num_particles = 2
        num_iterations = 2

        evaluation_function = f1_score

        pso_semi_supervised = pso_algorithm.PSO(particle_count=num_particles,
                                                is_semisupervised=True,
                                                distance_between_initial_particles=0.7,
                                                evaluation_metric=evaluation_function,
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

        try:
            best_model, best_results_semi_supervised = pso_semi_supervised.fit(X_train=x_train,
                                                                               X_test=x_test,
                                                                               Y_train=y_train,
                                                                               Y_test=y_test,
                                                                               maxiter=num_iterations,
                                                                               verbose=True,
                                                                               compare_models=should_compare_models,
                                                                               max_distance=0.05,
                                                                               agents=1)
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

                res_dict = {
                    "algorithm_name": test_case.algorithm.algorithm_name,
                    "dataset": test_case.dataset_name,
                    "anomaly_type": test_case.anomaly_type,
                    "noise_type": test_case.noise_type,
                    "labeled_anomaly_ratio": test_case.labeled_anomaly_ratio,
                    "noise_ratio": test_case.noise_ratio,
                    "apply_data_scaling": test_case.apply_data_scaling,
                    "scaler": test_case.scaler.__class__.__name__,
                    "roc_auc": roc_auc,
                    "recall": recall,
                    "precision": precision,
                    "accuracy": accuracy,
                    "f1-score": f1
                }

                df = pd.DataFrame(res_dict, index=[index])

                if index == 0:
                    df_for_csv = df.copy()
                else:
                    df_for_csv = pd.concat([df_for_csv, df])
            else:
                best_results = {**best_results_semi_supervised}

                print("################## START ########################")

                print(" !!!!!!!! GLOBAL BEST !!!!!!!!!!")
                print(
                    f"Best metric: {best_model.metric_best_i} - Best roc_auc: {roc_auc} - Best recall: {recall}" +
                    f"- Best accuracy: {accuracy} - Best F1: {f1} - Precision: {precision}")
                pprint(vars(best_model))
                print(" !!!!!!!! GLOBAL BEST !!!!!!!!!!")

                res_dict = {
                    "algorithm_name": test_case.algorithm.algorithm_name,
                    "dataset": test_case.dataset_name,
                    "anomaly_type": test_case.anomaly_type,
                    "noise_type": test_case.noise_type,
                    "labeled_anomaly_ratio": test_case.labeled_anomaly_ratio,
                    "noise_ratio": test_case.noise_ratio,
                    "apply_data_scaling": test_case.apply_data_scaling,
                    "scaler": test_case.scaler.__class__.__name__,
                    "roc_auc": roc_auc,
                    "recall": recall,
                    "precision": precision,
                    "accuracy": accuracy,
                    "f1-score": f1
                }

                df = pd.DataFrame(res_dict, index=[index])

                if index == 0:
                    df_for_csv = df.copy()
                else:
                    df_for_csv = pd.concat([df_for_csv, df])

                for key, value in best_results.items():
                    print(
                        "-------------------------------------------------------------------------------------")
                    print(
                        f'{key} -- {value.metric_i} -- Training time: {value.training_time}')
                    pprint(vars(value.model_i))
                    print(
                        "-------------------------------------------------------------------------------------")

                print("################### END #########################")
        except Exception:
            res_dict = {
                "algorithm_name": test_case.algorithm.algorithm_name,
                "dataset": test_case.dataset_name,
                "anomaly_type": test_case.anomaly_type,
                "noise_type": test_case.noise_type,
                "labeled_anomaly_ratio": test_case.labeled_anomaly_ratio,
                "noise_ratio": test_case.noise_ratio,
                "apply_data_scaling": test_case.apply_data_scaling,
                "scaler": test_case.scaler.__class__.__name__,
                "roc_auc": 0,
                "recall": 0,
                "precision": 0,
                "accuracy": 0,
                "f1-score": 0
            }

            df = pd.DataFrame(res_dict, index=[index])

            if index == 0:
                df_for_csv = df.copy()
            else:
                df_for_csv = pd.concat([df_for_csv, df])

        print("--- %s seconds ---" % (time.time() - start_time))

        index += 1

    dn = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    df_for_csv.to_csv(
        f"./algorithm_evaluation_results/{dn}_" +
        f"{test_case.dataset_name}_{test_case.anomaly_type}_{test_case.noise_type}_" +
        f"{test_case.labeled_anomaly_ratio}_{test_case.noise_ratio}_{test_case.apply_data_scaling}.csv")


if __name__ == "__main__":
    main()
