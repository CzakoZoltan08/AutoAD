# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 11:30:29 2019

@author: czzo
"""
import time
from pprint import pprint
from datetime import datetime
from csv import DictWriter, writer

import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
from sklearn.base import TransformerMixin
from sklearn.decomposition import PCA

from pyod.utils.utility import precision_n_scores
from autoad.algorithms.deep_sad.deepsad import DeepSAD
from autoad.algorithms.feawad.feawad import FEAWAD
from autoad.algorithms.ganomaly.ganomaly import GANomaly
from autoad.algorithms.prenet.prenet import PReNet
from autoad.algorithms.repen.repen import REPEN

from autoad.data_generators.anomaly_data_generator import AnomalyDataGenerator
from autoad.data_generators.anomaly_type import AnomalyType
from autoad.data_generators.dataset import Dataset
from autoad.data_generators.noise_type import NoiseType

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as impipe


class TestData:
    def __init__(self,
                 dataset_name,
                 anomaly_type,
                 noise_type,
                 labeled_anomaly_ratio,
                 noise_ratio,
                 algorithm,
                 apply_data_scaling: bool = False,
                 apply_data_rebalancing: bool = False,
                 apply_missing_data_filling: bool = False,
                 apply_dimensionality_reduction: bool = False,
                 scaler: TransformerMixin = MinMaxScaler(),
                 rebalance_pipeline=impipe(
                     steps=[('o', SMOTE(sampling_strategy=0.2))]),
                 fill_algorithm='Median',
                 dimensinality_reduction_algorithm=PCA(n_components=2)) -> None:
        self.dataset_name = dataset_name
        self.anomaly_type = anomaly_type
        self.noise_type = noise_type
        self.labeled_anomaly_ratio = labeled_anomaly_ratio
        self.noise_ratio = noise_ratio
        self.apply_data_scaling = apply_data_scaling
        self.scaler = scaler,
        self.apply_data_rebalancing = apply_data_rebalancing
        self.rebalance_pipeline = rebalance_pipeline
        self.apply_missing_data_filling = apply_missing_data_filling
        self.fill_algorithm = fill_algorithm
        self.apply_dimensionality_reduction = apply_dimensionality_reduction
        self.dimensinality_reduction_algorithm = dimensinality_reduction_algorithm
        self.algorithm = algorithm


# --- MAIN ---------------------------------------------------------------------+
def main():
    # noise_ratios = [x / 100.0 for x in range(0, 60, 10)]
    noise_ratios = [0]

    labeled_anomaly_ratios = [x / 100.0 for x in range(0, 110, 10)]
    # labeled_anomaly_ratios = [1.0]

    noise_types = [NoiseType.NONE]
    anomaly_types = [AnomalyType.LOCAL,
                     AnomalyType.GLOBAL, AnomalyType.CLUSTER]

    dataset_names = [x.value for x in Dataset]
    # dataset_names = dataset_names[1:]

    semisupervised_algorithms = [
        # GANomaly(),
        # DeepSAD(),
        REPEN(),
        # PReNet(),
        # FEAWAD(),
    ]

    field_names = ['algorithm_name', 'dataset', 'anomaly_type', 'noise_type',
                   'labeled_anomaly_ratio', 'noise_ratio',
                   'apply_data_scaling', 'scaler', 'apply_data_rebalancing',
                   'rebalance_pipeline', 'apply_missing_data_filling', 'fill_algorithm',
                   'apply_dimensionality_reduction', 'dimensinality_reduction_algorithm',
                   'roc_auc', 'precision_n_scores', 'exception']

    test_type = 'Semisupervised_algorithms'

    for dataset_name in dataset_names:
        for algorithm in semisupervised_algorithms:
            for anomaly_type in anomaly_types:
                dn = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
                file_name = f"./algorithm_evaluation_results/Semisupervised/Repen/{dn}_{test_type}_" + \
                    f"{dataset_name}_{anomaly_type}_{algorithm.__class__.__name__}_" + \
                    f"rebalance{False}.csv"

                with open(file_name, 'w') as f_object:
                    writer_object = writer(f_object)
                    writer_object.writerow(field_names)
                    f_object.close()

                all_test_cases = []
                test_case_count = 1
                test_case_index = 0

                for labeled_anomaly_ratio in labeled_anomaly_ratios:
                    for noise_type in noise_types:
                        for noise_ratio in noise_ratios:
                            # for apply_data_rebalancing in truth_values:
                            #     for under_sampling_algorithm in under_sampling_algorithms:
                            # for dimensinality_reduction_algorithm in dimensinality_reduction_algorithms:
                            #           for scaler in scalers:
                            #             for fill_algorithm in fill_algorithms:
                            #                 for apply_data_scaling in truth_values:
                            #                     for apply_missing_data_filling in truth_values:
                            #                         for apply_dimensionality_reduction in truth_values:
                            #                             for apply_data_rebalancing in truth_values:
                            all_test_cases.append(
                                TestData(
                                    dataset_name=dataset_name,
                                    anomaly_type=anomaly_type,
                                    noise_type=noise_type,
                                    labeled_anomaly_ratio=labeled_anomaly_ratio,
                                    noise_ratio=noise_ratio,
                                    apply_data_scaling=True,
                                    scaler=MinMaxScaler(),
                                    apply_data_rebalancing=False,
                                    rebalance_pipeline=impipe(
                                        [('u', RandomUnderSampler())]),
                                    apply_missing_data_filling=False,
                                    fill_algorithm='Zero',
                                    apply_dimensionality_reduction=False,
                                    dimensinality_reduction_algorithm=PCA(
                                        n_components=5),
                                    algorithm=algorithm
                                )
                            )
                            print(
                                f"Test Case: {test_case_count}")
                            test_case_count += 1

                # manual_test_cases = [
                #     TestData(
                #         dataset_name=Dataset.CARDIO,
                #         anomaly_type=AnomalyType.LOCAL,
                #         noise_type=NoiseType.NONE,
                #         labeled_anomaly_ratio=1.0,
                #         noise_ratio=0.1,
                #         apply_data_scaling=True,
                #         scaler=StandardScaler(),
                #         apply_data_rebalancing=True,
                #         rebalance_pipeline=impipe([('o', SMOTE(
                #             sampling_strategy=0.2)), ('u', RandomUnderSampler(sampling_strategy=0.3))]),
                #         apply_missing_data_filling=True,
                #         fill_algorithm='Zero',
                #         apply_dimensionality_reduction=True,
                #         dimensinality_reduction_algorithm=PCA(n_components=5),
                #         algorithm=kdeaf.get_algorithm(),
                #     )
                # ]

                index = 0
                df_for_csv = {}

                for test_case in all_test_cases:
                    try:
                        anomaly_data_generator = AnomalyDataGenerator()
                        dataset = anomaly_data_generator.generate(
                            dataset=test_case.dataset_name,
                            anomaly_type=test_case.anomaly_type,
                            noise_type=test_case.noise_type,
                            labeled_anomaly_ratio=test_case.labeled_anomaly_ratio,
                            noise_ratio=test_case.noise_ratio,
                            apply_data_scaling=test_case.apply_data_scaling,
                            scaler=test_case.scaler,
                            apply_data_rebalancing=test_case.apply_data_rebalancing,
                            rebalance_pipeline=test_case.rebalance_pipeline,
                            apply_missing_data_filling=test_case.apply_missing_data_filling,
                            fill_algorithm=test_case.fill_algorithm,
                            apply_dimensionality_reduction=test_case.apply_dimensionality_reduction,
                            dimensinality_reduction_algorithm=test_case.dimensinality_reduction_algorithm,
                            threshold=2000
                        )
                    except Exception:
                        continue

                    x_train = dataset['X_train']
                    y_train = dataset['y_train'].ravel()
                    x_test = dataset['X_test']
                    y_test = dataset['y_test'].ravel()

                    # num_particles = 3
                    # num_iterations = 5

                    # evaluation_function = roc_auc_score

                    # pso_unsupervised = pso_algorithm.PSO(particle_count=num_particles,
                    #                                      is_classification=True,
                    #                                      distance_between_initial_particles=0.7,
                    #                                      evaluation_metric=evaluation_function,
                    #                                      is_custom_algorithm_list=True,
                    #                                      algorithm_list=[test_case.algorithm])

                    # pso_supervised = \
                    #     pso_algorithm.PSO(particle_count=num_particles, is_semisupervised=False,
                    #                       distance_between_initial_particles=0.7, evaluation_metric=roc_auc_score)

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
                        best_model = test_case.algorithm.fit(x_train, y_train)
                        y_pred = best_model.predict(x_test)
                        roc_auc = round(roc_auc_score(
                            y_test, y_pred), ndigits=4)
                        prn = round(precision_n_scores(
                            y_test, y_pred), ndigits=4)

                        if should_compare_models is False:
                            print(
                                f"Best roc_auc: {roc_auc} - " +
                                f"Best precision_n_scores: {prn}")
                            pprint(vars(best_model))

                            res_dict = {
                                "algorithm_name": test_case.algorithm.__class__.__name__,
                                "dataset": test_case.dataset_name,
                                "anomaly_type": test_case.anomaly_type,
                                "noise_type": test_case.noise_type,
                                "labeled_anomaly_ratio": test_case.labeled_anomaly_ratio,
                                "noise_ratio": test_case.noise_ratio,
                                "apply_data_scaling": test_case.apply_data_scaling,
                                "scaler": test_case.scaler,
                                "apply_data_rebalancing": test_case.apply_data_rebalancing,
                                "rebalance_pipeline": test_case.rebalance_pipeline.get_params,
                                "apply_missing_data_filling": test_case.apply_missing_data_filling,
                                "fill_algorithm": test_case.fill_algorithm,
                                "apply_dimensionality_reduction": test_case.apply_dimensionality_reduction,
                                "dimensinality_reduction_algorithm":
                                    test_case.dimensinality_reduction_algorithm.__class__.__name__,
                                "roc_auc": roc_auc,
                                "precision_n_scores": prn,
                                "exception": ""
                            }

                            df = pd.DataFrame(res_dict, index=[index])

                            with open(file_name, 'a') as f_object:
                                dictwriter_object = DictWriter(
                                    f_object, fieldnames=field_names)
                                dictwriter_object.writerow(res_dict)
                                f_object.close()

                            test_case_index += 1
                            print(
                                f"!!!!!!!!!! Test {test_case_index} of {test_case_count} !!!!!!!!!!!!!!!!")

                            if index == 0:
                                df_for_csv = df.copy()
                            else:
                                df_for_csv = pd.concat([df_for_csv, df])
                        else:
                            print(
                                "################## START ########################")

                            print(" !!!!!!!! GLOBAL BEST !!!!!!!!!!")
                            print(
                                f"Best roc_auc: {roc_auc} - " +
                                f"Best precision_n_scores: {prn}")
                            pprint(vars(best_model))
                            print(" !!!!!!!! GLOBAL BEST !!!!!!!!!!")

                            res_dict = {
                                "algorithm_name": test_case.algorithm.__class__.__name__,
                                "dataset": test_case.dataset_name,
                                "anomaly_type": test_case.anomaly_type,
                                "noise_type": test_case.noise_type,
                                "labeled_anomaly_ratio": test_case.labeled_anomaly_ratio,
                                "noise_ratio": test_case.noise_ratio,
                                "apply_data_scaling": test_case.apply_data_scaling,
                                "scaler": test_case.scaler,
                                "apply_data_rebalancing": test_case.apply_data_rebalancing,
                                "rebalance_pipeline": test_case.rebalance_pipeline.get_params,
                                "apply_missing_data_filling": test_case.apply_missing_data_filling,
                                "fill_algorithm": test_case.fill_algorithm,
                                "apply_dimensionality_reduction": test_case.apply_dimensionality_reduction,
                                "dimensinality_reduction_algorithm":
                                    test_case.dimensinality_reduction_algorithm.__class__.__name__,
                                "roc_auc": roc_auc,
                                "precision_n_scores": prn,
                                "exception": ""
                            }

                            df = pd.DataFrame(res_dict, index=[index])

                            with open(file_name, 'a') as f_object:
                                dictwriter_object = DictWriter(
                                    f_object, fieldnames=field_names)
                                dictwriter_object.writerow(res_dict)
                                f_object.close()

                            test_case_index += 1
                            print(
                                f"!!!!!!!!!! Test {test_case_index} of {test_case_count} !!!!!!!!!!!!!!!!")

                            if index == 0:
                                df_for_csv = df.copy()
                            else:
                                df_for_csv = pd.concat([df_for_csv, df])

                            print(
                                "################### END #########################")
                    except Exception as e:
                        print(str(e))
                        res_dict = {
                            "algorithm_name": test_case.algorithm.__class__.__name__,
                            "dataset": test_case.dataset_name,
                            "anomaly_type": test_case.anomaly_type,
                            "noise_type": test_case.noise_type,
                            "labeled_anomaly_ratio": test_case.labeled_anomaly_ratio,
                            "noise_ratio": test_case.noise_ratio,
                            "apply_data_scaling": test_case.apply_data_scaling,
                            "scaler": test_case.scaler,
                            "apply_data_rebalancing": test_case.apply_data_rebalancing,
                            "rebalance_pipeline": test_case.rebalance_pipeline.get_params,
                            "apply_missing_data_filling": test_case.apply_missing_data_filling,
                            "fill_algorithm": test_case.fill_algorithm,
                            "apply_dimensionality_reduction": test_case.apply_dimensionality_reduction,
                            "dimensinality_reduction_algorithm":
                                test_case.dimensinality_reduction_algorithm.__class__.__name__,
                            "roc_auc": 0,
                            "precision_n_scores": 0,
                            "exception": str(e)
                        }

                        df = pd.DataFrame(res_dict, index=[index])

                        with open(file_name, 'a') as f_object:
                            dictwriter_object = DictWriter(
                                f_object, fieldnames=field_names)
                            dictwriter_object.writerow(res_dict)
                            f_object.close()

                        test_case_index += 1
                        print(
                            f"!!!!!!!!!! Test {test_case_index} of {test_case_count} !!!!!!!!!!!!!!!!")

                        if index == 0:
                            df_for_csv = df.copy()
                        else:
                            df_for_csv = pd.concat([df_for_csv, df])

                    print("--- %s seconds ---" % (time.time() - start_time))

                    index += 1


if __name__ == "__main__":
    main()
