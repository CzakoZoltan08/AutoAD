# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 11:30:29 2019

@author: czzo
"""
import time
from pprint import pprint
from datetime import datetime

import pandas as pd

from sklearn.preprocessing import RobustScaler, StandardScaler, MaxAbsScaler, \
    MinMaxScaler, QuantileTransformer, PowerTransformer, KernelCenterer, Normalizer, Binarizer
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, f1_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.base import TransformerMixin
from sklearn.decomposition import PCA

import AutomaticAI.ParticleSwarmOptimization as pso_algorithm
import AutomaticAI.IsolationForestAlgorithmFactory as ifaf
# import AutomaticAI.SemiSupervisedKNNAlgorithmFactory as ssknnaf

from autoad.data_generators.anomaly_data_generator import AnomalyDataGenerator
from autoad.data_generators.anomaly_type import AnomalyType
from autoad.data_generators.dataset import Dataset
from autoad.data_generators.noise_type import NoiseType

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as impipe

import AutomaticAI.AngleBaseOutlierDetectionAlgorithmFactory as abodaf
import AutomaticAI.ClusterBasedLocalOutlierFactorAlgorithmFactory as coblofaf
import AutomaticAI.HistogramBasedOutlierDetectionAlgorithmFactory as hbodaf
import AutomaticAI.SemiSupervisedKNNAlgorithmFactory as ssknnaf
import AutomaticAI.LightweightOnlineDetectorAlgorithmFactory as lodaaf
import AutomaticAI.LocalOutlierFactorAlgorithmFactory as lofaf
import AutomaticAI.LSTMOutlierDetectorAlgorithmFactory as lstmdaaf
import AutomaticAI.MultiObjectiveGenerativeAdversarialActiveLearningAlgorithmFactory as mogaalaf
import AutomaticAI.OneClassSVMAlgorithmFactory as ocsvmaf
import AutomaticAI.PCAAnomalyDetectorAlgorithmFactory as pcaadaf
import AutomaticAI.SubspaceOutlierDetectionAlgorithmFactory as sodaf
import AutomaticAI.SingleObjectiveGenerativeAdversarialActiveLearningAlgorithmFactory as sogaalaf
import AutomaticAI.VariationalAutoEncoderAlgorithmFactory as vaeaf
import AutomaticAI.FeatureBaggingOutlierDetectionAlgorithmFactory as fbodaf
import AutomaticAI.MinimumCovarianceDeterminantAlgorithmFactory as mcdaf
import AutomaticAI.LocallySelectiveCombinationAlgorithmFactory as lscaf
import AutomaticAI.InneAnomalyDetectorAlgorithmFactory as inneaf
import AutomaticAI.LMDDAnomalyDetectorAlgorithmFactory as lmddadaf
import AutomaticAI.KDEAnomalyDetectorAlgorithmFactory as kdeaf

import AutomaticAI.ExtraTreesClassifierAlgorithmFactory as etscaf
import AutomaticAI.SGDClassifierAlgorithmFactory as sgdcaf
import AutomaticAI.PassiveAgressiveClassifierAlgorithmFactory as pacaf
import AutomaticAI.DecisionTreeClassifierAlgorithmFactory as dtcaf
import AutomaticAI.ExtraTreeClassifierAlgorithmFactory as etcaf
import AutomaticAI.RandomForestAlgorithmFactory as rfaf
import AutomaticAI.KnnAlgorithmFactory as kaf
import AutomaticAI.RidgeClassifierAlgorithmFactory as rcaf
import AutomaticAI.GradientBoostingClassifierAlgorithmFactory as gbcaf
import AutomaticAI.XGBoostClassifierAlgorithmFactory as xgbcaf


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
    noise_ratios = [x / 100.0 for x in range(0, 100, 10)]
    noise_ratios.append(1.0)

    # labeled_anomaly_ratios = [x / 100.0 for x in range(10, 100, 10)]
    # labeled_anomaly_ratios.append(1.0)
    labeled_anomaly_ratios = [1.0]

    noise_types = [NoiseType.NONE, NoiseType.LABEL_ERROR,
                   NoiseType.DUPLICATES, NoiseType.IRRELEVANT_FEATURES]
    anomaly_types = [AnomalyType.LOCAL,
                     AnomalyType.GLOBAL, AnomalyType.CLUSTER]

    dataset_names = [x.value for x in Dataset]
    # dataset_names = dataset_names[-1:]

    # truth_values = [False, True]

    # scalers = []

    # scalers = [StandardScaler(), MinMaxScaler(),
    #            RobustScaler(), MaxAbsScaler(),
    #            QuantileTransformer(), PowerTransformer(),
    #            KernelCenterer(), Normalizer(), Binarizer()]

    # over_sampling_ratios = [x / 10.0 for x in range(1, 6, 1)]
    # under_sampling_ratios = [x / 10.0 for x in range(1, 6, 1)]

    # over_sampling_algorithms = [SMOTE(sampling_strategy=x)
    #                             for x in over_sampling_ratios]
    # under_sampling_algorithms = [RandomUnderSampler(
    #     sampling_strategy=x) for x in under_sampling_ratios]

    # under_sampling_algorithms = []

    # fill_algorithms = ['Zero', 'Mean', 'Median']
    # fill_algorithms = []

    # dimensinality_reduction_algorithms = []

    # dimensinality_reduction_algorithms = [PCA(n_components=x)
    #                                       for x in range(2, 100, 1)]

    anomaly_detection_unsupervised_algorithms = [
        lstmdaaf.get_algorithm(),
        # lofaf.get_algorithm(),
        # coblofaf.get_algorithm(),
        # ifaf.get_algorithm(),
        # ssknnaf.get_algorithm(),
        # abodaf.get_algorithm(),
        # hbodaf.get_algorithm(),
        # lodaaf.get_algorithm(),
        # mogaalaf.get_algorithm(),
        # ocsvmaf.get_algorithm(),
        # pcaadaf.get_algorithm(),
        # sodaf.get_algorithm(),
        # sogaalaf.get_algorithm(),
        # vaeaf.get_algorithm(),
        # fbodaf.get_algorithm(),
        # mcdaf.get_algorithm(),
        # lscaf.get_algorithm(),
        # inneaf.get_algorithm(),
        # lmddadaf.get_algorithm(),
        # kdeaf.get_algorithm(),
    ]

    supervised_algorithms = [
        kaf.get_algorithm(),
        rfaf.get_algorithm(),
        etcaf.get_algorithm(),
        dtcaf.get_algorithm(),
        rcaf.get_algorithm(),
        pacaf.get_algorithm(),
        gbcaf.get_algorithm(),
        sgdcaf.get_algorithm(),
        etscaf.get_algorithm(),
        xgbcaf.get_algorithm(),
    ]

    for dataset_name in dataset_names:
        for algorithm in anomaly_detection_unsupervised_algorithms:

            all_test_cases = []
            test_case_index = 1
            for anomaly_type in anomaly_types:
                for labeled_anomaly_ratio in labeled_anomaly_ratios:
                    for noise_type in noise_types:
                        for noise_ratio in noise_ratios:
                            # for scaler in scalers:
                            #     for under_sampling_algorithm in under_sampling_algorithms:
                            #         for dimensinality_reduction_algorithm in dimensinality_reduction_algorithms:
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
                                        [('u', RandomUnderSampler(sampling_strategy=0.3))]),
                                    apply_missing_data_filling=False,
                                    fill_algorithm='Zero',
                                    apply_dimensionality_reduction=False,
                                    dimensinality_reduction_algorithm=PCA(
                                        n_components=5),
                                    algorithm=algorithm
                                )
                            )
                            print(
                                f"Test Case: {test_case_index}")
                            test_case_index += 1

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
                        threshold=5000
                    )

                    x_train = dataset['X_train']
                    y_train = dataset['y_train'].ravel()
                    x_test = dataset['X_test']
                    y_test = dataset['y_test'].ravel()

                    num_particles = 3
                    num_iterations = 5

                    evaluation_function = roc_auc_score

                    test_type = 'Unsupervised_algorithms'

                    pso_unsupervised = pso_algorithm.PSO(particle_count=num_particles,
                                                         is_semisupervised=True,
                                                         distance_between_initial_particles=0.7,
                                                         evaluation_metric=evaluation_function,
                                                         is_custom_algorithm_list=True,
                                                         algorithm_list=[test_case.algorithm])

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
                        best_model, best_results_semi_supervised = pso_unsupervised.fit(X_train=x_train,
                                                                                        X_test=x_test,
                                                                                        Y_train=y_train,
                                                                                        Y_test=y_test,
                                                                                        maxiter=num_iterations,
                                                                                        verbose=True,
                                                                                        compare_models=should_compare_models,
                                                                                        max_distance=0.05,
                                                                                        agents=1)
                        y_pred = best_model.model_best_i.predict(x_test)
                        roc_auc = round(roc_auc_score(
                            y_test, y_pred), ndigits=4)
                        accuracy = round(accuracy_score(
                            y_test, y_pred), ndigits=4)
                        recall = round(recall_score(y_test, y_pred), ndigits=4)
                        f1 = round(f1_score(y_test, y_pred), ndigits=4)
                        precision = round(precision_score(
                            y_test, y_pred), ndigits=4)

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
                                "scaler": test_case.scaler,
                                "apply_data_rebalancing": test_case.apply_data_rebalancing,
                                "rebalance_pipeline": test_case.rebalance_pipeline.get_params,
                                "apply_missing_data_filling": test_case.apply_missing_data_filling,
                                "fill_algorithm": test_case.fill_algorithm,
                                "apply_dimensionality_reduction": test_case.apply_dimensionality_reduction,
                                "dimensinality_reduction_algorithm": test_case.dimensinality_reduction_algorithm.__class__.__name__,
                                "roc_auc": roc_auc,
                                "recall": recall,
                                "precision": precision,
                                "accuracy": accuracy,
                                "f1-score": f1,
                                "exception": ""
                            }

                            df = pd.DataFrame(res_dict, index=[index])

                            if index == 0:
                                df_for_csv = df.copy()
                            else:
                                df_for_csv = pd.concat([df_for_csv, df])
                        else:
                            best_results = {**best_results_semi_supervised}

                            print(
                                "################## START ########################")

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
                                "scaler": test_case.scaler,
                                "apply_data_rebalancing": test_case.apply_data_rebalancing,
                                "rebalance_pipeline": test_case.rebalance_pipeline.get_params,
                                "apply_missing_data_filling": test_case.apply_missing_data_filling,
                                "fill_algorithm": test_case.fill_algorithm,
                                "apply_dimensionality_reduction": test_case.apply_dimensionality_reduction,
                                "dimensinality_reduction_algorithm": test_case.dimensinality_reduction_algorithm.__class__.__name__,
                                "roc_auc": roc_auc,
                                "recall": recall,
                                "precision": precision,
                                "accuracy": accuracy,
                                "f1-score": f1,
                                "exception": ""
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

                            print(
                                "################### END #########################")
                    except Exception as e:
                        print(str(e))
                        res_dict = {
                            "algorithm_name": test_case.algorithm.algorithm_name,
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
                            "dimensinality_reduction_algorithm": test_case.dimensinality_reduction_algorithm.__class__.__name__,
                            "roc_auc": 0,
                            "recall": 0,
                            "precision": 0,
                            "accuracy": 0,
                            "f1-score": 0,
                            "exception": str(e)
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
                    f"./algorithm_evaluation_results/{dn}_{test_type}_" +
                    f"{test_case.dataset_name}_{test_case.anomaly_type}_{test_case.algorithm.algorithm_name}.csv")


if __name__ == "__main__":
    main()
