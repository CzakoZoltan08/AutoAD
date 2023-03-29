# Import libraries
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd

import os

import AutomaticAI.LocalOutlierFactorAlgorithmFactory as lofaf

import AutomaticAI.IsolationForestAlgorithmFactory as ifaf
import AutomaticAI.ClusterBasedLocalOutlierFactorAlgorithmFactory as coblofaf
import AutomaticAI.HistogramBasedOutlierDetectionAlgorithmFactory as hbodaf
import AutomaticAI.SemiSupervisedKNNAlgorithmFactory as ssknnaf
import AutomaticAI.LightweightOnlineDetectorAlgorithmFactory as lodaaf
import AutomaticAI.PCAAnomalyDetectorAlgorithmFactory as pcaadaf
import AutomaticAI.FeatureBaggingOutlierDetectionAlgorithmFactory as fbodaf
import AutomaticAI.LocallySelectiveCombinationAlgorithmFactory as lscaf
import AutomaticAI.InneAnomalyDetectorAlgorithmFactory as inneaf
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

from autoad.algorithms.deep_sad.deepsad import DeepSAD
from autoad.algorithms.ganomaly.ganomaly import GANomaly
from autoad.algorithms.prenet.prenet import PReNet
from autoad.algorithms.feawad.feawad import FEAWAD

from autoad.algorithms.abod import AngleBaseOutlierDetection
from autoad.algorithms.lmdd import LMDDAnomalyDetector
from autoad.algorithms.lstmod import LSTMOutlierDetector
from autoad.algorithms.mcd import MinimumCovarianceDeterminant
from autoad.algorithms.ocsvm import OneClassSVM
from autoad.algorithms.sod import SubspaceOutlierDetection
from autoad.algorithms.sogaal import SingleObjectiveGenerativeAdversarialActiveLearning
from autoad.algorithms.vae import VariationalAutoEncoder


anomaly_detection_unsupervised_algorithms = [
    lofaf.get_algorithm(),
    coblofaf.get_algorithm(),
    ifaf.get_algorithm(),
    ssknnaf.get_algorithm(),
    hbodaf.get_algorithm(),
    lodaaf.get_algorithm(),
    pcaadaf.get_algorithm(),
    fbodaf.get_algorithm(),
    lscaf.get_algorithm(),
    inneaf.get_algorithm(),
    kdeaf.get_algorithm(),
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
]

semisupervised_algorithms = [
    GANomaly(),
    DeepSAD(),
    # REPEN(),
    PReNet(),
    FEAWAD(),
]

anomaly_detection_unsupervised_algorithms_no_opt = [
    AngleBaseOutlierDetection(),  # test without hyperparameter optimization
    # test without hyperparameter optimization
    # MultiObjectiveGenerativeAdversarialActiveLearning(),
    OneClassSVM(),  # test without hyperparameter optimization
    SubspaceOutlierDetection(),  # test without hyperparameter optimization
    # test without hyperparameter optimization
    SingleObjectiveGenerativeAdversarialActiveLearning(),
    VariationalAutoEncoder(),  # test without hyperparameter optimization
    MinimumCovarianceDeterminant(),  # test without hyperparameter optimization
    LMDDAnomalyDetector(),  # test without hyperparameter optimization
    # LSTMOutlierDetector(),  # test without hyperparameter optimization
]


def box_plot(diagram_folder_path, df):
    # algorithm_names = [
    #     name.algorithm_name for name in anomaly_detection_unsupervised_algorithms]
    # algorithm_names = [
    #     name.algorithm_name for name in supervised_algorithms]
    algorithm_names = [
        algorithm.__class__.__name__ for algorithm in semisupervised_algorithms]

    # algorithm_names_opt = [
    #     name.algorithm_name for name in anomaly_detection_unsupervised_algorithms]

    # algorithm_names_no_opt = [
    #     algorithm.__class__.__name__ for algorithm in anomaly_detection_unsupervised_algorithms_no_opt]

    # algorithm_names = algorithm_names_opt + algorithm_names_no_opt

    roc_auc_list = []
    for algorithm_name in algorithm_names:
        noise_df = df[df['noise_ratio'] == 0]
        # anomaly_type_df = noise_df[noise_df['anomaly_type']
        #                            == 'AnomalyType.CLUSTER']
        filtered_df = noise_df[noise_df['algorithm_name']
                               == algorithm_name]
        lof_wine_local_roc_auc = filtered_df['roc_auc'].values.tolist()
        roc_auc_list.append(lof_wine_local_roc_auc)

    # Creating plot
    fig = plt.figure(figsize=(30, 15))

    ax = fig.add_subplot(111)
    ax.boxplot(roc_auc_list, patch_artist=True, vert=0)
    ax.set_title('Semisupervised Algorithms')
    ax.set_yticklabels(algorithm_names)
    ax.set_xlabel('ROCAUC')

    # show plot
    plt.show()

    anomaly_types = "AnomalyTypesAll"
    noise_types = "NoiseTypesNone"
    algoritm_type = "Semisupervised"
    dn = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    img_file_name = f"{dn}_{algoritm_type}_{anomaly_types}_{noise_types}.png"
    img_name = os.path.join(diagram_folder_path, img_file_name)
    fig.savefig(img_name, bbox_inches='tight')


def line_chart(diagram_folder_path, df):
    # algorithm_names = [
    #     name.algorithm_name for name in anomaly_detection_unsupervised_algorithms]
    # algorithm_names = [
    #     name.algorithm_name for name in supervised_algorithms]
    # algorithm_names = [
    #     algorithm.__class__.__name__ for algorithm in semisupervised_algorithms]

    algorithm_names_opt = [
        name.algorithm_name for name in anomaly_detection_unsupervised_algorithms]

    algorithm_names_no_opt = [
        algorithm.__class__.__name__ for algorithm in anomaly_detection_unsupervised_algorithms_no_opt]

    algorithm_names = algorithm_names_opt + algorithm_names_no_opt

    roc_auc_list = []
    steps = [x / 100.0 for x in range(0, 100, 10)]

    fig = plt.figure(figsize=(20, 10))
    ax = plt.subplot(111)

    for algorithm_name in algorithm_names:
        algorithm_df = df[df['algorithm_name']
                          == algorithm_name]
        noise_df = algorithm_df[algorithm_df['noise_type']
                                == 'NoiseType.IRRELEVANT_FEATURES']
        # anomaly_type_df = noise_df[noise_df['anomaly_type']
        #                            == 'AnomalyType.LOCAL']
        filtered_df = noise_df[noise_df['algorithm_name']
                               == algorithm_name]
        mean_roc_by_datasets = filtered_df.groupby(['noise_ratio'])[
            'roc_auc'].mean()
        roc_auc = mean_roc_by_datasets.values.tolist()
        roc_auc_list.append(roc_auc)

        ax.plot(steps, roc_auc[:len(steps)], label=algorithm_name)

    ax.set_xlabel('Label Ratio')
    ax.set_title('Irrelevant Features')
    # Set the y axis label of the current axis.
    ax.set_ylabel('ROCAUC Score')

    ax.set_xticks(steps)

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # show plot
    plt.show()

    anomaly_types = "AnomalyTypesAll"
    noise_types = "NoiseTypesIrrelevantFeatures"
    algoritm_type = "Unsupervised"
    dn = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    img_file_name = f"{dn}_linechart_{algoritm_type}_{anomaly_types}_{noise_types}.png"
    img_name = os.path.join(diagram_folder_path, img_file_name)
    fig.savefig(img_name, bbox_inches='tight')


def bar_chart(diagram_folder_path, df):
    # algorithm_names = [
    #     name.algorithm_name for name in anomaly_detection_unsupervised_algorithms]
    # algorithm_names = [
    #     name.algorithm_name for name in supervised_algorithms]
    algorithm_names = [
        algorithm.__class__.__name__ for algorithm in semisupervised_algorithms]

    # algorithm_names_opt = [
    #     name.algorithm_name for name in anomaly_detection_unsupervised_algorithms]

    # algorithm_names_no_opt = [
    #     algorithm.__class__.__name__ for algorithm in anomaly_detection_unsupervised_algorithms_no_opt]

    # algorithm_names = algorithm_names_opt + algorithm_names_no_opt

    roc_auc_list = []

    fig = plt.figure(figsize=(20, 10))
    ax = plt.subplot(111)

    for algorithm_name in algorithm_names:
        algorithm_df = df[df['algorithm_name']
                          == algorithm_name]
        noise_df = algorithm_df[algorithm_df['noise_ratio'] == 0]
        anomaly_type_df = noise_df[noise_df['anomaly_type']
                                   == 'AnomalyType.CLUSTER']
        filtered_df = anomaly_type_df[anomaly_type_df['algorithm_name']
                                      == algorithm_name]
        mean_roc = filtered_df['roc_auc'].mean()
        roc_auc_list.append(mean_roc)

    ax.bar(algorithm_names, roc_auc_list)

    ax.set_xlabel('Algorithms')
    ax.set_title('Cluster Anomalies')
    ax.set_xticklabels(algorithm_names, rotation=45, ha='right')
    # Set the y axis label of the current axis.
    ax.set_ylabel('ROCAUC')

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # show plot
    plt.show()

    anomaly_types = "AnomalyTypesCluster"
    noise_types = "NoiseTypesNone"
    algoritm_type = "Semisupervised"
    dn = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    img_file_name = f"{dn}_linechart_{algoritm_type}_{anomaly_types}_{noise_types}.png"
    img_name = os.path.join(diagram_folder_path, img_file_name)
    fig.savefig(img_name, bbox_inches='tight')


def main():
    folder_path = "./algorithm_evaluation_results/Unsupervised/"
    diagram_folder_path = "./algorithm_evaluation_results/Diagrams/Unsupervised/"
    _, _, files = next(os.walk(folder_path))
    file_count = len(files)
    dataframes_list = []

    for i in range(file_count):
        temp_df = pd.read_csv(os.path.join(folder_path, files[i]))
        dataframes_list.append(temp_df)

    df_res = pd.concat(dataframes_list)

    plt.rcParams['font.size'] = 24

    # box_plot(diagram_folder_path, df_res)
    line_chart(diagram_folder_path, df_res)
    # bar_chart(diagram_folder_path, df_res)

    print(df_res)


if __name__ == "__main__":
    main()
