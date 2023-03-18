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


def main():
    folder_path = "./algorithm_evaluation_results/Unsupervised/"
    diagram_folder_path = "./algorithm_evaluation_results/Diagrams/"
    _, _, files = next(os.walk(folder_path))
    file_count = len(files)
    dataframes_list = []

    for i in range(file_count):
        temp_df = pd.read_csv(os.path.join(folder_path, files[i]))
        dataframes_list.append(temp_df)

    df_res = pd.concat(dataframes_list)

    algorithm_names = [
        name.algorithm_name for name in anomaly_detection_unsupervised_algorithms]

    roc_auc_list = []
    for algorithm_name in algorithm_names:
        wine_df = df_res[df_res['noise_type'] != 'NoiseType.LABEL_ERROR']
        filtered_df = wine_df[wine_df['algorithm_name'] == algorithm_name]
        lof_wine_local_roc_auc = filtered_df['roc_auc'].values.tolist()
        roc_auc_list.append(lof_wine_local_roc_auc)

    # Creating plot
    fig = plt.figure(figsize=(30, 15))

    ax = fig.add_subplot(111)
    ax.boxplot(roc_auc_list, patch_artist=True, vert=0)
    ax.set_title('Unsupervised Algorithms')
    ax.set_yticklabels(algorithm_names)
    ax.set_xlabel('ROC AUC')

    # show plot
    # plt.show()

    anomaly_types = "AllAnomalyTypes"
    noise_types = "AllNoiseTypes"
    algoritm_type = "Unsupervised"
    dn = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    img_file_name = f"{dn}_{algoritm_type}_{anomaly_types}_{noise_types}.png"
    img_name = os.path.join(diagram_folder_path, img_file_name)
    fig.savefig(img_name, bbox_inches='tight')

    print(df_res)


if __name__ == "__main__":
    main()
