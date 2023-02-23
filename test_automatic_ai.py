# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 11:30:29 2019

@author: czzo
"""
import time

from sklearn.preprocessing import StandardScaler
from AutomaticAI import ParticleSwarmOptimization as pso_algorithm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
# from sklearn import datasets

from autoad.data_generators.anomaly_data_generator import AnomalyDataGenerator
# import sys
# sys.path.append('C:/University/Git/AutomaticAI_Flask')


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

    num_particles = 2
    num_iterations = 2

    pso = pso_algorithm.PSO(particle_count=num_particles, is_semisupervised=True,
                            distance_between_initial_particles=0.7, evaluation_metric=roc_auc_score)

    start_time = time.time()

    best_results = pso.fit(X_train=x_train,
                           X_test=x_test,
                           Y_train=y_train,
                           Y_test=y_test,
                           maxiter=num_iterations,
                           verbose=True,
                           compare_models=True,
                           max_distance=0.05,
                           agents=1)

    print("--- %s seconds ---" % (time.time() - start_time))

    print("##########################################")

    for key, value in best_results.items():
        print(
            f'{key} -- {value.metric_i} -- {value.model_i} -- Training time: {value.training_time}')

    print("##########################################")


if __name__ == "__main__":
    main()
