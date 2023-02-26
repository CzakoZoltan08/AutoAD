# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 12:44:59 2019

@author: czzo
"""

from AutomaticAI.Particle import Particle
from AutomaticAI.Utils import calculate_min_distance
from AutomaticAI.Utils import generate_initial_best_positions
from AutomaticAI.Utils import create_all_supported_algorithm_list
from AutomaticAI.Utils import create_all_supported_adaptive_algorithm_list
from AutomaticAI.Utils import create_all_supported_regresion_algorithm_list
from AutomaticAI.Utils import create_all_supported_semisupervised_algorithm_list
from AutomaticAI.Utils import create_custom_algorithm_list
from AutomaticAI.Utils import generate_initial_particle_positions
from AutomaticAI.Utils import generate_initial_particle_positions_for_adaptive_classification
from AutomaticAI.Utils import generate_initial_particle_positions_for_regression
from AutomaticAI.Utils import generate_initial_particle_positions_for_semisupervised_anomaly_detection
from AutomaticAI.Utils import generate_initial_particle_positions_for_custom_anomaly_detection

from AutomaticAI.Utils import get_classification_algorithm_mapping
from AutomaticAI.Utils import get_regression_algorithm_mapping
from AutomaticAI.Utils import get_adaptive_classification_algorithm_mapping
from AutomaticAI.Utils import get_semisupervised_algorithm_mapping
from AutomaticAI.Utils import get_custom_algorithm_mapping

from AutomaticAI.Utils import train_algorithm
from AutomaticAI.Utils import train_regression_algorithm
from AutomaticAI.Utils import traint_adaptive_algorithm
from AutomaticAI.Utils import train_semisupervised_algorithm
from AutomaticAI.Utils import evaluate_particle

from sklearn.metrics import f1_score

import numpy as np
import copy

import multiprocessing
from multiprocessing import cpu_count
from functools import partial


class PSO():
    def __init__(self,
                 particle_count,
                 distance_between_initial_particles=0.7,
                 is_classification=True,
                 evaluation_metric=f1_score,
                 is_maximization=True,
                 is_adaptive=False,
                 is_semisupervised=False,
                 is_custom_algorithm_list=False,
                 algorithm_list=[]):

        # establish the swarm
        self.current_epoch = 0
        self.swarm = []
        self.particle_count = particle_count

        self.evaluation_metric = evaluation_metric

        self.distance_between_initial_particles = distance_between_initial_particles

        self.cost_function = train_algorithm
        self.algorithm_mapping = get_classification_algorithm_mapping()
        algorithms = create_all_supported_algorithm_list(particle_count)
        self.is_unsupervised = False
        self.is_semisupervised = False

        initial_positions = []
        initial_positions = generate_initial_particle_positions(
            num_particles=particle_count,
            distance_between_initial_particles=distance_between_initial_particles)

        if is_custom_algorithm_list is True:
            self.cost_function = train_semisupervised_algorithm
            self.algorithm_mapping = None
            self.algorithm_mapping = get_custom_algorithm_mapping(
                algorithm_list)
            self.is_semisupervised = True
            algorithms = create_custom_algorithm_list(
                algorithm_list,
                particle_count)
            initial_positions = generate_initial_particle_positions_for_custom_anomaly_detection(
                algorithm_list,
                num_particles=particle_count,
                distance_between_initial_particles=distance_between_initial_particles)
        else:
            if is_classification is False:
                self.cost_function = train_regression_algorithm
                self.algorithm_mapping = None
                self.algorithm_mapping = get_regression_algorithm_mapping()
                algorithms = create_all_supported_regresion_algorithm_list(
                    particle_count)
                initial_positions = generate_initial_particle_positions_for_regression(
                    num_particles=particle_count,
                    distance_between_initial_particles=distance_between_initial_particles)

            if is_adaptive is True:
                self.cost_function = traint_adaptive_algorithm
                self.algorithm_mapping = None
                self.algorithm_mapping = get_adaptive_classification_algorithm_mapping()
                algorithms = create_all_supported_adaptive_algorithm_list(
                    particle_count)
                initial_positions = generate_initial_particle_positions_for_adaptive_classification(
                    num_particles=particle_count,
                    distance_between_initial_particles=distance_between_initial_particles)

            if is_semisupervised is True:
                self.cost_function = train_semisupervised_algorithm
                self.algorithm_mapping = None
                self.algorithm_mapping = get_semisupervised_algorithm_mapping()
                self.is_semisupervised = True
                algorithms = create_all_supported_semisupervised_algorithm_list(
                    particle_count)
                initial_positions = generate_initial_particle_positions_for_semisupervised_anomaly_detection(

                    num_particles=particle_count,
                    distance_between_initial_particles=distance_between_initial_particles)

        self.num_particles = len(algorithms)

        self.isMaximization = is_maximization
        self.initial_best_positions = generate_initial_best_positions(
            algorithms)

        for i in range(0, self.num_particles):
            self.swarm.append(Particle(algorithm=algorithms[i],
                                       hyper_parameter_list=initial_positions[i],
                                       evaluation_metric=evaluation_metric))

    def calculate_max_distance_between_particles(self):
        max_diff = abs(self.swarm[1].metric_best_i -
                       self.swarm[0].metric_best_i)
        min_element = self.swarm[0].metric_best_i

        arr_size = len(self.swarm)

        for i in range(1, arr_size):
            if (abs(self.swarm[i].metric_best_i - min_element) > max_diff):
                max_diff = abs(self.swarm[i].metric_best_i - min_element)

            if (self.swarm[i].metric_best_i < min_element):
                min_element = self.swarm[i].metric_best_i
        return max_diff

    def _remove_worst(self, verbose=False):
        if self.isMaximization:
            (m, i) = min((v.metric_best_i, i)
                         for i, v in enumerate(self.swarm))
        else:
            (m, i) = max((v.metric_best_i, i)
                         for i, v in enumerate(self.swarm))

        if verbose:
            print("\n* Particle {} Removed --- Algorithm Type: {} With Metric {} *".format(
                i, self.swarm[i].algorithm.algorithm_name, self.swarm[i].metric_best_i))

        self.swarm.pop(i)

    def _remove_worst_by_algorithm_name(self, algorithm_name, verbose=False):
        particles = [
            x for x in self.swarm if x.algorithm.algorithm_name == algorithm_name]
        if self.isMaximization:
            (m, i) = min((v.metric_best_i, i)
                         for i, v in enumerate(particles))
        else:
            (m, i) = max((v.metric_best_i, i)
                         for i, v in enumerate(particles))

        if verbose:
            print("\n* Particle {} Removed --- Algorithm Type: {} With Metric {} *".format(
                i, self.swarm[i].algorithm.algorithm_name, self.swarm[i].metric_best_i))

        algorithm_index = self.swarm.index(particles[i])

        self.swarm.pop(algorithm_index)

    def _add_to_best(self, verbose=False):
        best_particle = self._get_best_particle()

        algorithm_type = best_particle.algorithm.algorithm_type

        particles = self._get_particles_by_algorithm_type(algorithm_type)

        current_hyper_parameters = self._generate_list_of_hyper_parameters(
            particles)

        if verbose:
            print("\n* Particle Added -- Algorithm Type {} *".format(
                best_particle.algorithm.algorithm_name))

        return self._generate_new_particle(best_particle, current_hyper_parameters)

    def _add_by_algorithm_name(self, algorithm_name, verbose=False):
        particles = self._get_particles_by_algorithm_name(algorithm_name)

        current_hyper_parameters = self._generate_list_of_hyper_parameters(
            particles)

        if verbose:
            print("\n* Particle Added -- Algorithm Type {} *".format(algorithm_name))

        return self._generate_new_particle(particles[0], current_hyper_parameters)

    def _distinct_algorithm_names(self):
        return {x.algorithm.algorithm_name: x for x in self.swarm}.values()

    def _generate_new_particle(self, best_particle, current_hyper_parameters):
        dimensions = best_particle.algorithm.get_dimensions()
        distance_between_initial_particles = self.distance_between_initial_particles
        minimum_distance = 0
        while minimum_distance < distance_between_initial_particles:
            hyper_parameter_list = []
            for j in range(dimensions):
                min_bound = best_particle.algorithm.bounds[j][0]
                max_bound = best_particle.algorithm.bounds[j][1]
                parameter_value = np.random.uniform(min_bound, max_bound)
                hyper_parameter_list.append(parameter_value)
            minimum_distance = calculate_min_distance(
                current_hyper_parameters, hyper_parameter_list)

        new_algorithm = copy.deepcopy(best_particle.algorithm)

        return Particle(
            new_algorithm,
            hyper_parameter_list=hyper_parameter_list,
            evaluation_metric=self.evaluation_metric
        )

    def _get_best_particle(self):
        if self.isMaximization:
            (m, i) = max((v.metric_best_i, i)
                         for i, v in enumerate(self.swarm))
        else:
            (m, i) = min((v.metric_best_i, i)
                         for i, v in enumerate(self.swarm))
        return self.swarm[i]

    def _get_particles_by_algorithm_type(self, algorithm_type):
        return [particle for particle in self.swarm if particle.algorithm.algorithm_type == algorithm_type]

    def _get_particles_by_algorithm_name(self, algorithm_name):
        return [particle for particle in self.swarm if particle.algorithm.algorithm_name == algorithm_name]

    def _generate_list_of_hyper_parameters(self, particles):
        current_hyper_parameters = []
        for i in range(0, len(particles)):
            current_algorithm_hyper_parameters = particles[i].position_i
            current_hyper_parameters.append(current_algorithm_hyper_parameters)
        return current_hyper_parameters

    def _acceptance_criteria(self, particle_index, metric_best_g):
        if self.isMaximization:
            return self.swarm[particle_index].metric_best_i > metric_best_g
        else:
            return self.swarm[particle_index].metric_best_i < metric_best_g

    def _select_best_algorithms(self):
        current_algorithm_name = self.swarm[0].algorithm.algorithm_name
        best_results = {
            current_algorithm_name: self.swarm[0],
        }

        for particle in self.swarm[1:]:
            if (particle.algorithm.algorithm_name in best_results):
                current_result = best_results[particle.algorithm.algorithm_name].metric_i
                if particle.metric_i > current_result:
                    best_results[particle.algorithm.algorithm_name] = particle
            else:
                best_results[particle.algorithm.algorithm_name] = particle

        return best_results

    def fit(self,
            X_train,
            X_test,
            Y_train,
            Y_test,
            maxiter=20,
            compare_models=False,
            verbose=False,
            max_distance=0.05,
            semi_verbose=True,
            agents=None,
            chunksize=1,
            dbModel=None,
            job=None):

        self.current_epoch = 0
        metric_best_g = -1                   # best error for group

        if self.isMaximization is False:
            metric_best_g = 9999999999999999

        # best position for group
        pos_best_g = self.initial_best_positions
        model_best_g = any
        best_model_name = ""

        processes = cpu_count()
        print(f'CPU count {processes}')

        if agents is not None:
            processes = agents

        print(f'Agents count {processes}')

        with multiprocessing.Pool(processes=processes) as pool:
            # begin optimization loop
            i = 0
            while i < maxiter:
                self.current_epoch = i

                if dbModel is not None:
                    dbModel.totalEpochs = maxiter
                    dbModel.epoch = self.current_epoch
                    dbModel.save()

                if job is not None:
                    if dbModel is not None:
                        job.meta['model_id'] = str(dbModel.id)
                    job.meta['totalEpochs'] = maxiter
                    job.meta['earlyStop'] = False
                    job.meta['epoch'] = self.current_epoch
                    job.save_meta()

                if verbose or semi_verbose:
                    print("--- START EPOCH {} ---".format(i))
                # print i,err_best_g
                # cycle through particles in swarm and evaluate fitness

                result = []

                if self.is_unsupervised is not True:
                    for k in range(0, len(self.swarm)):
                        result_k = evaluate_particle(particle=self.swarm[k],
                                                     epoch=i,
                                                     verbose=verbose,
                                                     cost_function=self.cost_function,
                                                     X_train=X_train,
                                                     X_test=X_test,
                                                     Y_train=Y_train,
                                                     Y_test=Y_test)
                        result.append(result_k)
                else:
                    result = pool.map(partial(evaluate_particle,
                                              epoch=i,
                                              verbose=verbose,
                                              cost_function=self.cost_function,
                                              X_train=X_train,
                                              X_test=X_test,
                                              Y_train=None,
                                              Y_test=Y_test),
                                      self.swarm,
                                      chunksize)

                self.swarm = result

                for j in range(0, len(self.swarm)):
                    # determine if current particle is the best (globally)

                    if self._acceptance_criteria(j, metric_best_g):
                        golbal_best_position_index = self.algorithm_mapping[
                            self.swarm[j].algorithm.algorithm_name]
                        pos_best_g[golbal_best_position_index] = list(
                            self.swarm[j].position_i)
                        metric_best_g = float(self.swarm[j].metric_best_i)
                        model_best_g = self.swarm[j]
                        best_model_name = self.swarm[j].algorithm.algorithm_name

                    if verbose:
                        print("* Particle {} Algorithm Type {}: personal best metric={} Local best model: {} *".format(
                            j, self.swarm[j].algorithm.algorithm_name,
                            self.swarm[j].metric_best_i,
                            self.swarm[j].model_best_i))

                # max_distance_between_particle = self.calculate_max_distance_between_particles()

                # if max_distance_between_particle < max_distance:
                #     if verbose:
                #         print(f"Early stop - distance between particles is very low -
                #           {max_distance_between_particle} < {max_distance}")

                #     if dbModel is not None:
                #         dbModel.totalEpochs = maxiter
                #         dbModel.epoch = self.current_epoch
                #         dbModel.earlyStop = True
                #         dbModel.save()

                #     if job is not None:
                #         if dbModel is not None:
                #             job.meta['model_id'] = str(dbModel.id)
                #         job.meta['totalEpochs'] = maxiter
                #         job.meta['earlyStop'] = True
                #         job.meta['epoch'] = self.current_epoch
                #         job.save_meta()
                #     return metric_best_g, model_best_g, best_model_name

                if compare_models is False:
                    particle_to_remove = 1
                    if self.particle_count > 2:
                        particle_to_remove = int(self.particle_count/2)

                    for k in range(particle_to_remove):
                        self._remove_worst(verbose)
                        new_particle = self._add_to_best(verbose)

                        new_particle.evaluate(
                            self.cost_function,
                            X_train,
                            X_test,
                            Y_train,
                            Y_test,
                            epoch=i,
                            verbose=verbose)

                        self.swarm.append(new_particle)

                    if self.num_particles <= 0:
                        if verbose:
                            print("Early stop - number of particles 0")

                        return metric_best_g, model_best_g, best_model_name
                else:
                    particles = self._distinct_algorithm_names()

                    particle_to_remove = 1
                    if self.particle_count > 2 and self.particle_count <= 6:
                        particle_to_remove = int(self.particle_count/2)
                    else:
                        particle_to_remove = int(self.particle_count/3)

                    for particle in particles:
                        for k in range(particle_to_remove):
                            self._remove_worst_by_algorithm_name(
                                particle.algorithm.algorithm_name)
                        for k in range(particle_to_remove):
                            new_particle = self._add_by_algorithm_name(
                                particle.algorithm.algorithm_name)

                            new_particle.evaluate(
                                self.cost_function,
                                X_train,
                                X_test,
                                Y_train,
                                Y_test,
                                epoch=i,
                                verbose=verbose)

                            self.swarm.append(new_particle)

                # cycle through swarm and update velocities and position
                for j in range(0, self.num_particles):
                    golbal_best_position_index = self.algorithm_mapping[
                        self.swarm[j].algorithm.algorithm_name]
                    self.swarm[j].update_velocity(
                        pos_best_g[golbal_best_position_index])
                    self.swarm[j].update_position()
                i += 1
                if verbose or semi_verbose:
                    print("--- END EPOCH {} ---".format(i))

        if verbose or semi_verbose:
            # print final results
            print('FINAL PSO:')
            print(pos_best_g)
            print(metric_best_g)
            print(model_best_g)

        if dbModel is not None:
            dbModel.totalEpochs = maxiter
            dbModel.epoch = maxiter
            dbModel.save()

        if job is not None:
            if dbModel is not None:
                job.meta['model_id'] = str(dbModel.id)
            job.meta['totalEpochs'] = maxiter
            job.meta['earlyStop'] = False
            job.meta['epoch'] = maxiter
            job.save_meta()

        if compare_models is True:
            return model_best_g, self._select_best_algorithms()

        return metric_best_g, model_best_g, best_model_name
