#! /usr/bin/env/python3
"""
@author: Axel Bisi
@project: beh_model
@file: fit_global_glm.py
@time: 12/8/2023 3:28 PM
"""

# Imports
import os
import time
import numpy as np
import numpy.random as npr
npr.seed(0)
import pandas as pd
import pickle
from multiprocessing import Pool
from pathlib import Path

import ssm
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

from data_utils import create_data_lists
from plotting_utils import plot_model_glm_weights, plot_model_transition_matrix
from utils import get_expected_states, get_predicted_labels, calculate_predictive_accuracy, add_noise_to_weights


def process_split(split_idx, N_STATES, N_INSTANCES, dataset_path, result_path):
    result_dict_list = []

    # Load train and test datasets
    dataset_split_path = Path(dataset_path, f'dataset_{split_idx}')
    result_split_path = Path(result_path, f'model_{split_idx}')
    data_train = pickle.load(open(Path(dataset_split_path, 'data_train.pkl'), 'rb'))
    data_test = pickle.load(open(Path(dataset_split_path, 'data_test.pkl'), 'rb'))

    features = ['bias',
                'whisker',
                'auditory',
                'time_since_last_auditory_stim',
                'time_since_last_whisker_stim',
                'time_since_last_auditory_reward',
                'time_since_last_whisker_reward'
                ]

    # Create design matrices, split session-wise
    input_train, output_train, input_test, output_test = create_data_lists(data_train, data_test, features=features)

    prior_sigma = 1
    prior_alpha = 2

    for k_idx in range(1, N_STATES + 1):
        logger.info(f'Fitting global GLM-HMM with {k_idx} states...')
        model_split_n_state_path = os.path.join(result_split_path, f'{k_idx}_states')

        for instance_idx in range(N_INSTANCES):
            results_dir_iter = os.path.join(model_split_n_state_path, f'iter_{instance_idx}')
            os.makedirs(results_dir_iter, exist_ok=True)

            # Use 1-state model weights as initialization if they exist
            if k_idx > 1:
                path_to_init_weights = Path(result_split_path, f'1_states', f'iter_{instance_idx}', 'global_fit_glmhmm_results.npz')
                init_weights = np.load(path_to_init_weights, allow_pickle=True)['arr_0'].item()['weights']
                # Make it noisy
                init_weights = add_noise_to_weights(init_weights, noise_level=0.2)
                init_weights = np.repeat(init_weights, k_idx, axis=0)

            else:
                init_weights = None

            n_states = k_idx
            obs_dim = 1
            num_categories = 2
            input_dim = len(features)
            glmhmm = ssm.HMM(n_states, obs_dim, input_dim,
                             observations="input_driven_obs",
                             observation_kwargs=dict(C=num_categories, prior_sigma=prior_sigma),
                             transitions="sticky",
                             transition_kwargs=dict(alpha=prior_alpha, kappa=0))
            if init_weights is not None:
                glmhmm.observations.params = init_weights

            n_train_iters = 200
            tol = 10**-4
            ll_train = glmhmm.fit(output_train, inputs=input_train, method='em', num_iters=n_train_iters, tolerance=tol)
            recovered_weights = glmhmm.observations.params
            transition_matrix = glmhmm.transitions.transition_matrix

            ll_test = glmhmm.log_likelihood(output_test, input_test, None, None)

            expected_states_train = get_expected_states(glmhmm, outputs=output_train, inputs=input_train)
            posterior_probs_train = np.concatenate(expected_states_train, axis=0)
            pred_labels_train = get_predicted_labels(glmhmm, inputs=input_train, posteriors=posterior_probs_train)
            pred_acc_train = calculate_predictive_accuracy(output_train, pred_labels_train)

            expected_states_test = get_expected_states(glmhmm, outputs=output_test, inputs=input_test)
            posterior_probs_test = np.concatenate(expected_states_test, axis=0)
            pred_labels_test = get_predicted_labels(glmhmm, inputs=input_test, posteriors=posterior_probs_test)
            pred_acc_test = calculate_predictive_accuracy(output_test, pred_labels_test)

            logger.info(f'Predictive accuracy on train data: {pred_acc_train}')
            logger.info(f'Predictive accuracy on test data: {pred_acc_test}')

            plot_model_glm_weights(model=glmhmm, init_weights=init_weights, feature_names=features,
                                   save_path=results_dir_iter, file_name='global_weights', suffix=None, file_types=['png', 'svg'])
            plot_model_transition_matrix(model=glmhmm, save_path=results_dir_iter,
                                         file_name='transition_matrix', suffix=None, file_types=['png', 'svg'])

            result_dict = {
                'split_idx': split_idx,
                'n_states': k_idx,
                'instance_idx': instance_idx,
                'features': features,
                'weights': recovered_weights,
                'transition_matrix': transition_matrix,
                'll_train': ll_train,
                'll_test': ll_test,
                'output_train_preds': pred_labels_train,
                'output_test_preds': pred_labels_test,
                'predictive_acc_train': pred_acc_train,
                'predictive_acc_test': pred_acc_test,
            }
            result_dict_list.append(result_dict)

            # Save
            np.savez(os.path.join(results_dir_iter, 'global_fit_glmhmm_results.npz'), result_dict)

    return result_dict_list

if __name__ == '__main__':
    experimenter = 'Axel_Bisi'
    dataset_path = Path(f'\\\\sv-nas1.rcp.epfl.ch\\Petersen-Lab\\analysis\\{experimenter}\\beh_model\\datasets_time_new')
    result_path = Path(f'\\\\sv-nas1.rcp.epfl.ch\\Petersen-Lab\\analysis\\{experimenter}\\beh_model\\global_glmhmm_time_new\\full_models')
    # Create path if it does not exist
    os.makedirs(result_path, exist_ok=True)

    N_SPLITS = 10
    N_STATES = 5
    N_INSTANCES = 1

    start_time = time.time()
    with Pool(processes=min(N_SPLITS, os.cpu_count()-5)) as pool:
        all_results = pool.starmap(process_split, [(i, N_STATES, N_INSTANCES, dataset_path, result_path) for i in range(N_SPLITS)])

    flat_results = [item for sublist in all_results for item in sublist]
    res_df = pd.DataFrame(flat_results)
    res_df.to_hdf(os.path.join(result_path, 'global_fit_glmhmm_results.h5'), key='df', mode='w')

    logger.info(f'Script finished in {time.time() - start_time:.2f} seconds')