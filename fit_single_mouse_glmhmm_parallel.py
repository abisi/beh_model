#! /usr/bin/env/python3
"""
@author: Axel Bisi
@project: beh_model
@file: fit_single_mouse_glmhmm.py
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
from plotting_utils import remove_top_right_frame, save_figure_to_files, plot_model_glm_weights, \
    plot_model_transition_matrix, plot_single_session_predictions, plot_single_session_posterior_states
from utils import add_noise_to_weights, get_expected_states, get_predicted_labels
from utils import calculate_predictive_accuracy

def process_subject(mouse_id, global_data_train, global_data_test, root_path, path_to_global_weights):

    logger.info('Fitting GLM-HMM for subject: {}'.format(mouse_id))
    res = pd.DataFrame()

    # Create subject folders
    subject_folder_path = Path(root_path, mouse_id)
    subject_folder_path.mkdir(parents=True, exist_ok=True)
    dataset_folder_path = Path(subject_folder_path, 'datasets_time')
    dataset_folder_path.mkdir(parents=True, exist_ok=True)
    model_folder_path = Path(subject_folder_path, 'full_models_time')
    model_folder_path.mkdir(parents=True, exist_ok=True)

    # Get mouse data from dataset
    data_train = global_data_train[global_data_train['mouse_id'] == mouse_id]
    data_test = global_data_test[global_data_test['mouse_id'] == mouse_id]

    # Specify which global fit model weights to load: #TODO: loop over
    k_state = 1
    iter_idx = 0

    # Counter number of data-split model available i.e number of folders
    n_splits = len([name for name in os.listdir(path_to_global_weights) if os.path.isdir(os.path.join(path_to_global_weights, name))])
    for split_idx in range(n_splits):

        results_dir_iter = Path(
            os.path.join(model_folder_path, 'model_{}'.format(split_idx), '{}_states'.format(k_state),
                         'iter_{}'.format(iter_idx)))
        results_dir_iter.mkdir(parents=True, exist_ok=True)

        path_to_global_weights_iter = os.path.join(path_to_global_weights, 'model_{}'.format(split_idx),
                                              '{}_states'.format(k_state), 'iter_{}'.format(iter_idx))
        global_results = np.load(os.path.join(path_to_global_weights_iter, 'global_fit_glmhmm_results.npz'), allow_pickle=True)[
            'arr_0']
        global_results = global_results.item()
        k_state = global_results['n_states'] #TODO: loop over?
        split_idx = global_results['split_idx']
        iter_idx = global_results['instance_idx']
        weight_for_init = global_results['weights']
        features = global_results['features']

        # Make noisy GLM weights
        n_states = weight_for_init.shape[0]
        n_features = weight_for_init.shape[2]
        gauss_noise_weights = np.random.normal(0, 0.1, n_features)
        noisy_weights = weight_for_init + gauss_noise_weights

        # Make noisy transition matrix #TODO: unused, resolve
        sigma_trans = 0.05 * np.ones((k_state, k_state))
        gauss_noise_transition = np.random.normal(0, sigma_trans)
        transition_matrix = 0.95 * np.identity(k_state) + gauss_noise_transition

        # Normalize transition matrix (sum across rows) to get probabilities
        transition_matrix = transition_matrix / transition_matrix.sum(axis=1)[:, np.newaxis]

        # Create design matrices, split session-wise
        input_train, output_train, input_test, output_test = create_data_lists(data_train, data_test, features=features)

        prior_sigma = 2
        prior_alpha = 2

        # Initialize single-mouse GLM-HMM
        obs_dim = 1
        num_categories = 2
        input_dim = len(features)
        glmhmm = ssm.HMM(k_state, obs_dim, input_dim,
                         observations="input_driven_obs",
                         observation_kwargs=dict(C=num_categories, prior_sigma=prior_sigma),
                         transitions="sticky",
                         transition_kwargs=dict(alpha=prior_alpha, kappa=0),
                         )

        glmhmm.observations.params = noisy_weights

        # Fit GLM-HMM with MAP
        n_iters = 200  # max EM iterations (fitting will stop earlier if LL below tolerance)
        tol = 10 ** -4
        glmhmm.fit(output_train, inputs=input_train, method='em', num_iters=n_iters, tolerance=tol)
        ll_train = glmhmm.log_likelihood(output_train, input_train, None, None)
        recovered_weights = glmhmm.observations.params
        transition_matrix = glmhmm.transitions.transition_matrix

        # ------------------------
        # Performance on test data
        # ------------------------
        ll_test = glmhmm.log_likelihood(output_test, input_test, None, None)

        # -----------------------------
        # Calculate predictive accuracy
        # -----------------------------

        # Training data
        expected_states_train = get_expected_states(glmhmm, outputs=output_train, inputs=input_train)
        posterior_probs_train = np.concatenate(expected_states_train, axis=0)
        pred_labels_train = get_predicted_labels(glmhmm, inputs=input_train, posteriors=posterior_probs_train)
        pred_acc_train = calculate_predictive_accuracy(output_train, pred_labels_train)

        # Test data
        expected_states_test = get_expected_states(glmhmm, outputs=output_test, inputs=input_test)
        posterior_probs_test = np.concatenate(expected_states_test, axis=0)
        pred_labels_test = get_predicted_labels(glmhmm, inputs=input_test, posteriors=posterior_probs_test)
        pred_acc_test = calculate_predictive_accuracy(output_test, pred_labels_test)

        print('Predictive accuracy on train data: {}'.format(pred_acc_train))
        print('Predictive accuracy on test data: {}'.format(pred_acc_test))

        # -------------------------
        # Save single-mouse results
        # -------------------------

        result_dict = {
            'mouse_id': mouse_id,
            'split_idx': split_idx,
            'n_states': k_state,
            'instance_idx': iter_idx,
            'features': features,
            'weights': recovered_weights,
            'init_weights': noisy_weights,
            'transition_matrix': transition_matrix,
            'll_train': ll_train,
            'll_test': ll_test,
            'output_train_preds:': pred_labels_train,
            'output_test_preds': pred_labels_test,
            'predictive_acc_train': pred_acc_train,
            'predictive_acc_test': pred_acc_test,
            'reward_group': data_train['reward_group'].unique(),
        }
        np.savez(os.path.join(results_dir_iter, 'fit_glmhmm_results.npz'), result_dict)

        # ----------------------------------
        # Plot single-mouse model parameters
        # ----------------------------------
        plot_model_glm_weights(model=glmhmm, init_weights=noisy_weights, feature_names=features, save_path=results_dir_iter,
                               file_name='{}_glm_weights'.format(mouse_id), suffix=None, file_types=['png', 'svg'])

        plot_model_transition_matrix(model=glmhmm, save_path=results_dir_iter,
                                     file_name='{}_transition_matrix'.format(mouse_id), suffix=None,
                                     file_types=['png', 'svg'])

        # ----------------------------------
        # Plot single-mouse predictions
        # ----------------------------------

        # Stitch together train and test dataframes, according to trial number and session
        data_train['split'] = 'train'
        data_train['pred'] = np.concatenate(pred_labels_train, axis=0)
        data_test['split'] = 'test'
        data_test['pred'] = np.concatenate(pred_labels_test, axis=0)
        for state_id in range(k_state):
            data_train['posterior_state_{}'.format(state_id + 1)] = posterior_probs_train[:, state_id] #TODO: keep state indexing starting at 0
            data_test['posterior_state_{}'.format(state_id + 1)] = posterior_probs_test[:, state_id] #TODO: keep state indexing starting at 0
        data = pd.concat([data_train, data_test], axis=0)
        data = data.sort_values(by=['session_id', 'trial_id'])
        data = data.reset_index(drop=True)

        # Save mouse data with predictions as h5
        data.to_hdf(os.path.join(results_dir_iter, 'data_preds.h5'), key='data', mode='w')

        # Plot single-session predictions
        fig_folder = os.path.join(results_dir_iter, 'predictions')
        plot_single_session_predictions(data=data, save_path=fig_folder, file_name='predictions', suffix=None,
                                        file_types=['png', 'svg'])

        # Plot single-session posterior state estimations
        fig_folder = os.path.join(results_dir_iter, 'posterior_states')
        plot_single_session_posterior_states(data=data, save_path=fig_folder, file_name='posterior_states', suffix=None,
                                             file_types=['png', 'svg'])

    return res



if __name__ == '__main__':

    # Load train and test datasets
    experimenter = 'Axel_Bisi'
    root_path = Path(f'\\\\sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/{experimenter}/beh_model')
    global_dataset_path = root_path / 'datasets_time_new' / 'dataset_0'
    global_model_weight_path = root_path / 'global_glmhmm_time_new' / 'full_models'

    # Load general dataset to get subject IDs
    global_dataset_path = os.path.join(root_path, 'datasets_time_new', 'dataset_0')
    global_data_train = pickle.load(open(os.path.join(global_dataset_path, 'data_train.pkl'), 'rb'))
    global_data_test = pickle.load(open(os.path.join(global_dataset_path, 'data_test.pkl'), 'rb'))
    subject_ids_list = global_data_train['mouse_id'].unique()

    N_SPLITS = 1
    N_INSTANCES = 1

    start_time = time.time()
    with Pool(processes=os.cpu_count() - 1) as pool:
        all_results = pool.starmap(process_subject, [(subject_id, global_data_train, global_data_test, root_path,
                                                      global_model_weight_path) for subject_id in subject_ids_list])

    flat_results = [item for sublist in all_results for item in sublist]
    res_df = pd.DataFrame(flat_results)
    all_subjects_res_path = Path(root_path, 'all_subjects_glmhmm_time_new', 'all_subjects_glmhmm_results.h5')
    all_subjects_res_path.parent.mkdir(parents=True, exist_ok=True)
    res_df.to_hdf(all_subjects_res_path, key='df', mode='w')

    logger.info(f'Script finished in {time.time() - start_time:.2f} seconds.')

