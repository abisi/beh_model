#! /usr/bin/env/python3
"""
@author: Axel Bisi
@project: beh_model
@file: fit_single_mouse_glmhmm.py
@time: 12/8/2023 3:28 PM
"""


# Imports
import os
import numpy as np
import numpy.random as npr
npr.seed(0)
import pickle
import pandas as pd
from pathlib import Path

import ssm

from data_utils import create_data_lists
from plotting_utils import remove_top_right_frame, save_figure_to_files, plot_model_glm_weights, \
    plot_model_transition_matrix, plot_single_session_predictions, plot_single_session_posterior_states
from utils import add_noise_to_weights, get_expected_states, get_predicted_labels
from utils import calculate_predictive_accuracy



if __name__ == '__main__':

    # Load train and test datasets
    experimenter = 'Axel_Bisi'
    root_path = Path(f'\\\\sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/{experimenter}/beh_model')
    global_dataset_path = root_path / 'datasets' / 'dataset_0'
    global_model_weight_path = root_path / 'global_glmhmm' / 'full_models'

    # Load general dataset to get subject IDs
    global_dataset_path = os.path.join(root_path, 'datasets', 'dataset_0') # TODO: update in loop
    global_data_train = pickle.load(open(os.path.join(global_dataset_path, 'data_train.pkl'), 'rb'))
    global_data_test = pickle.load(open(os.path.join(global_dataset_path, 'data_test.pkl'), 'rb'))
    subject_ids_list = global_data_train['mouse_id'].unique()

    for subject_id in subject_ids_list:
        print('Fitting GLM-HMM for subject: {}'.format(subject_id))

        # Create subject folders
        subject_folder_path = Path(root_path, subject_id)
        subject_folder_path.mkdir(parents=True, exist_ok=True)
        dataset_folder_path = Path(subject_folder_path, 'datasets')
        dataset_folder_path.mkdir(parents=True, exist_ok=True)
        model_folder_path = Path(subject_folder_path, 'full_models')
        model_folder_path.mkdir(parents=True, exist_ok=True)

        # Get mouse data from dataset
        data_train = global_data_train[global_data_train['mouse_id'] == subject_id]
        data_test = global_data_test[global_data_test['mouse_id'] == subject_id]

        # Specifify which model to load
        split_idx = 0
        k_state = 2
        iter_idx = 0

        results_dir_iter = Path(os.path.join(model_folder_path, 'model_{}'.format(split_idx), '{}_states'.format(k_state), 'iter_{}'.format(iter_idx)))
        results_dir_iter.mkdir(parents=True, exist_ok=True)


        # Load global fit model weights
        path_to_global_weights = os.path.join(global_model_weight_path, 'model_{}'.format(split_idx), '{}_states'.format(k_state), 'iter_{}'.format(iter_idx))
        global_results = np.load(os.path.join(path_to_global_weights, 'global_fit_glmhmm_results.npz'), allow_pickle=True)['arr_0']
        global_results = global_results.item()
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
        n_states = k_state
        obs_dim = 1
        num_categories = 2
        input_dim = len(features)
        glmhmm = ssm.HMM(n_states, obs_dim, input_dim,
                         observations="input_driven_obs",
                         observation_kwargs=dict(C=num_categories, prior_sigma=prior_sigma),
                         transitions="sticky",
                         transition_kwargs=dict(alpha=prior_alpha,kappa=0)
                         )
        glmhmm.observations.params = noisy_weights


        # Fit GLM-HMM with MAP
        n_iters = 200  # max EM iterations (fitting will stop earlier if LL below tolerance)
        tol = 10 ** -3
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
        pred_acc_train, balanced_pred_acc_train = calculate_predictive_accuracy(output_train, pred_labels_train)

        # Test data
        expected_states_test = get_expected_states(glmhmm, outputs=output_test, inputs=input_test)
        posterior_probs_test = np.concatenate(expected_states_test, axis=0)
        pred_labels_test = get_predicted_labels(glmhmm, inputs=input_test, posteriors=posterior_probs_test)
        pred_acc_test, balanced_pred_acc_test = calculate_predictive_accuracy(output_test, pred_labels_test)

        print('Predictive accuracy on train data: {}'.format(pred_acc_train))
        print('Predictive accuracy on test data: {}'.format(pred_acc_test))


        # -------------------------
        # Save single-mouse results
        # -------------------------

        result_dict = {
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
        file_name='{}_glm_weights'.format(subject_id), suffix=None, file_types=['png', 'svg'])

        plot_model_transition_matrix(model=glmhmm, save_path=results_dir_iter,
        file_name='{}_transition_matrix'.format(subject_id), suffix=None, file_types=['png', 'svg'])

        # ----------------------------------
        # Plot single-mouse predictions
        # ----------------------------------

        # Stitch together train and test dataframes, according to trial number and session
        data_train['split'] = 'train'
        data_train['pred'] = np.concatenate(pred_labels_train, axis=0)
        data_test['split'] = 'test'
        data_test['pred'] = np.concatenate(pred_labels_test, axis=0)
        for state_id in range(k_state):
            data_train['posterior_state_{}'.format(state_id+1)] = posterior_probs_train[:, state_id]
            data_test['posterior_state_{}'.format(state_id+1)] = posterior_probs_test[:, state_id]
        data = pd.concat([data_train, data_test], axis=0)
        data = data.sort_values(by=['session_id', 'trial_id'])
        data = data.reset_index(drop=True)

        # Save mouse data with predictions as h5
        data.to_hdf(os.path.join(results_dir_iter, 'data_preds.h5'), key='data', mode = 'w')

        # Plot single-session predictions
        fig_folder = os.path.join(results_dir_iter, 'predictions')
        plot_single_session_predictions(data=data, save_path=fig_folder, file_name='predictions', suffix=None,
        file_types=['png', 'svg'])

        # Plot single-session posterior state estimations
        fig_folder = os.path.join(results_dir_iter, 'posterior_states')
        plot_single_session_posterior_states(data=data, save_path=fig_folder, file_name='posterior_states', suffix=None,
        file_types=['png', 'svg'])




