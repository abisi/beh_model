#! /usr/bin/env/python3
"""
@author: Axel Bisi
@project: beh_model
@file: fit_single_mouse_glmhmm_parallel.py
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
from itertools import product
from multiprocessing import Pool
from pathlib import Path

import ssm
import logging
import config

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

from data_utils import create_data_lists
from plotting_utils import (plot_model_glm_weights, plot_model_transition_matrix,
                            plot_single_session_predictions, plot_single_session_posterior_states)
from utils import build_glmhmm, get_expected_states, get_predicted_labels, calculate_predictive_accuracy


def process_single_fit(mouse_id, split_idx, k_state, iter_idx,
                       global_data_train, global_data_test,
                       root_path, path_to_global_weights):
    """
    Fit a single GLM-HMM for one (subject, split, n_states, instance) combination.

    :param mouse_id:               subject identifier
    :param split_idx:              dataset split index
    :param k_state:                number of hidden states
    :param iter_idx:               random initialisation instance index
    :param global_data_train:      full training dataframe (all mice)
    :param global_data_test:       full test dataframe (all mice)
    :param root_path:              root directory for saving results
    :param path_to_global_weights: path to global model weight directory
    :return: dict of results, or None if global weights are missing
    """
    logger.info(f'Subject {mouse_id} | split {split_idx} | {k_state} states | instance {iter_idx}')

    # ----------------------------------
    # Load corresponding global weights
    # ----------------------------------
    global_weights_path = (
        Path(path_to_global_weights)
        / f'model_{split_idx}'
        / f'{k_state}_states'
        / 'full'
        / f'iter_{iter_idx}'
        / 'global_fit_glmhmm_results.npz'
    )

    if not global_weights_path.exists():
        logger.warning(f'Global weights not found at {global_weights_path}, skipping.')
        return None

    global_results = np.load(global_weights_path, allow_pickle=True)['arr_0'].item()
    weight_for_init = global_results['weights']
    features = global_results['features']

    assert global_results['n_states'] == k_state, (
        f"Loaded n_states {global_results['n_states']} does not match expected {k_state}"
    )

    # ---------------------------
    # Prepare output directory
    # ---------------------------
    results_dir_iter = (
        Path(root_path, 'all_subjects_glmhmm', mouse_id, 'full_models')
        / f'model_{split_idx}'
        / f'{k_state}_states'
        / f'iter_{iter_idx}'
    )
    results_dir_iter.mkdir(parents=True, exist_ok=True)

    # ---------------------------
    # Filter to subject data
    # ---------------------------
    data_train = global_data_train[global_data_train['mouse_id'] == mouse_id].copy()
    data_test  = global_data_test[global_data_test['mouse_id'] == mouse_id].copy()

    # ---------------------------
    # Initialise noisy weights
    # ---------------------------
    noisy_weights = weight_for_init + np.random.normal(
        0, config.HMM_PARAMS['noise_level'], weight_for_init.shape
    )

    # ---------------------------
    # Build and fit GLM-HMM
    # ---------------------------
    input_train, output_train, input_test, output_test = create_data_lists(
        data_train, data_test, features=features
    )

    glmhmm = build_glmhmm(
        n_states=k_state,
        input_dim=len(features),
        prior_sigma=config.HMM_PARAMS['prior_sigma'],
        prior_alpha=config.HMM_PARAMS['prior_alpha'],
        kappa=config.HMM_PARAMS['kappa'],
    )
    glmhmm.observations.params = noisy_weights

    glmhmm.fit(
        output_train,
        inputs=input_train,
        method='em',
        num_iters=config.HMM_PARAMS['n_train_iters'],
        tolerance=config.HMM_PARAMS['tolerance'],
    )
    recovered_weights = glmhmm.observations.params
    transition_matrix = glmhmm.transitions.transition_matrix
    ll_train = glmhmm.log_likelihood(output_train, input_train, None, None)
    ll_test  = glmhmm.log_likelihood(output_test,  input_test,  None, None)

    # ---------------------------
    # Predictive accuracy
    # ---------------------------
    def _eval(outputs, inputs):
        posteriors  = np.concatenate(get_expected_states(glmhmm, outputs=outputs, inputs=inputs), axis=0)
        pred_labels = get_predicted_labels(glmhmm, inputs=inputs, posteriors=posteriors)
        acc, balanced_acc = calculate_predictive_accuracy(outputs, pred_labels)
        return posteriors, pred_labels, acc, balanced_acc

    posterior_probs_train, pred_labels_train, pred_acc_train, balanced_pred_acc_train = _eval(output_train, input_train)
    posterior_probs_test,  pred_labels_test,  pred_acc_test, balanced_pred_acc_test = _eval(output_test,  input_test)

    logger.info(f'  Train accuracy: {pred_acc_train:.4f} | Test accuracy: {pred_acc_test:.4f}')

    # ---------------------------
    # Save results
    # ---------------------------
    result_dict = {
        'mouse_id':             mouse_id,
        'split_idx':            split_idx,
        'n_states':             k_state,
        'instance_idx':         iter_idx,
        'features':             features,
        'weights':              recovered_weights,
        'init_weights':         noisy_weights,
        'transition_matrix':    transition_matrix,
        'll_train':             ll_train,
        'll_test':              ll_test,
        'output_train_labels': output_train,
        'output_test_labels':  output_test,
        'output_train_preds':   pred_labels_train,
        'output_test_preds':    pred_labels_test,
        'predictive_acc_train': pred_acc_train,
        'predictive_acc_test':  pred_acc_test,
        'reward_group':         data_train['reward_group'].unique(),
    }
    np.savez(results_dir_iter / 'fit_glmhmm_results.npz', result_dict)

    # ---------------------------
    # Plots
    # ---------------------------
    plot_model_glm_weights(
        model=glmhmm, init_weights=noisy_weights, feature_names=features,
        save_path=results_dir_iter,
        file_name=f'{mouse_id}_glm_weights',
        suffix=None, file_types=['pdf'],
    )
    plot_model_transition_matrix(
        model=glmhmm, save_path=results_dir_iter,
        file_name=f'{mouse_id}_transition_matrix',
        suffix=None, file_types=['png'],
    )

    # Assemble combined dataframe with predictions and posteriors
    data_train['split'] = 'train'
    data_train['pred']  = np.concatenate(pred_labels_train, axis=0)
    data_test['split']  = 'test'
    data_test['pred']   = np.concatenate(pred_labels_test, axis=0)

    for state_id in range(k_state):
        col = f'posterior_state_{state_id + 1}'
        data_train[col] = posterior_probs_train[:, state_id]
        data_test[col]  = posterior_probs_test[:, state_id]

    data = (
        pd.concat([data_train, data_test], axis=0)
        .sort_values(by=['session_id', 'trial_id'])
        .reset_index(drop=True)
    )
    data.to_hdf(results_dir_iter / 'data_preds.h5', key='data', mode='w')

    plot_single_session_predictions(
        data=data, save_path=results_dir_iter / 'predictions',
        file_name='predictions', suffix=None, file_types=['png', 'svg'],
    )
    plot_single_session_posterior_states(
        data=data, save_path=results_dir_iter / 'posterior_states',
        file_name='posterior_states', suffix=None, file_types=['png', 'svg'],
    )

    return result_dict


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':

    experimenter = 'Axel_Bisi'
    root_path = Path(f'\\\\sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/{experimenter}/combined_results/glm_hmm')
    global_dataset_path      = root_path / 'datasets_combined' / 'dataset_0'
    global_model_weight_path = root_path / 'global_glmhmm' #/ 'model_0'

    global_data_train = pickle.load(open(global_dataset_path / 'data_train.pkl', 'rb'))
    global_data_test  = pickle.load(open(global_dataset_path / 'data_test.pkl',  'rb'))
    subject_ids_list  = global_data_train['mouse_id'].unique()

    # Determine number of splits from available folders on disk
    N_SPLITS      = len([f for f in os.listdir(global_model_weight_path) if 'model_' in f])
    N_STATES_LIST = range(1, 1 + config.N_STATES)
    N_INSTANCES   = config.N_INSTANCES           # random restarts per configuration

    N_SPLITS = 2
    N_STATES_LIST = range(1,1+config.N_STATES)
    N_INSTANCES = 2


    # Build the full flat task list: one entry per (subject, split, k_state, instance)
    tasks = [
        (mouse_id, split_idx, k_state, iter_idx,
         global_data_train, global_data_test,
         root_path, global_model_weight_path)
        for mouse_id, split_idx, k_state, iter_idx
        in product(subject_ids_list, range(N_SPLITS), N_STATES_LIST, range(N_INSTANCES))
    ]
    logger.info(f'Total tasks to run: {len(tasks)}')

    start_time = time.time()
    with Pool(processes=max(1, os.cpu_count() - 5)) as pool:
        all_results = pool.starmap(process_single_fit, tasks)

    # Filter out skipped tasks (None) and collect into a dataframe
    flat_results = [r for r in all_results if r is not None]
    res_df = pd.DataFrame(flat_results)

    #out_path = root_path / 'all_subjects_glmhmm' / 'all_subjects_glmhmm_results.h5'
    out_path = root_path / 'all_subjects_glmhmm' / 'all_subjects_glmhmm_results.pkl'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    res_df.to_pickle(out_path)
    #res_df.to_hdf(out_path, key='df', mode='w')

    logger.info(f'Script finished in {time.time() - start_time:.2f} seconds.')
