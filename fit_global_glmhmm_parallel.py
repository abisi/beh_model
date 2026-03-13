#! /usr/bin/env/python3
"""
@author: Axel Bisi
@project: beh_model
@file: fit_global_glmhmm_parallel.py
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
from multiprocessing.pool import ThreadPool
from pathlib import Path

import ssm
import logging
import config

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

from data_utils import create_data_lists
from plotting_utils import plot_model_glm_weights, plot_model_transition_matrix
from utils import build_glmhmm, get_expected_states, get_predicted_labels, calculate_predictive_accuracy, add_noise_to_weights
from multicollinearity_utils import check_multicollinearity, plot_multicollinearity


def build_feature_sets(features, trial_types):
    """
    Build a dict of named feature sets to fit:
    Build a dict of named feature sets to fit:
      - 'full':          all features
      - 'drop_<feat>':   all features except <feat>  (leave-one-out)

    :param features: full list of feature names
    :return: dict mapping model_name -> feature list
    """
    sets = {'full': features}
    sets['bias_only'] = ['bias']
    mvt_features = ['jaw_distance', 'whisker_angle', 'nose_norm_distance', 'pupil_area']

    non_mvt_features = [f for f in features if f not in mvt_features]

    if trial_types == 'all_trials':
         trial_types = ['whisker', 'auditory']
         non_mvt_features = [f for f in features if f not in mvt_features]

         # Bias and trial types only
         sets['bias_trial_types_only'] = ['bias'] + trial_types
         # Remvoe trial types
         sets['drop_trial_type'] = [f for f in features if f not in trial_types]
         # Remove all history featuress
         sets['drop_history'] = mvt_features + ['bias', 'whisker', 'auditory']
    elif trial_types == 'whisker_trial':
        # Remove all history features
        sets['drop_history'] = mvt_features + ['bias']

        non_mvt_features = [f for f in features if f not in mvt_features and f not in ['whisker', 'auditory']]

    # Remove all mvt features
    sets['drop_mvt'] = non_mvt_features

    # Remove all trial info, keep only mvt
    sets['mvt_only'] = mvt_features

    # Bias and history only information
    sets['bias_history_only'] = ['bias'] + [f for f in features if 'time_since_last' in f]

    # Remove all features 1-by-1
    for feat in features:
        sets[f'drop_{feat}'] = [f for f in features if f != feat]

    return sets


def process_single_fit(split_idx, k_state, instance_idx, model_name, features,
                       dataset_path, result_path, reward_group):
    """
    Fit a single global GLM-HMM for one (split, n_states, instance, feature_set) combination.

    :param split_idx:     dataset split index
    :param k_state:       number of hidden states
    :param instance_idx:  random initialisation instance index
    :param model_name:    label for this feature set, e.g. 'full' or 'drop_whisker'
    :param features:      list of feature names to use for this fit
    :param dataset_path:  root path to dataset splits
    :param result_path:   root path for saving results
    :param reward_group:  reward group to filter by, e.g. 'R+' or 'R-'
    :return: result dict, or None on failure
    """
    logger.info(
        f'Split {split_idx} | {k_state} states | instance {instance_idx} | features: {model_name}'
    )

    # ----------------------------------
    # Load dataset split
    # ----------------------------------
    dataset_split_path = Path(dataset_path, f'dataset_{split_idx}')
    data_train = pickle.load(open(dataset_split_path / 'data_train.pkl', 'rb'))
    data_test  = pickle.load(open(dataset_split_path / 'data_test.pkl',  'rb'))
    print('Dataset columns', data_train.columns)

    fitted_trial_types = config.TRIAL_TYPES
    if fitted_trial_types == 'whisker_trial':
        data_train = data_train[data_train.whisker==1]
        data_test  = data_test[data_test.whisker==1]
        logger.info(f"Filtering dataset to trial types {fitted_trial_types}.")

    # Filter by reward group
    data_train = data_train[data_train['wh_reward'] == reward_group]
    data_test  = data_test[data_test['wh_reward']   == reward_group]
    n_mice_group = data_train.mouse_id.nunique()
    print(f"Filtering dataset to reward group {reward_group} with {n_mice_group} mice. ")


    # NOTE: remove all mouse_id that have no DLC data (nan) in DLC features
    dlc_cols = ['jaw_distance', 'whisker_angle', 'nose_norm_distance', 'pupil_area']
    # Find mouse_ids with any NaN in these columns in the training set
    mouse_ids_with_nan_train = data_train[data_train[dlc_cols].isna().all(axis=1)].mouse_id.unique()
    mouse_ids_with_nan_test = data_test[data_test[dlc_cols].isna().all(axis=1)].mouse_id.unique()
    mouse_ids_with_nan = np.union1d(mouse_ids_with_nan_train, mouse_ids_with_nan_test)
    # Filter out those mouse_ids from both train and test sets
    data_train = data_train[~data_train.mouse_id.isin(mouse_ids_with_nan)]
    data_test  = data_test[~data_test.mouse_id.isin(mouse_ids_with_nan)]
    print(f"Warning: filtering out mouse_ids {mouse_ids_with_nan} due to NaN values in DLC features. ")
    print(f"Remaining mouse_ids: {data_train.mouse_id.unique()}")

    input_train, output_train, input_test, output_test = create_data_lists(
        data_train, data_test, features=features
    )
    #print(input_test, output_test)

    # ----------------------------------
    # Prepare output directory
    # ----------------------------------
    results_dir_iter = (
        Path(result_path)
        / f'model_{split_idx}'
        / f'{k_state}_states'
        / f'reward_group_{reward_group}'
        / model_name          # 'full' or 'drop_<feat>'
        / f'iter_{instance_idx}'
    )
    results_dir_iter.mkdir(parents=True, exist_ok=True)

    # ----------------------------------
    # Weight initialisation
    # ----------------------------------
    # For k > 1: load the 1-state full-model weights and tile + perturb
    # For k = 1: start from random (SSM default)
    if k_state > 1:
        path_to_init_weights = (
            Path(result_path)
            / f'model_{split_idx}'
            / '1_states'
            / f'reward_group_{reward_group}'

            / model_name
            / f'iter_{instance_idx}'
            / 'global_fit_glmhmm_results.npz'
        )
        if not path_to_init_weights.exists():
            logger.warning(
                f'1-state init weights not found at {path_to_init_weights}. '
                f'Falling back to random init.'
            )
            init_weights = None
        else:
            base_weights = np.load(path_to_init_weights, allow_pickle=True)['arr_0'].item()['weights']
            noisy_base   = add_noise_to_weights(base_weights, noise_level=config.HMM_PARAMS['noise_level'])
            init_weights = np.repeat(noisy_base, k_state, axis=0)
    else:
        init_weights = None

    # ----------------------------------
    # Build and fit GLM-HMM
    # ----------------------------------
    glmhmm = build_glmhmm(
        n_states=k_state,
        input_dim=len(features),
        prior_sigma=config.HMM_PARAMS['prior_sigma'],
        prior_alpha=config.HMM_PARAMS['prior_alpha'],
        kappa=config.HMM_PARAMS['kappa'],
    )
    if init_weights is not None:
        glmhmm.observations.params = init_weights

    #TODO: create masks for missing values (dlc/real nans) or for dlc impute with median
    #masks_train = create_masks(input_train)
    ll_train = glmhmm.fit(
        output_train, inputs=input_train,
        method='em',
        num_iters=config.HMM_PARAMS['n_train_iters'],
        tolerance=config.HMM_PARAMS['tolerance'],
    )

    recovered_weights = glmhmm.observations.params
    transition_matrix = glmhmm.transitions.transition_matrix
    ll_test = glmhmm.log_likelihood(output_test, input_test, None, None)

    # ----------------------------------
    # Predictive accuracy
    # ----------------------------------
    def _eval(outputs, inputs):
        posteriors  = np.concatenate(get_expected_states(glmhmm, outputs=outputs, inputs=inputs), axis=0)
        pred_labels = get_predicted_labels(glmhmm, inputs=inputs, posteriors=posteriors)
        acc, balanced_acc = calculate_predictive_accuracy(outputs, pred_labels)
        return pred_labels, acc, balanced_acc

    pred_labels_train, pred_acc_train, balanced_pred_acc_train = _eval(output_train, input_train)
    pred_labels_test,  pred_acc_test, balanced_pred_acc_test  = _eval(output_test,  input_test)

    logger.info(f'  Train accuracy: {pred_acc_train:.4f} | Test accuracy: {pred_acc_test:.4f}')

    # ----------------------------------
    # Save results
    # ----------------------------------
    result_dict = {
        'split_idx':            split_idx,
        'n_states':             k_state,
        'instance_idx':         instance_idx,
        'model_name':           model_name,
        'reward_group': reward_group,
        'features':             features,
        'weights':              recovered_weights,
        'transition_matrix':    transition_matrix,
        'll_train':             ll_train,
        'll_test':              ll_test,
        'output_train_labels':  output_train,
        'output_test_labels':   output_test,
        'output_train_preds':   pred_labels_train,
        'output_test_preds':    pred_labels_test,
        'predictive_acc_train': pred_acc_train,
        'predictive_acc_test':  pred_acc_test,
        'balanced_predictive_acc_train': balanced_pred_acc_train,
        'balanced_predictive_acc_test': balanced_pred_acc_test,
    }
    np.savez(results_dir_iter / 'global_fit_glmhmm_results.npz', result_dict)

    # ----------------------------------
    # Plots
    # ----------------------------------
    plot_model_glm_weights(
        model=glmhmm, init_weights=init_weights, feature_names=features,
        save_path=results_dir_iter,
        file_name='global_weights', suffix=None, file_types=['pdf', 'eps'],
    )
    plot_model_transition_matrix(
        model=glmhmm, save_path=results_dir_iter,
        file_name='transition_matrix', suffix=None, file_types=['pdf', 'eps'],
    )

    return result_dict

def load_result(path):
    try:
        with np.load(path, allow_pickle=True) as data:
            arr = data["arr_0"]
            return arr[()]   # safer scalar extraction
    except Exception as e:
        print(f"\nFAILED FILE:\n{path}\nERROR:\n{e}\n")
        return None

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':

    experimenter = 'Axel_Bisi'
    dataset_path = Path(
        f'\\\\sv-nas1.rcp.epfl.ch\\Petersen-Lab\\analysis\\{experimenter}'
        f'\\combined_results\\glm_hmm\\datasets_combined_mvt'
    )
    if config.TRIAL_TYPES == 'whisker_trial':
        result_path = Path(
            f'\\\\sv-nas1.rcp.epfl.ch\\Petersen-Lab\\analysis\\{experimenter}'
            f'\\combined_results\\glm_hmm\\global_glmhmm_mvt_whisker_trials'
        )
    else:
        result_path = Path(
            f'\\\\sv-nas1.rcp.epfl.ch\\Petersen-Lab\\analysis\\{experimenter}'
            f'\\combined_results\\glm_hmm\\global_glmhmm_mvt'
        )
    result_path.mkdir(parents=True, exist_ok=True)

    N_SPLITS    = config.N_SPLITS
    N_STATES    = config.N_STATES
    N_INSTANCES = config.N_INSTANCES

    # Check multicolinearity of features
    sample_split_path = Path(dataset_path, f'dataset_0')
    data_train = pickle.load(open(sample_split_path / 'data_train.pkl', 'rb'))
    from multicollinearity_utils import check_multicollinearity, plot_multicollinearity
    results = check_multicollinearity(data_train, config.FEATURES)
    plot_multicollinearity(results, save_path=os.path.join(result_path, 'multicollinearity'))


    # Build all feature sets: full model + one leave-one-out set per feature
    print('Feature set using features:', config.FEATURES)
    feature_sets = build_feature_sets(config.FEATURES, config.TRIAL_TYPES)
    #for model_name, features in feature_sets.items():
    #    print(f'  {model_name}: {features}') # line by line

    reward_groups = [1, 0]
    reward_groups = config.REWARD_GROUPS

    def _make_tasks(k_states):
        return [
            (split_idx, k_state, instance_idx, model_name, features,
             dataset_path, result_path ,reward_group)
            for k_state, split_idx, instance_idx, (model_name, features), reward_group
            in product(
                k_states,
                range(N_SPLITS),
                range(N_INSTANCES),
                feature_sets.items(),
                reward_groups,

            )
        ]

    tasks_k1  = _make_tasks([1])
    tasks_kgt = _make_tasks(range(2, N_STATES + 1))
    logger.info(f'Total tasks to run: {len(tasks_k1) + len(tasks_kgt)} '
                f'({len(tasks_k1)} k=1 first, then {len(tasks_kgt)} k>1)')

    start_time = time.time()
    WORKERS = 30
    with Pool(processes=WORKERS) as pool:
        # k=1 must complete entirely before k>1 starts, as k>1 loads k=1 weights for init
        results_k1  = pool.starmap(process_single_fit, tasks_k1)
        results_kgt = pool.starmap(process_single_fit, tasks_kgt)
    all_results = results_k1 + results_kgt

    ## If results exist, fetch them, load them and save them as pickle
    #paths_from_tasks = [Path(result_path, f'model_{split_idx}', f'{k_state}_states', model_name, f'iter_{instance_idx}', 'global_fit_glmhmm_results.npz')
    #                    for split_idx, k_state, instance_idx, model_name, _, _, _ in tasks_k1 + tasks_kgt]
    #existing_paths = [p for p in paths_from_tasks if p.exists()]
    #logger.info(f'Found {len(existing_paths)} existing result files. Loading them...')
    #all_results = []
    ## Using multiprocessing, load results
    #with ThreadPool(processes=max(1, (os.cpu_count() or 1) - 2)) as pool:
    #    all_results = pool.map(load_result, existing_paths)

    # To dataframe
    res_df = pd.DataFrame(all_results)
    #cols_to_change = ['weights', 'transition_matrix', 'output_train_labels', 'output_test_labels',
    #                  'output_train_preds', 'output_test_preds', 'll_test','ll_train', 'features']
    #cols_to_drop = [ 'output_train_labels', 'output_test_labels',
    #                  'output_train_preds', 'output_test_preds' ]
    #for col in cols_to_change:
    #    res_df[col] = res_df[col].apply(lambda x: np.array(x) if isinstance(x, np.ndarray) else x)
    #res_df.drop(columns=cols_to_drop, inplace=True)


    # Save as pickle
    res_df.to_pickle(result_path / 'global_fit_glmhmm_results.pkl')
    #res_df.to_hdf(result_path / 'global_fit_glmhmm_results.h5', key='df', mode='w')


    logger.info(f'Script finished in {time.time() - start_time:.2f} seconds.')
