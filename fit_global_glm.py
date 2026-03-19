#! /usr/bin/env/python3
"""
@author: Axel Bisi
@project: beh_model
@file: fit_global_glm.py
@time: 12/8/2023 3:28 PM
"""


# Imports
import os
import numpy as np
import numpy.random as npr
npr.seed(0)
import pickle
import matplotlib.pyplot as plt
import ssm
from pathlib import Path
import time

from data_utils import create_data_lists
from plotting_utils import remove_top_right_frame, save_figure_to_files
from multicollinearity_utils import check_multicollinearity, plot_multicollinearity

from GLM import glm
from glm_utils import calculate_predictive_acc_glm
import config

if __name__ == '__main__':


    start_time = time.time()

    experimenter = 'Axel_Bisi'
    dataset_path = Path(
        f'\\\\sv-nas1.rcp.epfl.ch\\Petersen-Lab\\analysis\\{experimenter}'
        f'\\combined_results\\glm_hmm\\datasets_combined_mvt'
    )
    if config.TRIAL_TYPES == 'whisker_trial':
        result_path = Path(
            f'\\\\sv-nas1.rcp.epfl.ch\\Petersen-Lab\\analysis\\{experimenter}'
            f'\\combined_results\\glm_hmm\\global_glm_mvt_whisker_trials'
        )
    else:
        result_path = Path(
            f'\\\\sv-nas1.rcp.epfl.ch\\Petersen-Lab\\analysis\\{experimenter}'
            f'\\combined_results\\glm_hmm\\global_glm_mvt'
        )
    result_path.mkdir(parents=True, exist_ok=True)

    N_SPLITS    = config.N_SPLITS
    N_INSTANCES = config.N_INSTANCES

    # Check multicolinearity of features
    sample_split_path = Path(dataset_path, f'dataset_0')
    data_train = pickle.load(open(sample_split_path / 'data_train.pkl', 'rb'))

    results = check_multicollinearity(data_train, config.FEATURES)
    plot_multicollinearity(results, save_path=os.path.join(result_path, 'multicollinearity'))


    start_time = time.time()
    for split_idx in range(N_SPLITS):
        dataset_split_path = os.path.join(dataset_path, 'dataset_{}'.format(split_idx))
        result_split_path = os.path.join(result_path, 'model_{}'.format(split_idx))
        data_train = pickle.load(open(os.path.join(dataset_split_path, 'data_train.pkl'), 'rb'))
        data_test = pickle.load(open(os.path.join(dataset_split_path, 'data_test.pkl'), 'rb'))


        # Select features
        features = config.FEATURES
        print('Fitting global GLM with features: {}'.format(features))

        # Create design matrices, split session-wise
        input_train, output_train, input_test, output_test = create_data_lists(data_train, data_test, features=features)

        # Repeat for multiple instances
        ll_train_arr = np.zeros(N_INSTANCES)
        ll_test_arr = np.zeros(N_INSTANCES)
        recovered_weights_arr = np.zeros((N_INSTANCES, len(features)))

        for i in range(N_INSTANCES):

            # Folder for iteration
            results_dir_iter = os.path.join(result_split_path, 'iter_{}'.format(i))
            if not os.path.exists(results_dir_iter):
                os.makedirs(results_dir_iter)

            # Initialize one-state GLM
            n_states = 1
            obs_dim = 1     #dimensionality of what is being modelled, here only choice
            num_categories = 2
            input_dim = len(features)
            global_glm = ssm.HMM(n_states,
                                 obs_dim,
                                 input_dim,
                                 observations="input_driven_obs",
                                 observation_kwargs=dict(C=num_categories),
                                 transitions="standard", verbose=5)

            #global_glm = glm(M=input_dim, C=num_categories)
            #ll_train, recovered_weights = global_glm.fit_glm(datas=output_train,inputs=input_train, masks=None, tags=None)

            # Fit GLM-HMM with MLE
            n_iters = config.HMM_PARAMS['n_train_iters']  # max EM iterations (fitting will stop earlier if LL below tolerance)
            tol = config.HMM_PARAMS['tolerance']
            global_glm.fit(output_train, inputs=input_train, method='em', num_iters=n_iters, tolerance=tol, verbose=5)
            ll_train = global_glm.log_likelihood(output_train, input_train, None, None)
            recovered_weights = np.squeeze(global_glm.observations.params[0])

            # Retrieve and plot glm_weights
            fig, ax = plt.subplots(1,1,figsize=(5,5), dpi=200, facecolor='w', edgecolor='k')
            remove_top_right_frame(ax)
            ax.plot(range(input_dim),
                    recovered_weights,
                    marker='o',
                    linestyle='-',
                    lw=1.5)
            ax.set_xticks(range(input_dim), features, fontsize=12, rotation=45)
            ax.set_ylabel('Weight', fontsize=15)
            ax.set_xlabel('Covariate', fontsize=15)
            ax.set_title('Global weights', fontsize=15)
            ax.axhline(y=0, color="k", alpha=0.5, ls="--")

            # Save weight figure
            save_figure_to_files(fig, results_dir_iter, 'global_weights', suffix=None, file_types=['png','eps'], dpi=300)

            # Test sessions: one-state all-animal model
            ll_test = global_glm.log_likelihood(output_test, input_test, None, None)

            # Calculate predictive accuracy
            #new_glm = ssm.HMM(n_states,
            #                    obs_dim,
            #                    input_dim,
            #                    observations="input_driven_obs",
            #                    observation_kwargs=dict(C=num_categories),
            #                    transitions="standard", verbose=5)
            #new_glm.observations.params = global_glm.observations.params
            # Get logits
            #new_glm.observations.calculate_logits = lambda input: np.dot(new_glm.observations.params[0], input.T).T
            #acc_train = calculate_predictive_acc_glm(glm_weights=recovered_weights, inpt=np.array(input_train), y=np.array(output_train), idx_to_exclude=None)
            #acc_test = calculate_predictive_acc_glm(glm_weights=recovered_weights, inpt=input_test, y=output_test, idx_to_exclude=None)


            # Format dictionary of results, then save
            result_dict = {
                'covariates': features,
                'weights': recovered_weights,
                'll_train': ll_train,
               'll_test': ll_test,
            }
            print('Instance {} - ll_train: {}, ll_test: {}'.format(i, ll_train, ll_test))
            np.savez(os.path.join(results_dir_iter, 'global_fit_results.npz'), result_dict, allow_pickle=True)

            # Append results to arrays
            ll_train_arr[i] = ll_train
            ll_test_arr[i] = ll_test
            recovered_weights_arr[i,:] = recovered_weights

        # Make dictionary of results of all instances
        results_dict = {
            'covariates': features,
            'll_train_arr': ll_train_arr,
            'll_test_arr': ll_test_arr,
            'weights': recovered_weights
        }
        np.savez(os.path.join(result_split_path, 'global_fit_results_all_iters.npz'), results_dict, allow_pickle=True)
