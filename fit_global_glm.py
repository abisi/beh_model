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


from data_utils import create_data_lists
from plotting_utils import remove_top_right_frame, save_figure_to_files

from GLM import glm
from glm_utils import calculate_predictive_acc_glm


if __name__ == '__main__':


    # Load train and test datasets
    experimenter = 'Axel_Bisi'
    dataset_path = os.path.join('\\\\sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'analysis', experimenter, r'beh_model\global_glm\datasets')
    result_path = os.path.join('\\\\sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'analysis', experimenter, r'beh_model\global_glm\models')

    for split_idx in range(1):
        dataset_split_path = os.path.join(dataset_path, 'dataset_{}'.format(split_idx))
        result_split_path = os.path.join(result_path, 'model_{}'.format(split_idx))
        data_train = pickle.load(open(os.path.join(dataset_split_path, 'data_train.pkl'), 'rb'))
        data_test = pickle.load(open(os.path.join(dataset_split_path, 'data_test.pkl'), 'rb'))


        # Select features
        features = ['prev_choice', 'stimulus_type', 'prev_stimulus_type',
                        'auditory', 'prev_auditory', 'whisker', 'prev_whisker',
                        'reward_given', 'prev_reward_given']
        features = ['prev_choice',
                    'stimulus_type',
                    'prev_stimulus_type',
                    'prev_whisker',
                    'prev_reward_given']


        # Create design matrices, split session-wise
        input_train, output_train, input_test, output_test = create_data_lists(data_train, data_test, features=features)
        print('input_train shape:', input_train[0].shape)
        print('output_train shape:', output_train[0].shape)


        # Repeat for multiple instances
        n_instances = 1
        ll_train_arr = np.zeros(n_instances)
        ll_test_arr = np.zeros(n_instances)
        recovered_weights_arr = np.zeros((n_instances, len(features)))

        for i in range(n_instances):

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
            #ll_train, recovered_weights = global_glm.fit_glm(datas=[output_train],inputs=[input_train], masks=None, tags=None)

            # Fit GLM-HMM with MLE
            n_iters = 300  # max EM iterations (fitting will stop earlier if LL below tolerance)
            global_glm.fit(output_train, inputs=input_train, method='em', num_iters=n_iters, tolerance=10**-4)
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
