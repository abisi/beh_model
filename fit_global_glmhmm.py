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
import pandas as pd
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt

import ssm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from data_utils import create_data_lists
from plotting_utils import remove_top_right_frame, save_figure_to_files, plot_model_glm_weights, \
    plot_model_transition_matrix, plot_single_session_predictions, plot_single_session_posterior_states
from utils import add_noise_to_weights, get_expected_states, get_predicted_labels
from utils import calculate_predictive_accuracy

if __name__ == '__main__':


    # Paths
    experimenter = 'Axel_Bisi'
    dataset_path = os.path.join('\\\\sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'analysis', experimenter, r'beh_model\datasets')
    model_weight_path = os.path.join('\\\\sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'analysis', experimenter, r'beh_model\global_glm\models')
    result_path = os.path.join('\\\\sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'analysis', experimenter, r'beh_model\global_glmhmm\models')

    # Set run parameters
    N_SPLITS = 10
    N_STATES = 5
    N_INSTANCES = 10

    # Init. result storage
    result_dict_list = []

    logger.info("Starting training...")

    # Iterate over dataset splits
    for split_idx in range(N_SPLITS):

        # Load train and test datasets
        dataset_split_path = os.path.join(dataset_path, 'dataset_{}'.format(split_idx))
        model_weight_split_path = os.path.join(model_weight_path, 'model_{}'.format(split_idx))
        result_split_path = os.path.join(result_path, 'model_{}'.format(split_idx))
        data_train = pickle.load(open(os.path.join(dataset_split_path, 'data_train.pkl'), 'rb'))
        data_test = pickle.load(open(os.path.join(dataset_split_path, 'data_test.pkl'), 'rb'))

        split_ll_train_res = []
        split_ll_test_res = []
        split_weights_res = []

        # Show content of data
        logger.info('Number of sessions per mice:')
        logger.info(list(data_train.groupby('mouse_id').day.nunique()))
        logger.info('Number of trials per mice:')
        logger.info(list(data_train.groupby(['mouse_id', 'day']).trial_id.max().groupby('mouse_id').sum()))
        logger.info('Number of mice per reward group:')
        logger.info(list(data_train.groupby('reward_group').mouse_id.nunique()))

        # Select features
        features = ['prev_choice',
                    'auditory',
                    'prev_stim_auditory',
                    'whisker',
                    'prev_stim_whisker',
                    'prev_stim_reward_given']

        # Create design matrices, split session-wise
        input_train, output_train, input_test, output_test = create_data_lists(data_train, data_test, features=features)

        # Repeat for different number of states
        prior_sigma = 2
        prior_alpha = 2

        for k_idx in range(1, N_STATES+1):

            logger.info('Fitting global GLM-HMM with {} states...'.format(k_idx))
            # Create results directory for n_state
            model_split_n_state_path = os.path.join(result_split_path, '{}_states'.format(k_idx))

            for instance_idx in range(N_INSTANCES): #TODO: this has no effect because no noise
                # TODO: Across instances, must fix permutation of states

                # Create result folder per iteration
                results_dir_iter = os.path.join(model_split_n_state_path, 'iter_{}'.format(instance_idx))
                if not os.path.exists(results_dir_iter):
                    os.makedirs(results_dir_iter)

                # Load global fit single-state weights #TODO: think whether to load average of weights? they seem the same over dataset splits
                global_results = np.load(os.path.join(model_weight_split_path, 'iter_{}'.format(instance_idx), 'global_fit_results.npz'),
                                         allow_pickle=True)['arr_0']
                global_results = global_results.item()
                global_weights = global_results['weights']

                # Make noisy GLM weights
                # noisy_weights = add_noise_to_weights(global_weights, noise_level=0.2)

                # Make noisy transition matrix
                #sigma_trans = 0.05 * np.ones((k_idx, k_idx))
                #gauss_noise_transition = np.random.normal(0, sigma_trans)
                #transition_matrix = 0.95 * np.identity(k_idx) + gauss_noise_transition

                # Normalize transition matrix (sum across rows) to get probabilities
                #transition_matrix = transition_matrix / transition_matrix.sum(axis=1)[:, np.newaxis]

                # Initialize k_idx-state GLM-HMM
                n_states = k_idx
                obs_dim = 1
                num_categories = 2
                input_dim = len(features)
                glmhmm = ssm.HMM(n_states,
                                 obs_dim,
                                 input_dim,
                                 observations="input_driven_obs",
                                 observation_kwargs=dict(C=num_categories, prior_sigma=prior_sigma),
                                 transitions="sticky",
                                 transition_kwargs=dict(alpha=prior_alpha, kappa=0)
                                 )
                #noisy_weights = np.expand_dims(noisy_weights, axis=(0,1))
                # Now repeat along the first dimension
                #noisy_weights = np.repeat(noisy_weights, k_idx, axis=0)
                #glmhmm.observations.params = noisy_weights #TODO: resolve whether I do that

                # Fit GLM-HMM with MAP
                n_train_iters = 200  # max EM iterations (fitting will stop earlier if LL below tolerance)
                tol = 10 ** -3
                ll_train = glmhmm.fit(output_train, inputs=input_train, method='em', num_iters=n_train_iters, tolerance=tol)
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

                logger.info('Predictive accuracy on train data: {}'.format(pred_acc_train))
                logger.info('Predictive accuracy on test data: {}'.format(pred_acc_test))

                # --------------------------------------
                # Plot GLM weights and transition matrix
                # --------------------------------------

                plot_model_glm_weights(model=glmhmm, init_weights=None, feature_names=features, save_path=results_dir_iter,
                                       file_name='global_weights', suffix=None, file_types=['png','svg'])
                plot_model_transition_matrix(model=glmhmm, save_path=results_dir_iter,
                                             file_name='transition_matrix' , suffix=None, file_types=['png','svg'])


                # Plot for each session
                if split_idx == 0 and instance_idx == 0:
                    plot_sessions = True
                else:
                    plot_sessions = False

                # ------------------------------------------
                # Match session data and preds with metadata #TODO: WIP for chosen model only?
                # ------------------------------------------

                #session_res_df = pd.DataFrame()
                #session_res_df['session_id'] = data_train['session_id']
                #session_res_df['mouse_id'] = data_train['mouse_id']
                #session_res_df['behavior'] = data_train['behavior']
                #session_res_df['day'] = data_train['day']
                #session_res_df['output_train'] = [output_train]
                #session_res_df['pred_labels_train'] = [pred_labels_train]
                #session_res_df['output_test'] = [output_test]
                #session_res_df['pred_labels_test'] = pred_labels_test
                #session_res_df['posterior_probs_train'] = expected_states_train #split session-wise
                #session_res_df['posterior_probs_test'] = expected_states_test


                # ------------------------------
                # Plot single session predictions
                # -------------------------------

                fig_folder = os.path.join(results_dir_iter, 'pred_train')
                #plot_single_session_predictions(output_train, pred_labels_train, fig_folder, 'pred')

                fig_folder = os.path.join(results_dir_iter, 'pred_test')
                #plot_single_session_predictions(output_test, pred_labels_test, fig_folder, 'pred')

                # ----------------------------------------------------
                # Plot posterior states probabilities for each session #TODO: modularize?
                # ----------------------------------------------------

                plot_sessions = False
                if plot_sessions:
                    posterior_probs_train = get_expected_states(glmhmm, outputs=output_train, inputs=input_train)

                    fig_folder = os.path.join(results_dir_iter, 'posterior_states')
                    #plot_single_session_posterior_states(posterior_probs_train, fig_folder,
                    #                                     file_name='posterior_states', suffix=None,
                    #                                     file_types=['png', 'svg'])

                    # TODO: as a function, keep for now
                    for sess_id in range(len(output_train)):

                        # Create new figure
                        fig, axs = plt.subplots(2, 1, figsize=(30, 5), dpi=80, facecolor='w', edgecolor='k',
                                                gridspec_kw={'height_ratios': [10, 1]}, sharex=True)

                        # Find corresponding data in original dataframe
                        session_id_name = data_train['session_id'].unique()[sess_id]
                        data_session = data_train[data_train.session_id == session_id_name]

                        # Set title
                        mouse_name = data_train[data_train.session_id == session_id_name].mouse_id.unique()[0]
                        behavior = data_train[data_train.session_id == session_id_name].behavior.unique()[0]
                        day = data_train[data_train.session_id == session_id_name].day.unique()[0]
                        title_name = '{}, {}, {}'.format(mouse_name, behavior, day)
                        axs[0].set_title(title_name)

                        for k in range(n_states):
                            axs[0].plot(posterior_probs_train[sess_id][:, k], label="State " + str(k + 1), lw=2)

                        axs[0].set_ylim((-0.01, 1.01))
                        axs[0].set_xlabel("trial #", fontsize=15)
                        axs[0].set_ylabel("p(state)", fontsize=15)
                        axs[0].legend(loc="upper right", fontsize=12)

                        # Plot corresponding trial outcome
                        color_map = {'wh': 'forestgreen', 'wm': 'crimson',
                                     'ah': 'mediumblue', 'am': 'lightblue',
                                     'cr': 'lightgrey', 'fa': 'k'}
                        data_session['trial_type'] = np.nan #TODO: check if this works
                        data_session.loc[(data_session.stimulus_type == -1) & (data_session.choice == 1), 'trial_type'] = 'ah'
                        data_session.loc[(data_session.stimulus_type == -1) & (data_session.choice == 0), 'trial_type'] = 'am'
                        data_session.loc[(data_session.stimulus_type == 1) & (data_session.choice == 1), 'trial_type'] = 'wh'
                        data_session.loc[(data_session.stimulus_type == 1) & (data_session.choice == 0), 'trial_type'] = 'wm'
                        data_session.loc[(data_session.stimulus_type == 0) & (data_session.choice == 1), 'trial_type'] = 'fa'
                        data_session.loc[(data_session.stimulus_type == 0) & (data_session.choice == 0), 'trial_type'] = 'cr'


                        perf_map = {0: 'wm', 2: 'wh', 1: 'am', 3: 'ah', 4: 'cr', 5: 'fa'}
                        perf_map = {v: k for k, v in perf_map.items()}
                        perf_color = {
                            0: 'crimson',
                            1: 'lightblue',
                            2: 'forestgreen',
                            3: 'mediumblue',
                            4: 'lightgrey',
                            5: 'k'}
                        #data_session['perf'] = data_session['trial_type'].map(perf_map)
                        # Same code but without SettingWithCopyWarning
                        data_session = data_session.assign(perf=data_session['trial_type'].map(perf_map))     #TODO: check if this works
                        cmap = mpl.colors.LinearSegmentedColormap.from_list('perf_cmap', list(perf_color.values()), 6)

                        # define the bins and normalize
                        bounds = np.linspace(0, 6, 7)
                        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
                        axs[1].pcolor(np.expand_dims(data_session['perf'].values, axis=0), cmap=cmap, norm=norm,
                                      edgecolors=None, linewidths=0)

                        # Save figure
                        file_name = 'session_{}_posterior_states'.format(session_id_name)
                        save_figure_to_files(fig, save_path=fig_folder, file_name=file_name, suffix=None, file_types=['png','svg'], dpi=300)
                        plt.close()


                # Format dictionary of results, then save
                result_dict = {
                    'split_idx': split_idx,
                    'n_states': k_idx,
                    'instance_idx': instance_idx,
                    'features': features,
                    'weights': recovered_weights,
                    'transition_matrix': transition_matrix,
                    'll_train': ll_train,
                     'll_test': ll_test,
                    'output_train_preds:': pred_labels_train,
                    'output_test_preds': pred_labels_test,
                    'predictive_acc_train': pred_acc_train,
                    'predictive_acc_test': pred_acc_test,

                }
                result_dict_list.append(result_dict)

                np.savez(os.path.join(results_dir_iter, 'global_fit_glmhmm_results.npz'), result_dict)

            logger.info('Done with {} states'.format(k_idx))

        logger.info('Done with data split {}'.format(split_idx))

    # ----------------
    # Save all results
    # ----------------

    res_df = pd.DataFrame(result_dict_list)
    res_df.to_hdf(os.path.join(result_path, 'global_fit_glmhmm_results.h5'), key='df', mode='w')







