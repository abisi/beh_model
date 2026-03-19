#! /usr/bin/env/python3
"""
@author: Axel Bisi (converted from notebooks by Claude)
@project: beh_model
@file: plot_glmhmm_results.py
@description: Script to plot global and single-mouse GLM-HMM model performance
"""

import os
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import pandas as pd
import scipy as sp
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import euclidean, cityblock
from pathlib import Path

import plotting_utils
# Import custom utilities
from plotting_utils import remove_top_right_frame, save_figure_to_files, lighten_color
from utils import align_states_across_subjects

# Set random seed
npr.seed(0)

# ============================================================================
# REWARD GROUP CONSTANTS
# ============================================================================
REWARD_GROUP_COLORS = {0: 'crimson', 1: 'forestgreen'}
REWARD_GROUP_COLORS = {'R-': 'crimson', 'R+': 'forestgreen'}
REWARD_GROUP_NAMES  = {0: 'R-', 1: 'R+'}
REWARD_GROUP_NAMES  = {'R-': 'R-', 'R+': 'R+'}


# ============================================================================
# GLOBAL MODEL PERFORMANCE FUNCTIONS
# ============================================================================

def ll_to_bpt(ll, ll_null, n_trials): #TODO: it this correct for ll_null? fix potentially
    """Convert log-likelihood to bits per trial."""
    return (ll - ll_null) / (np.log(2) * n_trials)


def load_global_models(model_parent_path, model_types_filter=None):
    """
    Load global GLM-HMM model results.
    
    Parameters:
    -----------
    model_parent_path : str
        Path to parent directory containing model results
    model_types_filter : list, optional
        List of model types to include (e.g., ['full_models', 'wo_bias_models'])
        
    Returns:
    --------
    pd.DataFrame : Combined results from all models
    """

    # Load pickle file
    all_models_res_df = pd.read_pickle(os.path.join(model_parent_path, 'global_fit_glmhmm_results.pkl'))
    all_models_res_df['model_type'] = all_models_res_df['model_name']

    # Extract final training log-likelihood
    all_models_res_df['ll_train_final'] = all_models_res_df['ll_train'].apply(lambda x: x[-1] if isinstance(x, (list, np.ndarray)) else x)
    
    print('\nDataset summary:')
    print(f"Models: {all_models_res_df.model_type.unique()}")
    # For each model, show unique features
    for model in all_models_res_df.model_type.unique():
        features = all_models_res_df[all_models_res_df['model_type'] == model]['features'].iloc[0]
        print(f"Model: {model}, features: {features}")
    print(f"# models / type:\n{all_models_res_df.groupby('model_type').size()}")
    print(f"# states / type:\n{all_models_res_df.groupby('model_type')['n_states'].unique()}")
    print(f"# data splits / type:\n{all_models_res_df.groupby('model_type')['split_idx'].unique()}")
    
    return all_models_res_df


def compute_bits_per_trial(all_models_res_df):
    """
    Compute bits per trial from log-likelihood using Bernoulli null model.
    
    Parameters:
    -----------
    all_models_res_df : pd.DataFrame
        DataFrame containing model results
        
    Returns:
    --------
    pd.DataFrame : Input DataFrame with added 'bpt_train' and 'bpt_test' columns
    """
    # Get total number of trials
    n_trials_train = all_models_res_df.loc[0, 'output_train_preds']
    if isinstance(n_trials_train, (list, np.ndarray)):
        n_trials_train = sum([np.array(i).shape[0] for i in n_trials_train])
    
    n_trials_test = all_models_res_df.loc[0, 'output_test_preds']
    if isinstance(n_trials_test, (list, np.ndarray)):
        n_trials_test = sum([np.array(i).shape[0] for i in n_trials_test])
    
    # Calculate lick probability for null model
    p_lick_train = all_models_res_df.loc[0, 'output_train_preds']
    if isinstance(p_lick_train, (list, np.ndarray)):
        p_lick_train = np.mean([np.mean(i) for i in p_lick_train])
    
    print(f'p_lick_train: {p_lick_train}')
    
    # Calculate log-likelihood of the null model
    ll_null_train = n_trials_train * (p_lick_train * np.log(p_lick_train) + (1 - p_lick_train) * np.log(1 - p_lick_train))
    ll_null_test = n_trials_test * (p_lick_train * np.log(p_lick_train) + (1 - p_lick_train) * np.log(1 - p_lick_train))

    
    # Convert to bits per trial
    all_models_res_df['bpt_train'] = all_models_res_df['ll_train_final'].apply(
        lambda x: ll_to_bpt(x, ll_null_train, n_trials_train))
    all_models_res_df['bpt_test'] = all_models_res_df['ll_test'].apply(
        lambda x: ll_to_bpt(x, ll_null_test, n_trials_test))
    
    return all_models_res_df

def plot_global_model_metrics(all_models_res_df, result_figure_path):

    os.makedirs(result_figure_path, exist_ok=True)

    palette = {
        'full': 'k',
        'bias_only': 'gold',
        #'bias_history_only': 'coral',
        'drop_bias': sns.color_palette("Greys")[1],
        'drop_history': sns.color_palette("Greys")[2],
        #'drop_trial_type': sns.color_palette("Greys")[3],
        'drop_mvt': sns.color_palette("Greys")[4],
        #'drop_auditory': sns.color_palette("Blues")[1],
        #'drop_time_since_last_auditory_stim': sns.color_palette("Blues")[2],
        'drop_time_since_last_auditory_lick': sns.color_palette("Blues")[3],
        #'drop_whisker': sns.color_palette("Oranges")[1],
        #'drop_time_since_last_whisker_stim': sns.color_palette("Oranges")[2],
        'drop_time_since_last_whisker_lick': sns.color_palette("Oranges")[3],
        'drop_jaw_distance': sns.color_palette("RdPu")[1],
        #'drop_nose_norm_distance': sns.color_palette("RdPu")[2],
        'drop_whisker_angle': sns.color_palette("RdPu")[3],
        'drop_pupil_area': sns.color_palette("RdPu")[4],
        'mvt_only': sns.color_palette("RdPu")[5],
    }

    sns.set_style(rc={"lines.linewidth": 0.4})

    # ---------------------------
    # Helper plotting function
    # ---------------------------

    def plot_metric_subset(df, model_subset, file_name, reward_group=None):

        df = df[df['model_type'].isin(model_subset)]

        print('Dataset len', len(df))
        print('Reward groups', df['reward_group'].unique())

        # --- hue strategy ---
        # Combined view (reward_group=None): reward_group as hue, 2-colour palette.
        # Per-group view (reward_group given): filter to that group, model_type as hue.
        if reward_group is not None:
            df = df[df['reward_group'] == reward_group].copy()
            hue_col    = 'model_type'
            hue_order  = model_subset
            plot_palette = palette
            file_suffix  = f'_rg{reward_group}'
            title_suffix = f' – {REWARD_GROUP_NAMES[reward_group]}'
        else:
            hue_col    = 'reward_group'
            hue_order  = [0, 1]
            hue_order  = ['R+', 'R-']
            plot_palette = REWARD_GROUP_COLORS
            file_suffix  = ''
            title_suffix = ''

        figsize = (8,8)
        fig, axs = plt.subplots(4, 2, figsize=figsize, dpi=500, sharey='row', constrained_layout=True)
        for ax in axs.flat:
            ax.yaxis.set_tick_params(labelleft=True)

        axs[0,0].set_title('Train data' + title_suffix)
        sns.pointplot(
            x='n_states',
            y='ll_train_final',
            data=df,
            ax=axs[0,0],
            estimator=np.mean,
            errorbar='se',
            hue=hue_col,
            palette=plot_palette,
            hue_order=hue_order
        )

        axs[0,1].set_title('Test data' + title_suffix)
        sns.pointplot(
            x='n_states',
            y='ll_test',
            data=df,
            ax=axs[0,1],
            estimator=np.mean,
            errorbar='se',
            hue=hue_col,
            palette=plot_palette,
            hue_order=hue_order
        )
        axs[0,0].set_ylabel('Log-likelihood')
        axs[0,0].set_xlabel('Number of states')
        axs[0,1].set_ylabel('Log-likelihood')
        axs[0,1].set_xlabel('Number of states')

        axs[1,0].set_title('Train data' + title_suffix)
        sns.pointplot(
            x='n_states',
            y='predictive_acc_train',
            data=df,
            ax=axs[1,0],
            estimator=np.mean,
            errorbar='se',
            hue=hue_col,
            palette=plot_palette,
            hue_order=hue_order
        )

        axs[1,1].set_title('Test data' + title_suffix)
        sns.pointplot(
            x='n_states',
            y='predictive_acc_test',
            data=df,
            ax=axs[1,1],
            estimator=np.mean,
            errorbar='se',
            hue=hue_col,
            palette=plot_palette,
            hue_order=hue_order
        )
        axs[1,0].set_ylabel('Predictive accuracy')
        axs[1,1].set_xlabel('Number of states')
        axs[1,1].set_ylabel('Predictive accuracy')
        axs[1,0].set_xlabel('Number of states')

        axs[2,0].set_title('Train data' + title_suffix)
        sns.pointplot(
            x='n_states',
            y='balanced_predictive_acc_train',
            data=df,
            ax=axs[2,0],
            estimator=np.mean,
            errorbar='se',
            hue=hue_col,
            palette=plot_palette,
            hue_order=hue_order
        )

        axs[2,1].set_title('Test data' + title_suffix)
        sns.pointplot(
            x='n_states',
            y='balanced_predictive_acc_test',
            data=df,
            ax=axs[2,1],
            estimator=np.mean,
            errorbar='se',
            hue=hue_col,
            palette=plot_palette,
            hue_order=hue_order
        )

        axs[2,0].set_ylabel('Balanced accuracy')
        axs[2,1].set_xlabel('Number of states')
        axs[2,1].set_ylabel('Balanced accuracy')
        axs[2,0].set_xlabel('Number of states')


        axs[3,0].set_title('Train data' + title_suffix)
        sns.pointplot(
            x='n_states',
            y='bpt_train',
            data=df,
            ax=axs[3,0],
            estimator=np.mean,
            errorbar='se',
            hue=hue_col,
            palette=plot_palette,
            hue_order=hue_order
        )

        axs[3,1].set_title('Test data' + title_suffix)
        sns.pointplot(
            x='n_states',
            y='bpt_test',
            data=df,
            ax=axs[3,1],
            estimator=np.mean,
            errorbar='se',
            hue=hue_col,
            palette=plot_palette,
            hue_order=hue_order
        )
        axs[3,0].set_ylabel('Log-likelihood (bits/trial)')
        axs[3,0].set_xlabel('Number of states')
        axs[3,1].set_ylabel('Log- (bits/trial)')
        axs[3,1].set_xlabel('Number of states')

        # chance baseline – compute per reward_group if combined, else for this group
        def _chance(labels_col, df_sub):
            v = df_sub.iloc[0][labels_col]
            if isinstance(v, (list, np.ndarray)):
                v = np.mean([np.mean(i) for i in v])
            return max(float(v), 1 - float(v))

        if reward_group is not None:
            # Single-group: one baseline value each
            chance_train = _chance('output_train_labels', df)
            chance_test  = _chance('output_test_labels',  df)
            axs[1,0].axhline(chance_train, color='k', linestyle='--')
            axs[1,1].axhline(chance_test,  color='k', linestyle='--')
        else:
            # Combined: draw one baseline line per reward_group using its colour
            for rg in ['R+', 'R-']:

                df_rg = df[df['reward_group'] == rg]
                if df_rg.empty:
                    continue
                chance_train = _chance('output_train_labels', df_rg)
                chance_test  = _chance('output_test_labels',  df_rg)
                axs[1,0].axhline(chance_train, color=REWARD_GROUP_COLORS[rg], linestyle='--', alpha=0.6,
                                 label=f'chance {REWARD_GROUP_NAMES[rg]}')
                axs[1,1].axhline(chance_test,  color=REWARD_GROUP_COLORS[rg], linestyle='--', alpha=0.6,
                                 label=f'chance {REWARD_GROUP_NAMES[rg]}')

        axs[2,0].axhline(y=0.5, color='k', linestyle='--')
        axs[2,1].axhline(y=0.5, color='k', linestyle='--')

        for ax in axs.flat:
            remove_top_right_frame(ax)
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            #ax.legend(loc='center left',title='model' if reward_group is not None else 'reward group',bbox_to_anchor=(0.7, 0.5),frameon=False,fontsize=6,title_fontsize=8)
            # Hide legend
            ax.legend_.remove() if ax.legend_ else None

        # Add a single legend on top of the whole figure as one line
        legend_elements = []
        for model in model_subset:
            if model in palette:
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', label=model, markerfacecolor=palette[model], markersize=5))
        if legend_elements:
            fig.legend(
                handles=legend_elements,
                loc='upper center',
                ncol=len(legend_elements)//2,
                frameon=False,
                fontsize=10,
                title_fontsize=8,
                bbox_to_anchor=(0., 1.02, 1., .102),
                mode="expand",
                borderaxespad=0.
            )

        fig.tight_layout()
        plt.tight_layout()
        # align y labels
        fig.align_ylabels()
        fig.align_xlabels()

        output_dir = os.path.join(result_figure_path, 'performance')
        os.makedirs(output_dir, exist_ok=True)
        save_figure_to_files(
            fig=fig,
            save_path=output_dir,
            file_name=file_name + file_suffix,
            suffix=None,
            file_types=['pdf', 'eps'],
        )

        plt.close()

    # -------------------------------------------------
    # 1. Full vs leave-one-out (single variable)
    # -------------------------------------------------

    single_variable_drops = [
        'drop_bias',
        'drop_time_since_last_auditory_lick',
        'drop_time_since_last_whisker_lick',
        'drop_jaw_distance',
        'drop_whisker_angle',
        'drop_pupil_area',
        'full',
    ]

    plot_metric_subset(
        all_models_res_df,
        single_variable_drops,
        'models_leave_one_out'
    )
    for rg in ['R+', 'R-']:
        plot_metric_subset(all_models_res_df, single_variable_drops, 'models_leave_one_out', reward_group=rg)

    # -------------------------------------------------
    # 2. Full vs leave several groups
    # -------------------------------------------------

    group_drops = [
        'full',
        'drop_history',
        'drop_mvt',
    ]

    plot_metric_subset(
        all_models_res_df,
        group_drops,
        'models_drop_groups'
    )
    for rg in ['R+', 'R-']:
        plot_metric_subset(all_models_res_df, group_drops, 'models_drop_groups', reward_group=rg)

    # -------------------------------------------------
    # 3. Baseline models comparison
    # -------------------------------------------------

    baseline_models = [
        'full',
        'bias_only',
        'mvt_only'
    ]

    plot_metric_subset(
        all_models_res_df,
        baseline_models,
        'models_baselines'
    )
    for rg in ['R+', 'R-']:
        plot_metric_subset(all_models_res_df, baseline_models, 'models_baselines', reward_group=rg)

    return

def plot_global_model_metrics_old(all_models_res_df, result_figure_path):
    """
    Plot log-likelihood, predictive accuracy, and bits per trial for global models.
    
    Parameters:
    -----------
    all_models_res_df : pd.DataFrame
        DataFrame containing model results
    result_figure_path : str
        Path to save figures
    """
    os.makedirs(result_figure_path, exist_ok=True)

    palette = {
        'full': 'k',
        'bias_only': 'gold',
        'bias_history_only': 'coral',
        'drop_bias': sns.color_palette("Greys")[2],
        'drop_history': sns.color_palette("Greys")[3],
        'drop_trial_type': sns.color_palette("Greys")[4],
        'drop_mvt': sns.color_palette("Greys")[5],
        'drop_auditory': sns.color_palette("Blues")[1],
        'drop_time_since_last_auditory_stim': sns.color_palette("Blues")[2],
        'drop_time_since_last_auditory_lick': sns.color_palette("Blues")[3],
        'drop_whisker': sns.color_palette("Oranges")[1],
        'drop_time_since_last_whisker_stim': sns.color_palette("Oranges")[2],
        'drop_time_since_last_whisker_lick': sns.color_palette("Oranges")[3],
        'drop_jaw_distance': sns.color_palette("RdPu")[1],
        'drop_nose_norm_distance': sns.color_palette("RdPu")[2],
        'drop_whisker_angle': sns.color_palette("RdPu")[3],
        'drop_pupil_area': sns.color_palette("RdPu")[4],
        'mvt_only': sns.color_palette("RdPu")[5],

    }
    order= ['full', 'bias_only', 'bias_history_only', 'drop_bias', 'drop_history', 'drop_trial_type', 'drop_mvt',
            'drop_auditory', 'drop_time_since_last_auditory_stim', 'drop_time_since_last_auditory_lick',
            'drop_whisker', 'drop_time_since_last_whisker_stim', 'drop_time_since_last_whisker_lick',
            'drop_jaw_distance', 'drop_nose_norm_distance', 'drop_whisker_angle', 'drop_pupil_area', 'mvt_only']
    sns.set_style(rc={"lines.linewidth": 0.4})

    # Plot 1: Log-likelihood
    fig, axs = plt.subplots(1, 2, figsize=(10, 5), dpi=100)
    
    # Train data
    axs[0].set_title('Train data')
    sns.pointplot(x='n_states', y='ll_train_final', data=all_models_res_df, ax=axs[0], 
                  estimator=np.mean, errorbar='se', hue='model_type', palette=palette, hue_order=order)
    axs[0].set_ylabel('Log-likelihood')
    axs[0].set_xlabel('Number of states')
    
    # Test data
    axs[1].set_title('Test data')
    sns.pointplot(x='n_states', y='ll_test', data=all_models_res_df, ax=axs[1], 
                  estimator=np.mean, errorbar='se', hue='model_type', palette=palette, hue_order=order)
    axs[1].set_ylabel('Log-likelihood')
    axs[1].set_xlabel('Number of states')
    
    for ax in axs.flat:
        remove_top_right_frame(ax)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc='center left', title='model', bbox_to_anchor=(1, 0.5), 
                 frameon=False, fontsize=5, title_fontsize=8)
    
    fig.tight_layout()
    plt.subplots_adjust(wspace=1.2)
    save_figure_to_files(fig=fig, save_path=result_figure_path, file_name='models_ll', 
                        suffix=None, file_types=['pdf', 'eps'], dpi=200)
    plt.close()
    
    # Plot 2: Predictive accuracy
    fig, axs = plt.subplots(1, 2, figsize=(10, 5), dpi=100)
    
    # Train data
    axs[0].set_title('Train data')
    sns.pointplot(x='n_states', y='predictive_acc_train', data=all_models_res_df, ax=axs[0], 
                  estimator=np.mean, errorbar='sd', hue='model_type', palette=palette, hue_order=order)
    axs[0].set_ylabel('Predictive accuracy')
    axs[0].set_xlabel('Number of states')
    
    # Test data
    axs[1].set_title('Test data')
    sns.pointplot(x='n_states', y='predictive_acc_test', data=all_models_res_df, ax=axs[1], 
                  estimator=np.mean, errorbar='sd', hue='model_type', palette=palette, hue_order=order)
    axs[1].set_ylabel('Predictive accuracy')
    axs[1].set_xlabel('Number of states')

    # For train and test, splits, compute chance baseline accuracy as majority lick rate
    p_lick_train = all_models_res_df.loc[0, 'output_train_labels']
    if isinstance(p_lick_train, (list, np.ndarray)):
        p_lick_train = np.mean([np.mean(i) for i in p_lick_train])
    chance_acc_train = max(p_lick_train, 1 - p_lick_train)
    axs[0].axhline(y=chance_acc_train, color='k', linestyle='--', label='chance')

    p_lick_test = all_models_res_df.loc[0, 'output_test_labels']
    if isinstance(p_lick_test, (list, np.ndarray)):
        p_lick_test = np.mean([np.mean(i) for i in p_lick_test])
    chance_acc_test = max(p_lick_test, 1 - p_lick_test)
    axs[1].axhline(y=chance_acc_test, color='k', linestyle='--', label='chance')

    for ax in axs.flat:
        remove_top_right_frame(ax)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc='center left', title='model', bbox_to_anchor=(1, 0.5), 
                 frameon=False, fontsize=5, title_fontsize=8)
    
    fig.tight_layout()
    plt.subplots_adjust(wspace=1.2)
    save_figure_to_files(fig=fig, save_path=result_figure_path, file_name='models_pred_acc', 
                        suffix=None, file_types=['pdf', 'eps'], dpi=200)
    plt.close()
    
    # Plot 3: Bits per trial (if computed)
    if 'bpt_train' in all_models_res_df.columns:
        fig, axs = plt.subplots(1, 2, figsize=(10, 5), dpi=100)
        
        # Train data
        axs[0].set_title('Train data')
        sns.pointplot(x='n_states', y='bpt_train', data=all_models_res_df, ax=axs[0], 
                      estimator=np.mean, errorbar='sd', hue='model_type')
        axs[0].set_ylabel('Bits per trial')
        axs[0].set_xlabel('Number of states')
        
        # Test data
        axs[1].set_title('Test data')
        sns.pointplot(x='n_states', y='bpt_test', data=all_models_res_df, ax=axs[1], 
                      estimator=np.mean, errorbar='sd', hue='model_type')
        axs[1].set_ylabel('Bits per trial')
        axs[1].set_xlabel('Number of states')
        
        for ax in axs.flat:
            remove_top_right_frame(ax)
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            ax.legend(loc='center left', title='model', bbox_to_anchor=(1, 0.5), 
                     frameon=False, fontsize=5, title_fontsize=8)
        
        fig.tight_layout()
        plt.subplots_adjust(wspace=1.2)
        save_figure_to_files(fig=fig, save_path=result_figure_path,
                             file_name='models_bpt',
                            suffix=None, file_types=['pdf', 'eps'], dpi=200)
        plt.close()


    return


def analyze_global_weights(all_models_res_df, result_figure_path, model_type='full', reward_group=None):
    """
    Analyze and plot GLM weights across states.

    Stores weights per:
        (n_states, state_idx, feature)

    Averaging only across model instances/splits.

    Parameters
    ----------
    reward_group : int or None
        If None, use all reward groups (combined). If 0 or 1, filter to that group.
        Colors: 0 → crimson, 1 → forestgreen.
    """

    # Select model type
    df = all_models_res_df[all_models_res_df['model_type'] == model_type].copy()

    # Filter by reward group if specified
    if reward_group is not None:
        df = df[df['reward_group'] == reward_group].copy()
        rg_suffix     = f'_rg{reward_group}'
        weight_color  = REWARD_GROUP_COLORS[reward_group]
        rg_label      = REWARD_GROUP_NAMES[reward_group]
    else:
        rg_suffix    = ''
        weight_color = 'black'
        rg_label     = 'all groups'

    keep_cols = ['n_states', 'instance_idx', 'split_idx', 'features', 'weights', 'transition_matrix', 'reward_group']
    df = df[keep_cols]

    # One must permut across splits/iterations too ! -> show only first iteration of each split
    #df = df[df['instance_idx'] == 0]
    #df = df[df['split_idx'] == 0]

    print("Unique n_states:", df.n_states.unique())
    print("Unique instances:", df.instance_idx.unique())
    print("Unique splits:", df.split_idx.unique())

    # TODO: transition matrix permutation and plotting


    # Extract weights for each state
    rows = []

    for n_states in df.n_states.unique():

        print(f"Processing {n_states} states")

        sub_df = df[df['n_states'] == n_states]

        # Flatten each value of weights
        sub_df['weights'] = sub_df['weights'].apply(lambda x: np.array(x))

        # Explore weight matrix into weights, state_idx column, and feature column
        features = sub_df['features'].iloc[0]
        for _, row in sub_df.iterrows():
            w = row['weights']  # shape: (n_states, n_features)

            for state_idx in range(w.shape[0]):
                for feat_idx, feat in enumerate(features):
                    rows.append({
                        "n_states": row["n_states"],
                        "split_idx": row["split_idx"],
                        "instance_idx": row["instance_idx"],
                        "state_idx": state_idx,
                        "feature": feat,
                        "weight": w[state_idx, 0, feat_idx],
                        "reward_group": row["reward_group"]
                    })

    # Make it a dataframe
    weights_long_df = pd.DataFrame(rows)

    # Permute states across splits and instances using Hungarian algorithm
    import utils
    weights_long_df, permutations = utils.align_weights_dataframe(weights_long_df, use_mean_reference=False)

    # Verify alignment by plotting the weights across splits
    debug=True
    if debug:
        # Create a subplots to plot weights for each state and split
        n_states = weights_long_df.n_states.unique()[0]
        n_splits = weights_long_df.split_idx.nunique()
        fig, axs = plt.subplots(n_states, n_splits, figsize=(4*n_splits, 4*n_states), dpi=300)
        for state_idx in range(n_states):
            for split_idx in range(n_splits):
                ax = axs[state_idx, split_idx]
                data_sub = weights_long_df[(weights_long_df.state_idx == state_idx) & (weights_long_df.split_idx == split_idx)]
                sns.pointplot(
                    data=data_sub,
                    x="feature",
                    y="weight",
                    order=features,
                    ax=ax,
                    color='black',
                    errorbar='se',
                    estimator=np.mean,
                    legend=False
                )
                ax.axhline(0, color="gray", linestyle="--")
                ax.set_title(f"State {state_idx} - Split {split_idx}")
                ax.set_xticklabels(
                    ax.get_xticklabels(),
                    rotation=45,
                    ha="right"
                )
                remove_top_right_frame(ax)
        fig.tight_layout()
        # Save
        figname = 'weights_alignment_Across_spliots'


    tm_rows = []

    for n_states in df.n_states.unique():

        sub_df = df[df['n_states'] == n_states]

        # Convert transition matrices to numpy arrays
        sub_df['transition_matrix'] = sub_df['transition_matrix'].apply(lambda x: np.array(x))

        for _, row in sub_df.iterrows():
            tm = row['transition_matrix']  # shape: (n_states, n_states)

            # Get the permutation for this model (if n_states > 1)
            if n_states > 1:
                perm = permutations[(n_states, row['split_idx'], row['instance_idx'])]

                # Permute both rows and columns of transition matrix
                # tm[i,j] = P(state_j | state_i)
                # After permutation: tm_aligned = tm[np.ix_(perm, perm)]
                tm_aligned = tm[np.ix_(perm, perm)]
            else:
                tm_aligned = tm

            # Extract entries
            for from_state in range(n_states):
                for to_state in range(n_states):
                    tm_rows.append({
                        "n_states": row["n_states"],
                        "split_idx": row["split_idx"],
                        "instance_idx": row["instance_idx"],
                        "from_state": from_state,
                        "to_state": to_state,
                        "probability": tm_aligned[from_state, to_state],
                        "reward_group": row["reward_group"]
                    })

    # Make transition matrix dataframe
    tm_long_df = pd.DataFrame(tm_rows)

    # Average across splits and instances # then plot
    weights_df = (
            weights_long_df
            .groupby(["n_states", "state_idx", "feature", "reward_group"])["weight"]
            .mean()
            .reset_index()
        )

    tm_df = (
        tm_long_df
        .groupby(["n_states", "from_state", "to_state", "reward_group"])["probability"]
        .mean()
        .reset_index()
    )

    # Plotting
    for n_states in weights_df.n_states.unique():

        # Plot state weights
        # --------------------
        data_sub = weights_df[weights_df.n_states == n_states]
        fig, axs = plt.subplots(1, n_states, figsize=(4*n_states, 4), dpi=300)

        if n_states == 1:
            axs = [axs]

        # Determine ordering from state 1
        state_ref = data_sub[data_sub.state_idx == 1]
        ordered_features = (
            state_ref
            .assign(abs_w=lambda x: np.abs(x.weight))
            .sort_values("abs_w", ascending=False)
            .feature
            .values
        )

        print("Ordered features:", ordered_features)

        # Plot each state
        for state_idx in range(n_states):

            ax = axs[state_idx]
            state_data = data_sub[data_sub.state_idx == state_idx]
            if reward_group is None:
                hue = 'reward_group'
                hue_order = [0, 1]
                hue_order = ['R+', 'R-']
                palette = [REWARD_GROUP_COLORS[rg] for rg in hue_order]
                sns.pointplot(
                    data=state_data,
                    x="feature",
                    y="weight",
                    order=features,
                    ax=ax,
                    # color=weight_color,
                    errorbar='se',
                    estimator=np.mean,
                    hue=hue,
                    palette=palette,
                    legend=False
                )

            else:
                sns.pointplot(
                    data=state_data,
                    x="feature",
                    y="weight",
                    order=features,
                    ax=ax,
                    color=weight_color,
                    errorbar='se',
                    estimator=np.mean,
                    legend=False
                )

            ax.axhline(0, color="gray", linestyle="--")
            ax.set_title(f"State {state_idx} – {rg_label}")
            ax.set_ylabel("Weight")

            ax.set_xticklabels(
                ax.get_xticklabels(),
                rotation=45,
                ha="right"
            )

            remove_top_right_frame(ax)

        fig.tight_layout()

        figname = f"{model_type}_model_weights_{n_states}_states{rg_suffix}"
        output_dir = os.path.join(result_figure_path, "weights")
        os.makedirs(output_dir, exist_ok=True)
        plotting_utils.save_figure_with_options(
            fig,
            file_formats=["pdf", "eps"],
            filename=figname,
            output_dir=output_dir,
            dark_background=False
        )
        plt.close()

    for n_states in tm_df.n_states.unique():

        if n_states == 1:
            continue  # No meaningful transition matrix for 1 state

        #tm_data = tm_df[tm_df.n_states == n_states]
        tm_data = (
            tm_df[tm_df.n_states == n_states]
            .groupby(['from_state', 'to_state'], as_index=False)['probability']
            .mean()
        )

        # Reshape to matrix form
        tm_matrix = tm_data.pivot(index='from_state', columns='to_state', values='probability').values

        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(6, 5), dpi=300)

        # Plot heatmap
        im = ax.imshow(tm_matrix, cmap='Greys_r', vmin=0, vmax=1, aspect='auto')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.5)
        cbar.set_label('Probability', fontsize=12)

        # Set ticks and labels
        ax.set_xticks(range(n_states))
        ax.set_yticks(range(n_states))
        ax.set_xticklabels([f'State {i}' for i in range(n_states)], fontsize=10)
        ax.set_yticklabels([f'State {i}' for i in range(n_states)], fontsize=10)
        ax.set_xlabel(r'State $t$', fontsize=12)
        ax.set_ylabel(r'State $t-1$', fontsize=12)
        ax.set_title(f'Transition matrix – {rg_label}', fontsize=14)

        # Add text annotations with probabilities
        for i in range(n_states):
            for j in range(n_states):
                text = f'{tm_matrix[i, j]:.2f}'
                ax.text(j, i, text,
                        ha="center", va="center",
                        color="k" if tm_matrix[i, j] > 0.5 else "white",
                        fontsize=10)

        fig.tight_layout()

        figname = f"{model_type}_transition_matrix_{n_states}_states{rg_suffix}"
        output_dir = os.path.join(result_figure_path, "transition_matrices")
        os.makedirs(output_dir, exist_ok=True)
        plotting_utils.save_figure_with_options(
            fig,
            file_formats=["pdf", "eps"],
            filename=figname,
            output_dir=output_dir,
            dark_background=False
        )

        plt.close()

    return


# ============================================================================
# SINGLE MOUSE MODEL PERFORMANCE FUNCTIONS
# ============================================================================
# TO DO: average across splits and instances
# TODO: implement state permutation (compare methods)

def load_single_mouse_models(root_path, subject_ids, splits=10, n_states=[1, 2, 3, 4, 5, 6]):
    """
    Load single mouse GLM-HMM model results and trial data.
    
    Parameters:
    -----------
    root_path : str
        Root path to subject directories
    subject_ids : list
        List of subject IDs to load
    splits : int
        Number of data splits
    n_states : list
        List of state numbers to load
        
    Returns:
    --------
    tuple : (all_subjects_res DataFrame, all_subjects_trial_data DataFrame)
    """
    all_subjects_res = []
    all_subjects_trial_data = []
    
    for subject_id in subject_ids:
        for split_id in range(splits):
            for k_state in n_states:
                path_to_res_folder = os.path.join(root_path, subject_id, 'full_models',
                                                  f'model_{split_id}', f'{k_state}_states', 'iter_0')

                
                # Load model results
                path_to_res_file = os.path.join(path_to_res_folder, 'fit_glmhmm_results.npz')
                print(os.path.isfile(path_to_res_file))
                try:
                    res = np.load(path_to_res_file, allow_pickle=True)['arr_0'][()]
                    all_subjects_res.append(res)
                except Exception as err:
                    print(f"Could not load results for {subject_id}", err)
                    continue
                
                # Load trial data
                path_to_data_file = os.path.join(path_to_res_folder, 'data_preds.h5')
                try:
                    trial_data = pd.read_hdf(path_to_data_file)
                    trial_data['n_states'] = k_state
                    trial_data['split_idx'] = split_id
                    all_subjects_trial_data.append(trial_data)
                except:
                    print(f"Could not load trial data for {subject_id}")
                    continue
    
    # Create DataFrames
    all_subjects_res = pd.DataFrame(all_subjects_res)
    all_subjects_trial_data = pd.concat(all_subjects_trial_data, ignore_index=True)
    
    return all_subjects_res, all_subjects_trial_data


def plot_single_mouse_metrics(all_subjects_res, result_figure_path):
    """
    Plot log-likelihood and predictive accuracy for single mouse models.
    
    Parameters:
    -----------
    all_subjects_res : pd.DataFrame
        DataFrame containing single mouse model results
    result_figure_path : str
        Path to save figures
    """
    os.makedirs(result_figure_path, exist_ok=True)
    
    # Average per data splits first
    all_subjects_res_avg = all_subjects_res.groupby(['mouse_id', 'n_states']).agg({
        'll_train': 'mean', 
        'll_test': 'mean', 
        'predictive_acc_train': 'mean', 
        'predictive_acc_test': 'mean'
    }).reset_index()
    
    # Plot 1: Log-likelihood
    fig, axs = plt.subplots(1, 2, figsize=(6, 3), dpi=300)
    
    # Train data
    axs[0].set_title('Train data')
    sns.lineplot(x='n_states', y='ll_train', data=all_subjects_res_avg, ax=axs[0], 
                estimator=np.mean, errorbar='sd', err_style='bars', lw=3, c='k', 
                markers=True, marker='o', markeredgecolor=None)
    sns.lineplot(x='n_states', y='ll_train', data=all_subjects_res_avg, ax=axs[0], 
                units='mouse_id', estimator=None, lw=1, c='k', alpha=0.3)
    axs[0].set_ylabel('Log-likelihood')
    axs[0].set_xlabel('Number of states')
    axs[0].set_xticks([1, 2, 3, 4, 5])
    
    # Test data
    axs[1].set_title('Test data')
    sns.lineplot(x='n_states', y='ll_test', data=all_subjects_res_avg, ax=axs[1], 
                estimator=np.mean, errorbar='sd', err_style='bars', lw=3, c='k', 
                markers=True, marker='o', markeredgecolor=None)
    sns.lineplot(x='n_states', y='ll_test', data=all_subjects_res_avg, ax=axs[1], 
                units='mouse_id', estimator=None, lw=1, c='k', alpha=0.3)
    axs[1].set_ylabel('Log-likelihood')
    axs[1].set_xlabel('Number of states')
    axs[1].set_xticks([1, 2, 3, 4, 5])
    
    for ax in axs.flat:
        remove_top_right_frame(ax)
    
    fig.tight_layout()
    save_figure_to_files(fig=fig, save_path=result_figure_path,
                         file_name='single_mouse_ll',
                        suffix=None, file_types=['pdf', 'eps'], dpi=200)
    plt.close()
    
    # Plot 2: Predictive accuracy
    fig, axs = plt.subplots(1, 2, figsize=(6, 3), dpi=300)
    
    # Train data
    axs[0].set_title('Train data')
    sns.lineplot(x='n_states', y='predictive_acc_train', data=all_subjects_res_avg, ax=axs[0], 
                estimator=np.mean, errorbar='sd', err_style='bars', lw=3, c='k', 
                markers=True, marker='o', markeredgecolor=None)
    sns.lineplot(x='n_states', y='predictive_acc_train', data=all_subjects_res_avg, ax=axs[0], 
                units='mouse_id', estimator=None, lw=1, c='k', alpha=0.3)
    axs[0].set_ylabel('Predictive accuracy [%]')
    axs[0].set_xlabel('Number of states')
    axs[0].set_xticks([1, 2, 3, 4, 5])
    axs[0].yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    
    # Test data
    axs[1].set_title('Test data')
    sns.lineplot(x='n_states', y='predictive_acc_test', data=all_subjects_res_avg, ax=axs[1], 
                estimator=np.mean, errorbar='sd', err_style='bars', lw=3, c='k', 
                markers=True, marker='o', markeredgecolor=None)
    sns.lineplot(x='n_states', y='predictive_acc_test', data=all_subjects_res_avg, ax=axs[1], 
                units='mouse_id', estimator=None, lw=1, c='k', alpha=0.3)
    axs[1].set_ylabel('Predictive accuracy [%]')
    axs[1].set_xlabel('Number of states')
    axs[1].set_xticks([1, 2, 3, 4, 5])
    axs[1].yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    
    for ax in axs.flat:
        remove_top_right_frame(ax)
    
    fig.tight_layout()
    save_figure_to_files(fig=fig, save_path=result_figure_path, file_name='single_mouse_pred_acc', 
                        suffix=None, file_types=['pdf', 'eps'], dpi=200)
    plt.close()


# ============================================================================
# STATE ALIGNMENT HELPER FUNCTIONS
# ============================================================================

def cosine_similarity(A, B):
    """Compute cosine similarity between two vectors."""
    return np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))


def cosine_distance(A, B):
    """Compute cosine distance between two vectors."""
    return 1 - cosine_similarity(A, B)


def compute_distance_matrix(W1, W2, metric='euclidean'):
    """
    Compute distance matrix between two sets of 1D state weights.
    
    Parameters:
    -----------
    W1 : array
        1D weights for states in subject 1
    W2 : array
        1D weights for states in subject 2
    metric : str
        Distance metric ('euclidean', 'cosine', or 'manhattan')
        
    Returns:
    --------
    array : Distance matrix of shape (n_states, n_states)
    """
    n_states = len(W1)
    distance_matrix = np.zeros((n_states, n_states))
    
    for i in range(n_states):
        for j in range(n_states):
            if metric == 'cosine':
                distance_matrix[i, j] = cosine_distance(W1[i], W2[j])
            elif metric == 'euclidean':
                distance_matrix[i, j] = euclidean(W1[i], W2[j])
            elif metric == 'manhattan':
                distance_matrix[i, j] = cityblock(W1[i], W2[j])
            else:
                raise ValueError(f"Unknown metric: {metric}")
    
    return distance_matrix


def align_states(reference_weights, subject_weights, metric='euclidean'):
    """
    Align states using Hungarian algorithm.
    
    Parameters:
    -----------
    reference_weights : array
        Reference state weights
    subject_weights : array
        Subject state weights to align
    metric : str
        Distance metric to use
        
    Returns:
    --------
    array : Permutation indices
    """
    distance_matrix = compute_distance_matrix(reference_weights, subject_weights, metric=metric)
    row_ind, col_ind = linear_sum_assignment(distance_matrix)
    return col_ind


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    
    # Set matplotlib style
    sns.set_context("paper")
    sns.set_style("ticks")
    plt.rcParams['font.size'] = 12
    
    # ========================================================================
    # CONFIGURATION - UPDATE THESE PATHS
    # ========================================================================
    
    # For global model analysis
    global_model_parent_path = r'M:\analysis\Axel_Bisi\combined_results\glm_hmm\global_glmhmm_mvt_whisker_trials'
    global_result_figure_path = os.path.join(global_model_parent_path, 'figures')
    
    # For single mouse model analysis
    single_mouse_root_path = r'M:\analysis\Axel_Bisi\combined_results\glm_hmm'
    single_mouse_result_figure_path = os.path.join(single_mouse_root_path, 
                                                    'all_subjects_glmhmm',
                                                    'figures')
    
    # Subject IDs for single mouse analysis -> all mice in folder
    subject_ids = [folder for folder in os.listdir(single_mouse_root_path) if folder.startswith('AB') or folder.startswith('MH')]
    print('Subject IDs:', subject_ids)


    # Analysis parameters
    run_global_analysis = True
    run_single_mouse_analysis = False
    compute_bpt = True  # Compute bits per trial for global models
    
    # ========================================================================
    # GLOBAL MODEL ANALYSIS
    # ========================================================================
    
    if run_global_analysis:
        print("=" * 80)
        print("RUNNING GLOBAL MODEL ANALYSIS")
        print("=" * 80)
        
        try:
            # Load global models
            all_models_res_df = load_global_models(global_model_parent_path)
            
            # Compute bits per trial if requested, this requires null model predictions to be stored
            if compute_bpt:
                print("\nComputing bits per trial...")
                all_models_res_df = compute_bits_per_trial(all_models_res_df)
            
            # Plot metrics
            print("\nPlotting global model performance metrics...")
            plot_global_model_metrics(all_models_res_df, global_result_figure_path)
            
            # Analyze weights for full model
            print("\nAnalyzing GLM weights...")
            analyze_global_weights(all_models_res_df, global_result_figure_path, model_type='full')
            for rg in ['R+', 'R-']:
                print(f"\nAnalyzing GLM weights for reward_group={rg} ({REWARD_GROUP_NAMES[rg]})...")
                analyze_global_weights(all_models_res_df, global_result_figure_path, model_type='full', reward_group=rg)


            
            print(f"\nGlobal analysis complete. Figures saved to: {global_result_figure_path}")
            
        except Exception as e:
            print(f"Error in global model analysis: {e}")
            import traceback
            traceback.print_exc()
    
    # ========================================================================
    # SINGLE MOUSE MODEL ANALYSIS
    # ========================================================================
    
    if run_single_mouse_analysis:
        print("\n" + "=" * 80)
        print("RUNNING SINGLE MOUSE MODEL ANALYSIS")
        print("=" * 80)
        
        try:
            # Load single mouse models
            print("\nLoading single mouse models...")
            all_subjects_res, all_subjects_trial_data = load_single_mouse_models(
                single_mouse_root_path, 
                subject_ids, 
                splits=10,
                n_states=[1, 2, 3, 4, 5, 6]
            )
            
            # Filter to specific subject range if needed
            subjects_to_keep = [f'AB{str(z).zfill(3)}' for z in range(80, 157)]
            all_subjects_trial_data = all_subjects_trial_data[
                all_subjects_trial_data['mouse_id'].isin(subjects_to_keep)
            ]
            all_subjects_res = all_subjects_res[
                all_subjects_res['mouse_id'].isin(subjects_to_keep)
            ]
            
            # Plot metrics
            print("\nPlotting single mouse metrics...")
            plot_single_mouse_metrics(all_subjects_res, single_mouse_result_figure_path)
            
            print(f"\nSingle mouse analysis complete. Figures saved to: {single_mouse_result_figure_path}")
            
        except Exception as e:
            print(f"Error in single mouse model analysis: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
