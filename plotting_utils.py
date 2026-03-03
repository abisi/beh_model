#! /usr/bin/env/python3
"""
@author: Axel Bisi
@project: NWB_analysis
@file: plotting_utils.py
@time: 11/17/2023 4:13 PM
@description: Various plotting utilities for customizing plots.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.colors as mc
import colorsys



def remove_top_right_frame(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return

def color_to_rgba(color_name):
    """
    Converts color name to RGB.
    :param color_name:
    :return:
    """

    return colors.to_rgba(color_name)

def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.
    From: https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib
    :param color: Matplotlib color string.
    :param amount: Number between 0 and 1.
    :return:
    """

    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def adjust_lightness(color, amount=0.5):
    """
    Same as lighten_color but adjusts brightness to lighter color if amount>1 or darker if amount<1.
    Input can be matplotlib color string, hex string, or RGB tuple.
    From: https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib
    :param color: Matplotlib color string.
    :param amount: Number between 0 and 1.
    :return:
    """

    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


def make_cmap_n_from_color_lite2dark(color, N):
    """
    Make ListedColormap from matplotlib color of size N using the lighten_color function.
    :param color: Matplotlib color string.
    :param N: Number of colors to have in cmap.
    :return:
    """
    light_factors = np.linspace(0.2, 1, N)
    cmap = colors.ListedColormap(colors=[lighten_color(color, amount=i) for i in light_factors])
    return cmap


def save_figure_to_files(fig, save_path, file_name, suffix=None, file_types=list, dpi=500):
    """
    Save figure to file.
    :param fig: Figure to save.
    :param save_path: Path to save figure.
    :param file_name: Name of file.
    :param suffix: Suffix to add to file name.
    :param file_types: List of file types to save.
    :param dpi: Resolution of figure.
    :return:
    """

    if file_types is None:
        file_types = ['png', 'eps', 'pdf']

    if suffix is not None:
        file_name = file_name + '_' + suffix

    for file_type in file_types:
        file_format = '.{}'.format(file_type)
        file_path = os.path.join(save_path, file_name + file_format)

        if file_type == 'eps':
            fig.savefig(file_path, dpi=dpi, bbox_inches='tight', transparent=True)
        else:
            fig.savefig(file_path, dpi=dpi, bbox_inches='tight')
    return

def render_mpl_table(data, col_width=3.0, row_height=0.625, font_size=14,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    """
    Render a matplotlib table
    :param data:
    :param col_width:
    :param row_height:
    :param font_size:
    :param header_color:
    :param row_colors:
    :param edge_color:
    :param bbox:
    :param header_columns:
    :param ax:
    :param kwargs:
    :return:
    """
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')
    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in mpl_table._cells.items():
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
    return ax.get_figure(), ax


def plot_feature_matrix(feature_matrix, feature_names, title, save_path, file_name, suffix=None, file_types=None):
    """
    Plot feature matrix.
    :param feature_matrix:
    :param feature_names:
    :param title:
    :param save_path:
    :param file_name:
    :param suffix:
    :param file_types:
    :return:
    """

    fig, ax = plt.subplots(1, 1, figsize=(15, 10), dpi=200, facecolor='w', edgecolor='k')
    fig.suptitle(title, fontsize=20)
    ax.imshow(feature_matrix, cmap='viridis', aspect='auto')
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels(feature_names, fontsize=15)
    ax.set_xticks([])
    ax.set_xlabel('Time', fontsize=15)
    ax.set_ylabel('Feature', fontsize=15)

    save_figure_to_files(fig, save_path, file_name, suffix=suffix, file_types=file_types)
    return

def plot_model_glm_weights(model, init_weights, feature_names, save_path, file_name, suffix=None, file_types=None):
    """
    Plot GLM weights.
    :param model:
    :param feature_names:
    :param save_path:
    :param file_name:
    :param suffix:
    :param file_types:
    :return:
    """

    weights = model.observations.params
    input_dim = len(feature_names)

    # Get number of states from weights arrays
    if len(weights.shape) == 2:
        n_states = 1
    else:
        n_states = weights.shape[0]

    fig, axs = plt.subplots(1, n_states, figsize=(5*n_states, 5), dpi=400, facecolor='w', edgecolor='k',
                            sharey=True)
    fig.suptitle('GLM Weights', fontsize=20)

    if n_states == 1:
        axs = [axs]
    else:
        axs = axs.flatten()

    for idx, ax in enumerate(axs):
        remove_top_right_frame(ax)
        ax.plot(np.arange(input_dim),
                np.squeeze(weights[idx]),
                c='k',
                marker='o',
                linestyle='-',
                lw=1.5)
        if init_weights is not None:
            ax.plot(np.arange(input_dim),
                    np.squeeze(init_weights[idx]),
                    c='dimgrey',
                    marker='o',
                    linestyle='--',
                    lw=1.5)
        ax.set_xticks(np.arange(input_dim), feature_names, fontsize=12, rotation=90)
        ax.set_ylabel('Weight', fontsize=15)
        ax.set_xlabel('Features', fontsize=15)
        ax.set_title('State {}'.format(idx + 1), fontsize=15)
        ax.axhline(y=0, color="k", alpha=0.5, ls="--")

    save_figure_to_files(fig, save_path, file_name, suffix=suffix, file_types=file_types)
    plt.close()

    return

def plot_model_transition_matrix(model, save_path, file_name, suffix=None, file_types=None):
    """
    Plot GLM transition matrix.
    :param model:
    :return:
    """

    transition_matrix = model.transitions.transition_matrix

    fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=400, facecolor='w', edgecolor='k')
    fig.suptitle('Transition Matrix', fontsize=20)
    remove_top_right_frame(ax)
    ax.imshow(transition_matrix, cmap='Greys_r', norm=colors.Normalize(vmin=-0.3, vmax=1.0))
    ax.set_xticks(range(transition_matrix.shape[1]))
    ax.set_yticks(range(transition_matrix.shape[0]))
    ax.set_xlabel(r'State $t$', fontsize=15)
    ax.set_ylabel(r'State $t-1$', fontsize=15)

    # Add state labels
    ax.set_xticklabels([str(i) for i in range(transition_matrix.shape[1])], fontsize=12)
    ax.set_yticklabels([str(i) for i in range(transition_matrix.shape[0])], fontsize=12)

    # Add text annotation of transition probabilities
    for i in range(transition_matrix.shape[0]):
        for j in range(transition_matrix.shape[1]):
            text = ax.text(j, i, "{:.2f}".format(transition_matrix[i, j]),
                           ha="center", va="center", color="k")


    save_figure_to_files(fig, save_path, file_name, suffix=suffix, file_types=file_types)
    plt.close()

    return


def plot_single_session_predictions(data, save_path, file_name, suffix=None, file_types=None):
    """
    Plot single session predictions
    :param data: pd.DataFrame with session data
    :param save_path: path to save figures
    :param file_name: name of file
    :param suffix: suffix to add to file name
    :param file_types: list of file types to save

    :return:
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    rew_group_map = {0: 'R-', 1: 'R+', 2: 'R+ proba'}

    for session_id in data['session_id'].unique():

        fig, ax = plt.subplots(1, 1, figsize=(10, 5), dpi=300, facecolor='w', edgecolor='k')
        remove_top_right_frame(ax)

        sess_data = data[data['session_id'] == session_id]

        n_trials = len(sess_data)
        true_labels = sess_data['choice'].values
        predicted_labels = sess_data['pred'].values

        ax.scatter(range(n_trials), true_labels, c='k', edgecolors=None, alpha=0.7, s=10, marker='o', label='Data')
        ax.scatter(range(n_trials), predicted_labels, c='r', alpha=0.7, s=10, marker='x', label='Model')
        ax.set_ylabel('Choice', fontsize=15)
        ax.set_xlabel('Trials', fontsize=15)
        ax.set_yticks([0, 1], ['No lick', 'Lick'], fontsize=12)

        ax.legend(frameon=False, loc='center right')
        rew_group = sess_data['reward_group'].values[0]
        title = 'Predictions for {} - {}'.format(session_id, rew_group_map[rew_group])
        fig.suptitle(title, fontsize=15)


        save_figure_to_files(fig, save_path, file_name+'_{}'.format(session_id), suffix=suffix, file_types=file_types)
        plt.close()
    return

def plot_single_session_posterior_states(data, save_path, file_name, suffix=None, file_types=None):
    """
    Plot single session posterior states
    :param data: pd.DataFrame with session data
    :param save_path: path to save figures
    :param file_name: name of file
    :param suffix: suffix to add to file name
    :param file_types: list of file types to save

    :return:
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    rew_group_map = {0: 'R-', 1: 'R+', 2: 'R+ proba'}

    posterior_cols = [col for col in data.columns if 'posterior_state' in col]

    for session_id in data['session_id'].unique():

        fig, axs = plt.subplots(2, 1, figsize=(30, 5), dpi=80, facecolor='w', edgecolor='k',
                                gridspec_kw={'height_ratios': [10, 1]}, sharex=True)
        remove_top_right_frame(axs[0])

        sess_data = data[data['session_id'] == session_id]

        for idx, state_col in enumerate(posterior_cols):
            axs[0].plot(sess_data[state_col].values, label="State " + str(idx + 1), lw=2)

        axs[0].set_ylim((-0.01, 1.01))
        axs[0].set_xlabel("Trials", fontsize=15)
        axs[0].set_ylabel("P(state)", fontsize=15)
        axs[0].legend(frameon=False, loc="upper right", fontsize=12)

        # Plot corresponding trial outcome
        sess_data['trial_type'] = np.nan  # TODO: check if this works
        sess_data.loc[(sess_data.stimulus_type == -1) & (sess_data.choice == 1), 'trial_type'] = 'ah'
        sess_data.loc[(sess_data.stimulus_type == -1) & (sess_data.choice == 0), 'trial_type'] = 'am'
        sess_data.loc[(sess_data.stimulus_type == 1) & (sess_data.choice == 1), 'trial_type'] = 'wh'
        sess_data.loc[(sess_data.stimulus_type == 1) & (sess_data.choice == 0), 'trial_type'] = 'wm'
        sess_data.loc[(sess_data.stimulus_type == 0) & (sess_data.choice == 1), 'trial_type'] = 'fa'
        sess_data.loc[(sess_data.stimulus_type == 0) & (sess_data.choice == 0), 'trial_type'] = 'cr'

        perf_map = {0: 'wm', 2: 'wh', 1: 'am', 3: 'ah', 4: 'cr', 5: 'fa'}
        perf_map = {v: k for k, v in perf_map.items()}
        perf_color = {
            0: 'crimson',
            1: 'lightblue',
            2: 'forestgreen',
            3: 'mediumblue',
            4: 'lightgrey',
            5: 'k'}
        sess_data = sess_data.assign(perf=sess_data['trial_type'].map(perf_map))
        cmap = colors.LinearSegmentedColormap.from_list('perf_cmap', list(perf_color.values()), 6)

        # Define bins and normalize
        bounds = np.linspace(0, 6, 7)
        norm = colors.BoundaryNorm(bounds, cmap.N)
        axs[1].pcolor(np.expand_dims(sess_data['perf'].values, axis=0), cmap=cmap, norm=norm,
                      edgecolors=None, linewidths=0)

        save_figure_to_files(fig, save_path, file_name+'_{}'.format(session_id), suffix=suffix, file_types=file_types)
        plt.close()
    return



