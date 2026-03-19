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

def save_figure_with_options(figure, file_formats, filename, output_dir='', dark_background=False):
    """
    Save a Matplotlib figure in multiple file formats with options for dark background.
    :param figure: Figure object
    :param file_formats: list of file formats (e.g., ['png', 'pdf'])
    :param filename: filename without extension
    :param output_dir: directory to save figure
    :param dark_background: whether to change basic colors for dark background
    :return:
    """
    # Make transparent for dark background
    if dark_background:
        figure.patch.set_alpha(0)
        figure.set_facecolor('#f4f4ec')
        for ax in figure.get_axes():
            ax.set_facecolor('#f4f4ec')
        #plt.rcParams.update({'axes.facecolor': '#f4f4ec',  # very pale beige
        #                        'figure.facecolor': '#f4f4ec'})
        transparent = True
        filename = filename + '_transparent'
    else:
        transparent = False

    # Save the figure in each specified file format
    for file_format in file_formats:
        file_path = os.path.join(output_dir, f"{filename}.{file_format}")
        figure.savefig(file_path, transparent=transparent, bbox_inches='tight', dpi='figure')

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

        fig, ax = plt.subplots(1, 1, figsize=(5, 2.5), dpi=300, facecolor='w', edgecolor='k')
        remove_top_right_frame(ax)

        sess_data = data[data['session_id'] == session_id]

        n_trials = len(sess_data)
        true_labels = sess_data['choice'].values
        predicted_labels = sess_data['pred'].values

        def compress_binary(x, low=0.1, high=0.9):
            return x * (high - low) + low

        true_plot = compress_binary(true_labels)
        pred_plot = compress_binary(predicted_labels)

        ax.scatter(range(n_trials), true_plot, c='k', edgecolors=None, alpha=0.7, s=10, marker='o', label='Data')
        ax.scatter(range(n_trials), pred_plot, c='r', alpha=0.7, s=10, marker='x', label='Model')
        ax.set_ylabel('Choice', fontsize=12)
        ax.set_xlabel('Trials', fontsize=12)
        ax.set_yticks([0.1, 0.9], ['No lick', 'Lick'], fontsize=12)
        ax.set_ylim(0, 1)

        ax.legend(frameon=False, loc='center right')
        rew_group = sess_data['reward_group'].values[0]
        title = 'Predictions for {} - {}'.format(session_id, rew_group_map[rew_group])
        fig.suptitle(title, fontsize=12)


        save_figure_to_files(fig, save_path, file_name+'_{}'.format(session_id), suffix=suffix, file_types=file_types)
        plt.close()
    return

def plot_single_session_posterior_states(data, save_path, file_name, suffix=None, file_types=None): #too: plot comparison with Viterbi
    """
    Plot single session posterior states
    :param data: pd.DataFrame with session data
    :param save_path: path to save figures
    :param file_name: name of file
    :param suffix: suffix to add to file name
    :param file_types: list of file types to save

    :return:
    """
    # Create a cmap to map state indices to colors so that it's reusable
    state_index_cmap = {0:'#157BE9', 1:'#E9157B', 2:'#7BE915', 3: '#c5d642', 4:'#c5d642', 5:'#329fd1', 6:'#d642aa'}  # Example mapping, adjust as needed

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    rew_group_map = {0: 'R-', 1: 'R+', 2: 'R+ proba'}

    posterior_cols = [col for col in data.columns if 'posterior_state' in col]

    for session_id in data['session_id'].unique():

        fig, axs = plt.subplots(2, 1, figsize=(6, 3), dpi=400, facecolor='w', edgecolor='k',
                                gridspec_kw={'height_ratios': [12, 1]}, sharex=True)
        remove_top_right_frame(axs[0])

        sess_data = data[data['session_id'] == session_id]

        for idx, state_col in enumerate(posterior_cols):
            state_color = state_index_cmap[idx]  # Get color for this state index
            post_data = sess_data[state_col].values
            axs[0].plot(post_data, label="state " + str(idx + 1), lw=1.5, color=state_color)

            #Plot the same but smoother e.g. 5 trials smoothing
            window_size = 3
            post_smooth = np.convolve(sess_data[state_col].values, np.ones(window_size)/window_size, mode='same')
            #axs[0].plot(post_smooth, lw=1, linestyle='--', color=state_color)

        # Assign single-trial to most likely state #TODO: make it a function
        most_likely_state = np.argmax(sess_data[posterior_cols].values, axis=1)

        # If both states are 0.5, assign the previous state as most likely state to avoid too much noise in the plot
        for idx in range(1, len(most_likely_state)-1):
            if np.isclose(sess_data[posterior_cols].values[idx], 0.5).all():
                most_likely_state[idx] = most_likely_state[idx-1]

        # If states are only one trial long, merge them to the next state to avoid too much noise in the plot
        for idx in range(1, len(most_likely_state)-1):
            if most_likely_state[idx] != most_likely_state[idx-1] and most_likely_state[idx] != most_likely_state[idx+1]:
                most_likely_state[idx] = most_likely_state[idx+1]

        # Plot most likely state as a shaded background
        x = np.arange(len(sess_data))

        for idx in range(len(posterior_cols)):
            state_color = state_index_cmap[idx]
            mask = most_likely_state == idx

            start = None
            for i, m in enumerate(mask):
                if m and start is None:
                    start = i
                #elif not m and start is not None:
                #    axs[0].axvspan(start,i, ymin=0,ymax=0.5, color=state_color, alpha=0.1)
                #    start = None

            #if start is not None:
            #    axs[0].axvspan(start, len(mask), color=state_color, alpha=0.1)

        # --- Use Viterbi as most likely state for shading ---
        most_likely_state = sess_data["most_likely_state"].values  # already Viterbi
        x = np.arange(len(sess_data))

        for idx in range(len(posterior_cols)):
            state_color = state_index_cmap[idx]
            mask = most_likely_state == idx

            start = None
            for i, m in enumerate(mask):
                if m and start is None:
                    start = i
                elif not m and start is not None:
                    axs[0].axvspan(start, i, ymin=0.0,ymax=1.0,color=state_color, alpha=0.1)
                    start = None
            if start is not None:
                axs[0].axvspan(start, len(mask), color=state_color, alpha=0.1)

        axs[0].set_ylim((-0.01, 1.01))
        #axs[0].set_xlabel("Trials", fontsize=10)
        axs[0].set_ylabel("P(state)", fontsize=10)
        legend = axs[0].legend(frameon=True, loc="center right", fontsize=6)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(0.5)
        legend.get_frame().set_edgecolor("none")


        # Plot corresponding trial outcome
        sess_data['trial_type'] = np.nan
        sess_data.loc[(sess_data.auditory == 1) & (sess_data.choice == 1), 'trial_type'] = 'ah'
        sess_data.loc[(sess_data.auditory == 0) & (sess_data.choice == 0), 'trial_type'] = 'am'
        sess_data.loc[(sess_data.whisker == 1) & (sess_data.choice == 1), 'trial_type'] = 'wh'
        sess_data.loc[(sess_data.whisker == 1) & (sess_data.choice == 0), 'trial_type'] = 'wm'
        sess_data.loc[(sess_data.auditory == 0) & (sess_data.whisker == 0) & (sess_data.choice == 1), 'trial_type'] = 'fa'
        sess_data.loc[(sess_data.auditory == 0) & (sess_data.whisker == 0) & (sess_data.choice == 0), 'trial_type'] = 'cr'

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
        axs[1].set_xlabel("Trials", fontsize=10)
        # Remove all spines but the bottom one
        axs[1].spines['top'].set_visible(False)
        axs[1].spines['right'].set_visible(False)
        axs[1].spines['left'].set_visible(False)
        # Remove y ticks and labels
        axs[1].set_yticks([])
        axs[1].set_yticklabels([])
        # Keep x tickss


        # Reduce space between subplots
        plt.subplots_adjust(hspace=0.04)

        save_figure_to_files(fig, save_path, file_name+'_{}'.format(session_id), suffix=suffix, file_types=file_types)
        plt.close()
    return



def _annotate_feature_stats(ax, data, feats, rg_order, y_pad_frac=0.04):
    """
    For each feature position on the x-axis, run a between-group statistical
    test and draw a significance annotation above the data.

    Two groups  → Mann-Whitney U (two-sided).
    Three+ groups → Kruskal-Wallis.

    P-values are Bonferroni-corrected for the number of features tested.
    Significance stars: ns p≥0.05 | * p<0.05 | ** p<0.01 | *** p<0.001
    """
    from scipy.stats import mannwhitneyu, kruskal

    def _stars(p):
        if p < 0.001: return "***"
        if p < 0.01:  return "**"
        if p < 0.05:  return "*"
        return "ns"

    # ------------------------------------------------------------------
    # Pass 1: compute raw p-values for every feature
    # ------------------------------------------------------------------
    raw_pvalues = {}   # feat → p
    for feat in feats:
        groups = [
            data.loc[(data["feature"] == feat) & (data["reward_group"] == rg), "weight"].dropna().values
            for rg in rg_order
        ]
        groups = [g for g in groups if len(g) >= 2]
        if len(groups) < 2:
            continue
        try:
            print('WEIGHT STAT TEST DATAPOINTS', len(groups[0]), len(groups[1]))
            _, p = mannwhitneyu(groups[0], groups[1], alternative="two-sided")
            raw_pvalues[feat] = p
        except ValueError:
            continue

    if not raw_pvalues:
        return

    # ------------------------------------------------------------------
    # Bonferroni correction: multiply each p by the number of tests
    # ------------------------------------------------------------------
    n_tests = len(raw_pvalues)
    corrected_pvalues = {
        feat: min(p * n_tests, 1.0)
        for feat, p in raw_pvalues.items()
    }

    # ------------------------------------------------------------------
    # Pass 2: annotate using corrected p-values
    # ------------------------------------------------------------------
    y_min, y_max = ax.get_ylim()
    pad = (y_max - y_min) * y_pad_frac
    new_y_max = y_max

    for x_pos, feat in enumerate(feats):
        if feat not in corrected_pvalues:
            continue
        p_corr = corrected_pvalues[feat]

        feat_data = data[data["feature"] == feat]["weight"].dropna()
        y_top = feat_data.max() + pad

        ax.text(
            x_pos, y_top, _stars(p_corr),
            ha="center", va="bottom",
            fontsize=6, color="black",
        )
        new_y_max = max(new_y_max, y_top + pad)

    ax.set_ylim(y_min, new_y_max)