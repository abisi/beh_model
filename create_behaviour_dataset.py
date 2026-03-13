#! /usr/bin/env/python3
"""
@author: Axel Bisi
@project: beh_model
@file: create_behaviour_dataset.py
@time: 11/29/2023 3:30 PM
"""

# Imports
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from matplotlib import pyplot as plt
from multiprocessing import Pool
import scipy as sp

import config
import NWB_reader_functions as NWB_read
import plotting_utils
from data_utils import calculate_time_since_last_event, compute_time_since_last_event_norm
from utils import reindex_whisker_days

RESULT_PATH = r'M:\analysis\Axel_Bisi\combined_results'


# ── NWB-file discovery helper ──────────────────────────────────────────────────

def process_nwb_file(args):
    nwb_path, mouse_ids = args
    try:
        mouse = next((m for m in mouse_ids if m in nwb_path.name), None)
        if mouse is None:
            return None
        behavior_type, day = NWB_read.get_bhv_type_and_training_day_index(str(nwb_path))
        if behavior_type != 'whisker':
            return None
        return {
            'mouse_id':      mouse,
            'day':           day,
            'behavior_type': behavior_type,
            'nwb_name':      nwb_path.name,
            'nwb_file':      str(nwb_path),
        }
    except Exception as e:
        print(f'Error reading {nwb_path}: {e}')
        return None


# ── DLC preprocessing helpers ──────────────────────────────────────────────────

def remove_outliers_zscore(trace, threshold=10):
    """Replace outliers with NaNs based on z-score."""
    trace = np.array(trace, dtype=float)
    z = (trace - np.nanmean(trace)) / np.nanstd(trace)
    trace[np.abs(z) > threshold] = np.nan
    return trace


def interpolate_nans(trace):
    """Linearly interpolate NaNs in the trace."""
    trace = np.array(trace, dtype=float)
    nans = np.isnan(trace)
    if nans.all():
        return np.zeros_like(trace)
    trace[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), trace[~nans])
    return trace


def smooth_trace(trace, sigma=2):
    """Smooth the trace using a Gaussian filter."""
    return sp.ndimage.gaussian_filter1d(trace, sigma=sigma)


def smooth_trace_savgol(trace, window_length=11, polyorder=3):
    """Smooth the trace using Savitzky-Golay filter."""
    return sp.signal.savgol_filter(trace, window_length, polyorder)


def smooth_trace_median(trace, kernel_size=5):
    """Smooth the trace using a median filter."""
    return sp.signal.medfilt(trace, kernel_size=kernel_size)


def preprocess_dlc_trace(trace):
    """Preprocess DLC trace: remove outliers, interpolate, smooth."""
    trace = remove_outliers_zscore(trace, threshold=20)
    trace = interpolate_nans(trace)
    trace = smooth_trace(trace, sigma=5)
    if np.isnan(trace).any():
        print("Warning: Trace contains NaNs after preprocessing. Filling with zeros.")
        trace = np.nan_to_num(trace)
    return trace


# ── Vectorised DLC window extraction ──────────────────────────────────────────

def _extract_window_metrics(bp_data, bp_ts, start_times, t_start, t_end, is_pupil):
    """
    For each trial onset in `start_times`, extract the metric over [onset+t_start, onset+t_end).

    Uses np.searchsorted instead of per-trial boolean masking — ~10-30× faster
    for typical session sizes.

    Parameters
    ----------
    bp_data    : 1-D float array of DLC values
    bp_ts      : 1-D float array of DLC timestamps (monotonically increasing)
    start_times: 1-D float array of trial onset times
    t_start, t_end : window boundaries relative to trial onset (seconds)
    is_pupil   : if True use nanmean; otherwise use nanmean(|diff|) = motion energy

    Returns
    -------
    metrics : 1-D float array, length == len(start_times)
    """
    t_lo = start_times + t_start
    t_hi = start_times + t_end

    # Vectorised index boundaries (O(n_trials * log(n_timepoints)) instead of O(n_trials * n_timepoints))
    lo = np.searchsorted(bp_ts, t_lo, side='left')
    hi = np.searchsorted(bp_ts, t_hi, side='left')

    metrics = np.full(len(start_times), np.nan)
    for i, (l, h) in enumerate(zip(lo, hi)):
        if l >= h:
            continue
        sl = bp_data[l:h]
        try:
            metrics[i] = np.nanmean(sl) if is_pupil else np.nanmean(np.abs(np.diff(sl)))
        except Exception:
            pass  # leave as nan
    return metrics


# ── Per-session processing ─────────────────────────────────────────────────────

def _process_single_session(args):
    """
    Process one NWB session file into a behavioral DataFrame.

    Parameters
    ----------
    args : tuple of (nwb_file, mouse_info_df, n_trials_max)

    Returns
    -------
    pd.DataFrame, or None if the session should be skipped
    """
    nwb_file, mouse_info_df, n_trials_max = args
    print(f"Processing {nwb_file}...")

    # ---------------------------
    # Load and filter trial table
    # ---------------------------
    trial_df      = NWB_read.get_trial_table(nwb_file)
    behavior_type, day = NWB_read.get_bhv_type_and_training_day_index(nwb_file)
    trial_df['behavior'] = behavior_type
    trial_df['day']      = day

    trial_df = reindex_whisker_days(trial_df)

    if n_trials_max is not None:
        trial_df = trial_df.iloc[:n_trials_max]

    # Remove passive trials, early licks, and association trials
    trial_df = trial_df[(trial_df['context'] != 'passive') & (trial_df['perf'] != 6)]

    if behavior_type != 'whisker':
        return None

    mouse_id     = NWB_read.get_mouse_id(nwb_file)
    session_id   = NWB_read.get_session_id(nwb_file)
    reward_group = mouse_info_df.loc[mouse_info_df['mouse_id'] == mouse_id, 'reward_group'].values[0]

    # ---------------------------
    # Session metadata
    # ---------------------------
    data_df = pd.DataFrame(index=trial_df.index)
    data_df['mouse_id']   = mouse_id
    data_df['wh_reward']  = reward_group
    data_df['behavior']   = behavior_type
    data_df['session_id'] = session_id
    data_df['day']        = day

    # ---------------------------
    # Choice
    # ---------------------------
    data_df['trial_id']    = trial_df.index
    data_df['choice']      = trial_df['lick_flag']
    data_df['prev_choice'] = data_df['choice'].shift(1).astype('int32', errors='ignore')

    # ---------------------------
    # Stimulus encoding features
    # ---------------------------
    # Build all trial-type indicator columns in one pass
    tt = trial_df['trial_type']
    data_df['auditory']      = (tt == 'auditory_trial').astype(int)
    data_df['whisker']       = (tt == 'whisker_trial').astype(int)
    data_df['is_stimulus']   = ((tt == 'auditory_trial') | (tt == 'whisker_trial')).astype(int)
    data_df['prev_auditory'] = data_df['auditory'].shift(1).astype('int32', errors='ignore')
    data_df['prev_whisker']  = data_df['whisker'].shift(1).astype('int32', errors='ignore')

    data_df['lick_correct'] = trial_df['perf'].map({0: 0, 1: 0, 2: 1, 3: 1, 4: 0, 5: 0})

    # ---------------------------
    # Reward features (vectorised)
    # ---------------------------
    aud_reward = (data_df['auditory'] == 1) & (data_df['lick_correct'] == 1)
    wh_lick    = (data_df['whisker']  == 1) & (data_df['lick_correct'] == 1)

    data_df['reward_given'] = 0
    data_df.loc[aud_reward, 'reward_given'] = 1
    if reward_group == 'R+':
        data_df.loc[wh_lick, 'reward_given'] = 1
    # R- mice: whisker licks are never rewarded (reward_given stays 0)

    data_df['unrewarded_lick'] = (
        (data_df['lick_correct'] == 0) & (data_df['is_stimulus'] == 1)
    ).astype(int)

    # Previous-trial reward (including no-stim trials)
    data_df['prev_trial_reward_given'] = data_df['reward_given'].shift(1).astype('int32', errors='ignore')

    # Previous stimulus-trial reward — shift within stim rows only, then ffill
    stim_mask = data_df['is_stimulus'] == 1
    for src, dst in [('reward_given', 'prev_stim_reward_given'),
                     ('auditory',     'prev_stim_auditory'),
                     ('whisker',      'prev_stim_whisker')]:
        # .shift(1) on a filtered Series gives the previous stim-trial value at each stim position
        data_df.loc[stim_mask, dst] = data_df.loc[stim_mask, src].shift(1).values

    stim_cols = ['prev_stim_reward_given', 'prev_stim_auditory', 'prev_stim_whisker']
    # Forward-fill non-stim rows with the last stim value, then fill initial NaNs with 0
    data_df[stim_cols] = data_df[stim_cols].ffill().fillna(0)

    # ---------------------------
    # Time since last stimulus / lick / reward
    # ---------------------------
    trial_df['start_time_diff'] = trial_df['start_time'].diff(1).fillna(0)

    # Stimulus flags
    trial_df['whisker_stim_flag']  = (tt == 'whisker_trial').astype(int)
    trial_df['auditory_stim_flag'] = (tt == 'auditory_trial').astype(int)

    compute_time_since_last_event_norm(trial_df, 'whisker_stim_flag')
    compute_time_since_last_event_norm(trial_df, 'auditory_stim_flag')

    data_df['time_since_last_whisker_stim']  = trial_df['whisker_stim_flag_norm']
    data_df['time_since_last_auditory_stim'] = trial_df['auditory_stim_flag_norm']

    # Lick flags
    trial_df['whisker_lick_flag']  = ((trial_df['lick_flag'] == 1) & (tt == 'whisker_trial')).astype(int)
    trial_df['auditory_lick_flag'] = ((trial_df['lick_flag'] == 1) & (tt == 'auditory_trial')).astype(int)

    compute_time_since_last_event_norm(trial_df, 'whisker_lick_flag')
    compute_time_since_last_event_norm(trial_df, 'auditory_lick_flag')

    data_df['time_since_last_whisker_lick']  = trial_df['whisker_lick_flag_norm']
    data_df['time_since_last_auditory_lick'] = trial_df['auditory_lick_flag_norm']

    # Reward flags
    reward_filters = {'R+': ['auditory_trial', 'whisker_trial'], 'R-': ['auditory_trial']}
    trial_df['reward_flag'] = (
        (trial_df['lick_flag'] == 1) & (tt.isin(reward_filters[reward_group]))
    ).astype(int)
    trial_df['whisker_reward_flag']  = (
        ((trial_df['lick_flag'] == 1) & (tt == 'whisker_trial')).astype(int)
        if reward_group == 'R+' else 0
    )
    trial_df['auditory_reward_flag'] = (
        (trial_df['lick_flag'] == 1) & (tt == 'auditory_trial')
    ).astype(int)

    compute_time_since_last_event_norm(trial_df, 'reward_flag')
    compute_time_since_last_event_norm(trial_df, 'whisker_reward_flag')
    compute_time_since_last_event_norm(trial_df, 'auditory_reward_flag')

    data_df['time_since_last_reward']          = trial_df['reward_flag_norm']
    data_df['time_since_last_whisker_reward']  = trial_df['whisker_reward_flag_norm']
    data_df['time_since_last_auditory_reward'] = trial_df['auditory_reward_flag_norm']

    # ---------------------------
    # Reaction time (normalised)
    # ---------------------------
    rt = trial_df['lick_time'] - trial_df['response_window_start_time']
    rt_min, rt_max = rt.min(), rt.max()
    denom = rt_max - rt_min
    data_df['reaction_time_piezo'] = (rt - rt_min) / denom if denom > 0 else 0.0

    # ---------------------------
    # Cumulative reward (normalised within session) and recent reward rate
    # ---------------------------
    cum = data_df['reward_given'].cumsum()
    cum_max = cum.max()
    data_df['cumulative_reward'] = cum / cum_max if cum_max > 0 else 0.0

    window_len = 5
    data_df['recent_reward_count'] = data_df.groupby('session_id')['reward_given'].transform(lambda x: x.rolling(window=window_len, min_periods=1).sum())
    data_df['recent_reward_rate'] = data_df['recent_reward_count'] / window_len


    # ---------------------------
    # Constant bias term / ITI
    # ---------------------------
    data_df['bias'] = 1
    data_df['iti']  = trial_df['start_time_diff']

    # ---------------------------
    # DLC movement features (vectorised window extraction)
    # ---------------------------
    bodyparts = ['jaw_distance', 'nose_norm_distance', 'whisker_angle', 'pupil_area']
    dlc = NWB_read.get_dlc_data_dict(nwb_file)

    if dlc is None:
        print(f"No DLC data found for {nwb_file}. Skipping movement features.")
        return data_df

    start_times = trial_df['start_time'].values
    raw_metrics = {}  # store un-normalised values for batch normalisation below

    for bp in bodyparts:
        if bp not in dlc or isinstance(dlc[bp]['data'], float):
            continue

        bp_data = np.asarray(dlc[bp]['data'], dtype=float)
        bp_ts   = np.asarray(dlc[bp]['timestamps'], dtype=float)

        # --- Fix length mismatch ---
        if len(bp_data) != len(bp_ts):
            max_len = max(len(bp_data), len(bp_ts))
            def _pad(arr):
                if len(arr) >= max_len:
                    return arr
                out = np.full(max_len, np.nan)
                idx = np.random.choice(max_len, size=len(arr), replace=False)
                out[idx] = arr
                return out
            bp_data = _pad(bp_data)
            bp_ts   = _pad(bp_ts)

        # FIX: capture the return value (was silently discarded before)
        bp_data = preprocess_dlc_trace(bp_data)

        # Ensure timestamps are sorted for searchsorted
        if not np.all(np.diff(bp_ts[~np.isnan(bp_ts)]) >= 0):
            sort_idx = np.argsort(bp_ts)
            bp_ts, bp_data = bp_ts[sort_idx], bp_data[sort_idx]

        # Vectorised extraction (replaces the per-trial boolean-mask loop)
        window = (-0.5, 0) if bp == 'pupil_area' else (-0.2, 0)
        t_start, t_end = window
        metrics = _extract_window_metrics(
            bp_data, bp_ts, start_times, t_start, t_end, is_pupil=(bp == 'pupil_area')
        )

        raw_metrics[bp] = metrics

    # Batch min-max normalisation for all bodyparts
    for bp, metrics in raw_metrics.items():
        min_val = np.nanmin(metrics)
        max_val = np.nanmax(metrics)
        denom   = max_val - min_val
        data_df[bp] = (metrics - min_val) / denom if denom > 0 else 0.0

    plot_dataset = False
    if plot_dataset:
        features = config.FEATURES

        # Report any NaN/inf columns
        for col in features:
            if col not in data_df.columns:
                continue
            if np.isnan(data_df[col]).any():
                nan_rows = data_df[np.isnan(data_df[col])]
                print(f'NaN in "{col}":\n{nan_rows[["mouse_id", "session_id", "day", "trial_id"]]}')

        cols_to_plot = [
            'choice', 'reaction_time_piezo',
            'auditory', 'time_since_last_auditory_stim', 'time_since_last_auditory_lick',
            'whisker', 'time_since_last_whisker_stim', 'time_since_last_whisker_lick',
            'jaw_distance', 'nose_norm_distance', 'whisker_angle', 'pupil_area',
            'cumulative_reward', 'recent_reward_rate',
        ]

        for (mouse_id, day), session_data in data_df.groupby(['mouse_id', 'day']):
            feature_mat = session_data[cols_to_plot].values
            n_rows = feature_mat.shape[1]

            for i, col in enumerate(cols_to_plot):
                col_vals = feature_mat[:, i]
                if np.isinf(col_vals).any():
                    print(f'Inf in "{col}" — mouse={mouse_id}, day={day}')
                if np.isnan(col_vals).any():
                    print(f'NaN in "{col}" — mouse={mouse_id}, day={day}')

            start_id, n_trials = 20, 50
            mat = feature_mat[start_id:start_id + n_trials].T

            group1 = slice(0, 1)
            group2 = slice(1, 2)
            group3 = slice(2, 5)
            group4 = slice(5, 8)
            group5 = slice(8, n_rows)
            cmap_mvt = 'RdPu'
            cmap_mvt.set_bad('grey')

            fig, ax = plt.subplots(1, 1, dpi=400)
            im = ax.imshow(mat[group1], aspect='equal', cmap='Greys',
                           interpolation='none', extent=[0, n_trials, group1.stop, group1.start])
            ax.imshow(mat[group2], aspect='equal', cmap='copper',
                      interpolation='none', extent=[0, n_trials, group2.stop, group2.start])
            ax.imshow(mat[group3], aspect='equal', cmap='Blues',
                      interpolation='none', extent=[0, n_trials, group3.stop, group3.start])
            ax.imshow(mat[group4], aspect='equal', cmap='Oranges',
                      interpolation='none', extent=[0, n_trials, group4.stop, group4.start])
            ax.imshow(mat[group5], aspect='equal', cmap=cmap_mvt,
                      interpolation='none', extent=[0, n_trials, group5.stop, group5.start])

            cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.1, fraction=0.05, shrink=0.3)
            cbar.set_ticks([0, 1])
            cbar.set_label('Feature value', fontsize=6, labelpad=-0.5)
            cbar.ax.tick_params(labelsize=6)
            ax.set_xlabel('Trials', fontsize=6)
            ax.set_ylim(n_rows, 0)
            ax.set_yticks(np.arange(n_rows) + 0.5)
            ax.set_yticklabels(cols_to_plot, fontsize=4)
            ax.tick_params(which='major', labelsize=4)
            fig.tight_layout()

            out_dir = dataset_folder / 'single_sessions'
            out_dir.mkdir(parents=True, exist_ok=True)
            for ext in ['pdf', 'eps']:
                plt.savefig(out_dir / f'feature_matrix_{mouse_id}_day{day}.{ext}',
                            bbox_inches='tight', dpi='figure')
            plt.close(fig)

    return data_df



def create_behavior_dataset(nwb_list, mouse_info_df, n_trials_max=None, n_workers=None):
    """
    Build the behavioral dataset by processing NWB sessions in parallel.

    Parameters
    ----------
    nwb_list      : list of NWB file paths
    mouse_info_df : DataFrame with per-mouse metadata
    n_trials_max  : cap on trials per session (None = unlimited)
    n_workers     : parallel workers (None → os.cpu_count()-2)

    Returns
    -------
    pd.DataFrame of all trials concatenated across sessions
    """
    if n_workers is None:
        n_workers = max(1, os.cpu_count() - 2)

    task_args = [(nwb_file, mouse_info_df, n_trials_max) for nwb_file in nwb_list]

    #with Pool(processes=n_workers) as pool:
    #    session_results = pool.map(_process_single_session, task_args)
    with Pool(processes=n_workers) as pool:
         session_results = list(tqdm(pool.imap(_process_single_session, task_args), total=len(task_args), desc='Dataset creation:'))



    bhv_data = [df for df in session_results if df is not None]
    if not bhv_data:
        raise RuntimeError('No whisker sessions found in the provided NWB file list.')

    bhv_data_df = pd.concat(bhv_data, ignore_index=True)

    # Encode reward group as integer and merge
    reward_group_map = {'R+': 1, 'R-': 0}
    mouse_info_copy = mouse_info_df.copy()
    mouse_info_copy['reward_group'] = mouse_info_copy['reward_group'].map(reward_group_map)
    bhv_data_df = bhv_data_df.merge(
        mouse_info_copy[['mouse_id', 'reward_group']], on='mouse_id', how='left'
    )

    # Cast integer columns
    int_cols = [
        'choice', 'prev_choice', 'auditory', 'whisker',
        'reward_given', 'prev_trial_reward_given', 'prev_stim_reward_given',
    ]
    bhv_data_df[int_cols] = bhv_data_df[int_cols].astype('int32', errors='ignore')

    # Sanity-check feature ranges
    features = config.FEATURES
    for f in features:
        if f not in bhv_data_df.columns:
            continue
        lo, hi = bhv_data_df[f].min(), bhv_data_df[f].max()
        print(f'Feature "{f}": min={lo:.3f}, max={hi:.3f}')
        if lo < 0 or hi > 1:
            print(f'  Warning: "{f}" has values outside [0, 1].')

    nan_cols = bhv_data_df.columns[bhv_data_df.isna().any()].tolist()
    if nan_cols:
        print(f'Warning: NaN values remain in columns: {nan_cols}')

    # Fill any residual NaNs with 0
    print('Filling remaining NaNs with 0.')
    bhv_data_df.fillna(0, inplace=True)

    return bhv_data_df


# ── Train / test split ─────────────────────────────────────────────────────────

def split_dataset(dataset, fraction_training=config.FRACTION_TRAINING):
    """
    Split dataset into training and test sets by random trial sampling.

    Parameters
    ----------
    dataset           : pd.DataFrame of all trials
    fraction_training : fraction of trials allocated to training

    Returns
    -------
    (data_train, data_test)
    """
    test_size = 1.0 - fraction_training
    if not 0 < test_size < 1:
        raise ValueError(f'test_size must be between 0 and 1, got {test_size:.2f}')

    n          = len(dataset)
    n_test     = int(np.floor(test_size * n))
    rng        = np.random.default_rng(42)          # reproducible, modern API
    test_idx   = np.sort(rng.choice(n, n_test, replace=False))
    train_idx  = np.setdiff1d(np.arange(n), test_idx)

    return dataset.iloc[train_idx], dataset.iloc[test_idx]


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    output_path   = config.OUTPUT_PATH / 'glm_hmm'
    mouse_info_df = pd.read_excel(config.INFO_PATH / 'joint_mouse_reference_weight.xlsx')
    mouse_info_df.rename(columns={'mouse_name': 'mouse_id'}, inplace=True)

    mouse_info_df = mouse_info_df[
        (mouse_info_df['exclude'] == 0) &
        (mouse_info_df['reward_group'].isin(config.REWARD_GROUPS))
    ]

    # Discover available NWB files
    all_nwb_paths = list(config.NWB_PATH.iterdir())
    nwb_name_set  = {p.name.split('_')[0] for p in all_nwb_paths}   # set lookup: O(1) vs O(n)

    mouse_ids = [
        m for m in mouse_info_df['mouse_id'].unique()
        if m in nwb_name_set
    ]

    dataset_folder = output_path / ('datasets_combined_mvt' if len(mouse_ids) > 1 else mouse_ids[0])
    dataset_folder.mkdir(parents=True, exist_ok=True)

    create_dataset = True
    plot_dataset   = True

    if create_dataset:

        # Discover whisker NWB files in parallel
        args_list    = [(nwb_path, mouse_ids) for nwb_path in all_nwb_paths]
        with Pool(processes=os.cpu_count() - 1) as pool:
            results = pool.map(process_nwb_file, args_list)

        nwb_files_df = pd.DataFrame([r for r in results if r is not None])

        # Keep only subjects with enough whisker sessions
        session_counts = nwb_files_df.groupby('mouse_id')['nwb_name'].count()
        valid_subjects = session_counts[session_counts >= config.MIN_WHISKER_DAYS].index
        nwb_files_df   = nwb_files_df[nwb_files_df['mouse_id'].isin(valid_subjects)]
        nwb_list       = nwb_files_df['nwb_file'].tolist()


        # Build dataset (parallelised over sessions)
        dataset = create_behavior_dataset(nwb_list, mouse_info_df, n_trials_max=config.N_TRIALS_MAX)
        print('Unique mice in data:', dataset['mouse_id'].unique())
        print('Unique sessions per mouse:\n', dataset.groupby('mouse_id')['session_id'].nunique())

        # Save dataset splits
        for split_idx in range(config.N_SPLITS):
            data_train, data_test = split_dataset(dataset)
            split_path = dataset_folder / f'dataset_{split_idx}'
            split_path.mkdir(parents=True, exist_ok=True)
            dataset.to_pickle(split_path / 'dataset.pkl')
            data_train.to_pickle(split_path / 'data_train.pkl')
            data_test.to_pickle(split_path / 'data_test.pkl')
            print(f'Datasets saved in {split_path}')

    if plot_dataset:

        dataset_path = Path(r"M:\analysis\Axel_Bisi\combined_results\glm_hmm\datasets_combined_mvt\dataset_0\dataset.pkl")
        dataset = pd.read_pickle(dataset_path)
        features = config.FEATURES

        # Report any NaN/inf columns
        for col in features:
            if col not in dataset.columns:
                continue
            if np.isnan(dataset[col]).any():
                nan_rows = dataset[np.isnan(dataset[col])]
                print(f'NaN in "{col}":\n{nan_rows[["mouse_id", "session_id", "day", "trial_id"]]}')

        cols_to_plot = [
            'choice', 'reaction_time_piezo',
            'auditory', 'time_since_last_auditory_stim', 'time_since_last_auditory_lick',
            'whisker',  'time_since_last_whisker_stim',  'time_since_last_whisker_lick',
            'jaw_distance', 'nose_norm_distance', 'whisker_angle', 'pupil_area',
            'cumulative_reward', 'recent_reward_rate',
        ]
        cols_to_plot = [col for col in cols_to_plot if col in dataset.columns]

        for (mouse_id, day), session_data in dataset.groupby(['mouse_id', 'day']):
            feature_mat = session_data[cols_to_plot].values
            n_rows = feature_mat.shape[1]

            for i, col in enumerate(cols_to_plot):
                col_vals = feature_mat[:, i]
                if np.isinf(col_vals).any():
                    print(f'Inf in "{col}" — mouse={mouse_id}, day={day}')
                if np.isnan(col_vals).any():
                    print(f'NaN in "{col}" — mouse={mouse_id}, day={day}')

            start_id, n_trials = 0, 70

            group1 = slice(0, 1)
            group2 = slice(1, 2)
            group3 = slice(2, 5)
            group4 = slice(5, 8)
            group5 = slice(8, n_rows)
            cmap_mvt = plt.get_cmap('RdPu')
            cmap_mvt.set_bad('grey')


            def _make_feature_fig(mat, n_trials, n_rows, cols_to_plot):
                fig, ax = plt.subplots(1, 1, dpi=400)
                im = ax.imshow(mat[group1], aspect='equal', cmap='Greys',
                               interpolation='none', extent=[0, n_trials, group1.stop, group1.start])
                ax.imshow(mat[group2], aspect='equal', cmap='copper',
                          interpolation='none', extent=[0, n_trials, group2.stop, group2.start])
                ax.imshow(mat[group3], aspect='equal', cmap='Blues',
                          interpolation='none', extent=[0, n_trials, group3.stop, group3.start])
                ax.imshow(mat[group4], aspect='equal', cmap='Oranges',
                          interpolation='none', extent=[0, n_trials, group4.stop, group4.start])
                ax.imshow(mat[group5], aspect='equal', cmap=cmap_mvt,
                          interpolation='none', extent=[0, n_trials, group5.stop, group5.start])
                cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.1, fraction=0.05, shrink=0.3)
                cbar.set_ticks([0, 1])
                cbar.set_label('Feature value', fontsize=6, labelpad=-0.5)
                cbar.ax.tick_params(labelsize=6)
                ax.set_xlabel('Trials', fontsize=6)
                ax.set_ylim(n_rows, 0)
                ax.set_yticks(np.arange(n_rows) + 0.5)
                ax.set_yticklabels(cols_to_plot, fontsize=4)
                ax.tick_params(which='major', labelsize=4)
                fig.tight_layout()
                return fig


            out_dir = dataset_folder / 'single_sessions'
            out_dir.mkdir(parents=True, exist_ok=True)

            # --- Figure 1: all trials ---
            mat_all = feature_mat[start_id:start_id + n_trials].T
            fig_all = _make_feature_fig(mat_all, n_trials, n_rows, cols_to_plot)
            for ext in ['pdf', 'eps']:
                fig_all.savefig(out_dir / f'feature_matrix_{mouse_id}_day{day}.{ext}',
                                bbox_inches='tight', dpi='figure')
            plt.close(fig_all)

            # --- Figure 2: whisker trials only ---
            whisker_data = session_data[session_data['whisker'] == 1]
            feature_mat_wh = whisker_data[cols_to_plot].values
            n_trials_wh = min(n_trials, len(feature_mat_wh))
            mat_wh = feature_mat_wh[start_id:start_id + n_trials_wh].T
            fig_wh = _make_feature_fig(mat_wh, n_trials_wh, n_rows, cols_to_plot)
            for ext in ['pdf', 'eps']:
                fig_wh.savefig(out_dir / f'feature_matrix_{mouse_id}_day{day}_whisker.{ext}',
                               bbox_inches='tight', dpi='figure')
            plt.close(fig_wh)

    print('Dataset creation done.')
