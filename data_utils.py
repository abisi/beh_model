#! /usr/bin/env/python3
"""
@author: Axel Bisi
@project: beh_model
@file: data_utils.py
@time: 12/8/2023 3:32 PM
"""
import numpy as np
import pandas as pd

def cumsum_excluding_first(x):
    """Cumulative sum excluding first row."""
    return x.iloc[1:].cumsum()


def compute_time_since_last_event_norm(trial_df, event_flag_col):
    """
    Compute time since last event (seconds), keeping the event trial itself as time since previous event.
    Inverted min-max normalization: recent events = 1, far events = 0.

    Rules:
    - Before first event → 0
    - Event trial itself → retains time since previous event
    - Only resets after the next event
    - No NaNs
    """

    times = trial_df['start_time'].values
    flags = trial_df[event_flag_col].values.astype(bool)

    ts_last = np.zeros_like(times, dtype=float)
    last_event_time = None

    for i, (t, f) in enumerate(zip(times, flags)):
        if last_event_time is None:
            ts_last[i] = 0.0  # before first event
        else:
            ts_last[i] = t - last_event_time

        if f:
            # Update last_event_time, but do NOT reset ts_last at this trial
            last_event_time = t

    trial_df[f'{event_flag_col}_ts'] = ts_last

    # Invert and min-max normalize
    min_val, max_val = ts_last.min(), ts_last.max()
    if max_val - min_val > 0:
        trial_df[f'{event_flag_col}_norm'] = 1 - ((ts_last - min_val) / (max_val - min_val))
    else:
        trial_df[f'{event_flag_col}_norm'] = 1.0

    # Override all trials before the first event occured and set to 0
    first_event_idx = np.where(flags)[0]
    if len(first_event_idx) > 0:
        first_event_idx = first_event_idx[0]
        trial_df.loc[:first_event_idx, f'{event_flag_col}_norm'] = 0.0


    return trial_df

def calculate_time_since_last_event(trial_df, event_col):
    """
    Calculate time since last reward/lick for different reward types.
    :param trial_df:
    :param event_col:
    :return:
    """
    # Get rows where no rewards were obtained i.e. intervals between two reward times
    no_reward_ids = trial_df[trial_df[event_col].isna()].index

    # At every reward, create a new group (in order to cumsum per group only later)
    trial_df['nonan_group'] = trial_df[event_col].notna().cumsum()

    # Calculate cumulative time between rewards, per group, excluding first row
    # This gives the (increasing) time since last reward, reset at each reward
    trial_df['start_time_diff_cumsum'] = trial_df.groupby('nonan_group')['start_time_diff'].apply(
        lambda x: pd.concat([pd.Series([None]), cumsum_excluding_first(x)])).reset_index(drop=True)  # the pd.concat command ensures indices are correct after groupby

    # Get list of indices where to place these cumulative ITIs - exclude group 0 because by definition it has never seen a reward
    ids_to_fill = no_reward_ids.intersection(trial_df[trial_df['nonan_group'] != 0].index)

    # Add these times to the reward column
    trial_df.loc[ids_to_fill, event_col] = trial_df.loc[ids_to_fill, 'start_time_diff_cumsum']

    # Ensure float type
    trial_df[event_col] = trial_df[event_col].astype(float)  # Ensure type consistency
    return

def create_data_lists(data_train, data_test, features):

    if 'choice' not in data_train.columns:
        raise ValueError('data_train must contain a column named "choice"')

    if data_test is not None:
        if 'choice' not in data_test.columns:
            raise ValueError('data_test must contain a column named "choice"')

    # Create design matrix
    inputs_train = []
    outputs_train = []
    inputs_test = []
    outputs_test = []

    for idx, sess_id in enumerate(data_train.session_id.unique()):
        # Get session data
        session_data_train = data_train[data_train.session_id == sess_id]
        # Get session inputs and outputs
        session_inputs_train = session_data_train[features].values
        session_outputs_train = np.expand_dims(session_data_train['choice'].values, axis=1)
        # Append to list
        inputs_train.append(session_inputs_train)
        outputs_train.append(session_outputs_train)

    if data_test is not None:
        for idx, sess_id in enumerate(data_test.session_id.unique()):
            session_data_test = data_test[data_test.session_id == sess_id]
            session_inputs_test = session_data_test[features].values
            session_outputs_test = np.expand_dims(session_data_test['choice'].values, axis=1)
            inputs_test.append(session_inputs_test)
            outputs_test.append(session_outputs_test)

    if data_test is not None:
        return inputs_train, outputs_train, inputs_test, outputs_test
    else:
        return inputs_train, outputs_train
