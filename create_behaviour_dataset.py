#! /usr/bin/env/python3
"""
@author: Axel Bisi
@project: ssm
@file: create_behaviour_dataset.py
@time: 11/29/2023 3:30 PM
"""

# Imports
import os
import socket
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from data_utils import calculate_time_since_last_reward
import NWB_reader_functions as NWB_read

#import nwb_wrappers.nwb_reader_functions as NWB_read
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

def create_behavior_dataset(nwb_list, mouse_info_df, params):
    """
    Create a behavior dataset from a list of NWB files.
    :param nwb_list:
    :return:
    """

    # Get parameters
    n_trials_max = params['n_trials_max']

    # Create behavior dataset #TODO: include whisker ON sessions as whisker day 3
    bhv_data =[]

    # Get each session data
    for nwb_file in nwb_list:

        data_df = pd.DataFrame()

        # Get trial data
        trial_df = NWB_read.get_trial_table(nwb_file)
        trial_df = trial_df.iloc[:n_trials_max, :]  # limit number of trials

        # Remove passive trials, early licks, asso trials etc
        trial_df = trial_df[(trial_df['context'] != 'passive') & (trial_df['perf'] != 6)]

        mouse_id = NWB_read.get_mouse_id(nwb_file)
        behavior_type, day = NWB_read.get_bhv_type_and_training_day_index(nwb_file)
        session_id = NWB_read.get_session_id(nwb_file)
        #reward_group = NWB_read.get_session_metadata(nwb_file)['wh_reward'] # keep for now
        reward_group = mouse_info_df[mouse_info_df['mouse_id'] == mouse_id]['reward_group'].values[0]

        if behavior_type not in ['whisker']:
            continue


        # Add metadata
        data_df.index = trial_df.index
        data_df['mouse_id'] = [mouse_id for trial in range(len(trial_df.index))]
        data_df['wh_reward'] = [reward_group for trial in range(len(trial_df.index))]
        data_df['behavior'] = [behavior_type for trial in range(len(trial_df.index))]  # auditory or whisker
        data_df['session_id'] = [session_id for trial in range(len(trial_df.index))]
        data_df['day'] = [day for trial in range(len(trial_df.index))]  # day relative to start of whisker

        # Add trial index and choice target
        data_df['trial_id'] = trial_df.index
        data_df['choice'] = trial_df['lick_flag'] # whether mouse licks or not
        data_df['prev_choice'] = data_df['choice'].shift(1).astype('int32', errors='ignore') #ignore nan of first row

        # Add stimulus encoding features
        data_df['stimulus_type'] = trial_df['trial_type'].map({'auditory_trial': -1, 'whisker_trial': 1, 'no_stim_trial':0})
        data_df['prev_stimulus_type'] = data_df['stimulus_type'].shift(1).astype('int32', errors='ignore')
        data_df['auditory'] = trial_df['trial_type'].map({'auditory_trial': 1, 'whisker_trial': -1, 'no_stim_trial': -1})
        data_df['prev_auditory'] = data_df['auditory'].shift(1).astype('int32', errors='ignore')
        data_df['whisker'] = trial_df['trial_type'].map({'auditory_trial': -1, 'whisker_trial': 1, 'no_stim_trial': -1})
        data_df['prev_whisker'] = data_df['whisker'].shift(1).astype('int32', errors='ignore')
        data_df['lick_correct'] = trial_df['perf'].map({0:0, 1:0, 2:1, 3:1, 4:0, 5:0})
        data_df['is_stimulus'] = trial_df['trial_type'].map({'auditory_trial': 1, 'whisker_trial': 1, 'no_stim_trial': 0})

        # Add reward-related features
        data_df.loc[data_df['auditory']==1, 'reward_given'] = data_df['lick_correct']
        data_df.loc[data_df['whisker']==1, 'reward_given'] = data_df['lick_correct'] * data_df['wh_reward'] * trial_df['reward_available'] # TODO: check if proba reward groups are ok doing like that
        data_df.loc[data_df['is_stimulus']==0, 'reward_given'] = 0
        data_df['reward_given'] = data_df['reward_given'].map({1:1, 0:-1})
        # Reward at previous trial, including no stim trials
        data_df['prev_trial_reward_given'] = data_df['reward_given'].shift(1).astype('int32', errors='ignore')
        # Reward at previous stimulus trial, excluding thus no stim trials
        data_df_stim = data_df[data_df['is_stimulus'] == 1] # get only stim trials
        data_df_stim['prev_stim_reward_given'] = data_df_stim['reward_given'].shift(1) # get if last stim trial was rewarded
        data_df_stim['prev_stim_auditory'] = data_df_stim['auditory'].shift(1) # get if last stim trial was auditory
        data_df_stim['prev_stim_whisker'] = data_df_stim['whisker'].shift(1) # get if last stim trial was whisker
        data_df = data_df.merge(data_df_stim[['prev_stim_reward_given', 'prev_stim_auditory', 'prev_stim_whisker']], left_index=True, right_index=True, how='left') # merge previous values back to original dataframe
        # Fill nan with 0 for prev_stim_reward_given
        data_df[['prev_stim_reward_given', 'prev_stim_auditory', 'prev_stim_whisker']].fillna(0, inplace=True) # fill nan with 0s for non stim trials # TODO: check if ok

        # For these columns, go back along the rows and copy the value of the first non-nan value #TODO: redo this
        data_df['prev_stim_reward_given'] = data_df['prev_stim_reward_given'].fillna(method='ffill')
        data_df['prev_stim_auditory'] = data_df['prev_stim_auditory'].fillna(method='ffill')
        data_df['prev_stim_whisker'] = data_df['prev_stim_whisker'].fillna(method='ffill')

        # Time since last auditory stimulus
        data_df['time_since_last_auditory_stim'] = trial_df['start_time'].diff(1) * data_df['auditory']
        # Time since last whisker stimulus
        data_df['time_since_last_whisker_stim'] = trial_df['start_time'].diff(1) * data_df['whisker']

        # Get inter-trial interval
        trial_df['start_time_diff'] = trial_df['start_time'].diff(1)

        # Get time since last whisker/auditory stimulus
        whisker_trial_df = trial_df[trial_df['trial_type'] == 'whisker_trial']
        auditory_trial_df = trial_df[trial_df['trial_type'] == 'auditory_trial']

        for col, df in zip(['time_since_last_whisker_stim', 'time_since_last_auditory_stim'], [whisker_trial_df, auditory_trial_df]):
            trial_df.loc[df.index, col] = df['lick_time'].diff(1).fillna(0)  # first stimulus is set to 0

        # Calculate and fill 'time_since_last_reward' for all reward types
        for col in ['time_since_last_whisker_stim', 'time_since_last_auditory_stim']:
            calculate_time_since_last_reward(trial_df, col)

        # Inverse min-max normalization of the 'time_since_last_stim' columns to the range [0, 1] where 0 is far away in time, 1 is recent
        for col in ['time_since_last_whisker_stim', 'time_since_last_auditory_stim']:
            min_val, max_val = trial_df[col].min(), trial_df[col].max()
            trial_df[f'{col}_norm'] = 1 - ((trial_df[col] - min_val) / (max_val - min_val))

        # Add time since last stimulus to data_df
        data_df['time_since_last_whisker_stim'] = trial_df['time_since_last_whisker_stim_norm']
        data_df['time_since_last_auditory_stim'] = trial_df['time_since_last_auditory_stim_norm']

        # -----------------------
        # Time since last rewards
        # -----------------------
        # Initial filtering based on reward group
        reward_filters = {
            'R+': ['auditory_trial', 'whisker_trial'],
            'R-': ['auditory_trial'],
            'R+proba': ['auditory_trial', 'whisker_trial']
        }

        if reward_group in reward_filters:
            reward_trial_df = trial_df[
                (trial_df['lick_flag'] == 1) & (trial_df['trial_type'].isin(reward_filters[reward_group]))]
            if reward_group == 'R+proba':
                reward_trial_df = reward_trial_df[reward_trial_df['reward_available'] == 1]

        # Split into whisker and auditory trials
        whisker_reward_trial_df = reward_trial_df[reward_trial_df['trial_type'] == 'whisker_trial']
        auditory_reward_trial_df = reward_trial_df[reward_trial_df['trial_type'] == 'auditory_trial']

        # Get time between rewards for each trial
        for col, df in zip(
                ['time_since_last_reward', 'time_since_last_whisker_reward', 'time_since_last_auditory_reward'],
                [reward_trial_df, whisker_reward_trial_df, auditory_reward_trial_df]):
            trial_df.loc[df.index, col] = df['lick_time'].diff(1).fillna(0) #first reward is set to 0 to split non-nan groups

        # Get inter-trial interval
        #trial_df['start_time_diff'] = trial_df['start_time'].diff(1)

        # Calculate and fill 'time_since_last_reward' for all reward types
        for col in ['time_since_last_reward', 'time_since_last_whisker_reward', 'time_since_last_auditory_reward']:
            calculate_time_since_last_reward(trial_df, col)

        # Inverse min-max normalization of the 'time_since_last_reward' columns to the range [0, 1] where 0 is far away in time, 1 is recent
        for col in ['time_since_last_reward', 'time_since_last_whisker_reward', 'time_since_last_auditory_reward']:
            min_val, max_val = trial_df[col].min(), trial_df[col].max()
            trial_df[f'{col}_norm'] = 1 - ((trial_df[col] - min_val) / (max_val - min_val))

        # Add time since last reward to data_df
        data_df['time_since_last_reward'] = trial_df['time_since_last_reward_norm']
        data_df['time_since_last_whisker_reward'] = trial_df['time_since_last_whisker_reward_norm']
        data_df['time_since_last_auditory_reward'] = trial_df['time_since_last_auditory_reward_norm']

        # Add reaction time as possible output for multinomial GLMs
        data_df['reaction_time'] = trial_df['lick_time'] - trial_df['response_window_start_time']

        # Add bias term
        data_df['bias'] = 1

        bhv_data.append(data_df)

    # Make as dataframe
    bhv_data_df = pd.concat(bhv_data, ignore_index=True)

    # Add additional mouse information
    mouse_info_df['reward_group'] = mouse_info_df['reward_group'].map({'R+': 1, 'R-': 0, 'R+proba': 2})
    bhv_data_df = bhv_data_df.merge(mouse_info_df[['mouse_id', 'reward_group']], on='mouse_id', how='left')

    # Format data
    cols = ['choice', 'prev_choice', 'stimulus_type', 'auditory', 'whisker', 'reward_given', 'prev_trial_reward_given',
            'prev_stim_reward_given']
    bhv_data_df[cols] = bhv_data_df[cols].astype('int32', errors='ignore')

    # Make NaN into zeros because has to be finite (early sessions when no previous trials exist)
    bhv_data_df.fillna(0, inplace=True)

    return bhv_data_df



def split_dataset(dataset, fraction_training=0.8):
    """
    Split dataset, either multi-mouse or single mouse, into training and test sets.
    This does not split for cross-validation datasets.
    :param dataset: pandas dataframe
    :param fraction_training: fraction of dataset to use for training
    :return:
    """

    test_size = 1 - fraction_training
    if not 0 < test_size < 1:
        raise ValueError("Test size must be between 0 and 1")


    n_samples = len(dataset)
    n_test_samples = int(np.floor(test_size * n_samples))
    np.random.seed(42)

    test_indices = np.sort(np.random.choice(n_samples, n_test_samples, replace=False))
    train_indices = np.setdiff1d(np.arange(n_samples), test_indices)  # get all others

    data_train = dataset.iloc[train_indices]
    data_test = dataset.iloc[test_indices]

    ## Check if input is a multi-mouse dataset
    #if len(dataset['mouse_id'].unique()) > 1:
    #    multi_mouse = True
    #else:
    #    multi_mouse = False
#
    ## Multi-mouse dataset
    #if multi_mouse:
    #    dataset_prefix = 'global_'
    #    # Keep a bit of each mouse in dataset, trials shuffle per mouse
#
    #        train_folds = []
    #        test_fold = []
#
    #        # Mouse stratification: keep a bit of each mouse in dataset
    #        n_fold = int(1/(1-fraction_training))
    #        skf = StratifiedKFold(n_splits=n_fold, shuffle=False, random_state=None) #5-splits to get 80-20%
    #        for fold_idx, (train_index, test_index) in enumerate(skf.split(dataset, dataset['mouse_id'])):
#
    #            # Make each split
    #            data_train, data_test = dataset.iloc[train_index], dataset.iloc[test_index]
#
    #            # Create training and test set with 80-20% split
    #            if fold_idx in range(n_fold-1):
    #                train_folds.append(data_test) #yes, test!
    #            elif fold_idx == n_fold-1:
    #                test_fold.append(data_test)
#
    #        # Concatenate all folds to make final datasets
    #        data_train = pd.concat(train_folds, ignore_index=True)
    #        data_test = pd.concat(test_fold, ignore_index=True)
#
    ## Single mouse dataset
    #else:
#
    #    n_samples = len(dataset)
    #    n_test_samples = int(np.floor(test_size * n_samples))
    #    np.random.seed(42)
#
    #    test_indices = np.sort(np.random.choice(n_samples, n_test_samples, replace=False))
    #    train_indices = np.setdiff1d(np.arange(n_samples), test_indices) # get all others
#
    #    data_train = dataset.iloc[train_indices]
    #    data_test = dataset.iloc[test_indices]

    return data_train, data_test


if __name__ == '__main__':

    experimenter = 'Axel_Bisi'
    host = socket.gethostname()
    if 'haas' in host:
        ROOT_PATH_AXEL = os.path.join(r'/mnt/lsens-analysis', 'Axel_Bisi', 'NWB_combined')
        ROOT_PATH_MYRIAM = os.path.join(r'/mnt/lsens-analysis', 'Myriam_Hamon', 'NWBFull_new')
        # ROOT_PATH_MYRIAM = os.path.join(r'/mnt/lsens-analysis', 'Myriam_Hamon', 'NWB')
        ROOT_PATH_MYRIAM = os.path.join(r'/mnt/lsens-analysis', 'Axel_Bisi', 'NWB_combined')
        INFO_PATH = os.path.join(r'/mnt/z_LSENS', 'Share', f'{experimenter}_Share', 'dataset_info')
        OUTPUT_PATH = os.path.join(r'/mnt/lsens-analysis', experimenter, 'combined_results')
        PROC_DATA_PATH = os.path.join(r'/mnt/lsens-analysis', experimenter, 'combined_data', 'processed_data')
    elif host == 'SV-07M-005':
        ROOT_PATH_AXEL = os.path.join(r'/Volumes', 'Petersen-Lab', 'analysis', 'Axel_Bisi', 'NWB_combined')
        ROOT_PATH_MYRIAM = os.path.join(r'/Volumes', 'Petersen-Lab', 'analysis', 'Myriam_Hamon', 'NWBFull_new')
        ROOT_PATH_MYRIAM = os.path.join(r'/Volumes', 'Petersen-Lab', 'analysis', 'Myriam_Hamon', 'NWB')
        ROOT_PATH_MYRIAM = os.path.join(r'/Volumes', 'Petersen-Lab', 'analysis', 'Axel_Bisi', 'NWB_combined')
        INFO_PATH = os.path.join(r'/Volumes', 'Petersen-Lab', 'z_LSENS', 'Share', 'Axel_Bisi_Share',
                                 'dataset_info')
        OUTPUT_PATH = os.path.join(r'/Volumes', 'Petersen-Lab', 'analysis', experimenter, 'combined_results')
        PROC_DATA_PATH = os.path.join(r'/Volumes', 'Petersen-Lab', 'analysis', experimenter, 'combined_data',
                                      'processed_data')
    else:
        ROOT_PATH_AXEL = os.path.join(r'\\sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'analysis', 'Axel_Bisi', 'NWB_combined')
        ROOT_PATH_MYRIAM = os.path.join(r'\\sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'analysis', 'Myriam_Hamon',
                                        'NWBFull_new')
        ROOT_PATH_MYRIAM = os.path.join(r'\\sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'analysis', 'Myriam_Hamon', 'NWB')
        ROOT_PATH_MYRIAM = os.path.join(r'\\sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'analysis', 'Axel_Bisi',
                                        'NWB_combined')
        INFO_PATH = os.path.join(r'\\sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'z_LSENS', 'Share', 'Axel_Bisi_Share',
                                 'dataset_info')
        OUTPUT_PATH = os.path.join(r'\\sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'analysis', experimenter,
                                   'combined_results')
        PROC_DATA_PATH = os.path.join(r'\\sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'analysis', experimenter,
                                      'combined_data', 'processed_data')

    #experimenter = 'Axel_Bisi'
    ## Paths
    #info_path = os.path.join('\\\\sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'analysis', experimenter, 'mice_info')
    #root_path = os.path.join('\\\\sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'analysis', experimenter, r'NWB_Combined')

    output_path = os.path.join(OUTPUT_PATH, 'glm_hmm')

    mouse_info_df = pd.read_excel(os.path.join(INFO_PATH, 'joint_mouse_reference_weight.xlsx'))
    mouse_info_df.rename(columns={'mouse_name': 'mouse_id'}, inplace=True)

    # Filter for usable mice
    mouse_info_df = mouse_info_df[
        (mouse_info_df['exclude'] == 0) &
        (mouse_info_df['reward_group'].isin(['R+', 'R-']))
        ]

    all_nwb_names = []
    all_nwb_mice = []
    all_nwb_names = os.listdir(ROOT_PATH_AXEL)
    all_nwb_mice.extend([name.split('_')[0] for name in all_nwb_names])

    # Filter by available NWB files
    mouse_ids = mouse_info_df['mouse_id'].unique()
    mouse_ids = [mouse for mouse in mouse_ids if any(mouse in name for name in all_nwb_mice)]

    ## Choose list of subject IDs
    #mouse_ids = range(80,156+1)
    ## Make list of subject if strings using zfill starting with 'AB
    #mouse_ids =['AB{}'.format(str(i).zfill(3)) for i in mouse_ids]
    #included_mice = mouse_info_df[mouse_info_df['exclude'] == 0]['mouse_id'].unique()
    #mouse_ids = [s for s in mouse_ids if s in included_mice]

    nwb_list = []

    # Check if multi-mouse data
    if len(mouse_ids) > 1:
        dataset_folder = os.path.join(output_path, 'datasets_combined')
        if not os.path.exists(dataset_folder):
            os.makedirs(dataset_folder)
    elif len(mouse_ids) == 1:
        dataset_folder = os.path.join(output_path, mouse_ids[0])
        if not os.path.exists(dataset_folder):
            os.makedirs(dataset_folder)

    # Get NWB files
    nwb_files_dict = []
    for subject in mouse_ids:
        # Get list of NWB files to use
        for nwb_name in all_nwb_names:
            nwb_dict = {}

            if subject in nwb_name:

                nwb_file = os.path.join(ROOT_PATH_AXEL, nwb_name)
                try:
                    behavior_type, day = NWB_read.get_bhv_type_and_training_day_index(nwb_file)
                except Exception as e:
                    print(e, 'Error in file {}'.format(nwb_file))
                    continue
                # Keep only whisker sessions
                if behavior_type == 'whisker':
                    nwb_dict['mouse_id'] = subject
                    nwb_dict['day'] = day
                    nwb_dict['behavior_type'] = behavior_type
                    nwb_dict['nwb_name'] = nwb_name
                    nwb_dict['nwb_file'] = nwb_file
                    nwb_files_dict.append(nwb_dict)

    # Keep only files for which the subject id has two files
    nwb_files_df = pd.DataFrame(nwb_files_dict)
    nwb_files_df['day'] = nwb_files_df['day'].map({4:2}) # when particle days done between 2 days

    # Keep only subjects with more than 2 whisker days
    nwb_files_per_subject = nwb_files_df.groupby('mouse_id')['nwb_name'].count() # no. files per subject
    nwb_files_per_subject = nwb_files_per_subject[nwb_files_per_subject >= 2]
    nwb_files_df = nwb_files_df[nwb_files_df['mouse_id'].isin(nwb_files_per_subject.index)]

    # Get list of NWB files
    nwb_list = nwb_files_df['nwb_file'].tolist()

    # Parameters
    params = {
        'fraction_training': 0.8,
         'n_trials_max': None
    }

    # Create behavior dataset
    dataset = create_behavior_dataset(nwb_list, mouse_info_df=mouse_info_df, params=params)

    print('Unique mice in data', dataset['mouse_id'].unique())
    print('Unique sessions per mouse', dataset.groupby('mouse_id')['session_id'].nunique())


    debug = False # for a single mouse
    if debug:

        # Make matrix using cols
        cols_to_plot = ['choice',
                        'prev_choice',
                        'bias',
                        'auditory', 'prev_auditory',
                        'whisker', 'prev_whisker',
                        'time_since_last_whisker_stim',
                        'time_since_last_auditory_stim',
                         'time_since_last_whisker_reward',
                         'time_since_last_auditory_reward',
                        ]
        dataset[cols_to_plot].head()
        feature_mat = dataset[cols_to_plot].values

        # Plot feature matrix
        fig, ax = plt.subplots(1,1, dpi=400)
        im = ax.imshow(feature_mat[0:50,:].T, aspect='equal', cmap='Greys', interpolation='None')
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.1, fraction=0.05)
        cbar.set_ticks([-1,0,1])
        cbar.set_label('Feature value')
        ax.set_xlabel('Trials')
        ax.set_yticks(range(len(cols_to_plot)), cols_to_plot, rotation=0, fontsize=6)

        fig.tight_layout()
        plt.savefig(os.path.join(dataset_folder, 'feature_matrix.png'), bbox_inches='tight', dpi='figure')
        plt.savefig(os.path.join(dataset_folder, 'feature_matrix.svg'), bbox_inches='tight', dpi='figure')

    # Create dataset splits
    n_splits = 10
    for split_idx in range(n_splits):
        data_train, data_test = split_dataset(dataset, fraction_training=params['fraction_training'])

        # Save behavior dataset
        data_folder = 'dataset_{}'.format(split_idx)
        dataset_path = os.path.join(dataset_folder, data_folder)
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)
        dataset.to_pickle(os.path.join(dataset_path, 'dataset.pkl'))
        data_train.to_pickle(os.path.join(dataset_path, 'data_train.pkl'))
        data_test.to_pickle(os.path.join(dataset_path, 'data_test.pkl'))

        print('Datasets saved in {}'.format(dataset_path))

    print('Dataset creation done.')




