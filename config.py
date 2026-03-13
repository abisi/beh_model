#! /usr/bin/env/python3
"""
@author: Axel Bisi
@project: beh_model
@file: config.py
@description: Central configuration — paths, hyperparameters, and dataset settings.
              Paths are resolved automatically based on the current hostname.
"""

import socket
from pathlib import Path

# ── Identity ──────────────────────────────────────────────────────────────────

EXPERIMENTER = 'Axel_Bisi'

# ── Host-based path resolution ────────────────────────────────────────────────

_HOST = socket.gethostname()

if 'haas' in _HOST:
    NWB_PATH    = Path('/mnt/lsens-analysis', EXPERIMENTER, 'NWB_combined')
    INFO_PATH   = Path('/mnt', 'share_internal', f'{EXPERIMENTER}_Share', 'dataset_info')
    OUTPUT_PATH = Path('/mnt/lsens-analysis', EXPERIMENTER, 'combined_results')
elif _HOST == 'SV-07M-005':
    NWB_PATH    = Path('/Volumes/Petersen-Lab/analysis', EXPERIMENTER, 'NWB_combined')
    INFO_PATH   = Path('/Volumes/Petersen-Lab/share_internal', f'{EXPERIMENTER}_Share', 'dataset_info')
    OUTPUT_PATH = Path('/Volumes/Petersen-Lab/analysis', EXPERIMENTER, 'combined_results')
else:
    NWB_PATH    = Path(r'\\sv-nas1.rcp.epfl.ch\Petersen-Lab\analysis', EXPERIMENTER, 'NWB_combined')
    INFO_PATH   = Path(r'\\sv-nas1.rcp.epfl.ch\Petersen-Lab\share_internal', f'{EXPERIMENTER}_Share', 'dataset_info')
    OUTPUT_PATH = Path(r'\\sv-nas1.rcp.epfl.ch\Petersen-Lab\analysis', EXPERIMENTER, 'combined_results')

# ── Dataset creation ──────────────────────────────────────────────────────────

FRACTION_TRAINING = 0.8
N_TRIALS_MAX      = None        # None = no trial limit per session
MIN_WHISKER_DAYS  = 1           # minimum number of whisker sessions to include a subject
REWARD_GROUPS     = ['R+', 'R-']  # reward groups to include

# ── GLM-HMM fitting ───────────────────────────────────────────────────────────

N_SPLITS    = 3
N_STATES    = 6
N_INSTANCES = 1

TRIAL_TYPES = 'whisker_trial' #    'all_trials',


HMM_PARAMS = {
    'prior_sigma':   0.3,
    'prior_alpha':   2,
    'kappa': 0,
    'n_train_iters': 300,
    'tolerance':     1e-4,
    'noise_level':   0.1,
}

if TRIAL_TYPES == 'whisker_trial':
    FEATURES = [
        'bias',
        'time_since_last_auditory_lick',
        'time_since_last_whisker_lick',
        'jaw_distance',
        'whisker_angle',
        'pupil_area',
        'cumulative_reward'
    ]
else:
    FEATURES = [
        'bias',
        'auditory',
        'whisker',
        'time_since_last_auditory_lick',
        'time_since_last_whisker_lick',
        'jaw_distance',
        'whisker_angle',
        'pupil_area'
        'cumulative_reward'

    ]