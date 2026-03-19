#! /usr/bin/env/python3
"""
@author: Axel Bisi
@project: beh_model
@file: utils.py
@time: 8/2/2024 1:28 PM
"""


# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
from scipy.optimize import linear_sum_assignment
import ssm
import autograd.numpy.random as npr
npr.seed(42)

import plotting_utils

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

def reindex_whisker_days(bhv_data):
    """
    Identify and rearrange mice with fewer than `min_days` whisker days.
    Combines whisker_on_1 and whisker_on_2 sessions into a single whisker day entry.
    Updates and corrects day ids for few mice.
    :param bhv_data: trial table
    :return
    """
    min_days = 3
    bhv_data = bhv_data.copy()

    # Identify mice with fewer than `min_days` whisker days
    mice_n_whisker_days = (
        bhv_data.loc[bhv_data['behavior'] == 'whisker']
        .groupby('mouse_id')['day']
        .nunique()
    )
    mice_few_days = mice_n_whisker_days[mice_n_whisker_days < min_days].index

    # Identify mice that have whisker_on control sessions -> make it whisker days
    mice_control_days = bhv_data.loc[bhv_data['behavior'].isin(['whisker_on_1', 'whisker_on_2'])]['mouse_id'].unique()

    # Filter to relevant mice
    mice_to_rearrange = [m for m in mice_few_days if m in mice_control_days]

    print(f'Re-indexing whisker days for: {mice_to_rearrange}')
    for mouse_id in mice_to_rearrange:

        # Extract relevant sessions
        on_1_df = bhv_data.loc[(bhv_data['mouse_id'] == mouse_id) & (bhv_data['behavior'] == 'whisker_on_1')]
        on_2_df = bhv_data.loc[(bhv_data['mouse_id'] == mouse_id) & (bhv_data['behavior'] == 'whisker_on_2')]

        # Skip if missing one of the sessions
        if on_1_df.empty or on_2_df.empty:
            continue

        # Adjust session info for on_2 to follow on_1
        new_session_id = on_1_df['session_id'].iloc[0]
        offset = 100 + on_1_df.iloc[-1]['stop_time']

        on_2_df = on_2_df.copy()
        on_2_df.loc[:, 'session_id'] = new_session_id

        # Shift all time columns (except piezo_lick_times)
        time_cols = [c for c in on_2_df.columns if 'time' in c and c != 'piezo_lick_times']
        on_2_df.loc[:, time_cols] = on_2_df[time_cols].apply(lambda col: col + offset)

        # Merge sessions and relabel
        whisker_day_merged = pd.concat([on_1_df, on_2_df], ignore_index=True)
        whisker_day_merged.loc[:, 'behavior'] = 'whisker'
        whisker_day_merged.loc[:, 'day'] = 2 if mouse_id != 'AB073' else 1

        # Remove old control sessions and append merged data
        drop_behaviors = ['whisker_on_1', 'whisker_on_2', 'whisker_off', 'whisker_off_1']
        bhv_data = bhv_data.loc[~((bhv_data['mouse_id'] == mouse_id) & (bhv_data['behavior'].isin(drop_behaviors)))]
        bhv_data = pd.concat([bhv_data, whisker_day_merged], ignore_index=True)

    # Fix special case for AB073
    mask = (bhv_data['mouse_id'] == 'AB073') & (bhv_data['day'] == 4)
    bhv_data.loc[mask, 'day'] = 2

    # Reindex AB155 days from 0,1,2,3 to 0,0,1,2
    mask = bhv_data.mouse_id=='AB155'
    day_map = {zip(range(4),[0,0,1,2])}
    bhv_data[mask] = bhv_data[mask].replace({'day': day_map})

    return bhv_data

def build_feature_sets(features, trial_types):
    """
    Build a dict of named feature sets to fit:
    Build a dict of named feature sets to fit:
      - 'full':          all features
      - 'drop_<feat>':   all features except <feat>  (leave-one-out)

    :param features: full list of feature names
    :return: dict mapping model_name -> feature list
    """

    sets={}
    sets['full'] = features
    #sets['bias_only'] = ['bias']
    mvt_features = ['jaw_distance', 'whisker_angle', 'nose_norm_distance', 'pupil_area']
    mvt_features = ['jaw_distance', 'whisker_angle', 'pupil_area']

    non_mvt_features = [f for f in features if f not in mvt_features]

    if trial_types == 'all_trials':
        trial_types = ['whisker', 'auditory']
        non_mvt_features = [f for f in features if f not in mvt_features]
        # Bias and trial types only
        #sets['bias_trial_types_only'] = ['bias'] + trial_types
        # Remvoe trial types
        #sets['drop_trial_type'] = [f for f in features if f not in trial_types]
        # Remove all history featuress
        #sets['drop_history'] = mvt_features + ['bias', 'whisker', 'auditory']
    elif trial_types == 'whisker':
        # Remove all history features
        #sets['drop_history'] = mvt_features + ['bias']
        #sets['drop_mvt'] = non_mvt_features
        non_mvt_features = [f for f in features if f not in mvt_features and f not in ['whisker', 'auditory']]

    ## Remove all mvt features
    #sets['drop_mvt'] = non_mvt_features
#
    ## Remove all trial info, keep only mvt
    #sets['mvt_only'] = mvt_features
#
    ## Bias and history only information
    #sets['bias_history_only'] = ['bias'] + [f for f in features if 'time_since_last' in f]

    # Remove all features 1-by-1
    #for feat in features:
    #    sets[f'drop_{feat}'] = [f for f in features if f != feat]

    # Add model with bias
    if 'bias' not in features:
        sets['full_with_bias'] = features + ['bias']

    return sets

def build_glmhmm(n_states, input_dim, prior_sigma, prior_alpha, kappa,
                 obs_dim=1, num_categories=2):
    """ Build input-driven HMM i.e. GLM-HMM with specified parameters. """
    model = ssm.HMM(
        n_states, obs_dim, input_dim,
        observations="input_driven_obs",
        observation_kwargs=dict(C=num_categories, prior_sigma=prior_sigma),
        transitions="sticky",
        transition_kwargs=dict(alpha=prior_alpha, kappa=kappa),
    )
    return model

def add_noise_to_weights(weights, noise_level=0.2):
    """
    Add noise to weights
    :param weights: weights to add noise to
    :param noise_level: level of noise to add
    :return: weights with added noise
    """

    #n_weights = weights.shape[0]
    #gauss_noise_weights = np.random.normal(0, noise_level, n_weights)
    #weights = weights + gauss_noise_weights
    return weights + np.random.normal(0, noise_level, weights.shape)
    return weights

def get_expected_states(model, outputs, inputs):
    """
    Get expected states from model
    :param model: model to get expected states from
    :param outputs: list of outputs
    :param inputs: list of inputs
    :return:
    """

    expected_states = [model.expected_states(data=data, input=inpt)[0]
    for data, inpt in zip(outputs, inputs)]

    return expected_states

def get_predicted_labels(model, inputs, posteriors):
    """
    Get predicted labels from model
    :param model: model to get predicted labels from
    :param inputs: list of inputs
    :param posteriors: list of posteriors
    :return:
    """

    # Get lick probabilities from model for every trial
    prob_lick = [np.exp(model.observations.calculate_logits(input=inpt)) for inpt in inputs]
    prob_lick = np.concatenate(prob_lick, axis=0)
    prob_lick = prob_lick[:, :, 1]

    # Multiply posterior probabilities and prob_lick
    final_prob_lick = np.sum(np.multiply(posteriors, prob_lick), axis=1)

    # Get the predicted label for each time step
    predicted_labels = np.around(final_prob_lick, decimals=0).astype('int')

    # Segment predictions per session
    predicted_labels = np.split(predicted_labels, np.cumsum([len(inpt) for inpt in inputs])[:-1])

    return predicted_labels


def calculate_predictive_accuracy(true_labels, predicted_labels):
    """
    Calculate predictive accuracy and balanced accuracy
    :param true_labels: list of arrays or single array with true labels
    :param predicted_labels: list of arrays or single array with predicted labels
    :return: predictive_acc, balanced_acc
    """
    # First flatten lists of sessions to get a unique vector for all the data
    true_labels = np.concatenate(true_labels, axis=0)
    predicted_labels = np.concatenate(predicted_labels, axis=0)

    # Calculate predictive accuracy
    predictive_acc = np.sum(true_labels[:, 0] == predicted_labels) / len(true_labels)

    # Calculate balanced accuracy (average of per-class recall)
    # For class 0 (no lick)
    class_0_mask = true_labels[:, 0] == 0
    class_0_correct = np.sum((true_labels[class_0_mask, 0] == predicted_labels[class_0_mask]))
    class_0_total = np.sum(class_0_mask)
    class_0_recall = class_0_correct / class_0_total if class_0_total > 0 else 0

    # For class 1 (lick)
    class_1_mask = true_labels[:, 0] == 1
    class_1_correct = np.sum((true_labels[class_1_mask, 0] == predicted_labels[class_1_mask]))
    class_1_total = np.sum(class_1_mask)
    class_1_recall = class_1_correct / class_1_total if class_1_total > 0 else 0

    # Balanced accuracy is the average of per-class recalls
    balanced_acc = (class_0_recall + class_1_recall) / 2

    return predictive_acc, balanced_acc

def compute_distance_matrix(w1, w2, method='euclidean'):
    """
    Compute the distance matrix between states of two subjects (1D weights).
    :param w1: first 1d vector
    :param w2: second 1s vector
    :param method:
    :return:
    """
    if method=='euclidean':
        # Compute the Euclidean distance between states
        distance = np.linalg.norm(w1 - w2)
    elif method=='cosine':
        # Compute the cosine similarity between states
        distance = 1 - np.dot(w1[:, None], w2) / (np.linalg.norm(w1[:, None]) * np.linalg.norm(w2))
    else:
        raise ValueError(f"Invalid method: {method}")
    return distance

def compute_permutations_from_viterbi(viterbi_dict, n_states):
    """
    Compute within-mouse state permutations from Viterbi-decoded sequences.

    :param viterbi_dict: {(split_idx, inst_idx): z_array (n_trials,)}
                         All arrays must be the same length (full trial set).
    :param n_states: int
    :return: {(n_states, split_idx, inst_idx): perm_array}
    """
    if n_states == 1:
        return {(n_states, s, i): np.array([0])
                for (s, i) in viterbi_dict}

    model_ids = sorted(viterbi_dict.keys())
    # Reference = first model (equivalent to use_mean_reference=False)
    ref_id    = model_ids[0]
    z_ref     = viterbi_dict[ref_id]

    permutations = {}
    for (split_idx, inst_idx) in model_ids:
        z = viterbi_dict[(split_idx, inst_idx)]
        perm = find_permutation(z_ref, z, K1=n_states, K2=n_states)
        permutations[(n_states, split_idx, inst_idx)] = perm

    return permutations

def align_weights_dataframe(df, use_mean_reference=True):
    """
    Align GLM-HMM states in a long-format dataframe before averaging.

    States are aligned **separately within each reward group**, so the reference
    is never contaminated by mice from the other group.

    Usage:
        aligned_df, permutations = align_weights_dataframe(df)
        # Now you can safely average within a reward group
        mean_weights = aligned_df.groupby(
            ['reward_group', 'n_states', 'state_idx', 'feature'])['weight'].mean()
        # Use permutations to align transition matrices
        for (n_states, split_idx, inst_idx), perm in permutations.items():
            tm_aligned = tm[np.ix_(perm, perm)]

    :param df: DataFrame with columns:
               ['n_states', 'split_idx', 'instance_idx', 'state_idx',
                'feature', 'weight', 'reward_group']
    :param use_mean_reference: If True, align to within-group mean weights;
                               if False, align to the first model of each group.
    :return: (aligned_df, permutations_dict)
             - aligned_df        : DataFrame with aligned state_idx values
             - permutations_dict : {(n_states, split_idx, instance_idx): permutation_array}
    """
    required_cols = ['n_states', 'split_idx', 'instance_idx', 'state_idx',
                     'feature', 'weight', 'reward_group']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")

    aligned_dfs       = []
    all_permutations  = {}   # {(n_states, split_id, inst_id): perm}

    for n_states in sorted(df['n_states'].unique()):

        # Skip alignment for 1-state models – nothing to permute
        if n_states == 1:
            aligned_dfs.append(df[df['n_states'] == n_states].copy())
            continue

        df_nstates = df[df['n_states'] == n_states].copy()

        # ------------------------------------------------------------------
        # Align within each reward group independently
        # ------------------------------------------------------------------
        aligned_nstates_dfs = []

        for reward_group in sorted(df_nstates['reward_group'].unique()):

            df_subset = df_nstates[df_nstates['reward_group'] == reward_group].copy()

            # Unique models in this reward group
            models = df_subset[['split_idx', 'instance_idx']].drop_duplicates()

            # Extract weight matrices for each model
            weight_matrices = []
            model_ids       = []

            for _, row in models.iterrows():
                split_id = row['split_idx']
                inst_id  = row['instance_idx']

                model_data = df_subset[
                    (df_subset['split_idx']    == split_id) &
                    (df_subset['instance_idx'] == inst_id)
                ]

                # Pivot to weight matrix: rows = states, cols = features
                weight_matrix = model_data.pivot(
                    index='state_idx', columns='feature', values='weight'
                )
                weight_matrices.append(weight_matrix.values)
                model_ids.append((split_id, inst_id))

            # Compute within-group reference
            if use_mean_reference:
                reference_weights = np.mean(weight_matrices, axis=0)
            else:
                reference_weights = weight_matrices[0]

            # Align each model to the group reference
            permutations = []

            for (split_id, inst_id), weights in zip(model_ids, weight_matrices):
                perm = align_states(reference_weights, weights)
                permutations.append(perm)

                # Store permutation (key is unique because split/instance IDs
                # already belong to exactly one reward group)
                all_permutations[(n_states, split_id, inst_id)] = perm

                # Build inverse permutation: original_state -> aligned_state
                inverse_perm            = np.empty_like(perm)
                inverse_perm[perm]      = np.arange(len(perm))

                # Remap state_idx for this model
                mask = (
                    (df_subset['split_idx']    == split_id) &
                    (df_subset['instance_idx'] == inst_id)
                )
                df_subset.loc[mask, 'state_idx'] = (
                    df_subset.loc[mask, 'state_idx'].map(lambda x: inverse_perm[x])
                )

            # Alignment diagnostics
            unique_perms = np.unique(permutations, axis=0)
            if len(unique_perms) < len(model_ids):
                consistency = 100 * (1 - (len(unique_perms) - 1) / len(model_ids))

            aligned_nstates_dfs.append(df_subset)

        # Recombine reward groups for this n_states level
        aligned_dfs.append(pd.concat(aligned_nstates_dfs, ignore_index=True))

    aligned_df = pd.concat(aligned_dfs, ignore_index=True)
    return aligned_df, all_permutations

# Function to align states of a subject to a reference using the Hungarian algorithm
def align_states(reference_weights, subject_weights):
    """
    Align the states of a subject to a reference using the Hungarian algorithm.
    :param reference_weights:
    :param subject_weights:
    :return:
    """
    ## Compute the distance matrix between the reference and the subject's states
    #print('distance between', reference_weights.shape, subject_weights.shape)
    #distance_matrix = compute_distance_matrix(reference_weights, subject_weights) #code of before
#
    ## Use Hungarian algorithm to find the best matching
    #row_ind, col_ind = linear_sum_assignment(distance_matrix)
#
    ## Return the permuted state indices for the subject
    #return col_ind

    n_ref = reference_weights.shape[0]
    n_sub = subject_weights.shape[0]

    # Build proper pairwise distance matrix
    distance_matrix = np.array([
        [compute_distance_matrix(reference_weights[i], subject_weights[j])
         for j in range(n_sub)]
        for i in range(n_ref)
    ])
    row_ind, col_ind = linear_sum_assignment(distance_matrix)
    return col_ind


def align_states_across_subjects(w_subjects, use_mean_reference=True):
    """
    Align state indices across multiple subjects based on the similarity of their 1D weights.

    W_subjects: List of arrays where each array contains the 1D weights for each state of a subject.
    use_mean_reference: If True, align states to the mean state vector across all subjects.
                        If False, align to the first subject.

    Returns: List of permuted weight arrays, where each subject's states are aligned.
    """
    # Number of subjects
    n_subjects = len(w_subjects)

    # Number of states per subject (assumed constant across subjects)
    n_states = len(w_subjects[0])

    # Compute the reference states (either from the mean or the first subject)
    if use_mean_reference:
        # Compute the mean state vector across all subjects
        reference_weights = np.mean(w_subjects, axis=0)
    else:
        # Use the first subject as the reference
        reference_weights = w_subjects[0]

    # Initialize the list to store permuted weights for each subject
    aligned_subjects = []

    for s in range(n_subjects):
        # Align the states of the current subject to the reference
        permuted_indices = align_states(reference_weights, w_subjects[s])

        # Permute the subject's weights based on the matching indices
        permuted_weights = w_subjects[s][permuted_indices]

        # Store the permuted weights
        aligned_subjects.append(permuted_weights)

    return aligned_subjects


def assign_most_likely_state(sess_data, posterior_cols, tol=1e-6):
    """
    Compute most likely state per trial with smoothing rules:
    1. Assign argmax state
    2. If all posteriors are equal (flat distribution), inherit previous state
    3. Remove single-trial states by merging with next state

    Parameters
    ----------
    sess_data : pandas.DataFrame
        Data containing posterior probabilities.
    posterior_cols : list of str
        Column names for posterior probabilities.
    tol : float
        Tolerance for equality check.

    Returns
    -------
    np.ndarray
         Most likely state sequence.
    """
    posteriors = sess_data[posterior_cols].values
    most_likely_state = np.argmax(posteriors, axis=1)

    # Step 2: handle flat posterior (all states equal)
    for idx in range(1, len(most_likely_state) - 1):
        row = posteriors[idx]
        if np.all(np.isclose(row, row[0], atol=tol)):
            most_likely_state[idx] = most_likely_state[idx - 1]

    # Step 3: remove single-trial states
    for idx in range(1, len(most_likely_state) - 1):
        if (
                most_likely_state[idx] != most_likely_state[idx - 1]
                and most_likely_state[idx] != most_likely_state[idx + 1]
        ):
            most_likely_state[idx] = most_likely_state[idx + 1]

    return most_likely_state


def compute_pairwise_distance(w1, w2, method='euclidean'):
    """
    Compute the distance between two 1D weight vectors.
    :param w1:     1D array of shape (M,)
    :param w2:     1D array of shape (M,)
    :param method: 'euclidean' or 'cosine'
    :return:       scalar distance
    """
    if method == 'euclidean':
        return np.linalg.norm(w1 - w2)
    elif method == 'cosine':
        return 1 - np.dot(w1, w2) / (np.linalg.norm(w1) * np.linalg.norm(w2))
    else:
        raise ValueError(f"Invalid method: {method}")


# ===========================================================================
# SSM-style overlap-based permutation (trial-space alignment)
# ===========================================================================

def compute_state_overlap(z1, z2, K1=None, K2=None):
    """
    Compute co-occurrence counts between two hard state-assignment sequences.
    :param z1: (T,) int array — state assignments from model 1
    :param z2: (T,) int array — state assignments from model 2
    :return:   (K1, K2) overlap matrix
    """
    assert z1.dtype == int and z2.dtype == int
    assert z1.shape == z2.shape
    assert z1.min() >= 0 and z2.min() >= 0

    K1 = z1.max() + 1 if K1 is None else K1
    K2 = z2.max() + 1 if K2 is None else K2
    overlap = np.zeros((K1, K2))
    for k1 in range(K1):
        for k2 in range(K2):
            overlap[k1, k2] = np.sum((z1 == k1) & (z2 == k2))
    return overlap

    #K1 = z1.max() + 1 if K1 is None else K1
    #K2 = z2.max() + 1 if K2 is None else K2
    ## One-hot encode both sequences and matrix-multiply
    #Z1 = (z1[:, None] == np.arange(K1))  # (T, K1)
    #Z2 = (z2[:, None] == np.arange(K2))  # (T, K2)
    #return Z1.T @ Z2  # (K1, K2)


def find_permutation(z1, z2, K1=None, K2=None):
    """
    Find the permutation of z2's states that best matches z1 via the
    Hungarian algorithm applied to the state-overlap matrix.
    :param z1: (T,) int array — reference state assignments
    :param z2: (T,) int array — subject state assignments
    :return:   (K,) int array — perm such that z2[perm] aligns to z1
    """
    overlap = compute_state_overlap(z1, z2, K1=K1, K2=K2)
    K1, K2 = overlap.shape
    tmp, perm = linear_sum_assignment(-overlap)
    assert np.all(tmp == np.arange(K1)), "All indices should have been matched!"
    if K1 < K2:
        unused = np.array(list(set(np.arange(K2)) - set(perm)))
        perm = np.concatenate((perm, unused))
    return perm


# ===========================================================================
# Weight-space alignment (used for cross-mouse, or fallback)
# ===========================================================================

def align_states(reference_weights, subject_weights, method='euclidean'):
    """
    Align the states of a subject to a reference using the Hungarian algorithm
    applied to a pairwise weight-distance matrix.
    :param reference_weights: (K, M) array
    :param subject_weights:   (K, M) array
    :return:                  (K,) permutation array col_ind
    """
    n_ref = reference_weights.shape[0]
    n_sub = subject_weights.shape[0]

    # Vectorised Euclidean: shape (n_ref, n_sub)
    if method == 'euclidean':
        distance_matrix = np.linalg.norm(
            reference_weights[:, None, :] - subject_weights[None, :, :],
            axis=-1
        )
    else:
        distance_matrix = np.array([
            [compute_pairwise_distance(reference_weights[i], subject_weights[j], method=method)
             for j in range(n_sub)]
            for i in range(n_ref)
        ])

    _, col_ind = linear_sum_assignment(distance_matrix)
    return col_ind


# ===========================================================================
# Viterbi-based within-mouse permutation computation
# ===========================================================================

def compute_permutations_from_viterbi(viterbi_dict, n_states):
    """
    Compute within-mouse state permutations from Viterbi-decoded sequences.

    Uses the SSM overlap approach (trial-space alignment) rather than
    weight-space alignment, so permutations are grounded in actual state
    occupancy rather than parameter similarity.

    :param viterbi_dict: {(split_idx, inst_idx): z_array (n_trials,) int}
                         All arrays must have the same length (full trial set,
                         same trial order across splits).
    :param n_states:     int — number of hidden states K
    :return:             {(n_states, split_idx, inst_idx): perm_array}
    """
    if n_states == 1:
        return {(1, s, i): np.array([0]) for (s, i) in viterbi_dict}

    model_ids = sorted(viterbi_dict.keys())
    ref_id = model_ids[0]
    z_ref = viterbi_dict[ref_id].astype(int)

    permutations = {}
    for (split_idx, inst_idx) in model_ids:
        z = viterbi_dict[(split_idx, inst_idx)].astype(int)
        perm = find_permutation(z_ref, z, K1=n_states, K2=n_states)
        permutations[(n_states, split_idx, inst_idx)] = perm

    return permutations


# ===========================================================================
# DataFrame-level alignment (within-mouse, across splits × instances)
# ===========================================================================

def align_weights_dataframe(df, use_mean_reference=True, permutations=None):
    """
    Align GLM-HMM states in a long-format weight DataFrame before averaging.

    States are aligned separately within each reward group. Permutations can
    be supplied (e.g. from Viterbi sequences) to bypass weight-based alignment.

    Usage:
        # With Viterbi permutations (preferred for within-mouse alignment):
        viterbi_perms = compute_permutations_from_viterbi(viterbi_dict, n_states)
        aligned_df, permutations = align_weights_dataframe(df, permutations=viterbi_perms)

        # Weight-based fallback:
        aligned_df, permutations = align_weights_dataframe(df, use_mean_reference=False)

        # Average safely after alignment:
        mean_weights = aligned_df.groupby(
            ['reward_group', 'n_states', 'state_idx', 'feature'])['weight'].mean()

        # Align transition matrices consistently:
        for (n_states, split_idx, inst_idx), perm in permutations.items():
            tm_aligned = tm[np.ix_(perm, perm)]

    :param df:                 DataFrame with columns:
                               ['n_states', 'split_idx', 'instance_idx',
                                'state_idx', 'feature', 'weight', 'reward_group']
    :param use_mean_reference: if True and no permutations supplied, align to
                               within-group mean weights; if False, align to
                               the first model.
    :param permutations:       optional {(n_states, split_idx, inst_idx): perm_array}
                               If provided, weight-based alignment is skipped.
    :return: (aligned_df, permutations_dict)
    """
    required_cols = ['n_states', 'split_idx', 'instance_idx',
                     'state_idx', 'feature', 'weight', 'reward_group']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")

    aligned_dfs = []
    all_permutations = {}

    for n_states in sorted(df['n_states'].unique()):

        if n_states == 1:
            aligned_dfs.append(df[df['n_states'] == n_states].copy())
            continue

        df_nstates = df[df['n_states'] == n_states].copy()

        aligned_nstates_dfs = []

        for reward_group in sorted(df_nstates['reward_group'].unique()):

            df_subset = df_nstates[df_nstates['reward_group'] == reward_group].copy()
            models = df_subset[['split_idx', 'instance_idx']].drop_duplicates()

            weight_matrices = []
            model_ids = []

            for _, row in models.iterrows():
                split_id = row['split_idx']
                inst_id = row['instance_idx']

                model_data = df_subset[
                    (df_subset['split_idx'] == split_id) &
                    (df_subset['instance_idx'] == inst_id)
                    ]
                weight_matrix = model_data.pivot(
                    index='state_idx', columns='feature', values='weight'
                )
                weight_matrices.append(weight_matrix.values)
                model_ids.append((split_id, inst_id))

            # Reference for weight-based fallback
            if use_mean_reference:
                reference_weights = np.mean(weight_matrices, axis=0)
            else:
                reference_weights = weight_matrices[0]

            for (split_id, inst_id), weights in zip(model_ids, weight_matrices):
                key = (n_states, split_id, inst_id)

                # Use supplied permutation (e.g. Viterbi) if available
                if permutations is not None and key in permutations:
                    perm = permutations[key]
                else:
                    perm = align_states(reference_weights, weights)

                all_permutations[key] = perm

                inverse_perm = np.empty_like(perm)
                inverse_perm[perm] = np.arange(len(perm))

                mask = (
                        (df_subset['split_idx'] == split_id) &
                        (df_subset['instance_idx'] == inst_id)
                )
                df_subset.loc[mask, 'state_idx'] = (
                    df_subset.loc[mask, 'state_idx'].map(lambda x: inverse_perm[x])
                )

            aligned_nstates_dfs.append(df_subset)

        aligned_dfs.append(pd.concat(aligned_nstates_dfs, ignore_index=True))

    aligned_df = pd.concat(aligned_dfs, ignore_index=True)
    return aligned_df, all_permutations


def _load_viterbi_dict(
        path_map: dict[tuple, Path],
        valid_keys: set[tuple],
        n_states: int,
) -> tuple[dict, bool]:
    """
    Load Viterbi-decoded state sequences from data_preds.h5 for all valid
    (split_idx, inst_idx) pairs belonging to one mouse.

    Tries 'most_likely_state' column first; falls back to argmax of
    'posterior_state_*' columns if absent.

    All sequences must have the same length (full trial set, same trial order
    across splits) — if any mismatch is detected, returns (empty, False) so
    the caller can fall back to weight-based alignment.

    :param path_map:   {(si, inst): Path to model directory}
    :param valid_keys: subset of path_map keys whose NPZ file exists
    :param n_states:   number of hidden states K (unused here, kept for symmetry)
    :return:           (viterbi_dict, sequences_ok)
                       viterbi_dict : {(si, inst): z array (T,) int}
                       sequences_ok : False if any file is missing or lengths differ
    """
    viterbi_dict: dict[tuple, np.ndarray] = {}
    ref_len: int | None = None

    for (si, inst) in valid_keys:
        h5 = path_map[(si, inst)] / "data_preds.h5"
        if not h5.exists():
            logger.warning(f"    data_preds.h5 not found: {h5}")
            return {}, False

        try:
            data = pd.read_hdf(h5)
        except Exception as e:
            logger.warning(f"    Could not read {h5}: {e}")
            return {}, False

        if "most_likely_state" in data.columns:
            z = data["most_likely_state"].values.astype(int)
        else:
            post_cols = sorted(c for c in data.columns if c.startswith("posterior_state_"))
            if not post_cols:
                logger.warning(f"    No state columns found in {h5}")
                return {}, False
            z = data[post_cols].values.argmax(axis=1).astype(int)

        if ref_len is None:
            ref_len = len(z)
        elif len(z) != ref_len:
            logger.warning(
                f"    Sequence length mismatch: expected {ref_len}, got {len(z)} in {h5}"
            )
            return {}, False

        viterbi_dict[(si, inst)] = z

    return viterbi_dict, True


def _build_weight_df(
        path_map: dict[tuple, Path],
        valid_keys: set[tuple],
        mouse_id: str,
        rg: str,
        n_states: int,
        feats: list[str],
) -> pd.DataFrame:
    """
    Build a long-form weight DataFrame for one mouse from NPZ result files.

    :param path_map:   {(si, inst): Path to model directory}
    :param valid_keys: subset of path_map keys whose NPZ file exists
    :param mouse_id:   subject ID string
    :param rg:         reward group string, e.g. 'R+' or 'R-'
    :param n_states:   number of hidden states K
    :param feats:      ordered list of feature names, length M
    :return:           DataFrame with columns:
                       [mouse_id, reward_group, n_states, split_idx,
                        instance_idx, state_idx, feature, weight]
    """
    n_feats = len(feats)
    feat_tile = feats * n_states  # repeating feature names across states
    state_ids = np.repeat(np.arange(n_states), n_feats).tolist()

    rows = []
    for (si, inst) in valid_keys:
        f = path_map[(si, inst)] / "fit_glmhmm_results.npz"
        if not f.exists():
            continue
        res = np.load(f, allow_pickle=True)["arr_0"].item()
        w = np.array(res["weights"])  # (K, 1, M)
        w_flat = w[:, 0, :].ravel()  # (K*M,)

        rows.extend(zip(
            [mouse_id] * (n_states * n_feats),
            [rg] * (n_states * n_feats),
            [n_states] * (n_states * n_feats),
            [si] * (n_states * n_feats),
            [inst] * (n_states * n_feats),
            state_ids,
            feat_tile,
            w_flat.tolist(),
        ))

    return pd.DataFrame(rows, columns=[
        "mouse_id", "reward_group", "n_states", "split_idx",
        "instance_idx", "state_idx", "feature", "weight",
    ])


def _mean_weight_matrix(
        aligned_df: pd.DataFrame,
        n_states: int,
        feats: list[str],
) -> np.ndarray:
    """
    Compute the (K, M) mean weight matrix from an aligned long-form weight
    DataFrame, averaging across splits and instances.

    :param aligned_df: long-form weight DataFrame after align_weights_dataframe
    :param n_states:   number of hidden states K
    :param feats:      ordered list of feature names, length M
    :return:           (K, M) float array
    """
    pivot = (
        aligned_df.groupby(["state_idx", "feature"])["weight"]
        .mean()
        .unstack("feature")
        .reindex(index=range(n_states), columns=feats, fill_value=0.0)
    )
    return pivot.values


def align_weights_dataframe(
        df: pd.DataFrame,
        use_mean_reference: bool = True,
        permutations: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Align GLM-HMM states in a long-format weight DataFrame before averaging.

    States are aligned separately within each reward group. Permutations can
    be supplied (e.g. from Viterbi sequences) to bypass weight-based alignment.

    :param df:                 DataFrame with columns:
                               [n_states, split_idx, instance_idx, state_idx,
                                feature, weight, reward_group]
    :param use_mean_reference: if True and no permutations supplied, align to
                               within-group mean weights; if False, align to
                               the first model.
    :param permutations:       optional {(n_states, split_idx, inst_idx): perm}
                               where perm[new_s] = old_s.
                               If provided, weight-based alignment is skipped.
    :return: (aligned_df, permutations_dict)
             aligned_df       : DataFrame with remapped state_idx values
             permutations_dict: {(n_states, split_idx, inst_idx): perm array}
    """
    required_cols = [
        "n_states", "split_idx", "instance_idx",
        "state_idx", "feature", "weight", "reward_group",
    ]
    if not all(c in df.columns for c in required_cols):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")

    aligned_dfs = []
    all_permutations = {}

    for n_states in sorted(df["n_states"].unique()):

        if n_states == 1:
            aligned_dfs.append(df[df["n_states"] == n_states].copy())
            continue

        df_nstates = df[df["n_states"] == n_states].copy()
        aligned_nstates_dfs = []

        for reward_group in sorted(df_nstates["reward_group"].unique()):

            df_subset = df_nstates[df_nstates["reward_group"] == reward_group].copy()
            models = df_subset[["split_idx", "instance_idx"]].drop_duplicates()

            weight_matrices = []
            model_ids = []

            for _, row in models.iterrows():
                split_id = row["split_idx"]
                inst_id = row["instance_idx"]
                model_data = df_subset[
                    (df_subset["split_idx"] == split_id) &
                    (df_subset["instance_idx"] == inst_id)
                    ]
                weight_matrix = model_data.pivot(
                    index="state_idx", columns="feature", values="weight"
                )
                weight_matrices.append(weight_matrix.values)
                model_ids.append((split_id, inst_id))

            if use_mean_reference:
                reference_weights = np.mean(weight_matrices, axis=0)
            else:
                reference_weights = weight_matrices[0]

            for (split_id, inst_id), weights in zip(model_ids, weight_matrices):
                key = (n_states, split_id, inst_id)

                if permutations is not None and key in permutations:
                    perm = permutations[key]
                else:
                    perm = align_states(reference_weights, weights)

                all_permutations[key] = perm

                inv_perm = np.empty_like(perm)
                inv_perm[perm] = np.arange(len(perm))

                mask = (
                        (df_subset["split_idx"] == split_id) &
                        (df_subset["instance_idx"] == inst_id)
                )
                df_subset.loc[mask, "state_idx"] = (
                    df_subset.loc[mask, "state_idx"].map(lambda x: inv_perm[x])
                )

            aligned_nstates_dfs.append(df_subset)

        aligned_dfs.append(pd.concat(aligned_nstates_dfs, ignore_index=True))

    return pd.concat(aligned_dfs, ignore_index=True), all_permutations

def _cross_mouse_alignment_from_lick_rate(
    path_map_per_mouse: dict[str, dict[tuple, Path]],
    within_inv_perms:   dict[str, dict[tuple, np.ndarray]],
    n_states:           int,
    trial_type:         str = "whisker_trial",
) -> dict[str, np.ndarray]:
    """
    Compute cross-mouse state permutations by sorting each mouse's states
    by their mean lick rate on whisker trials (ascending).

    State 0 = lowest whisker lick rate, state K-1 = highest.
    Works for any K — for K=2 this directly separates low/high lick states.

    :param path_map_per_mouse: {mouse_id: {(si, inst): model_dir Path}}
    :param within_inv_perms:   {mouse_id: {(si, inst): within_inv_perm}}
                               Already-computed within-mouse inv_perms so
                               lick rates are computed on aligned state labels.
    :param n_states:           K
    :param trial_type:         trial type column value to filter on
    :return:                   {mouse_id: cross_inv_perm (K,) int array}
                               new_label = cross_inv_perm[within_aligned_label]
    """
    cross_inv_perms: dict[str, np.ndarray] = {}

    for mouse_id, si_inst_path_map in path_map_per_mouse.items():
        valid_keys = {
            k for k, p in si_inst_path_map.items()
            if (p / "data_preds.h5").exists()
        }
        if not valid_keys:
            continue

        # Accumulate per-state lick counts across all splits x instances
        lick_counts  = np.zeros(n_states)
        trial_counts = np.zeros(n_states)

        for (si, inst) in valid_keys:
            h5 = si_inst_path_map[(si, inst)] / "data_preds.h5"
            try:
                df = pd.read_hdf(h5)
            except Exception as e:
                logger.warning(f"  Could not read {h5}: {e}")
                continue

            # Apply within-mouse inv_perm to get aligned state labels
            inv_perm = within_inv_perms[mouse_id].get((si, inst))
            if inv_perm is not None:
                df["aligned_state"] = inv_perm[df["most_likely_state"].values]
            else:
                df["aligned_state"] = df["most_likely_state"]


            # Filter to whisker trials only
            #whisker = df[df["trial_type"] == trial_type]
            whisker = df #already whisker only

            for s in range(n_states):
                state_trials = whisker[whisker["aligned_state"] == s]
                lick_counts[s]  += state_trials["choice"].sum()
                trial_counts[s] += len(state_trials)

        # Lick rate per state on whisker trials
        with np.errstate(invalid="ignore"):
            lick_rates = np.where(trial_counts > 0, lick_counts / trial_counts, 0.0)
        logger.info(f"  [{mouse_id} | K={n_states}] "
                    f"Whisker lick rates per aligned state: {np.round(lick_rates, 3)}")

        # Sort states by lick rate ascending → state 0 = low, K-1 = high
        # sort_order[new_label] = old_label (this is perm, not inv_perm)
        sort_order = np.argsort(lick_rates)

        # Invert to get inv_perm: new_label = inv_perm[old_label]
        cross_inv              = np.empty(n_states, dtype=int)
        cross_inv[sort_order]  = np.arange(n_states)
        cross_inv_perms[mouse_id] = cross_inv

    return cross_inv_perms

def save_permutations(all_perms: dict, output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(all_perms, f)
    logger.info(f"Saved {len(all_perms)} permutations → {output_path}")

def load_permutations(input_path: Path) -> dict:
    input_path = Path(input_path)
    if not input_path.exists():
        logger.warning(f"Permutation file not found: {input_path}")
        return {}
    with open(input_path, "rb") as f:
        all_perms = pickle.load(f)
    logger.info(f"Loaded {len(all_perms)} permutations from {input_path}")
    return all_perms


