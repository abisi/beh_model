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
from scipy.optimize import linear_sum_assignment
import ssm
import autograd.numpy.random as npr
npr.seed(42)

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
        sets['drop_history'] = mvt_features + ['bias']
        sets['drop_mvt'] = non_mvt_features
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
        print(f"\nAligning {n_states}-state models...")

        # Skip alignment for 1-state models – nothing to permute
        if n_states == 1:
            print("  Skipping – no alignment needed for 1 state")
            aligned_dfs.append(df[df['n_states'] == n_states].copy())
            continue

        df_nstates = df[df['n_states'] == n_states].copy()

        # ------------------------------------------------------------------
        # Align within each reward group independently
        # ------------------------------------------------------------------
        aligned_nstates_dfs = []

        for reward_group in sorted(df_nstates['reward_group'].unique()):
            print(f"  Reward group {reward_group}:")

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
            print(f"    Aligned {len(model_ids)} models")
            print(f"    Found {len(unique_perms)} unique permutations")
            if len(unique_perms) < len(model_ids):
                consistency = 100 * (1 - (len(unique_perms) - 1) / len(model_ids))
                print(f"    Alignment consistency: {consistency:.1f}%")

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


