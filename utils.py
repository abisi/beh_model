#! /usr/bin/env/python3
"""
@author: Axel Bisi
@project: beh_model
@file: utils.py
@time: 8/2/2024 1:28 PM
"""


# Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

def add_noise_to_weights(weights, noise_level=0.2):
    """
    Add noise to weights
    :param weights: weights to add noise to
    :param noise_level: level of noise to add
    :return: weights with added noise
    """

    n_weights = weights.shape[0]
    gauss_noise_weights = np.random.normal(0, noise_level, n_weights)
    weights = weights + gauss_noise_weights

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
    Calculate predictive accuracy
    :param true_labels:
    :param predicted_labels:
    :return:
    """
    # First flatten lists of sessions to get a unique vector for all the data
    true_labels = np.concatenate(true_labels, axis=0)
    predicted_labels = np.concatenate(predicted_labels, axis=0)

    # Calculate predictive accuracy
    predictive_acc = np.sum(true_labels[:,0] == predicted_labels) / len(true_labels)

    return predictive_acc

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


# Function to align states of a subject to a reference using the Hungarian algorithm
def align_states(reference_weights, subject_weights):
    """
    Align the states of a subject to a reference using the Hungarian algorithm.
    :param reference_weights:
    :param subject_weights:
    :return:
    """
    # Compute the distance matrix between the reference and the subject's states
    print('distance between', reference_weights.shape, subject_weights.shape)
    distance_matrix = compute_distance_matrix(reference_weights, subject_weights)

    # Use Hungarian algorithm to find the best matching
    row_ind, col_ind = linear_sum_assignment(distance_matrix)

    # Return the permuted state indices for the subject
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

    print(reference_weights, reference_weights.shape)
    for s in range(n_subjects):
        # Align the states of the current subject to the reference
        print(w_subjects[s], w_subjects[s].shape)
        permuted_indices = align_states(reference_weights, w_subjects[s])

        # Permute the subject's weights based on the matching indices
        permuted_weights = w_subjects[s][permuted_indices]

        # Store the permuted weights
        aligned_subjects.append(permuted_weights)

    return aligned_subjects


