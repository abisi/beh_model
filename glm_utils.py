#! /usr/bin/env/python3
"""
@author: Axel Bisi
@project: beh_model
@file: glm_utils.py
@time: 7/18/2024 3:28 PM
"""
import autograd.numpy as np
import autograd.numpy.random as npr
import matplotlib.pyplot as plt
from GLM import glm

import ssm

npr.seed(65)

def calculate_predictive_acc_glm(glm_weights, inpt, y, idx_to_exclude):
    M = inpt.shape[1]
    C = 2
    # Calculate test loglikelihood
    from GLM import glm
    new_glm = glm(M, C)
    # Set parameters to fit parameters:
    new_glm.params = glm_weights
    # time dependent logits:
    prob_right = np.exp(new_glm.calculate_logits(inpt))
    prob_right = prob_right[:, 0, 1]
    # Get the predicted label for each time step:
    predicted_label = np.around(prob_right, decimals=0).astype('int')
    # Examine at appropriate idx
    predictive_acc = np.sum(
        y[idx_to_exclude,
          0] == predicted_label[idx_to_exclude]) / len(idx_to_exclude)
    return predictive_acc

def calculate_predictive_accuracy(inputs, datas, train_masks, hmm_params, K,
                                  permutation, transition_alpha, prior_sigma,
                                  y, idx_to_exclude):
    # Run forward algorithm on hmm with these parameters and collect gammas:
    M = inputs[0].shape[1]
    D = datas[0].shape[1]
    this_hmm = ssm.HMM(K,
                       D,
                       M,
                       observations="input_driven_obs",
                       observation_kwargs=dict(C=2, prior_sigma=prior_sigma),
                       transitions="sticky",
                       transition_kwargs=dict(alpha=transition_alpha, kappa=0))
    this_hmm.params = hmm_params
    # Get expected states:
    expectations = [
        this_hmm.expected_states(data=data,
                                 input=input,
                                 mask=np.expand_dims(mask, axis=1))[0]
        for data, input, mask in zip(datas, inputs, train_masks)
    ]
    # Convert this now to one array:
    posterior_probs = np.concatenate(expectations, axis=0)
    posterior_probs = posterior_probs[:, permutation]
    prob_right = [
        np.exp(this_hmm.observations.calculate_logits(input=input))
        for data, input, train_mask in zip(datas, inputs, train_masks)
    ]
    prob_right = np.concatenate(prob_right, axis=0)
    # Now multiply posterior probs and prob_right:
    prob_right = prob_right[:, :, 1]
    # Now multiply posterior probs and prob_right and sum over latent axis:
    final_prob_right = np.sum(np.multiply(posterior_probs, prob_right), axis=1)
    # Get the predicted label for each time step:
    predicted_label = np.around(final_prob_right, decimals=0).astype('int')
    # Examine at appropriate idx
    predictive_acc = np.sum(
        y[idx_to_exclude,
          0] == predicted_label[idx_to_exclude]) / len(idx_to_exclude)
    return predictive_acc

def load_data(animal_file):
    container = np.load(animal_file, allow_pickle=True)
    data = [container[key] for key in container]
    inpt = data[0]
    y = data[1]
    session = data[2]
    return inpt, y, session


def fit_glm(inputs, datas, M, C):
    new_glm = glm(M, C)
    new_glm.fit_glm(datas, inputs, masks=None, tags=None)
    # Get loglikelihood of training data:
    loglikelihood_train = new_glm.log_marginal(datas, inputs, None, None)
    recovered_weights = new_glm.Wk
    return loglikelihood_train, recovered_weights


# Append column of zeros to weights matrix in appropriate location
def append_zeros(weights):
    weights_tranpose = np.transpose(weights, (1, 0, 2))
    weights = np.transpose(
        np.vstack([
            weights_tranpose,
            np.zeros((1, weights_tranpose.shape[1], weights_tranpose.shape[2]))
        ]), (1, 0, 2))
    return weights


def load_session_fold_lookup(file_path):
    container = np.load(file_path, allow_pickle=True)
    data = [container[key] for key in container]
    session_fold_lookup_table = data[0]
    return session_fold_lookup_table


def load_animal_list(list_file):
    container = np.load(list_file, allow_pickle=True)
    data = [container[key] for key in container]
    animal_list = data[0]
    return animal_list


def plot_input_vectors(Ws,
                       figure_directory,
                       title='true',
                       save_title="true",
                       labels_for_plot=[]):
    K = Ws.shape[0]
    K_prime = Ws.shape[1]
    M = Ws.shape[2] - 1
    fig = plt.figure(figsize=(7, 9), dpi=80, facecolor='w', edgecolor='k')
    plt.subplots_adjust(left=0.15,
                        bottom=0.27,
                        right=0.95,
                        top=0.95,
                        wspace=0.3,
                        hspace=0.3)

    for j in range(K):
        for k in range(K_prime - 1):
            # plt.subplot(K, K_prime, 1+j*K_prime+k)
            plt.plot(range(M + 1), -Ws[j][k], marker='o')
            plt.plot(range(-1, M + 2), np.repeat(0, M + 3), 'k', alpha=0.2)
            plt.axhline(y=0, color="k", alpha=0.5, ls="--")
            if len(labels_for_plot) > 0:
                plt.xticks(list(range(0, len(labels_for_plot))),
                           labels_for_plot,
                           rotation=90,
                           fontsize=12)
            else:
                plt.xticks(list(range(0, 3)),
                           ['Stimulus', 'Past Choice', 'Bias'],
                           rotation=90,
                           fontsize=12)
            plt.ylim((-3, 6))

    fig.text(0.04,
             0.5,
             "Weight",
             ha="center",
             va="center",
             rotation=90,
             fontsize=15)
    fig.suptitle("GLM Weights: " + title, y=0.99, fontsize=14)
    fig.savefig(figure_directory + 'glm_weights_' + save_title + '.png')