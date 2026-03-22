#!/usr/bin/env python3
"""
run_pipeline.py  –  Unified GLM-HMM pipeline
=============================================
Stages (run selectively via --stages flag or CFG booleans):
  1. dataset   – build behavioural dataset from NWB files and create train/test splits
  2. global    – fit global (all-mouse) GLM-HMM for each
                 (split × n_states × instance × feature_set × reward_group)
  3. single    – fit per-mouse GLM-HMM initialised from the best global instance
  4. all       – run all three stages in order

Feature sets
------------
One "full" model is fit using all features.  Additionally one leave-one-out
model is fit per feature (and per trial-type block if TRIAL_TYPES is set),
so you can quantify each feature's contribution by comparing test LL / accuracy
against the full model.

Usage examples
--------------
  python run_pipeline.py                          # all stages, all feature sets
  python run_pipeline.py --stages global single
  python run_pipeline.py --stages single --n_states 3 --reward_groups 1
"""

# ── stdlib ────────────────────────────────────────────────────────────────────
import os
import time
import pickle
import logging
import argparse
import socket
from itertools import product
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm

# ── third-party ───────────────────────────────────────────────────────────────
import numpy as np
import numpy.random as npr
import pandas as pd
import ssm
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from scipy.optimize import linear_sum_assignment

import utils
# ── project ───────────────────────────────────────────────────────────────────
from create_behaviour_dataset import create_behavior_dataset, split_dataset
from data_utils import create_data_lists
from plotting_utils import (
    _annotate_feature_stats,
    plot_model_glm_weights,
    plot_model_transition_matrix,
    plot_single_session_predictions,
    plot_single_session_posterior_states,
)
from utils import (
    build_feature_sets,
    get_expected_states,
    get_predicted_labels,
    calculate_predictive_accuracy,
    add_noise_to_weights,
)
import NWB_reader_functions as NWB_read

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
def _resolve_paths():
    host = socket.gethostname()
    if "haas" in host:
        root = Path("/mnt/lsens-analysis/Axel_Bisi")
        info = Path("/mnt/share_internal/Axel_Bisi_Share/dataset_info")
    elif host == "SV-07M-005":
        root = Path("/Volumes/Petersen-Lab/analysis/Axel_Bisi")
        info = Path("/Volumes/Petersen-Lab/share_internal/Axel_Bisi_Share/dataset_info")
    else:
        root = Path(r"\\sv-nas1.rcp.epfl.ch\Petersen-Lab\analysis\Axel_Bisi")
        info = Path(r"\\sv-nas1.rcp.epfl.ch\Petersen-Lab\share_internal\Axel_Bisi_Share\dataset_info")
    return root, info



_ROOT, _INFO = _resolve_paths()
_OUTPUT = _ROOT / "combined_results" / "glm_hmm"

CFG = dict(
    # ── paths ────────────────────────────────────────────────────────────────
    nwb_root      = _ROOT / "NWB_combined",
    info_path     = _INFO,
    dataset_path  = _OUTPUT  / "datasets",
    global_path   = _OUTPUT  / "global_glmhmm",
    single_path   = _OUTPUT  / "single_mouse_glmhmm",

    # ── stage switches ────────────────────────────────────────────────────────
    run_dataset   = True,
    run_global    = True,
    run_single    = True,
    run_permute   = True,

    # ── dataset params ─────────────────────────────────────────────────────────
    n_splits          = 5, #10,  # dataset splits
    fraction_training = 0.8,
    n_trials_max      = 10000,        # set to int to cap trials per session

    # ── model params ───────────────────────────────────────────────────────────
    n_states_list = [1, 2, 3, 4, 5, 6],
    n_instances   = 1,  #5, # number of model instantiations
    n_train_iters = 300,
    tolerance     = 1e-4,
    prior_sigma   = 2.0, #2.0
    prior_alpha   = 2.0,
    kappa         = 0,             # sticky HMM self-transition bias (0 = no stickiness)
    noise_level = 0.05,              # std dev of noise added to weights for random restarts

    reward_groups = [1,0],
    # ── features ──────────────────────────────────────────────────────────────
    features = [
        "bias",
        #"whisker",
        #"auditory",
        #"time_since_last_auditory_reward",
        #"time_since_last_whisker_reward",
        "time_since_last_auditory_lick", #integrative
        "time_since_last_whisker_lick",
        "prev_wh_choice", #directly comparable across R1 and R-
        "prev_auditory",
        #'jaw_distance',
        'whisker_angle',
        #'pupil_area',
    ],
    # ── trial types to fit ───────────────────────────────────────────────────────────
    trial_types="whisker",  # "all_trials", or "whisker"
    # ── parallelism ───────────────────────────────────────────────────────────
    n_workers = max(1, os.cpu_count() - 2),
)

# If fit only whisker trials, change features and update paths
if CFG["trial_types"] == "whisker":
    CFG["features"] = [f for f in CFG["features"] if f not in ["bias", "auditory", "whisker"]] #remove trial types
    CFG["global_path"] = _OUTPUT  / "global_glmhmm_whisker"
    CFG["single_path"] = _OUTPUT  / "single_mouse_glmhmm_whisker"


state_index_cmap = {0:'#9234eb', 1:'#e3b81b', 2:'#c5d642', 3: '#e58757', 4:'#fd3f56', 5:'#329fd1', 6:'#d642aa'}  # Example mapping, adjust as needed


# ─────────────────────────────────────────────────────────────────────────────
# PATH HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _rg_label(reward_group) -> str:
    """Map numeric reward-group code to a short directory-safe string."""
    return {1: "Rplus", 0: "Rminus"}.get(int(reward_group), f"rg{reward_group}")


def global_model_dir(cfg, split_idx, n_states, instance_idx,
                     model_name: str, reward_group) -> Path:
    return (
        cfg["global_path"]
        / _rg_label(reward_group)
        / model_name
        / f"model_{split_idx}"
        / f"{n_states}_states"
        / f"iter_{instance_idx}"
    )


def single_model_dir(cfg, mouse_id, split_idx, n_states, instance_idx,
                     model_name: str, reward_group) -> Path:
    return (
        cfg["single_path"]
        / _rg_label(reward_group)
        / model_name
        / mouse_id
        / f"model_{split_idx}"
        / f"{n_states}_states"
        / f"iter_{instance_idx}"
    )


def split_data_dir(cfg, split_idx):
    return cfg["dataset_path"] / f"dataset_{split_idx}"


# ─────────────────────────────────────────────────────────────────────────────
# SHARED MODEL UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
def build_glmhmm(n_states: int, input_dim: int, cfg: dict) -> ssm.HMM:
    return ssm.HMM(
        n_states, 1, input_dim,
        observations="input_driven_obs",
        observation_kwargs=dict(C=2, prior_sigma=cfg["prior_sigma"]),
        transitions="sticky",
        transition_kwargs=dict(alpha=cfg["prior_alpha"], kappa=cfg["kappa"]),
    )


def _null_log_likelihood(outputs, p_lick=None):
    """LL of a bias-only Bernoulli null model.
    If p_lick is provided, uses that fixed rate (e.g. from training set).
    Otherwise estimates p_lick from outputs."""
    y = np.concatenate(outputs, axis=0)[:, 0]
    if p_lick is None:
        p_lick = y.mean()
    p = np.clip(p_lick, 1e-9, 1 - 1e-9)
    return np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

def _bits_per_trial(ll_model, ll_null, n_trials):
    return (ll_model - ll_null) / (n_trials * np.log(2))

def fit_and_evaluate(glmhmm, input_train, output_train, input_test, output_test, cfg):
    """Fit with EM, then return metrics + arrays for downstream use."""
    glmhmm.fit(
        output_train, inputs=input_train,
        method="em", num_iters=cfg["n_train_iters"], tolerance=cfg["tolerance"],
    )

    ll_train = glmhmm.log_likelihood(output_train, input_train, None, None)
    ll_test  = glmhmm.log_likelihood(output_test,  input_test,  None, None)

    def _posteriors_preds_acc(outputs, inputs):
        states = get_expected_states(glmhmm, outputs=outputs, inputs=inputs)
        post   = np.concatenate(states, axis=0)
        preds  = get_predicted_labels(glmhmm, inputs=inputs, posteriors=post)
        acc, bal_acc    = calculate_predictive_accuracy(outputs, preds)
        return post, preds, acc, bal_acc

    post_train, preds_train, acc_train, bacc_train = _posteriors_preds_acc(output_train, input_train)
    post_test,  preds_test,  acc_test,  bacc_test  = _posteriors_preds_acc(output_test,  input_test)

    # Null LL fixed at training lick rate for both splits — test bits per trial
    # is then a pure out-of-sample metric, with the null not seeing test labels
    p_lick_train = np.concatenate(output_train, axis=0)[:, 0].mean()
    ll_null_train = _null_log_likelihood(output_train, p_lick=p_lick_train)
    ll_null_test = _null_log_likelihood(output_test, p_lick=p_lick_train) # this is intentional
    n_train = sum(len(o) for o in output_train)
    n_test = sum(len(o) for o in output_test)
    bpt_train = _bits_per_trial(ll_train, ll_null_train, n_train)
    bpt_test = _bits_per_trial(ll_test, ll_null_test, n_test)

    metric_dict = dict(
        ll_train=ll_train,               ll_test=ll_test,
        acc_train=acc_train,             acc_test=acc_test,
        balanced_acc_train=bacc_train,   balanced_acc_test=bacc_test,
        bpt_train=bpt_train,              bpt_test=bpt_test,
        post_train=post_train,           post_test=post_test,
        preds_train=preds_train,         preds_test=preds_test,
        weights=glmhmm.observations.params,
        transition_matrix=glmhmm.transitions.transition_matrix,
    )


    ## Add trial-type metrics to the result dict
    #input_train['trial_type'] = 'no_stim_trial'
    #input_test['trial_type'] = 'no_stim_trial'
    #input_train.loc[input_train['whisker'] == 1, 'trial_type'] = 'whisker_trial'
    #input_train.loc[input_train['auditory'] == 1  , 'trial_type'] = 'auditory_trial'
    #input_test.loc[input_test['whisker'] == 1, 'trial_type'] = 'whisker_trial'
    #input_test.loc[input_test['auditory'] == 1, 'trial_type'] = 'auditory_trial'
    #output_train['trial_type'] = input_train['trial_type']
    #output_test['trial_type'] = input_test['trial_type']
    ## For all trial types separately, calculate accuracy and bits per trial against the same null (overall lick rate)
    #ttypes_metrics = {}
    #for ttype in input_train['trial_type'].unique():
    #    idx_train = input_train['trial_type'] == ttype
    #    idx_test = input_test['trial_type'] == ttype
    #    acc_tt, bacc_tt = calculate_predictive_accuracy(output_train[idx_train], preds_train[idx_train])
    #    acc_tt_test, bacc_tt_test = calculate_predictive_accuracy(output_test[idx_test], preds_test[idx_test])
    #    ll_tt_train = glmhmm.log_likelihood(output_train[idx_train], input_train[idx_train], None, None)
    #    ll_tt_test  = glmhmm.log_likelihood(output_test[idx_test],  input_test[idx_test],  None, None)
    #    n_tt_train = idx_train.sum()
    #    n_tt_test = idx_test.sum()
    #    bpt_tt_train = _bits_per_trial(ll_tt_train, ll_null_train, n_tt_train)
    #    bpt_tt_test = _bits_per_trial(ll_tt_test, ll_null_test, n_tt_test)
    #    ttypes_metrics[ttype] = dict(
    #        acc_train=acc_tt, bacc_train=bacc_tt, bpt_train=bpt_tt_train,
    #        acc_test=acc_tt_test, bacc_test=bacc_tt_test, bpt_test=bpt_tt_test,
    #    )
#
    #metric_dict['trial_type_metrics'] = ttypes_metrics

    return dict(
        ll_train=ll_train,               ll_test=ll_test,
        acc_train=acc_train,             acc_test=acc_test,
        balanced_acc_train=bacc_train,   balanced_acc_test=bacc_test,
        bpt_train=bpt_train,              bpt_test=bpt_test,
        post_train=post_train,           post_test=post_test,
        preds_train=preds_train,         preds_test=preds_test,
        weights=glmhmm.observations.params,
        transition_matrix=glmhmm.transitions.transition_matrix,
    )
    #return metric_dict

def noisy_copy(weights: np.ndarray, sigma: float = 0.1) -> np.ndarray:
    return weights + np.random.normal(0.0, sigma, weights.shape)


def load_split(cfg, split_idx):
    d = split_data_dir(cfg, split_idx)
    return (
        pickle.load(open(d / "data_train.pkl", "rb")),
        pickle.load(open(d / "data_test.pkl",  "rb")),
    )


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1  –  DATASET CREATION
# ─────────────────────────────────────────────────────────────────────────────
def _check_nwb_file(args):
    name, fpath = args
    try:
        btype, _ = NWB_read.get_bhv_type_and_training_day_index(fpath)
        return fpath if btype == "whisker" else None
    except Exception as exc:
        logger.warning(f"  Skipping {name}: {exc}")
        return None

def stage_create_dataset(cfg):
    logger.info("=" * 60)
    logger.info("STAGE 1 – Dataset creation")
    logger.info("=" * 60)

    mouse_info_df = pd.read_excel(cfg["info_path"] / "joint_mouse_reference_weight.xlsx")
    mouse_info_df.rename(columns={"mouse_name": "mouse_id"}, inplace=True)
    mouse_info_df = mouse_info_df[
        (mouse_info_df["exclude"] == 0) &
        (mouse_info_df["reward_group"].isin(["R+", "R-"]))
    ]

    all_nwb = os.listdir(cfg["nwb_root"])
    valid_mice = [
        m for m in mouse_info_df["mouse_id"].unique()
        if any(m in n for n in all_nwb)
    ]

    candidates = [
        (name, str(cfg["nwb_root"] / name))
        for mouse in valid_mice
        for name in all_nwb
        if mouse in name
    ]

    with Pool(processes=os.cpu_count() - 2) as pool:
        results = pool.map(_check_nwb_file, candidates)

    nwb_list = [r for r in results if r is not None]

    #nwb_list = []
    #for mouse in valid_mice:
    #    for name in all_nwb:
    #        if mouse not in name:
    #            continue
    #        fpath = str(cfg["nwb_root"] / name)
    #        try:
    #            btype, _ = NWB_read.get_bhv_type_and_training_day_index(fpath)
    #            if btype == "whisker":
    #                nwb_list.append(fpath)
    #        except Exception as exc:
    #            logger.warning(f"  Skipping {name}: {exc}")

    logger.info(f"Found {len(nwb_list)} whisker NWB files from {len(valid_mice)} mice.")

    # ── Build dataset ─────────────────────────────────────────────────────────
    params = dict(n_trials_max=cfg["n_trials_max"])
    dataset = create_behavior_dataset(nwb_list, mouse_info_df, cfg["n_trials_max"])
    print('Dataset created')

    # Check multicolinearity of features
    # ----------------------------------
    from multicollinearity_utils import check_multicollinearity, plot_multicollinearity
    results = check_multicollinearity(dataset, cfg["features"])
    multicol_path = cfg["global_path"].parent / "figures" / "multicollinearity"
    multicol_path.mkdir(parents=True, exist_ok=True)
    plot_multicollinearity(results, save_path=multicol_path / "multicollinearity")


    logger.info(
        f"Dataset size: {len(dataset):,} trials | "
        f"{dataset['mouse_id'].nunique()} mice | "
        f"{dataset['session_id'].nunique()} sessions"
    )

    for split_idx in range(cfg["n_splits"]):
        out_dir = split_data_dir(cfg, split_idx)
        #if (out_dir / "data_train.pkl").exists():
        #    logger.info(f"  Split {split_idx}: already exists, skipping.")
        #    continue
        out_dir.mkdir(parents=True, exist_ok=True)
        data_train, data_test = split_dataset(dataset, cfg["fraction_training"])
        dataset.to_pickle(out_dir / "dataset.pkl")
        data_train.to_pickle(out_dir / "data_train.pkl")
        data_test.to_pickle(out_dir  / "data_test.pkl")
        logger.info(
            f"  Split {split_idx}: {len(data_train):,} train | {len(data_test):,} test."
        )

    logger.info("Stage 1 complete.\n")


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2  –  GLOBAL MODEL FITTING
# ─────────────────────────────────────────────────────────────────────────────
def _fit_global_worker(args):
    """
    Worker for one (split_idx, n_states, instance_idx, model_name, features, reward_group).

    For n_states > 1 the worker loads the best-LL instance from the k=1 result
    of the *same* (split, model_name, reward_group) combination to initialise
    weights.  Because pool.map for k=1 tasks completes entirely before k>1
    tasks are submitted, those files are guaranteed to exist.
    """
    (split_idx, n_states, instance_idx,
     model_name, features, cfg, reward_group) = args

    seed = (split_idx * 10_000 + n_states * 1_000
            + instance_idx * 100 + int(reward_group) * 10
            + abs(hash(model_name)) % 10)
    npr.seed(seed)

    out_dir     = global_model_dir(cfg, split_idx, n_states, instance_idx,
                                   model_name, reward_group)
    result_file = out_dir / "global_fit_glmhmm_results.npz"

    tag = (f"global | rg={_rg_label(reward_group)} {model_name} "
           f"split={split_idx} K={n_states} inst={instance_idx}")

    #if result_file.exists():
    #    logger.info(f"  [skip] {tag}")
    #    return np.load(result_file, allow_pickle=True)["arr_0"].item()

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Data (filtered to reward group) ───────────────────────────────────────
    data_train_all, data_test_all = load_split(cfg, split_idx)

    # If fit only whisker trials, filter
    if cfg["trial_types"] == "whisker":
        data_train_all = data_train_all[data_train_all['whisker'] == 1]
        data_test_all = data_test_all[data_test_all['whisker'] == 1]

        # Add win-stay/lose-shift interaction
        data_train_all['wsls'] = data_train_all['prev_choice'] * data_train_all['prev_trial_reward_given']
        data_test_all['wsls'] = data_test_all['prev_choice'] * data_test_all['prev_trial_reward_given']

    data_train = data_train_all[data_train_all["reward_group"] == reward_group]
    data_test  = data_test_all[ data_test_all["reward_group"]  == reward_group]

    if len(data_train) == 0:
        logger.warning(f"  [skip – no data for cond] {tag}")
        return None

    input_train, output_train, input_test, output_test = create_data_lists(
        data_train, data_test, features=features
    )

    # ── Weight initialisation from best k=1 instance ──────────────────────────
    init_weights = None
    if n_states > 1:
        best_ll, best_inst = -np.inf, 0
        for inst in range(cfg["n_instances"]):
            f = (global_model_dir(cfg, split_idx, 1, inst, model_name, reward_group)
                 / "global_fit_glmhmm_results.npz")
            if not f.exists():
                continue
            res_ll = np.load(f, allow_pickle=True)["arr_0"].item().get("ll_test", -np.inf)
            if res_ll > best_ll:
                best_ll, best_inst = res_ll, inst

        k1_file = (global_model_dir(cfg, split_idx, 1, instance_idx, model_name, reward_group)
                   / "global_fit_glmhmm_results.npz")
        if k1_file.exists():
            w1 = np.load(k1_file, allow_pickle=True)["arr_0"].item()["weights"]
            # w1: (1, C-1, M) → tile to (K, C-1, M) then add noise
            init_weights = noisy_copy(np.repeat(w1, n_states, axis=0), sigma=0.2)
        else:
            logger.warning(f"  k=1 weights not found for {tag}, using random init.")

    # ── Build & fit ───────────────────────────────────────────────────────────
    glmhmm = build_glmhmm(n_states, len(features), cfg)
    if init_weights is not None:
        glmhmm.observations.params = init_weights

    metrics = fit_and_evaluate(
        glmhmm, input_train, output_train, input_test, output_test, cfg
    )

    result_dict = dict(
        split_idx=split_idx,
        n_states=n_states,
        instance_idx=instance_idx,
        model_name=model_name,
        features=features,
        reward_group=reward_group,
        init_weights=init_weights,
        **metrics,
    )
    np.savez(result_file, result_dict)

    plot_model_glm_weights(
        model=glmhmm, init_weights=init_weights,
        feature_names=features, save_path=out_dir,
        file_name="global_weights", suffix=None, file_types=["png", "svg"],
    )
    plot_model_transition_matrix(
        model=glmhmm, save_path=out_dir,
        file_name="transition_matrix", suffix=None, file_types=["png", "svg"],
    )

    logger.info(
        f"  {tag}  ll_train={metrics['ll_train']:.1f}  "
        f"acc_train={metrics['acc_train']:.3f} (bal={metrics['balanced_acc_train']:.3f})  "
        f"acc_test={metrics['acc_test']:.3f} (bal={metrics['balanced_acc_test']:.3f})"
    )
    return result_dict


def _make_global_tasks(cfg, k_states, feature_sets):
    """Build flat list of worker arg-tuples for the given k values."""
    return [
        (split_idx, k_state, instance_idx, model_name, features, cfg, reward_group)
        for k_state, split_idx, instance_idx, (model_name, features), reward_group
        in product(
            k_states,
            range(cfg["n_splits"]),
            range(cfg["n_instances"]),
            feature_sets.items(),
            cfg["reward_groups"],
        )
    ]


def stage_fit_global(cfg):
    logger.info("=" * 60)
    logger.info("STAGE 2 – Global model fitting")
    logger.info("=" * 60)

    feature_sets = build_feature_sets(cfg["features"], cfg.get("trial_types"))
    logger.info(f"Feature sets ({len(feature_sets)}): {list(feature_sets.keys())}")

    all_k = sorted(cfg["n_states_list"])
    tasks_k1  = _make_global_tasks(cfg, [1],          feature_sets)
    tasks_kgt = _make_global_tasks(cfg, all_k[1:],    feature_sets)

    logger.info(
        f"Total tasks: {len(tasks_k1) + len(tasks_kgt)}  "
        f"({len(tasks_k1)} k=1 first, then {len(tasks_kgt)} k>1)"
    )

    workers = min(cfg["n_workers"], max(len(tasks_k1), 1))
    with Pool(processes=workers) as pool:
        # k=1 must complete entirely before k>1 starts (init weight dependency)
        results_k1  = pool.map(_fit_global_worker, tasks_k1)
        results_kgt = pool.map(_fit_global_worker, tasks_kgt)

    all_results = [r for r in results_k1 + results_kgt if r is not None]

    scalar_keys = ["split_idx", "n_states", "instance_idx", "model_name", "reward_group",
        "ll_train", "ll_test",
        "acc_train", "acc_test",
        "balanced_acc_train", "balanced_acc_test",
        "bpt_train", "bpt_test",
    ]
    summary_df = pd.DataFrame(
        [{k: r[k] for k in scalar_keys if k in r} for r in all_results]
    )
    out_h5 = cfg["global_path"] / "global_fit_glmhmm_summary.h5"
    out_h5.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_hdf(out_h5, key="df", mode="w")
    logger.info(f"Stage 2 complete. Summary → {out_h5}\n")
    return summary_df


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3  –  SINGLE-MOUSE MODEL FITTING
# ─────────────────────────────────────────────────────────────────────────────
def _fit_single_worker(args):
    """
    Worker for one (mouse_id, split_idx, n_states, model_name, features, reward_group).
    Selects the best global instance (highest test LL) for that combination.
    Same two-phase k=1 / k>1 ordering is enforced by the caller.
    """
    (mouse_id, split_idx, instance_idx, n_states,
     model_name, features, cfg, reward_group) = args

    # ── Early exit: skip entirely if this mouse has no data for this reward group
    # This prevents result folders being created for the wrong reward group.
    data_train_all, data_test_all = load_split(cfg, split_idx)

    # If fit only whisker trials, filter
    if cfg["trial_types"] == "whisker":
        data_train_all = data_train_all[data_train_all['whisker'] == 1]
        data_test_all = data_test_all[data_test_all['whisker'] == 1]

        # Add win-stay/lose-shift interaction
        data_train_all['wsls'] = data_train_all['prev_choice'] * data_train_all['prev_trial_reward_given']
        data_test_all['wsls'] = data_test_all['prev_choice'] * data_test_all['prev_trial_reward_given']

    data_train = data_train_all[
        (data_train_all["mouse_id"] == mouse_id) &
        (data_train_all["reward_group"] == reward_group)
        ].copy()
    data_test = data_test_all[
        (data_test_all["mouse_id"] == mouse_id) &
        (data_test_all["reward_group"] == reward_group)
        ].copy()



    if len(data_train) == 0: # Note: this to skip mouse when iterating over the reward group they dont belong to
        #print('No data for mouse', mouse_id, 'reward group', reward_group, 'split', split_idx)
        return None  # silent — this is expected for most (mouse, reward_group) combos

    seed = (abs(hash(mouse_id)) % 10_000 + split_idx * 100
            + n_states * 10 + int(reward_group))
    npr.seed(seed)

    tag = (f"single | rg={_rg_label(reward_group)} {model_name} "
           f"{mouse_id} split={split_idx} K={n_states}")

    ## ── Best global instance for this (split, K, model, rg) ────────────────── #TODO: ignore
    #best_ll, best_inst = -np.inf, 0
    #for inst in range(cfg["n_instances"]):
    #    f = (global_model_dir(cfg, split_idx, n_states, inst, model_name, reward_group)
    #         / "global_fit_glmhmm_results.npz")
    #    if split_idx != 0:
    #        print('Loading split', split_idx, 'from', f)
    #        # Assert it exists
    #        assert f.exists(), f"Expected global result file not found: {f}"
    #    #if not f.exists():
    #    #    continue
    #    res_ll = np.load(f, allow_pickle=True)["arr_0"].item().get("ll_test", -np.inf)
    #    if res_ll > best_ll:
    #        best_ll, best_inst = res_ll, inst

    global_file = (global_model_dir(cfg, split_idx, n_states, instance_idx, model_name, reward_group)
                   / "global_fit_glmhmm_results.npz")

    if not global_file.exists():
        logger.warning(f"  [skip – no global result] {tag}")
        return None

    global_w = np.load(global_file, allow_pickle=True)["arr_0"].item()["weights"]

    out_dir = single_model_dir(cfg, mouse_id, split_idx, n_states,
                               instance_idx, model_name, reward_group)
    #assert out_dir.exists(), f"Best global result file not found: {out_dir}"
    result_file = out_dir / "fit_glmhmm_results.npz"
    print('Result file', result_file)

    #if result_file.exists():
    #    logger.info(f"  [skip] {tag} inst={best_inst}")
    #    return np.load(result_file, allow_pickle=True)["arr_0"].item()

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Mouse + reward-group filtered data ──────────────────────────────────── #to clean this up
    data_train_all, data_test_all = load_split(cfg, split_idx)

    # If fit only whisker trials, filter
    if cfg["trial_types"] == "whisker":
        data_train_all = data_train_all[data_train_all['whisker'] == 1]
        data_test_all = data_test_all[data_test_all['whisker'] == 1]

        # Add win-stay/lose-shift interaction
        data_train_all['wsls'] = data_train_all['prev_choice'] * data_train_all['prev_trial_reward_given']
        data_test_all['wsls'] = data_test_all['prev_choice'] * data_test_all['prev_trial_reward_given']

    data_train = data_train_all[
        (data_train_all["mouse_id"] == mouse_id) &
        (data_train_all["reward_group"] == reward_group)
    ].copy()
    data_test = data_test_all[
        (data_test_all["mouse_id"] == mouse_id) &
        (data_test_all["reward_group"] == reward_group)
    ].copy()

    if len(data_train) == 0:
        logger.warning(f"  [skip – no data for cond] {tag}")
        return None

    input_train, output_train, input_test, output_test = create_data_lists(
        data_train, data_test, features=features
    )

    init_weights = noisy_copy(global_w, sigma=0.1)
    glmhmm = build_glmhmm(n_states, len(features), cfg)
    glmhmm.observations.params = init_weights

    metrics = fit_and_evaluate(
        glmhmm, input_train, output_train, input_test, output_test, cfg
    )



    # ── Annotate dataframes for plotting ──────────────────────────────────────
    data_train["split"] = "train"
    data_train["pred"]  = np.concatenate(metrics["preds_train"])
    data_test["split"]  = "test"
    data_test["pred"]   = np.concatenate(metrics["preds_test"])
    for s in range(n_states):
        data_train[f"posterior_state_{s+1}"] = metrics["post_train"][:, s]
        data_test[ f"posterior_state_{s+1}"] = metrics["post_test"][:, s]

    data = (
        pd.concat([data_train, data_test])
        .sort_values(["session_id", "trial_id"])
        .reset_index(drop=True)
    )

    # Get most likely state sequence per session using Viterbi decoding
    state_seqs = []
    for session_id, df_sess in data.groupby("session_id"):
        df_sess = df_sess.sort_values("trial_id")

        y = df_sess["choice"].values[:, None] #ssm expects "D with timesteps
        u = df_sess[features].values

        z_viterbi = glmhmm.most_likely_states(y, input=u)
        state_seqs.append(pd.Series(z_viterbi, index=df_sess.index))

    # Combine back into dataframe
    data["most_likely_state"] = pd.concat(state_seqs).sort_index()

    data.to_hdf(out_dir / "data_preds.h5", key="data", mode="w")

    result_dict = dict(
        mouse_id=mouse_id,
        split_idx=split_idx,
        n_states=n_states,
        instance_idx=instance_idx,
        model_name=model_name,
        features=features,
        reward_group=reward_group,
        init_weights=init_weights,
        **metrics,
    )
    np.savez(result_file, result_dict)

    # Single fit plots
    # -----------------


    plot_model_glm_weights(
        model=glmhmm, init_weights=init_weights,
        feature_names=features, save_path=out_dir,
        file_name=f"{mouse_id}_glm_weights",
        suffix=None, file_types=["png", "svg"],
    )
    plot_model_transition_matrix(
        model=glmhmm, save_path=out_dir,
        file_name=f"{mouse_id}_transition_matrix",
        suffix=None, file_types=["png", "svg"],
    )
    plot_single_session_predictions(
        data=data, save_path=out_dir / "predictions",
        file_name="predictions", file_types=["png"],
    )
    plot_single_session_posterior_states(
        data=data, save_path=out_dir / "posterior_states",
        file_name="posterior_states", file_types=["png"],
    )

    logger.info(
        f"  {tag} inst={instance_idx}  "
        f"acc_train={metrics['acc_train']:.3f} (bal={metrics['balanced_acc_train']:.3f})  "
        f"acc_test={metrics['acc_test']:.3f} (bal={metrics['balanced_acc_test']:.3f})"
    )
    return result_dict


def _make_single_tasks(cfg, k_states, feature_sets, mouse_ids):
    return [
        (mouse_id, split_idx, instance_idx, k_state, model_name, features, cfg, reward_group)
        for k_state, split_idx, instance_idx, mouse_id, (model_name, features), reward_group
        in product(
            k_states,
            range(cfg["n_splits"]),
            range(cfg["n_instances"]),
            mouse_ids,
            feature_sets.items(),
            cfg["reward_groups"],
        )
    ]


def stage_fit_single(cfg, n_states_to_fit=None):
    logger.info("=" * 60)
    logger.info("STAGE 3 – Single-mouse model fitting")
    logger.info("=" * 60)

    feature_sets = build_feature_sets(cfg["features"], cfg.get("trial_types"))

    # For single-mouse plotting, keep only the full feature set
    feature_sets = {k: v for k, v in feature_sets.items() if k == "full"}
    logger.info("For single-mouse fits, using only feature sets: ['full']")

    all_k = sorted(
        [n_states_to_fit] if n_states_to_fit is not None else cfg["n_states_list"]
    )

    data_train, _ = load_split(cfg, 0)
    mouse_ids = data_train["mouse_id"].unique().tolist()

    logger.info(f"Mice ({len(mouse_ids)}): {mouse_ids}")

    tasks_k1  = _make_single_tasks(cfg, [1],       feature_sets, mouse_ids)
    tasks_kgt = _make_single_tasks(cfg, all_k[1:], feature_sets, mouse_ids)


    logger.info(
        f"Total tasks: {len(tasks_k1) + len(tasks_kgt)}  "
        f"({len(tasks_k1)} k=1 first, then {len(tasks_kgt)} k>1)"
    )

    workers = min(cfg["n_workers"], max(len(tasks_k1), 1))
    with Pool(processes=workers) as pool:
        results_k1  = pool.map(_fit_single_worker, tasks_k1)
        results_kgt = pool.map(_fit_single_worker, tasks_kgt)

    # COunt number of None results
    n_none_k1 = sum(r is None for r in results_k1)
    all_results = [r for r in results_k1 + results_kgt if r is not None]

    scalar_keys = [
        "mouse_id", "split_idx", "n_states", "instance_idx",
        "model_name", "reward_group",
        "ll_train", "ll_test",
        "acc_train", "acc_test",
        "balanced_acc_train", "balanced_acc_test",
        "bpt_train", "bpt_test",
    ]
    summary_df = pd.DataFrame(
        [{k: r[k] for k in scalar_keys if k in r} for r in all_results]
    )

    out_h5 = cfg["single_path"] / "all_subjects_glmhmm_summary.h5"
    out_h5.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_hdf(out_h5, key="df", mode="w")
    logger.info(f"Stage 3 complete. Summary → {out_h5}\n")
    return summary_df


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 4  –  PERFORMANCE FIGURES
# ─────────────────────────────────────────────────────────────────────────────

# Reward-group int → display string mapping (mirrors plot_glmhmm_results.py)
_RG_STR   = {1: "R+", 0: "R-"}
_RG_COLOR = {"R+": "forestgreen", "R-": "crimson"}


def _prep_global_df(cfg) -> pd.DataFrame:
    """
    Load the global summary HDF5 and normalise column names / types so the
    plotting helpers below can work with a single canonical schema.
    """
    h5 = cfg["global_path"] / "global_fit_glmhmm_summary.h5"
    if not h5.exists():
        raise FileNotFoundError(f"Global summary not found: {h5}  – run stage 2 first.")
    df = pd.read_hdf(h5, key="df")

    # ll_train from ssm.fit() is a list of per-iteration values; keep the last one
    if "ll_train" in df.columns:
        df["ll_train_final"] = df["ll_train"].apply(
            lambda x: x[-1] if isinstance(x, (list, np.ndarray)) else x
        )
    else:
        df["ll_train_final"] = np.nan

    # Rename accuracy columns to names expected by plotting helpers
    df = df.rename(columns={
        "acc_train":          "predictive_acc_train",
        "acc_test":           "predictive_acc_test",
        "balanced_acc_train": "balanced_predictive_acc_train",
        "balanced_acc_test":  "balanced_predictive_acc_test",
    })

    # Convert numeric reward_group → display string
    df["reward_group"] = df["reward_group"].map(lambda x: _RG_STR.get(int(x), str(x)))
    df["model_type"]   = df["model_name"]

    return df


def _prep_single_df(cfg) -> pd.DataFrame:
    """Load the single-mouse summary HDF5 with the same normalisation."""
    h5 = cfg["single_path"] / "all_subjects_glmhmm_summary.h5"
    if not h5.exists():
        raise FileNotFoundError(f"Single-mouse summary not found: {h5}  – run stage 3 first.")
    df = pd.read_hdf(h5, key="df")

    df = df.rename(columns={
        "acc_train":          "predictive_acc_train",
        "acc_test":           "predictive_acc_test",
        "balanced_acc_train": "balanced_predictive_acc_train",
        "balanced_acc_test":  "balanced_predictive_acc_test",
    })
    df["reward_group"] = df["reward_group"].map(lambda x: _RG_STR.get(int(x), str(x)))
    return df


def _compute_bpt(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add bpt_train / bpt_test columns (bits per trial above a Bernoulli null).
    Uses the first row to estimate lick probability; works on the global df.
    """
    def _ll_to_bpt(ll, ll_null, n):
        return (ll - ll_null) / (np.log(2) * n) if n > 0 else np.nan

    # Estimate p_lick from training predictions stored in the first row
    first = df.iloc[0]
    preds = first.get("preds_train", None)
    if preds is None or not isinstance(preds, (list, np.ndarray)):
        logger.warning("bpt: preds_train not available in summary – skipping bpt computation.")
        return df

    p_lick = float(np.mean([np.mean(p) for p in preds]))
    p_lick = np.clip(p_lick, 1e-6, 1 - 1e-6)

    n_train = sum(len(p) for p in preds)
    preds_test = first.get("preds_test", None)
    n_test  = sum(len(p) for p in preds_test) if preds_test is not None else n_train

    ll_null_train = n_train * (p_lick * np.log(p_lick) + (1 - p_lick) * np.log(1 - p_lick))
    ll_null_test  = n_test  * (p_lick * np.log(p_lick) + (1 - p_lick) * np.log(1 - p_lick))

    df["bpt_train"] = df["ll_train_final"].apply(lambda x: _ll_to_bpt(x, ll_null_train, n_train))
    df["bpt_test"]  = df["ll_test"].apply(       lambda x: _ll_to_bpt(x, ll_null_test,  n_test))
    return df


def _auto_palette(model_names: list) -> dict:
    """
    Build a colour palette for model names.  'full' is always black; LOO models
    get colours from a seaborn palette so any feature-set name is handled.
    """
    loo = [m for m in model_names if m != "full"]
    colours = sns.color_palette("tab10", len(loo))
    palette = {name: col for name, col in zip(loo, colours)}
    palette["full"] = "black"
    return palette


def plot_global_performance(df: pd.DataFrame, figure_path: Path,
                             model_subset: list | None = None):
    """
    4-row × 2-col panel (LL / acc / balanced-acc / bpt) × (train / test),
    one curve per model type, split by reward group.
    Produces:
      • one combined figure (both reward groups as hue)
      • one figure per reward group (model_type as hue)
    """
    from plotting_utils import remove_top_right_frame, save_figure_to_files
    out_dir = figure_path / "performance"
    out_dir.mkdir(parents=True, exist_ok=True)
    if model_subset is None:
        model_subset = sorted(df["model_type"].unique().tolist(),
                              key=lambda x: (x != "full", x))   # 'full' first
    palette = _auto_palette(model_subset)
    assert 'bpt_train' in df.columns and 'bpt_test' in df.columns, "bpt columns not found in dataframe – check _compute_bpt() was run"
    has_bpt = "bpt_train" in df.columns # should be default
    n_rows = 4 if has_bpt else 3
    row_metrics = [             #todo add bpt if available
        ("ll_train_final",                "ll_test",                    "Log-likelihood"),
        ("predictive_acc_train",          "predictive_acc_test",        "Predictive accuracy"),
        ("balanced_predictive_acc_train", "balanced_predictive_acc_test","Balanced accuracy"),
        ("bpt_train", "bpt_test", "Bits / trial"),
    ]
    #if has_bpt:
    #    row_metrics.append(("bpt_train", "bpt_test", "Bits / trial"))
    reward_groups_present = df["reward_group"].unique().tolist()
    def _make_fig(sub_df, hue_col, hue_order, plot_palette, title_suffix, fname):
        fig, axs = plt.subplots(n_rows, 2, figsize=(8, 3 * n_rows),
                                dpi=300, sharey="row", constrained_layout=True)
        for ax in axs.flat:
            ax.yaxis.set_tick_params(labelleft=True)
        for row_idx, (y_train, y_test, ylabel) in enumerate(row_metrics):
            for col_idx, (y_col, split_label) in enumerate(
                    [(y_train, "Train"), (y_test, "Test")]):
                ax = axs[row_idx, col_idx]
                if y_col not in sub_df.columns:
                    ax.set_visible(False)
                    continue
                sns.pointplot(
                    x="n_states", y=y_col, data=sub_df,
                    ax=ax,
                    estimator=np.mean,
                    errorbar="se",
                    hue=hue_col,
                    palette=plot_palette,
                    hue_order=hue_order,
                )
                ax.set_title(f"{split_label}{title_suffix}", fontsize=9)
                ax.set_ylabel(ylabel, fontsize=8)
                ax.set_xlabel("Number of states", fontsize=8)
                # chance baseline on accuracy rows
                if "acc" in y_col and "balanced" not in y_col:
                    ax.axhline(0.5, color="grey", linestyle="--", lw=0.8)
                if "balanced" in y_col:
                    ax.axhline(0.5, color="grey", linestyle="--", lw=0.8)
                remove_top_right_frame(ax)
                if ax.legend_:
                    ax.legend_.remove()
        # Single shared legend at top
        handles = [
            plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=plot_palette.get(m, "grey"),
                       label=m, markersize=5)
            for m in hue_order if m in plot_palette
        ]
        if handles:
            fig.legend(handles=handles, loc="upper center",
                       ncol=min(len(handles), 4), frameon=False,
                       fontsize=7, bbox_to_anchor=(0.5, 1.02))
        fig.align_ylabels()
        fig.align_xlabels()
        save_figure_to_files(fig=fig, save_path=str(out_dir),
                             file_name=fname, suffix=None,
                             file_types=["pdf", "eps"], dpi=300)
        plt.close()
    sub = df[df["model_type"].isin(model_subset)]
    # 1. Combined: reward_group as hue just for the FULL model
    _make_fig(sub[sub.model_type=='full'],
              hue_col="reward_group",
              hue_order=[rg for rg in ["R+", "R-"] if rg in reward_groups_present],
              plot_palette=_RG_COLOR,
              title_suffix="",
              fname="global_performance_full_by_rg")
    # 2. Per reward group: model_type as hue
    for rg in reward_groups_present:
        sub_rg = sub[sub["reward_group"] == rg]
        if sub_rg.empty:
            continue
        _make_fig(sub_rg,
                  hue_col="model_type",
                  hue_order=model_subset,
                  plot_palette=palette,
                  title_suffix=f" – {rg}",
                  fname=f"global_performance_{rg}")
    logger.info(f"  Global performance figures saved to {out_dir}")


def plot_per_mouse_performance(df_single: pd.DataFrame, figure_path: Path,
                                model_name: str = "full"):
    """
    For a given model_name (feature set), plot per-mouse LL and accuracy
    across n_states, with individual mouse trajectories + group mean±SD.

    Produces one figure per reward group (and one combined).
    Mirrors plot_single_mouse_metrics() from plot_glmhmm_results.py but:
      • accepts any model_name
      • colours individual mouse lines by reward group
      • plots balanced accuracy as a third panel
    """
    from plotting_utils import remove_top_right_frame, save_figure_to_files

    out_dir = figure_path / "per_mouse_performance" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    sub = df_single[df_single["model_name"] == model_name].copy()
    if sub.empty:
        logger.warning(f"  plot_per_mouse_performance: no data for model_name='{model_name}'")
        return

    rgs_present = sub["reward_group"].unique().tolist()

    # Average across splits/instances for each (mouse, n_states, reward_group)
    agg = (
        sub.groupby(["mouse_id", "n_states", "reward_group"])[
            ["ll_train", "ll_test",
             "predictive_acc_train", "predictive_acc_test",
             "balanced_predictive_acc_train", "balanced_predictive_acc_test",
                "bpt_train", "bpt_test",]
        ]
        .mean()
        .reset_index()
    )

    k_vals = sorted(agg["n_states"].unique())

    metrics = [
        ("ll_test",                       "Test log-likelihood"),
        ("predictive_acc_test",           "Predictive accuracy (test)"),
        ("balanced_predictive_acc_test",  "Balanced accuracy (test)"),
        ("bpt_test",                       "Bits/trial (test)"),
    ]

    def _make_panel(data, color, fname):

        fig, axs = plt.subplots(1, len(metrics), figsize=(4.5 * len(metrics), 4),
                                dpi=300, constrained_layout=True)
        for ax, (col, ylabel) in zip(axs, metrics):
            if col not in data.columns:
                ax.set_visible(False)
                continue

            # Individual mouse lines
            for mouse_id, mouse_df in data.groupby("mouse_id"):
                ax.plot(mouse_df["n_states"], mouse_df[col],
                        color=color, alpha=0.25, lw=0.8)

            # Group mean ± SEM
            grp = data.groupby("n_states")[col].agg(["mean", "std"]).reset_index()
            ax.errorbar(grp["n_states"], grp["mean"], yerr=grp["std"],
                        color=color, lw=2, marker="o", capsize=3,
                        markeredgecolor="white", zorder=5)

            if "acc" in col:
                ax.axhline(0.5, color="grey", linestyle="--", lw=0.8)
                ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))

            ax.set_xlabel("Number of states", fontsize=9)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.set_xticks(k_vals)
            remove_top_right_frame(ax)

        return fig

    # One figure per reward group
    for rg in rgs_present:
        sub_rg = agg[agg["reward_group"] == rg]
        if sub_rg.empty:
            continue
        color = _RG_COLOR.get(rg, "steelblue")
        fig = _make_panel(sub_rg, color, fname=f"per_mouse_{model_name}_{rg}")
        fig.suptitle(f"Per-mouse performance – {model_name} – {rg}", fontsize=10)
        save_figure_to_files(fig=fig, save_path=str(out_dir),
                             file_name=f"per_mouse_{rg}",
                             suffix=None, file_types=["pdf", "eps"], dpi=300)
        plt.close()

    # Combined figure: both reward groups overlaid, coloured by group
    fig, axs = plt.subplots(1, len(metrics), figsize=(4.5 * len(metrics), 4),
                            dpi=300, constrained_layout=True)
    for ax, (col, ylabel) in zip(axs, metrics):
        if col not in agg.columns:
            ax.set_visible(False)
            continue
        for rg in rgs_present:
            color = _RG_COLOR.get(rg, "steelblue")
            sub_rg = agg[agg["reward_group"] == rg]
            for _, mouse_df in sub_rg.groupby("mouse_id"):
                ax.plot(mouse_df["n_states"], mouse_df[col],
                        color=color, alpha=0.20, lw=0.8)
            grp = sub_rg.groupby("n_states")[col].agg(["mean", "std"]).reset_index()
            ax.errorbar(grp["n_states"], grp["mean"], yerr=grp["std"],
                        color=color, lw=2, marker="o", capsize=3,
                        markeredgecolor="white", label=rg, zorder=5)
        if "acc" in col:
            ax.axhline(0.5, color="grey", linestyle="--", lw=0.8)
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
        ax.set_xlabel("Number of states", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xticks(k_vals)
        remove_top_right_frame(ax)

    axs[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Per-mouse performance – {model_name}", fontsize=10)
    save_figure_to_files(fig=fig, save_path=str(out_dir),
                         file_name="per_mouse_combined",
                         suffix=None, file_types=["pdf", "eps"], dpi=300)
    plt.close()
    logger.info(f"  Per-mouse performance figures saved to {out_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADERS  (weights + trial data from individual NPZ / HDF5 files)
# ─────────────────────────────────────────────────────────────────────────────

def _load_global_weights_long(cfg) -> pd.DataFrame:
    """
    Walk every global NPZ result file and return a long-form DataFrame with
    columns: model_name, reward_group, n_states, split_idx, instance_idx,
             state_idx, feature, weight.
    State indices are permuted via Hungarian-algorithm alignment so states are
    comparable across splits and instances within each (model_name, rg, K).
    """
    feature_sets = build_feature_sets(cfg["features"], cfg.get("trial_types"))
    all_rows = []

    for rg_int in cfg["reward_groups"]:
        rg = _RG_STR.get(int(rg_int), str(rg_int))

        for model_name in feature_sets:
            for n_states in cfg["n_states_list"]:

                # ----------------------------------------------------------
                # Pass 1: collect all weights for this (model, rg, K) to
                # compute permutation indices across splits × instances.
                # ----------------------------------------------------------
                weight_rows = []
                for split_idx in range(cfg["n_splits"]):
                    for instance_idx in range(cfg["n_instances"]):
                        f = (global_model_dir(cfg, split_idx, n_states,
                                              instance_idx, model_name, rg_int)
                             / "global_fit_glmhmm_results.npz")
                        if not f.exists():
                            continue
                        res   = np.load(f, allow_pickle=True)["arr_0"].item()
                        w     = np.array(res["weights"])  # (K, C-1, M)
                        feats = list(res.get("features", cfg["features"]))
                        for s in range(w.shape[0]):
                            for fi, feat in enumerate(feats):
                                weight_rows.append(dict(
                                    model_name=model_name,
                                    reward_group=rg,
                                    n_states=n_states,
                                    split_idx=split_idx,
                                    instance_idx=instance_idx,
                                    state_idx=s,
                                    feature=feat,
                                    weight=float(w[s, 0, fi]),
                                ))

                if not weight_rows:
                    continue

                weight_df = pd.DataFrame(weight_rows)

                # permut_ids keyed by (n_states, split_idx, instance_idx)
                weight_df, permut_ids = utils.align_weights_dataframe(weight_df, use_mean_reference=False)


                # ----------------------------------------------------------
                # Pass 2: re-read weights and append rows with remapped
                # state_idx according to the per-(split, instance) permutation.
                # ----------------------------------------------------------
                for split_idx in range(cfg["n_splits"]):
                    for instance_idx in range(cfg["n_instances"]):
                        f = (global_model_dir(cfg, split_idx, n_states,
                                              instance_idx, model_name, rg_int)
                             / "global_fit_glmhmm_results.npz")
                        if not f.exists():
                            continue
                        res   = np.load(f, allow_pickle=True)["arr_0"].item()
                        w     = np.array(res["weights"])  # (K, C-1, M)
                        feats = list(res.get("features", cfg["features"]))

                        perm_key = (n_states, split_idx, instance_idx)
                        perm     = permut_ids.get(perm_key)  # perm[new_s] = old_s

                        for s in range(w.shape[0]):
                            aligned_s = int(np.where(perm == s)[0][0]) if perm is not None else s
                            for fi, feat in enumerate(feats):
                                all_rows.append(dict(
                                    model_name=model_name,
                                    reward_group=rg,
                                    n_states=n_states,
                                    split_idx=split_idx,
                                    instance_idx=instance_idx,
                                    state_idx=aligned_s,
                                    feature=feat,
                                    weight=float(w[s, 0, fi]),
                                ))

    return pd.DataFrame(all_rows)



def _load_single_weights_long(cfg) -> pd.DataFrame:
    """
    Walk every single-mouse NPZ result file and return a long-form DataFrame
    with the same schema as _load_global_weights_long plus mouse_id.
    All splits and instances are loaded. State indices are permuted in two stages:

      1. Within-mouse alignment (across splits × instances) via Hungarian algorithm.
      2. Across-mouse alignment (per reward_group × model_name × n_states) via
         Hungarian algorithm applied to the per-mouse mean weight vectors.
         This ensures state indices are comparable across mice.
    """
    feature_sets = build_feature_sets(cfg["features"], cfg.get("trial_types"))

    # Buffer rows per (rg, model_name, n_states) so cross-mouse alignment
    # can be applied once all mice in the group have been collected.
    # Keys: (rg, model_name, n_states)  Values: list of row dicts
    group_rows: dict[tuple, list] = {}

    for rg_int in cfg["reward_groups"]:
        rg = _RG_STR.get(int(rg_int), str(rg_int))

        for model_name in feature_sets:

            if model_name != 'full':
                continue


            feats_cfg = feature_sets[model_name]

            for n_states in cfg["n_states_list"]:
                single_base = cfg["single_path"] / _rg_label(rg_int) / model_name
                if not single_base.exists():
                    continue

                group_key = (rg, model_name, n_states)
                if group_key not in group_rows:
                    group_rows[group_key] = []

                for mouse_dir in single_base.iterdir():
                    mouse_id = mouse_dir.name

                    # ----------------------------------------------------------
                    # Step 1: Collect weights across ALL splits × instances and
                    # compute within-mouse permutation indices.
                    # ----------------------------------------------------------
                    weight_rows = []
                    for split_idx in range(cfg["n_splits"]):
                        for inst in range(cfg["n_instances"]):
                            f = (single_model_dir(cfg, mouse_id, split_idx, n_states,
                                                  inst, model_name, rg_int)
                                 / "fit_glmhmm_results.npz")
                            if not f.exists():
                                continue
                            res   = np.load(f, allow_pickle=True)["arr_0"].item()
                            w     = np.array(res["weights"])   # (K, C-1, M)
                            feats = list(res.get("features", feats_cfg))
                            for s in range(w.shape[0]):


                                for fi, feat in enumerate(feats):
                                    weight_rows.append(dict(
                                        mouse_id=mouse_id,
                                        reward_group=rg,
                                        n_states=n_states,
                                        split_idx=split_idx,
                                        instance_idx=inst,
                                        state_idx=s,
                                        feature=feat,
                                        weight=float(w[s, 0, fi]),
                                    ))

                    if not weight_rows:
                        continue

                    weight_df = pd.DataFrame(weight_rows)
                    weight_df, permut_ids = utils.align_weights_dataframe(
                        weight_df, use_mean_reference=False
                    )
                    print(f'  [{mouse_id} | {model_name} | K={n_states} | {rg}] '
                          f'permutation ids: {permut_ids}')


                    # Collect Viterbi sequences to permute using this
                    viterbi_dict = {}
                    f_h5 = (single_model_dir(cfg, mouse_id, split_idx, n_states,
                                             inst, model_name, rg_int)
                            / "data_preds.h5")
                    if not f_h5.exists():
                        continue
                    data = pd.read_hdf(f_h5, key='data')
                    z = data['most_likely_state'].values.astype(int)
                    viterbi_dict[(split_idx, inst)] = z

                    # ----------------------------------------------------------
                    # Step 2: Re-iterate splits × instances; emit rows with
                    # state_idx remapped by within-mouse permutation.
                    # ----------------------------------------------------------
                    for split_idx in range(cfg["n_splits"]):
                        for inst in range(cfg["n_instances"]):
                            f = (single_model_dir(cfg, mouse_id, split_idx, n_states,
                                                  inst, model_name, rg_int)
                                 / "fit_glmhmm_results.npz")
                            if not f.exists():
                                continue
                            res   = np.load(f, allow_pickle=True)["arr_0"].item()
                            w     = np.array(res["weights"])   # (K, C-1, M)
                            feats = list(res.get("features", feats_cfg))

                            perm_key = (n_states, split_idx, inst)
                            perm     = permut_ids.get(perm_key)  # perm[new_s] = old_s

                            for s in range(w.shape[0]):
                                aligned_s = (int(np.where(perm == s)[0][0])
                                             if perm is not None else s)
                                for fi, feat in enumerate(feats):
                                    group_rows[group_key].append(dict(
                                        mouse_id=mouse_id,
                                        model_name=model_name,
                                        reward_group=rg,
                                        n_states=n_states,
                                        split_idx=split_idx,
                                        instance_idx=inst,
                                        state_idx=aligned_s,
                                        feature=feat,
                                        weight=float(w[s, 0, fi]),
                                    ))

    # ------------------------------------------------------------------
    # Step 3: Cross-mouse alignment, independently per group_key.
    # For each mouse compute its mean weight vector per state (averaged
    # across splits × instances after within-mouse alignment), then run
    # the Hungarian algorithm against the grand mean across mice.
    # The resulting permutation is applied to state_idx in the rows.
    # ------------------------------------------------------------------
    all_rows = []

    for (rg, model_name, n_states), rows in group_rows.items():
        if not rows:
            continue

        df = pd.DataFrame(rows)
        mice = df["mouse_id"].unique()

        # Mean weight vector per (mouse, state, feature) — collapses splits/instances
        mouse_state_means = (
            df.groupby(["mouse_id", "state_idx", "feature"])["weight"]
            .mean()
            .reset_index()
        )
        feats = sorted(mouse_state_means["feature"].unique())

        # Build (K, M) weight matrix per mouse
        mouse_weight_matrices: dict[str, np.ndarray] = {}
        for mouse_id in mice:
            mdf = mouse_state_means[mouse_state_means["mouse_id"] == mouse_id]
            w_mat = np.zeros((n_states, len(feats)))
            for s in range(n_states):
                for fi, feat in enumerate(feats):
                    val = mdf.loc[
                        (mdf["state_idx"] == s) & (mdf["feature"] == feat), "weight"
                    ].values
                    w_mat[s, fi] = val[0] if len(val) else 0.0
            mouse_weight_matrices[mouse_id] = w_mat

        # Grand mean across mice as alignment reference
        ref = np.mean(list(mouse_weight_matrices.values()), axis=0)  # (K, M)

        # Per-mouse permutation: perm[new_s] = old_s
        mouse_perms: dict[str, np.ndarray] = {}
        for mouse_id, w_mat in mouse_weight_matrices.items():
            D = np.stack([
                np.linalg.norm(ref[i] - w_mat, axis=1)   # distance from ref state i to each mouse state
                for i in range(n_states)
            ])  # (K_ref, K_mouse)
            _, col_ind = linear_sum_assignment(D)
            mouse_perms[mouse_id] = col_ind
            print(f'  Cross-mouse perm [{mouse_id} | {model_name} | K={n_states} | {rg}]: '
                  f'{col_ind}')

        # Apply cross-mouse permutation to state_idx
        for row in rows:
            perm = mouse_perms.get(row["mouse_id"])
            if perm is not None:
                old_s = row["state_idx"]
                new_s = int(np.where(perm == old_s)[0][0])
                row = {**row, "state_idx": new_s}
            all_rows.append(row)

    return pd.DataFrame(all_rows)

def _load_single_weights_long_permut(cfg, all_perms: dict) -> pd.DataFrame:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    feature_sets = build_feature_sets(cfg["features"], cfg.get("trial_types"))
    # ── Only "full" model is used — skip the loop entirely ────────────────────
    if "full" not in feature_sets:
        return pd.DataFrame()
    feats_cfg  = feature_sets["full"]
    model_name = "full"

    # ── 1. Collect all valid tasks upfront ─────────────────────────────────────
    tasks = []  # (f, mouse_id, rg, n_states, si, inst, feats_cfg)

    for rg_int in cfg["reward_groups"]:
        rg = _RG_STR.get(int(rg_int), str(rg_int))
        single_base = cfg["single_path"] / _rg_label(rg_int) / model_name
        if not single_base.exists():
            continue

        for mouse_dir in single_base.iterdir():
            mouse_id = mouse_dir.name
            for si in range(cfg["n_splits"]):
                for inst in range(cfg["n_instances"]):
                    for n_states in cfg["n_states_list"]:
                        f = (single_model_dir(cfg, mouse_id, si, n_states, inst, model_name, rg_int)
                             / "fit_glmhmm_results.npz")
                        if f.exists():
                            tasks.append((f, mouse_id, rg, rg_int, n_states, si, inst))

    if not tasks:
        return pd.DataFrame()

    # ── 2. Worker: load one NPZ → small DataFrame (no Python loops over weights)
    def _process_one(task) -> pd.DataFrame | None:
        f, mouse_id, rg, rg_int, n_states, si, inst, = task
        try:
            res = np.load(f, allow_pickle=True)["arr_0"].item()
        except Exception as e:
            logger.warning(f"  Could not load {f}: {e}")
            return None

        w     = np.array(res["weights"])                    # (K, 1, M)
        feats = list(res.get("features", feats_cfg))
        K, _, M = w.shape
        w2d   = w[:, 0, :]                                  # (K, M)

        # Apply inv_perm to row order in one shot instead of per-state branching
        inv_perm = all_perms.get((rg, model_name, n_states, mouse_id, si, inst))
        if inv_perm is not None:
            row_order = [int(inv_perm[s]) for s in range(K)]
        else:
            row_order = list(range(K))

        # Build index arrays with np.repeat / np.tile — no Python loop over cells
        state_indices = np.repeat(row_order, M)             # [s0,s0,...,s1,s1,...]
        feature_vals  = np.tile(feats, K)                   # [f0,f1,...,f0,f1,...]
        weight_vals   = w2d[np.arange(K)].ravel()           # flatten in row order

        return pd.DataFrame({
            "mouse_id"    : mouse_id,
            "model_name"  : model_name,
            "reward_group": rg,
            "n_states"    : n_states,
            "split_idx"   : si,
            "instance_idx": inst,
            "state_idx"   : state_indices,
            "feature"     : feature_vals,
            "weight"      : weight_vals,
        })

    # ── 3. Parallel I/O ────────────────────────────────────────────────────────
    max_workers = min(len(tasks), 10)
    frames: list[pd.DataFrame] = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_process_one, t): t for t in tasks}
        for fut in as_completed(futures):
            result = fut.result()
            if result is not None:
                frames.append(result)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _load_single_weights_long_viterbi(cfg) -> pd.DataFrame:
    """
    Walk every single-mouse NPZ result file and return a long-form DataFrame.

    Alignment is performed in two stages:
      1. Within-mouse: Viterbi-based (trial-space) alignment across splits ×
         instances, falling back to weight-based if sequences are unavailable
         or mismatched.
      2. Cross-mouse: weight-based alignment of per-mouse mean weight vectors
         to the grand mean, independently per (reward_group, model_name, n_states).

    :param cfg: pipeline config dict with keys:
                  features, trial_types, reward_groups, n_states_list,
                  single_path, n_splits, n_instances
    :return: long-form DataFrame of aligned weights
    """
    from pathlib import Path

    feature_sets = build_feature_sets(cfg["features"], cfg.get("trial_types"))

    # Buffer rows per (rg, model_name, n_states) for cross-mouse alignment
    group_rows: dict[tuple, list] = {}

    for rg_int in cfg["reward_groups"]:
        rg = _RG_STR.get(int(rg_int), str(rg_int))

        for model_name in feature_sets:

            if model_name != 'full':
                continue

            feats_cfg = feature_sets[model_name]

            for n_states in cfg["n_states_list"]: #note: 1-state model keep entry in permut dict
                single_base = cfg["single_path"] / _rg_label(rg_int) / model_name
                if not single_base.exists():
                    continue

                group_key = (rg, model_name, n_states)
                if group_key not in group_rows:
                    group_rows[group_key] = []

                for mouse_dir in single_base.iterdir():
                    mouse_id = mouse_dir.name

                    # ----------------------------------------------------------
                    # Step 1a: Collect weights across all splits × instances
                    # ----------------------------------------------------------
                    weight_rows = []
                    for split_idx in range(cfg["n_splits"]):
                        for inst in range(cfg["n_instances"]):
                            f = (single_model_dir(cfg, mouse_id, split_idx, n_states,
                                                  inst, model_name, rg_int)
                                 / "fit_glmhmm_results.npz")
                            if not f.exists():
                                continue
                            res = np.load(f, allow_pickle=True)["arr_0"].item()
                            w = np.array(res["weights"])  # (K, 1, M)
                            feats = list(res.get("features", feats_cfg))
                            for s in range(w.shape[0]):
                                for fi, feat in enumerate(feats):
                                    weight_rows.append(dict(
                                        mouse_id=mouse_id,
                                        reward_group=rg,
                                        n_states=n_states,
                                        split_idx=split_idx,
                                        instance_idx=inst,
                                        state_idx=s,
                                        feature=feat,
                                        weight=float(w[s, 0, fi]),
                                    ))

                    if not weight_rows:
                        continue

                    weight_df = pd.DataFrame(weight_rows)

                    # ----------------------------------------------------------
                    # Step 1b: Load Viterbi sequences from data_preds.h5
                    # ----------------------------------------------------------
                    viterbi_dict = {}
                    ref_len = None
                    sequences_ok = True

                    for split_idx in range(cfg["n_splits"]):
                        for inst in range(cfg["n_instances"]):
                            f_h5 = (single_model_dir(cfg, mouse_id, split_idx, n_states,
                                                     inst, model_name, rg_int)
                                    / "data_preds.h5")
                            if not f_h5.exists():
                                sequences_ok = False
                                break

                            data = pd.read_hdf(f_h5, key='data')

                            # Resolve Viterbi column name
                            if 'most_likely_state' in data.columns:
                                z = data['most_likely_state'].values.astype(int)
                            else:
                                # Fall back to argmax of soft posteriors
                                posterior_cols = sorted(
                                    [c for c in data.columns if c.startswith('posterior_state_')]
                                )
                                if not posterior_cols:
                                    sequences_ok = False
                                    break
                                z = data[posterior_cols].values.argmax(axis=1).astype(int)

                            # All sequences must have the same length
                            if ref_len is None:
                                ref_len = len(z)
                            elif len(z) != ref_len:
                                sequences_ok = False
                                break

                            viterbi_dict[(split_idx, inst)] = z

                        if not sequences_ok:
                            break

                    # ----------------------------------------------------------
                    # Step 1c: Compute within-mouse permutations
                    # ----------------------------------------------------------
                    if sequences_ok and len(viterbi_dict) > 0:
                        viterbi_perms = utils.compute_permutations_from_viterbi(
                            viterbi_dict, n_states
                        )
                        print(f'  [{mouse_id} | {model_name} | K={n_states} | {rg}] '
                              f'Using Viterbi-based alignment')
                    else:
                        viterbi_perms = None
                        print(f'  [{mouse_id} | {model_name} | K={n_states} | {rg}] '
                              f'Falling back to weight-based alignment')

                    weight_df, permut_ids = utils.align_weights_dataframe(
                        weight_df,
                        use_mean_reference=False,
                        permutations=viterbi_perms,
                    )
                    print(f'  [{mouse_id} | {model_name} | K={n_states} | {rg}] '
                          f'permutation ids: {permut_ids}')

                    # ----------------------------------------------------------
                    # Step 1d: Re-iterate splits × instances; emit aligned rows
                    # ----------------------------------------------------------
                    for split_idx in range(cfg["n_splits"]):
                        for inst in range(cfg["n_instances"]):
                            f = (single_model_dir(cfg, mouse_id, split_idx, n_states,
                                                  inst, model_name, rg_int)
                                 / "fit_glmhmm_results.npz")
                            if not f.exists():
                                continue
                            res = np.load(f, allow_pickle=True)["arr_0"].item()
                            w = np.array(res["weights"])  # (K, 1, M)
                            feats = list(res.get("features", feats_cfg))

                            perm_key = (n_states, split_idx, inst)
                            perm = permut_ids.get(perm_key)

                            for s in range(w.shape[0]):
                                aligned_s = (int(np.where(perm == s)[0][0])
                                             if perm is not None else s)
                                for fi, feat in enumerate(feats):
                                    group_rows[group_key].append(dict(
                                        mouse_id=mouse_id,
                                        model_name=model_name,
                                        reward_group=rg,
                                        n_states=n_states,
                                        split_idx=split_idx,
                                        instance_idx=inst,
                                        state_idx=aligned_s,
                                        feature=feat,
                                        weight=float(w[s, 0, fi]),
                                    ))

    # --------------------------------------------------------------------------
    # Step 2: Cross-mouse alignment per (rg, model_name, n_states)
    # --------------------------------------------------------------------------
    all_rows = []

    for (rg, model_name, n_states), rows in group_rows.items():
        if not rows:
            continue

        df = pd.DataFrame(rows)
        mice = df["mouse_id"].unique()

        mouse_state_means = (
            df.groupby(["mouse_id", "state_idx", "feature"])["weight"]
            .mean()
            .reset_index()
        )
        feats = sorted(mouse_state_means["feature"].unique())

        # Build (K, M) mean weight matrix per mouse
        mouse_weight_matrices: dict[str, np.ndarray] = {}
        for mouse_id in mice:
            mdf = mouse_state_means[mouse_state_means["mouse_id"] == mouse_id]
            w_mat = np.zeros((n_states, len(feats)))
            for s in range(n_states):
                for fi, feat in enumerate(feats):
                    val = mdf.loc[
                        (mdf["state_idx"] == s) & (mdf["feature"] == feat), "weight"
                    ].values
                    w_mat[s, fi] = val[0] if len(val) else 0.0
            mouse_weight_matrices[mouse_id] = w_mat

        # Grand mean as reference
        ref = np.mean(list(mouse_weight_matrices.values()), axis=0)  # (K, M)

        # Per-mouse permutation via weight-based alignment to grand mean
        mouse_perms: dict[str, np.ndarray] = {}
        for mouse_id, w_mat in mouse_weight_matrices.items():
            D = np.linalg.norm(
                ref[:, None, :] - w_mat[None, :, :], axis=-1
            )  # (K_ref, K_mouse)
            _, col_ind = linear_sum_assignment(D)
            mouse_perms[mouse_id] = col_ind
            print(f'  Cross-mouse perm [{mouse_id} | {model_name} | K={n_states} | {rg}]: '
                  f'{col_ind}')

        # Apply cross-mouse permutation to state_idx
        for row in rows:
            perm = mouse_perms.get(row["mouse_id"])
            old_s = row["state_idx"]
            new_s = int(np.where(perm == old_s)[0][0]) if perm is not None else old_s
            all_rows.append({**row, "state_idx": new_s})

    return pd.DataFrame(all_rows)


def _load_single_trial_data_old(cfg, model_name: str, n_states: int) -> pd.DataFrame:
    feature_sets = build_feature_sets(cfg["features"], cfg.get("trial_types"))
    feats = feature_sets.get(model_name, cfg["features"])
    n_feats = len(feats)
    n_splits = cfg["n_splits"]
    n_instances = cfg["n_instances"]

    # OPT 1 — hoist constants that are identical for every (split, inst, mouse) iteration
    post_cols_ordered = [f"posterior_state_{i + 1}" for i in range(n_states)]
    feat_tile = [feats[fi] for fi in np.tile(np.arange(n_feats), n_states)]  # length K*M

    dfs = []

    for rg_int in cfg["reward_groups"]:
        rg = _RG_STR.get(int(rg_int), str(rg_int))
        single_base = cfg["single_path"] / _rg_label(rg_int) / model_name
        if not single_base.exists():
            continue

        for mouse_dir in single_base.iterdir():
            mouse_id = mouse_dir.name

            # OPT 2 — compute every path once; reuse in Step 1 and Step 2
            path_map = {
                (si, inst): single_model_dir(cfg, mouse_id, si, n_states, inst, model_name, rg_int)
                for si in range(n_splits)
                for inst in range(n_instances)
            }

            # ------------------------------------------------------------------
            # Step 1: build weight DataFrame for state alignment
            # OPT 3 — list of tuples (not dicts) + np.repeat/tile to eliminate
            #          the inner Python loop over states × features
            # ------------------------------------------------------------------
            weight_tuples = []
            valid_keys = set()  # OPT 6 — only load h5 for keys that had a valid npz

            for (si, inst), base in path_map.items():
                f = base / "fit_glmhmm_results.npz"
                if not f.exists():
                    continue
                res = np.load(f, allow_pickle=True)["arr_0"].item()
                w = np.array(res["weights"])   # (K, 1, M)
                K = w.shape[0]
                w_flat = w[:, 0, :].ravel()    # (K*M,) — drop obs_dim axis

                # OPT 3 cont. — vectorised row generation; no per-row Python loop
                state_ids = np.repeat(np.arange(K), n_feats).tolist()
                weight_vals = w_flat.tolist()
                feat_labels = feat_tile[:K * n_feats]  # handle K < n_states gracefully

                weight_tuples.extend(zip(
                    [mouse_id] * (K * n_feats),
                    [rg]       * (K * n_feats),
                    [n_states] * (K * n_feats),
                    [si]       * (K * n_feats),
                    [inst]     * (K * n_feats),
                    state_ids,
                    feat_labels,
                    weight_vals,
                ))
                valid_keys.add((si, inst))

            if not weight_tuples:
                continue

            # OPT 4 — DataFrame from tuples is ~3× faster than from list-of-dicts
            weight_df = pd.DataFrame(weight_tuples, columns=[
                "mouse_id", "reward_group", "n_states", "split_idx",
                "instance_idx", "state_idx", "feature", "weight",
            ])
            weight_df, permut_ids = utils.align_weights_dataframe(
                weight_df, use_mean_reference=True
            )

            # ------------------------------------------------------------------
            # Step 2: load h5 files and apply permutation
            # OPT 6 — iterate valid_keys only; skip (split, inst) with no npz
            # ------------------------------------------------------------------
            for (si, inst) in valid_keys:
                h5 = path_map[(si, inst)] / "data_preds.h5"
                if not h5.exists():
                    continue
                try:
                    df = pd.read_hdf(h5)

                    # OPT 5 — direct assignment avoids the full-copy made by assign()
                    df["mouse_id"]     = mouse_id
                    df["reward_group"] = rg
                    df["n_states"]     = n_states
                    df["model_name"]   = model_name
                    df["split_idx"]    = si
                    df["instance_idx"] = inst

                    # Apply state permutation
                    perm_key = (n_states, si, inst)
                    if perm_key in permut_ids:
                        perm = permut_ids[perm_key]
                        old_cols = [f"posterior_state_{perm[i] + 1}" for i in range(len(perm))]
                        present = [c for c in post_cols_ordered if c in df.columns]
                        if old_cols and len(old_cols) == len(present):
                            df[present] = df[old_cols].to_numpy()

                    # Rolling smooth + dominant state
                    present_post = [c for c in post_cols_ordered if c in df.columns]
                    if present_post:
                        # OPT 7 — operate on the raw numpy array; avoids pandas
                        #          overhead for repeated small-df rolling calls
                        arr = df[present_post].to_numpy(dtype=float)
                        smoothed = (
                            pd.DataFrame(arr, columns=present_post)
                            .rolling(window=5, min_periods=1, center=True)
                            .mean()
                            .to_numpy()
                        )
                        df[present_post] = smoothed
                        df["dominant_state"] = smoothed.argmax(axis=1)

                    dfs.append(df)
                except Exception as e:
                    logger.warning(f"  Could not load {h5}: {e}")

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def _load_single_trial_data_permut(cfg, model_name: str, all_perms: dict) -> pd.DataFrame:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    n_splits    = cfg["n_splits"]
    n_instances = cfg["n_instances"]

    # ── 1. Collect all valid tasks upfront ─────────────────────────────────────
    # Avoids building a path_map dict only to discard most of it.
    Task = tuple  # (h5, mouse_id, rg, n_states, si, inst, post_cols_ordered)
    tasks: list[Task] = []

    for n_states in cfg["n_states_list"]:
        post_cols_ordered = [f"posterior_state_{i + 1}" for i in range(n_states)]

        for rg_int in cfg["reward_groups"]:
            rg = _RG_STR.get(int(rg_int), str(rg_int))
            single_base = cfg["single_path"] / _rg_label(rg_int) / model_name
            if not single_base.exists():
                continue

            for mouse_dir in single_base.iterdir():
                mouse_id = mouse_dir.name
                for si in range(n_splits):
                    for inst in range(n_instances):
                        p  = single_model_dir(cfg, mouse_id, si, n_states, inst, model_name, rg_int)
                        h5 = p / "data_preds.h5"
                        if h5.exists():
                            tasks.append((h5, mouse_id, rg, n_states, si, inst, post_cols_ordered))

    if not tasks:
        return pd.DataFrame()

    # ── 2. Worker: load + transform one file ───────────────────────────────────
    def _process_one(task: Task) -> pd.DataFrame | None:
        h5, mouse_id, rg, n_states, si, inst, post_cols_ordered = task
        try:
            df = pd.read_hdf(h5)
        except Exception as e:
            logger.warning(f"  Could not load {h5}: {e}")
            return None

        # Scalar metadata in one shot
        df = df.assign(
            mouse_id     = mouse_id,
            reward_group = rg,
            n_states     = n_states,
            model_name   = model_name,
            split_idx    = si,
            instance_idx = inst,
        )

        # Permutation remapping
        inv_perm = all_perms.get((rg, model_name, n_states, mouse_id, si, inst))
        if inv_perm is not None:
            present = [c for c in post_cols_ordered if c in df.columns]
            if present:
                old_cols    = [f"posterior_state_{inv_perm[i] + 1}" for i in range(len(inv_perm))]
                df[present] = df[old_cols].to_numpy()
            if "most_likely_state" in df.columns:
                df["most_likely_state"] = inv_perm[df["most_likely_state"].to_numpy()]

        # Dominant state
        if "most_likely_state" in df.columns:
            df["dominant_state"] = df["most_likely_state"]
        else:
            logger.warning(
                f"  most_likely_state not found in {h5}; falling back to posterior argmax"
            )
            present_post = [c for c in post_cols_ordered if c in df.columns]
            if present_post:
                df["dominant_state"] = df[present_post].to_numpy().argmax(axis=1)

        return df

    # ── 3. Parallel I/O ────────────────────────────────────────────────────────
    # Threads (not processes) because HDF5 reads are I/O-bound and GIL is released
    # during the underlying C-level disk read.  Cap workers to avoid thrashing the
    # NAS; 8–12 is typically the sweet spot for a network-mounted share.
    max_workers = min(len(tasks), 10)
    dfs: list[pd.DataFrame] = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_process_one, t): t for t in tasks}
        for fut in as_completed(futures):
            result = fut.result()
            if result is not None:
                dfs.append(result)

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def _load_single_trial_data(cfg, model_name: str, n_states: int) -> pd.DataFrame:
    feature_sets = build_feature_sets(cfg["features"], cfg.get("trial_types"))
    feats = feature_sets.get(model_name, cfg["features"])
    n_feats = len(feats)
    n_splits = cfg["n_splits"]
    n_instances = cfg["n_instances"]

    post_cols_ordered = [f"posterior_state_{i + 1}" for i in range(n_states)]
    feat_tile = [feats[fi] for fi in np.tile(np.arange(n_feats), n_states)]

    dfs = []

    for rg_int in cfg["reward_groups"]:
        rg = _RG_STR.get(int(rg_int), str(rg_int))
        single_base = cfg["single_path"] / _rg_label(rg_int) / model_name
        if not single_base.exists():
            continue

        for mouse_dir in single_base.iterdir():
            mouse_id = mouse_dir.name

            path_map = {
                (si, inst): single_model_dir(cfg, mouse_id, si, n_states, inst, model_name, rg_int)
                for si in range(n_splits)
                for inst in range(n_instances)
            }

            # ------------------------------------------------------------------
            # Step 1: build weight DataFrame for state alignment
            # ------------------------------------------------------------------
            weight_tuples = []
            valid_keys = set()

            for (si, inst), base in path_map.items():
                f = base / "fit_glmhmm_results.npz"
                if not f.exists():
                    continue
                res = np.load(f, allow_pickle=True)["arr_0"].item()
                w = np.array(res["weights"])   # (K, 1, M)
                K = w.shape[0]
                w_flat = w[:, 0, :].ravel()    # (K*M,)

                state_ids = np.repeat(np.arange(K), n_feats).tolist()
                weight_vals = w_flat.tolist()
                feat_labels = feat_tile[:K * n_feats]

                weight_tuples.extend(zip(
                    [mouse_id] * (K * n_feats),
                    [rg]       * (K * n_feats),
                    [n_states] * (K * n_feats),
                    [si]       * (K * n_feats),
                    [inst]     * (K * n_feats),
                    state_ids,
                    feat_labels,
                    weight_vals,
                ))
                valid_keys.add((si, inst))

            if not weight_tuples:
                continue

            weight_df = pd.DataFrame(weight_tuples, columns=[
                "mouse_id", "reward_group", "n_states", "split_idx",
                "instance_idx", "state_idx", "feature", "weight",
            ])
            weight_df, permut_ids = utils.align_weights_dataframe(
                weight_df, use_mean_reference=True
            )

            # ------------------------------------------------------------------
            # Step 2: load h5 files and apply permutation
            # ------------------------------------------------------------------
            for (si, inst) in valid_keys:
                h5 = path_map[(si, inst)] / "data_preds.h5"
                if not h5.exists():
                    continue
                try:
                    df = pd.read_hdf(h5)

                    df["mouse_id"]     = mouse_id
                    df["reward_group"] = rg
                    df["n_states"]     = n_states
                    df["model_name"]   = model_name
                    df["split_idx"]    = si
                    df["instance_idx"] = inst

                    # ----------------------------------------------------------
                    # Apply state permutation
                    # perm[new_idx] = old_idx, so:
                    #   - posterior columns: new position i <- old position perm[i]
                    #   - most_likely_state (integer label): build inv_perm where
                    #     inv_perm[old_idx] = new_idx, then remap each integer label
                    # ----------------------------------------------------------
                    perm_key = (n_states, si, inst)
                    if perm_key in permut_ids:
                        perm = permut_ids[perm_key]

                        # Permute posterior probability columns
                        old_cols = [f"posterior_state_{perm[i] + 1}" for i in range(len(perm))]
                        present = [c for c in post_cols_ordered if c in df.columns]
                        if old_cols and len(old_cols) == len(present):
                            df[present] = df[old_cols].to_numpy()

                        # Permute most_likely_state using the inverse permutation:
                        # inv_perm maps old state index -> new (aligned) state index
                        if "most_likely_state" in df.columns:
                            inv_perm = np.empty(len(perm), dtype=int)
                            inv_perm[perm] = np.arange(len(perm))
                            df["most_likely_state"] = inv_perm[df["most_likely_state"].to_numpy()]

                    # ----------------------------------------------------------
                    # Use Viterbi most_likely_state as the dominant state
                    # (already permuted above)
                    # ----------------------------------------------------------
                    if "most_likely_state" in df.columns:
                        df["dominant_state"] = df["most_likely_state"]
                    else:
                        logger.warning(
                            f"  most_likely_state not found in {h5}; "
                            "falling back to posterior argmax"
                        )
                        # Fallback: posterior argmax (no smoothing)
                        present_post = [c for c in post_cols_ordered if c in df.columns]
                        if present_post:
                            df["dominant_state"] = (
                                df[present_post].to_numpy().argmax(axis=1)
                            )

                    # ----------------------------------------------------------
                    # PREVIOUS VERSION (posterior argmax with rolling smoothing):
                    # kept here for reference — delete once Viterbi path is stable
                    # ----------------------------------------------------------
                    # present_post = [c for c in post_cols_ordered if c in df.columns]
                    # if present_post:
                    #     # Operate on raw numpy array to avoid pandas overhead
                    #     arr = df[present_post].to_numpy(dtype=float)
                    #     smoothed = (
                    #         pd.DataFrame(arr, columns=present_post)
                    #         .rolling(window=5, min_periods=1, center=True)
                    #         .mean()
                    #         .to_numpy()
                    #     )
                    #     df[present_post] = smoothed
                    #     df["dominant_state"] = smoothed.argmax(axis=1)
                    # ----------------------------------------------------------

                    dfs.append(df)
                except Exception as e:
                    logger.warning(f"  Could not load {h5}: {e}")

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def _load_single_trial_data_viterbi(cfg, model_name: str, n_states: int) -> pd.DataFrame:
    feature_sets = build_feature_sets(cfg["features"], cfg.get("trial_types"))
    feats = feature_sets.get(model_name, cfg["features"])
    n_feats = len(feats)
    n_splits = cfg["n_splits"]
    n_instances = cfg["n_instances"]

    post_cols_ordered = [f"posterior_state_{i + 1}" for i in range(n_states)]
    feat_tile = [feats[fi] for fi in np.tile(np.arange(n_feats), n_states)]

    dfs = []

    for rg_int in cfg["reward_groups"]:
        rg = _RG_STR.get(int(rg_int), str(rg_int))
        single_base = cfg["single_path"] / _rg_label(rg_int) / model_name
        if not single_base.exists():
            continue

        for mouse_dir in single_base.iterdir():
            mouse_id = mouse_dir.name


            path_map = {
                (si, inst): single_model_dir(cfg, mouse_id, si, n_states, inst, model_name, rg_int)
                for si in range(n_splits)
                for inst in range(n_instances)
            }

            # ------------------------------------------------------------------
            # Step 1: build weight DataFrame for state alignment
            # ------------------------------------------------------------------
            weight_tuples = []
            valid_keys = set()

            for (si, inst), base in path_map.items():
                f = base / "fit_glmhmm_results.npz"
                if not f.exists():
                    continue
                res = np.load(f, allow_pickle=True)["arr_0"].item()
                w = np.array(res["weights"])   # (K, 1, M)
                K = w.shape[0]
                w_flat = w[:, 0, :].ravel()    # (K*M,)

                state_ids   = np.repeat(np.arange(K), n_feats).tolist()
                weight_vals = w_flat.tolist()
                feat_labels = feat_tile[:K * n_feats]

                weight_tuples.extend(zip(
                    [mouse_id] * (K * n_feats),
                    [rg]       * (K * n_feats),
                    [n_states] * (K * n_feats),
                    [si]       * (K * n_feats),
                    [inst]     * (K * n_feats),
                    state_ids,
                    feat_labels,
                    weight_vals,
                ))
                valid_keys.add((si, inst))

            if not weight_tuples:
                continue

            weight_df = pd.DataFrame(weight_tuples, columns=[
                "mouse_id", "reward_group", "n_states", "split_idx",
                "instance_idx", "state_idx", "feature", "weight",
            ])

            # ------------------------------------------------------------------
            # Step 1b: load Viterbi sequences from data_preds.h5
            # ------------------------------------------------------------------
            viterbi_dict = {}
            ref_len      = None
            sequences_ok = True

            for (si, inst) in valid_keys:
                h5 = path_map[(si, inst)] / "data_preds.h5"
                if not h5.exists():
                    sequences_ok = False
                    break
                try:
                    data = pd.read_hdf(h5)
                    if "most_likely_state" in data.columns:
                        z = data["most_likely_state"].values.astype(int)
                    else:
                        present_post = sorted(
                            [c for c in data.columns if c.startswith("posterior_state_")]
                        )
                        if not present_post:
                            sequences_ok = False
                            break
                        z = data[present_post].values.argmax(axis=1).astype(int)

                    if ref_len is None:
                        ref_len = len(z)
                    elif len(z) != ref_len:
                        sequences_ok = False
                        break

                    viterbi_dict[(si, inst)] = z
                except Exception as e:
                    logger.warning(f"  Could not load Viterbi from {h5}: {e}")
                    sequences_ok = False
                    break

            if sequences_ok and viterbi_dict:
                viterbi_perms = utils.compute_permutations_from_viterbi(
                    viterbi_dict, n_states
                )
                logger.info(f"  [{mouse_id} | {model_name} | K={n_states} | {rg}] "
                            f"Using Viterbi-based alignment")
            else:
                viterbi_perms = None
                logger.warning(f"  [{mouse_id} | {model_name} | K={n_states} | {rg}] "
                               f"Falling back to weight-based alignment")

            weight_df, permut_ids = utils.align_weights_dataframe(
                weight_df,
                use_mean_reference=True,
                permutations=viterbi_perms,  # None → weight-based fallback
            )

            # ------------------------------------------------------------------
            # Step 2: load h5 files and apply permutation
            # (unchanged from here down)
            # ------------------------------------------------------------------
            for (si, inst) in valid_keys:
                h5 = path_map[(si, inst)] / "data_preds.h5"
                if not h5.exists():
                    continue
                try:
                    df = pd.read_hdf(h5)

                    df["mouse_id"]     = mouse_id
                    df["reward_group"] = rg
                    df["n_states"]     = n_states
                    df["model_name"]   = model_name
                    df["split_idx"]    = si
                    df["instance_idx"] = inst

                    perm_key = (n_states, si, inst)
                    if perm_key in permut_ids:
                        perm = permut_ids[perm_key]

                        old_cols = [f"posterior_state_{perm[i] + 1}" for i in range(len(perm))]
                        present  = [c for c in post_cols_ordered if c in df.columns]
                        if old_cols and len(old_cols) == len(present):
                            df[present] = df[old_cols].to_numpy()

                        if "most_likely_state" in df.columns:
                            inv_perm = np.empty(len(perm), dtype=int)
                            inv_perm[perm] = np.arange(len(perm))
                            df["most_likely_state"] = inv_perm[df["most_likely_state"].to_numpy()]

                    if "most_likely_state" in df.columns:
                        df["dominant_state"] = df["most_likely_state"]
                    else:
                        logger.warning(
                            f"  most_likely_state not found in {h5}; "
                            "falling back to posterior argmax"
                        )
                        present_post = [c for c in post_cols_ordered if c in df.columns]
                        if present_post:
                            df["dominant_state"] = (
                                df[present_post].to_numpy().argmax(axis=1)
                            )

                    dfs.append(df)
                except Exception as e:
                    logger.warning(f"  Could not load {h5}: {e}")

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS 1 – State alignment diagnostics (Hungarian algorithm)
# ─────────────────────────────────────────────────────────────────────────────

def plot_state_alignment_diagnostics(cfg, figure_path: Path, model_name: str = "full"):
    """
    For each (model_name, n_states > 1, reward_group):

      1. Load raw weights long-form for all (split_idx, instance_idx).
      2. Run utils.align_weights_dataframe (Hungarian algorithm) to find the
         permutation that maps each (split, instance) to a common reference.
      3. Produce three diagnostic figures per combination:

         Fig A – "before" grid: rows = split × instance, cols = K states.
                 Each cell is a bar chart of raw (unaligned) feature weights.
         Fig B – "after"  grid: same layout but with states reordered by the
                 computed permutation — good alignment → same state profile
                 in the same column across all rows.
         Fig C – permutation map: heatmap showing which original state index was
                 mapped to which aligned state index for every (split, instance).
         Fig D – summary: mean ± SE aligned weights per state, one subplot
                 per state, averaged over all splits and instances.

    Requires utils.align_weights_dataframe(df, use_mean_reference) which
    returns (aligned_df, permutations_dict) where permutations_dict is keyed
    by (n_states, split_idx, instance_idx).
    """
    from utils import align_weights_dataframe
    from plotting_utils import remove_top_right_frame, save_figure_to_files

    out_base = figure_path / "state_alignment" / model_name

    feature_sets = build_feature_sets(cfg["features"], cfg.get("trial_types"))
    feats = feature_sets.get(model_name, cfg["features"])

    for rg_int in cfg["reward_groups"]:
        rg    = _RG_STR.get(int(rg_int), str(rg_int))
        color = _RG_COLOR.get(rg, "steelblue")
        out_dir = out_base / rg
        out_dir.mkdir(parents=True, exist_ok=True)

        for n_states in sorted(cfg["n_states_list"]):
            if n_states == 1:
                continue  # alignment is trivial for K=1

            # ── Collect raw weights: all splits × instances ───────────────────
            rows = []
            for split_idx in range(cfg["n_splits"]):
                for instance_idx in range(cfg["n_instances"]):
                    f = (global_model_dir(cfg, split_idx, n_states,
                                         instance_idx, model_name, rg_int)
                         / "global_fit_glmhmm_results.npz")

                    if not f.exists():
                        continue

                    res = np.load(f, allow_pickle=True)["arr_0"].item()
                    w   = np.array(res["weights"])   # (K, C-1, M)
                    for s in range(w.shape[0]):
                        for fi, feat in enumerate(feats):
                            rows.append(dict(
                                n_states=n_states,
                                split_idx=split_idx,
                                instance_idx=instance_idx,
                                state_idx=s,
                                feature=feat,
                                weight=float(w[s, 0, fi]),
                            ))

            if not rows:
                continue

            wdf_raw = pd.DataFrame(rows)
            print('WEIGHT DATAFRAME', wdf_raw.head())

            # ── Run alignment ─────────────────────────────────────────────────
            wdf_aligned, permutations = align_weights_dataframe(
                wdf_raw.copy(), use_mean_reference=False
            )

            # Collect (split, instance) combos actually present
            si_pairs = (
                wdf_raw[["split_idx", "instance_idx"]]
                .drop_duplicates()
                .sort_values(["split_idx", "instance_idx"])
                .values.tolist()
            )
            n_rows_grid = len(si_pairs)

            def _weight_matrix(df, si_pairs, n_states, feats):
                """Return dict (split,inst) → ndarray (K, M) in state_idx order."""
                out = {}
                for si, ii in si_pairs:
                    sub = df[(df.split_idx == si) & (df.instance_idx == ii)]
                    mat = np.zeros((n_states, len(feats)))
                    for s in range(n_states):
                        row = sub[sub.state_idx == s]
                        for fi, feat in enumerate(feats):
                            val = row[row.feature == feat]["weight"].values
                            mat[s, fi] = val[0] if len(val) else 0.0
                    out[(si, ii)] = mat
                return out

            mats_raw     = _weight_matrix(wdf_raw,     si_pairs, n_states, feats)
            mats_aligned = _weight_matrix(wdf_aligned, si_pairs, n_states, feats)

            # shared y-limits across all cells for comparability
            all_w = np.concatenate([m.ravel() for m in mats_raw.values()])
            ylim  = (float(all_w.min()) - 0.1, float(all_w.max()) + 0.1)

            # ─────────────────────────────────────────────────────────────────
            # Fig A – before alignment
            # ─────────────────────────────────────────────────────────────────
            fig_a, axs_a = plt.subplots(
                n_rows_grid, n_states,
                figsize=(2.8 * n_states, 2.0 * n_rows_grid),
                dpi=180, constrained_layout=True, squeeze=False,
            )
            for row_i, (si, ii) in enumerate(si_pairs):
                mat = mats_raw[(si, ii)]
                for col_i in range(n_states):
                    ax = axs_a[row_i, col_i]
                    ax.bar(range(len(feats)), mat[col_i],
                           color=color, alpha=0.70, width=0.7)
                    ax.axhline(0, color="k", lw=0.5, ls="--")
                    ax.set_ylim(ylim)
                    ax.set_xticks(range(len(feats)))
                    ax.set_xticklabels(
                        feats if row_i == n_rows_grid - 1 else [],
                        rotation=45, ha="right", fontsize=5,
                    )
                    ax.set_ylabel("w" if col_i == 0 else "", fontsize=6)
                    ax.set_title(
                        f"State {col_i+1}" if row_i == 0 else "",
                        fontsize=7,
                    )
                    if col_i == n_states - 1:
                        ax.set_ylabel(f"sp{si} i{ii}", fontsize=5,
                                      rotation=0, labelpad=30, va="center")
                        ax.yaxis.set_label_position("right")
                    remove_top_right_frame(ax)

            fig_a.suptitle(
                f"BEFORE alignment | {model_name} K={n_states} | {rg}",
                fontsize=9,
            )
            save_figure_to_files(
                fig=fig_a, save_path=str(out_dir),
                file_name=f"A_before_K{n_states}",
                suffix=None, file_types=["pdf", "png"], dpi=180,
            )
            plt.close()

            # ─────────────────────────────────────────────────────────────────
            # Fig B – after alignment (same layout, permuted state columns)
            # ─────────────────────────────────────────────────────────────────
            fig_b, axs_b = plt.subplots(
                n_rows_grid, n_states,
                figsize=(2.8 * n_states, 2.0 * n_rows_grid),
                dpi=180, constrained_layout=True, squeeze=False,
            )
            for row_i, (si, ii) in enumerate(si_pairs):
                mat = mats_aligned[(si, ii)]
                for col_i in range(n_states):
                    ax = axs_b[row_i, col_i]
                    ax.bar(range(len(feats)), mat[col_i],
                           color=color, alpha=0.70, width=0.7)
                    ax.axhline(0, color="k", lw=0.5, ls="--")
                    ax.set_ylim(ylim)
                    ax.set_xticks(range(len(feats)))
                    ax.set_xticklabels(
                        feats if row_i == n_rows_grid - 1 else [],
                        rotation=45, ha="right", fontsize=5,
                    )
                    ax.set_ylabel("w" if col_i == 0 else "", fontsize=6)
                    ax.set_title(
                        f"State {col_i+1}" if row_i == 0 else "",
                        fontsize=7,
                    )
                    if col_i == n_states - 1:
                        ax.set_ylabel(f"sp{si} i{ii}", fontsize=5,
                                      rotation=0, labelpad=30, va="center")
                        ax.yaxis.set_label_position("right")
                    remove_top_right_frame(ax)

            fig_b.suptitle(
                f"AFTER alignment | {model_name} K={n_states} | {rg}",
                fontsize=9,
            )
            save_figure_to_files(
                fig=fig_b, save_path=str(out_dir),
                file_name=f"B_after_K{n_states}",
                suffix=None, file_types=["pdf", "png"], dpi=180,
            )
            plt.close()

            # ─────────────────────────────────────────────────────────────────
            # Fig C – permutation map heatmap
            # Each cell (row=split×inst, col=aligned state) shows the original
            # state index that was mapped there.  Diagonal = no permutation.
            # ─────────────────────────────────────────────────────────────────
            perm_matrix = np.full((n_rows_grid, n_states), np.nan)
            for row_i, (si, ii) in enumerate(si_pairs):
                key = (n_states, si, ii)
                if key in permutations:
                    perm_matrix[row_i] = permutations[key]

            fig_c, ax_c = plt.subplots(
                figsize=(2.0 * n_states, 0.55 * n_rows_grid + 1.0),
                dpi=180, constrained_layout=True,
            )
            im = ax_c.imshow(perm_matrix, cmap="tab10",
                             vmin=0, vmax=n_states - 1,
                             aspect="auto", interpolation="nearest")

            # Annotate each cell with the mapped original state index
            for row_i in range(n_rows_grid):
                for col_i in range(n_states):
                    val = perm_matrix[row_i, col_i]
                    if not np.isnan(val):
                        ax_c.text(col_i, row_i, str(int(val)),
                                  ha="center", va="center",
                                  fontsize=7, color="white")

            ax_c.set_xticks(range(n_states))
            ax_c.set_xticklabels([f"Aligned\nstate {s}" for s in range(n_states)],
                                 fontsize=7)
            ax_c.set_yticks(range(n_rows_grid))
            ax_c.set_yticklabels([f"sp{si} i{ii}" for si, ii in si_pairs],
                                 fontsize=6)
            ax_c.set_xlabel("Aligned state index", fontsize=8)
            ax_c.set_ylabel("Split / instance", fontsize=8)

            plt.colorbar(im, ax=ax_c, shrink=0.6,
                         label="Original state index")
            fig_c.suptitle(
                f"Permutation map | {model_name} K={n_states} | {rg}",
                fontsize=9,
            )
            save_figure_to_files(
                fig=fig_c, save_path=str(out_dir),
                file_name=f"C_permutation_map_K{n_states}",
                suffix=None, file_types=["pdf", "png"], dpi=180,
            )
            plt.close()

            # ─────────────────────────────────────────────────────────────────
            # Fig D – summary: mean ± SE of aligned weights, one subplot / state
            # ─────────────────────────────────────────────────────────────────
            fig_d, axs_d = plt.subplots(
                1, n_states,
                figsize=(3.2 * n_states, 3.2),
                dpi=220, constrained_layout=True, squeeze=False,
            )
            for s_i in range(n_states):
                ax = axs_d[0, s_i]
                sub = wdf_aligned[wdf_aligned.state_idx == s_i]
                grp = (
                    sub.groupby("feature")["weight"]
                    .agg(["mean", "sem"])
                    .reindex(feats)
                    .reset_index()
                )
                ax.bar(range(len(feats)), grp["mean"],
                       yerr=grp["sem"], color=color,
                       alpha=0.75, capsize=3, width=0.6,
                       error_kw={"lw": 0.8})
                ax.axhline(0, color="k", lw=0.5, ls="--")
                ax.set_xticks(range(len(feats)))
                ax.set_xticklabels(feats, rotation=45, ha="right", fontsize=6)
                ax.set_ylabel("Weight (mean ± SE)" if s_i == 0 else "",
                              fontsize=7)
                ax.set_title(f"State {s_i+1}", fontsize=8)
                remove_top_right_frame(ax)

            n_si = len(si_pairs)
            fig_d.suptitle(
                f"Aligned weights summary | {model_name} K={n_states} | {rg}"
                f"  (n={n_si} split×inst)",
                fontsize=9,
            )
            save_figure_to_files(
                fig=fig_d, save_path=str(out_dir),
                file_name=f"D_aligned_summary_K{n_states}",
                suffix=None, file_types=["pdf", "png"], dpi=220,
            )
            plt.close()

    logger.info(f"  [1] State alignment diagnostics → {out_base}")


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS 2 – Per-mouse weight diagnostic: splits × states grid
# ─────────────────────────────────────────────────────────────────────────────

def plot_mouse_weight_splits(cfg, figure_path: Path, model_name: str = "full"):
    """
    Plot GLM-HMM weights before and after alignment.

    Layout:
        rows = (split_idx, instance_idx) + mean of aligned states (bottom row)
        cols = states
        colors = before / after alignment
        bottom row shows mean ± std of aligned/permuted states as a single bar per state
    """

    import seaborn as sns
    from plotting_utils import remove_top_right_frame, save_figure_to_files

    out_base = figure_path / "weight_diagnostics" / model_name
    feature_sets = build_feature_sets(cfg["features"], cfg.get("trial_types"))
    feats = feature_sets.get(model_name, cfg["features"])

    sns.set_style("whitegrid")

    for rg_int in cfg["reward_groups"]:

        rg = _RG_STR.get(int(rg_int), str(rg_int))
        single_base = cfg["single_path"] / _rg_label(rg_int) / model_name
        if not single_base.exists():
            continue

        for mouse_dir in sorted(single_base.iterdir()):
            mouse_id = mouse_dir.name

            for n_states in sorted(cfg["n_states_list"]):

                rows = []

                # ----------------------------------
                # LOAD WEIGHTS
                # ----------------------------------
                for split_idx in range(cfg["n_splits"]):
                    for inst in range(cfg["n_instances"]):

                        f = (
                            single_model_dir(
                                cfg,
                                mouse_id,
                                split_idx,
                                n_states,
                                inst,
                                model_name,
                                rg_int,
                            )
                            / "fit_glmhmm_results.npz"
                        )

                        if not f.exists():
                            continue

                        res = np.load(f, allow_pickle=True)["arr_0"].item()
                        w = np.array(res["weights"])  # (K,1,M)

                        for s in range(w.shape[0]):
                            for fi, feat in enumerate(feats):
                                rows.append(
                                    dict(
                                        mouse_id=mouse_id,
                                        reward_group=rg,
                                        n_states=n_states,
                                        split_idx=split_idx,
                                        instance_idx=inst,
                                        state_idx=s,
                                        feature=feat,
                                        weight=w[s, 0, fi],
                                    )
                                )

                weight_df = pd.DataFrame(rows)
                if weight_df.empty:
                    continue

                # ----------------------------------
                # ALIGN STATES
                # ----------------------------------
                before_df = weight_df.copy()
                before_df["alignment"] = "before"

                aligned_df, _ = utils.align_weights_dataframe(
                    weight_df, use_mean_reference=False
                )
                aligned_df["alignment"] = "after"

                plot_df = pd.concat([before_df, aligned_df], ignore_index=True)

                # ----------------------------------
                # ROW IDENTIFIERS FOR SPLIT × INSTANCE
                # ----------------------------------
                model_rows = (
                    plot_df[["split_idx", "instance_idx"]]
                    .drop_duplicates()
                    .sort_values(["split_idx", "instance_idx"])
                )
                model_rows["row_id"] = (
                    "S" + model_rows["split_idx"].astype(str)
                    + "-I" + model_rows["instance_idx"].astype(str)
                )

                row_lookup = dict(
                    zip(
                        zip(model_rows.split_idx, model_rows.instance_idx),
                        model_rows.row_id,
                    )
                )
                plot_df["row_id"] = plot_df.apply(
                    lambda r: row_lookup[(r.split_idx, r.instance_idx)], axis=1
                )

                # ----------------------------------
                # COMPUTE BOTTOM ROW: MEAN ± STD ACROSS ALIGNED MODELS
                # ----------------------------------
                aligned_only = aligned_df.copy()

                mean_stats_df = (
                    aligned_only.groupby(["state_idx", "feature"], as_index=False)["weight"]
                    .agg(["mean", "std"])
                    .reset_index()
                )
                mean_stats_df["row_id"] = "mean"
                mean_stats_df["split_idx"] = -1
                mean_stats_df["instance_idx"] = -1
                mean_stats_df = mean_stats_df.rename(columns={"mean": "weight_mean", "std": "weight_std"})

                row_order = model_rows["row_id"].tolist() + ["mean"]
                n_rows = len(row_order)
                n_cols = n_states

                fig, axs = plt.subplots(
                    n_rows,
                    n_cols,
                    figsize=(3 * n_cols, 2.2 * n_rows),
                    dpi=200,
                    sharex=True,
                    sharey=True,
                    squeeze=False,
                )

                # ----------------------------------
                # PLOT EACH CELL
                # ----------------------------------
                for row_i, row_id in enumerate(row_order):
                    for col_i in range(n_states):
                        ax = axs[row_i, col_i]

                        if row_id == "mean":
                            df_plot = mean_stats_df[mean_stats_df["state_idx"] == col_i]
                            if df_plot.empty:
                                ax.axis("off")
                                continue

                            # Single bar per state, mean ± std
                            ax.bar(
                                df_plot["feature"],
                                df_plot["weight_mean"],
                                yerr=df_plot["weight_std"],
                                color="steelblue",
                                alpha=0.7,
                                capsize=3,
                            )
                        else:
                            df_plot = plot_df[
                                (plot_df["row_id"] == row_id)
                                & (plot_df["state_idx"] == col_i)
                            ]
                            if df_plot.empty:
                                ax.axis("off")
                                continue

                            # Individual bars before vs after alignment
                            sns.barplot(
                                data=df_plot,
                                x="feature",
                                y="weight",
                                hue="alignment",
                                ax=ax,
                                errorbar=None,
                            )

                        ax.axhline(0, color="k", lw=0.6, ls="--")

                        if row_i == n_rows - 1:
                            ax.set_xticklabels(
                                feats, rotation=45, ha="right", fontsize=7
                            )
                        else:
                            ax.set_xticklabels([])

                        if col_i == 0:
                            ax.set_ylabel(row_id)
                        else:
                            ax.set_ylabel("")

                        if row_i == 0:
                            ax.set_title(f"State {col_i+1}")

                        remove_top_right_frame(ax)

                handles, labels = ax.get_legend_handles_labels()
                fig.legend(handles, labels, loc="upper right")

                fig.suptitle(
                    f"{mouse_id} | {model_name} | K={n_states} | {rg}",
                    fontsize=10,
                )

                out_dir = out_base / rg / mouse_id
                out_dir.mkdir(parents=True, exist_ok=True)

                save_figure_to_files(
                    fig=fig,
                    save_path=str(out_dir),
                    file_name=f"weight_alignment_K{n_states}",
                    suffix=None,
                    file_types=["pdf", "png"],
                    dpi=200,
                )

                plt.close(fig)

    logger.info(f"  [1] Per-mouse weight diagnostics → {out_base}")


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS 2 – Average posterior probability curve across trials, per day
# ─────────────────────────────────────────────────────────────────────────────
def _compute_whisker_alignment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add two columns to df:
      - trial_id : cumcount within session, re-zeroed at the first
                           whisker trial (negative = before first whisker trial)
      - whisker_trial_idx: counter that increments only on whisker trials
                           (NaN on non-whisker trials)
    Requires columns: mouse_id, session_id, whisker (1 = whisker trial).
    """
    # Absolute within-session position (0-indexed)
    print('Computing whisker alignment...')
    df = df.copy()
    df["_abs_pos"] = df.groupby(["mouse_id", "session_id"]).cumcount()

    # Position of the first whisker trial per session
    first_wh = (
        df[df["whisker"] == 1]
        .groupby(["mouse_id", "session_id"])["_abs_pos"]
        .min()
        .rename("_first_wh_pos")
        .reset_index()
    )
    df = df.merge(first_wh, on=["mouse_id", "session_id"], how="left")
    # Sessions with no whisker trial at all → offset 0 (they will be excluded anyway)
    df["_first_wh_pos"] = df["_first_wh_pos"].fillna(0).astype(int)

    # Aligned trial index (0 = first whisker trial)
    df["trial_id"] = df["_abs_pos"] - df["_first_wh_pos"]

    # Whisker-trial-only counter within session
    df["whisker_trial_idx"] = np.where(
        df["whisker"] == 1,
        df[df["whisker"] == 1].groupby(["mouse_id", "session_id"]).cumcount(),
        np.nan,
    )

    return df.drop(columns=["_abs_pos", "_first_wh_pos"])


def _draw_posterior_panel(ax, sub_df, post_cols, state_colors,
                          x_col: str, max_trials: int, n_states: int):
    """
    Compute two-step (within-mouse then across-mice) mean ± SEM for every
    state posterior and draw overlaid lines + shaded bands on `ax`.

    x_col     : column to use as x-axis (trial_id or whisker_trial_idx)
    max_trials: upper bound on x_col (lower bound is always 0)
    """
    sub_df = sub_df[(sub_df[x_col] >= 0) & (sub_df[x_col] < max_trials)].copy()
    sub_df[x_col] = sub_df[x_col].astype(int)

    if sub_df.empty:
        return 0

    # Step 1: within-mouse average across sessions
    mouse_avg = (
        sub_df.groupby(["mouse_id", x_col])[post_cols]
        .mean()
        .reset_index()
    )
    # Step 2: across-mice mean ± SEM
    grp   = mouse_avg.groupby(x_col)[post_cols]
    means = grp.mean()
    sems  = grp.sem(ddof=1).fillna(0)
    trial_idx = means.index.values

    ax.axhline(1 / n_states, color="grey", lw=0.6, ls="--", label="uniform prior")
    for s_i, pcol in enumerate(post_cols):
        color = state_colors[s_i]
        m = means[pcol].values
        e = sems[pcol].values
        ax.plot(trial_idx, m, color=color, lw=1.8, label=f"State {s_i + 1}")
        ax.fill_between(trial_idx, m - e, m + e, color=color, alpha=0.15)

    ax.set_xlim(0, max_trials - 1)
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("p(state)", fontsize=9)
    ax.legend(frameon=False, fontsize=7, loc="upper right")

    return mouse_avg["mouse_id"].nunique()

def _draw_posterior_panel_single_mouse(
    ax, sub: pd.DataFrame, post_cols: list, state_colors: list,
    x_col: str, max_trials: int,
) -> int:
    """
    Draw one thin, semi-transparent line per mouse per state.
    Averages first across splits/instances for each (mouse, x_col) position,
    then plots each mouse independently.
    Returns the number of mice drawn.
    """
    mice = sub["mouse_id"].unique()

    for state_idx, col in enumerate(post_cols):
        color = state_colors[state_idx]
        for mouse_id in mice:
            mouse_sub = sub[sub["mouse_id"] == mouse_id]

            # Average across splits and instances for this mouse
            curve = (
                mouse_sub[mouse_sub[x_col] < max_trials]
                .groupby(x_col)[col]
                .mean()
                .sort_index()
            )
            if curve.empty:
                continue

            ax.plot(
                curve.index,
                curve.values,
                color=color,
                linewidth=0.6,
                alpha=0.4,
                rasterized=True,
            )

    # Legend: one entry per state using a thicker representative line
    for state_idx, col in enumerate(post_cols):
        ax.plot([], [], color=state_colors[state_idx], linewidth=1.5,
                label=f"State {state_idx + 1}")

    ax.set_ylabel("P(state)", fontsize=9)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=False, fontsize=7, loc="upper right")

    return len(mice)


def _draw_posterior_panel_interpolated(ax, sub, post_cols, state_colors, n_grid=100):
    """
    Interpolate each session's posteriors to a normalised [0, 1] grid,
    average within-mouse, then plot grand mean ± SEM across mice.

    Grouping key: (mouse_id, session_id, split_idx, instance_idx) so that
    each unique fitting instance × session contributes one curve.

    Returns the number of mice contributing.
    """
    x_grid = np.linspace(0, 1, n_grid)

    # Build the grouping key from whichever columns are present
    group_keys = ["mouse_id", "session_id"]
    for col in ("split_idx", "instance_idx"):
        if col in sub.columns:
            group_keys.append(col)

    mouse_curves: dict[str, list[np.ndarray]] = {}

    for keys, sess_df in sub.groupby(group_keys):
        mouse_id = keys[0]  # first element is always mouse_id
        n_trials = len(sess_df)
        if n_trials < 2:
            continue

        x_orig = np.linspace(0, 1, n_trials)

        # Interpolate every posterior column to the common grid
        interp_mat = np.zeros((n_grid, len(post_cols)))
        for j, col in enumerate(post_cols):
            y = sess_df[col].to_numpy(dtype=float)
            interp_mat[:, j] = np.interp(x_grid, x_orig, y)

        mouse_curves.setdefault(mouse_id, []).append(interp_mat)

    if not mouse_curves:
        return 0

    # Within-mouse average  →  (n_mice, n_grid, n_states)
    mouse_means = np.array(
        [np.mean(curves, axis=0) for curves in mouse_curves.values()]
    )

    grand_mean = mouse_means.mean(axis=0)  # (n_grid, n_states)
    n_mice = len(mouse_means)
    sem = (mouse_means.std(axis=0, ddof=1) / np.sqrt(n_mice)
           if n_mice > 1 else np.zeros_like(grand_mean))

    for j, col in enumerate(post_cols):
        color = state_colors[j]
        ax.plot(x_grid, grand_mean[:, j],
                color=color, lw=1.5, label=f"State {j + 1}")
        ax.fill_between(x_grid,
                        grand_mean[:, j] - sem[:, j],
                        grand_mean[:, j] + sem[:, j],
                        color=color, alpha=0.2)

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.01, 1.01)
    ax.set_ylabel("P(state)", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    return n_mice


def plot_posterior_curves_by_day(cfg, figure_path: Path, trial_df: pd.DataFrame, model_name: str = "full",
                                 max_trials: int = 300):
    """
    For each (reward_group, training day) and each K in n_states_list, plot the
    trial-averaged posterior probability of every state on a single axis, with
    the inter-mouse SEM shown as a shaded band.

    Six figures are saved per (reward_group, day, K):
      1. All trials — mean ± SEM across mice
      2. Whisker trials only — mean ± SEM across mice
      3. All trials — thin single-mouse curves
      4. Whisker trials only — thin single-mouse curves
      5. All trials — interpolated to normalised [0, 1] grid, mean ± SEM
      6. Whisker trials only — interpolated to normalised [0, 1] grid, mean ± SEM
    """
    from plotting_utils import remove_top_right_frame, save_figure_to_files

    out_base = figure_path / "posterior_curves" / model_name

    if trial_df.empty:
        logger.warning("  plot_posterior_curves: trial_df is empty, skipping.")
        return

    # Filter to requested model
    trial_df = trial_df[trial_df["model_name"] == model_name].copy()
    if trial_df.empty:
        logger.warning(f"  plot_posterior_curves: no data for model_name='{model_name}'.")
        return

    for n_states in sorted(cfg["n_states_list"]):

        # Filter to current K and derive posterior columns from that subset only
        sub_k = trial_df[trial_df["n_states"] == n_states]
        if sub_k.empty:
            print(f" No trial data for {n_states}-states models")
            continue

        # Build expected column names directly from n_states
        post_cols = [f"posterior_state_{s + 1}" for s in range(n_states)]

        # Guard: verify they actually exist and are non-null for this K
        missing = [c for c in post_cols if c not in sub_k.columns]
        if missing:
            logger.warning(
                f"  Missing posterior columns for K={n_states}: {missing} — skipping."
            )
            continue

        state_colors = sns.color_palette("tab10", n_states)
        state_colors = state_index_cmap

        rgs = sorted(sub_k["reward_group"].unique())
        days = sorted(sub_k["day"].unique()) if "day" in sub_k.columns else [None]

        for rg in rgs:
            for day in days:

                sub = sub_k[sub_k["reward_group"] == rg]

                if day is not None:
                    sub = sub[sub["day"] == day]
                if sub.empty:
                    continue

                day_str = f"day{day}" if day is not None else "alldays"
                out_dir = out_base / rg / day_str
                out_dir.mkdir(parents=True, exist_ok=True)

                # ── Figure 1: all trials, mean ± SEM ─────────────────────────
                fig1, ax1 = plt.subplots(figsize=(7, 3.5), dpi=200,
                                         constrained_layout=True)
                n_mice = _draw_posterior_panel(
                    ax1, sub, post_cols, state_colors,
                    x_col="trial_id",
                    max_trials=max_trials,
                    n_states=n_states,
                )
                ax1.set_xlabel("Trial in session", fontsize=9)
                remove_top_right_frame(ax1)
                fig1.suptitle(
                    f"Posterior | {model_name} K={n_states} | {rg} | {day_str}"
                    f"  (n={n_mice} mice, all trials)",
                    fontsize=9,
                )
                save_figure_to_files(
                    fig=fig1, save_path=str(out_dir),
                    file_name=f"posterior_curve_K{n_states}_all_trials",
                    suffix=None, file_types=["pdf", "png"], dpi=200,
                )
                plt.close(fig1)

                # ── Figure 3: all trials, single-mouse curves ─────────────────
                fig3, ax3 = plt.subplots(figsize=(7, 3.5), dpi=200,
                                         constrained_layout=True)
                n_mice_sm = _draw_posterior_panel_single_mouse(
                    ax3, sub, post_cols, state_colors,
                    x_col="trial_id",
                    max_trials=max_trials,
                )
                ax3.set_xlabel("Trial in session", fontsize=9)
                remove_top_right_frame(ax3)
                fig3.suptitle(
                    f"Posterior (per mouse) | {model_name} K={n_states} | {rg} | {day_str}"
                    f"  (n={n_mice_sm} mice, all trials)",
                    fontsize=9,
                )
                save_figure_to_files(
                    fig=fig3, save_path=str(out_dir),
                    file_name=f"posterior_curve_K{n_states}_all_trials_per_mouse",
                    suffix=None, file_types=["pdf", "png"], dpi=200,
                )
                plt.close(fig3)

                # ── Figure 5: all trials, interpolated mean ± SEM ────────────
                fig5, ax5 = plt.subplots(figsize=(7, 3.5), dpi=200,
                                         constrained_layout=True)
                n_mice_interp = _draw_posterior_panel_interpolated(
                    ax5, sub, post_cols, state_colors, n_grid=100,
                )
                ax5.set_xlabel(
                    "Normalised trial position  (0 = first trial, 1 = last)",
                    fontsize=9,
                )
                remove_top_right_frame(ax5)
                fig5.suptitle(
                    f"Posterior (interpolated) | {model_name} K={n_states} | {rg} | {day_str}"
                    f"  (n={n_mice_interp} mice, all trials)",
                    fontsize=9,
                )
                save_figure_to_files(
                    fig=fig5, save_path=str(out_dir),
                    file_name=f"posterior_curve_K{n_states}_all_trials_interp",
                    suffix=None, file_types=["pdf", "png"], dpi=200,
                )
                plt.close(fig5)

                # ── Whisker subset (Figures 2, 4, 6) ─────────────────────────
                sub_wh = sub[sub["whisker"] == 1].copy()
                if sub_wh.empty:
                    continue

                sub_wh["whisker_trial_idx"] = (
                    sub_wh.groupby(["mouse_id", "session_id", "split_idx", "instance_idx"])
                    .cumcount()
                )

                # ── Figure 2: whisker trials only, mean ± SEM ────────────────
                fig2, ax2 = plt.subplots(figsize=(7, 3.5), dpi=200,
                                         constrained_layout=True)
                n_mice_wh = _draw_posterior_panel(
                    ax2, sub_wh, post_cols, state_colors,
                    x_col="whisker_trial_idx",
                    max_trials=max_trials,
                    n_states=n_states,
                )
                ax2.set_xlabel("Whisker trial in session", fontsize=9)
                remove_top_right_frame(ax2)
                fig2.suptitle(
                    f"Posterior (whisker trials) | {model_name} K={n_states} | {rg} | {day_str}"
                    f"  (n={n_mice_wh} mice)",
                    fontsize=9,
                )
                save_figure_to_files(
                    fig=fig2, save_path=str(out_dir),
                    file_name=f"posterior_curve_K{n_states}_whisker_trials_first{max_trials}",
                    suffix=None, file_types=["pdf", "png"], dpi=200,
                )
                plt.close(fig2)

                # ── Figure 4: whisker trials only, single-mouse curves ────────
                fig4, ax4 = plt.subplots(figsize=(7, 3.5), dpi=200,
                                         constrained_layout=True)
                n_mice_wh_sm = _draw_posterior_panel_single_mouse(
                    ax4, sub_wh, post_cols, state_colors,
                    x_col="whisker_trial_idx",
                    max_trials=max_trials,
                )
                ax4.set_xlabel("Whisker trial in session", fontsize=9)
                remove_top_right_frame(ax4)
                fig4.suptitle(
                    f"Posterior (per mouse, whisker trials) | {model_name} K={n_states} | {rg} | {day_str}"
                    f"  (n={n_mice_wh_sm} mice)",
                    fontsize=9,
                )
                save_figure_to_files(
                    fig=fig4, save_path=str(out_dir),
                    file_name=f"posterior_curve_K{n_states}_whisker_trials_per_mouse",
                    suffix=None, file_types=["pdf", "png"], dpi=200,
                )
                plt.close(fig4)

                # ── Figure 6: whisker trials only, interpolated mean ± SEM ───
                fig6, ax6 = plt.subplots(figsize=(7, 3.5), dpi=200,
                                         constrained_layout=True)
                n_mice_wh_interp = _draw_posterior_panel_interpolated(
                    ax6, sub_wh, post_cols, state_colors, n_grid=100,
                )
                ax6.set_xlabel(
                    "Normalised whisker-trial position  (0 = first, 1 = last)",
                    fontsize=9,
                )
                remove_top_right_frame(ax6)
                fig6.suptitle(
                    f"Posterior (interpolated, whisker trials) | {model_name} K={n_states}"
                    f" | {rg} | {day_str}  (n={n_mice_wh_interp} mice)",
                    fontsize=9,
                )
                save_figure_to_files(
                    fig=fig6, save_path=str(out_dir),
                    file_name=f"posterior_curve_K{n_states}_whisker_trials_interp",
                    suffix=None, file_types=["pdf", "png"], dpi=200,
                )
                plt.close(fig6)

    logger.info(f"  [2] Posterior curves by day → {out_base}")


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS 3 – Global model weights per state, hue = reward_group
# ─────────────────────────────────────────────────────────────────────────────

def plot_global_weights_by_rg(cfg, figure_path: Path):
    """
    For each model_type and each K, plot one subplot per state showing mean±SE
    weights averaged across all splits and instances.  Hue = reward_group.

    One figure per (model_name, K).
    """
    from plotting_utils import remove_top_right_frame, save_figure_to_files

    out_base = figure_path / "global_weights_by_rg"

    logger.info("  [3] Loading global weights …")
    wdf = _load_global_weights_long(cfg)
    if wdf.empty:
        logger.warning("  [3] No global weight data found.")
        return

    rg_order  = [rg for rg in ["R+", "R-", "R+proba"] if rg in wdf["reward_group"].unique()]
    rg_palette = [_RG_COLOR[rg] for rg in rg_order]

    for model_name in wdf["model_name"].unique():
        sub_m = wdf[wdf["model_name"] == model_name]
        feats = list(sub_m["feature"].unique())  # preserve original order
        # restore feature order from config
        feats = [f for f in cfg["features"] if f in feats]

        for n_states in sorted(sub_m["n_states"].unique()):
            sub = sub_m[sub_m["n_states"] == n_states]

            fig, axs = plt.subplots(
                1, n_states,
                figsize=(3.5 * n_states, 3.5),
                dpi=250, constrained_layout=True,
                squeeze=False, sharey=True
            )

            for s_i in range(n_states):
                ax   = axs[0, s_i]
                data = sub[sub["state_idx"] == s_i]
                sns.pointplot(
                    data=data, x="feature", y="weight",
                    order=feats, hue="reward_group",
                    hue_order=rg_order, palette=rg_palette,
                    estimator=np.mean, errorbar="se",
                    ax=ax, dodge=True, legend=(s_i == 0),
                )
                ax.axhline(0, color="grey", lw=0.6, ls="--")
                ax.set_title(f"State {s_i+1}", fontsize=8)
                ax.set_xlabel("")
                ax.set_ylabel("Weight" if s_i == 0 else "", fontsize=8)
                ax.set_xticklabels(feats, rotation=45, ha="right", fontsize=8)
                remove_top_right_frame(ax)
                ax.yaxis.set_tick_params(labelleft=True)

                if s_i == 0 and ax.legend_:
                    ax.legend(frameon=False, fontsize=7, title="Reward group",
                              title_fontsize=7)
                elif ax.legend_:
                    ax.legend_.remove()

            fig.suptitle(f"Global weights | {model_name} | K={n_states}", fontsize=9)

            out_dir = out_base / model_name
            out_dir.mkdir(parents=True, exist_ok=True)
            save_figure_to_files(
                fig=fig, save_path=str(out_dir),
                file_name=f"global_weights_K{n_states}",
                suffix=None, file_types=["pdf", "eps"], dpi=250,
            )
            plt.close()

    logger.info(f"  [3] Global weights by reward group → {out_base}")

def plot_global_weights(cfg, figure_path: Path):
    """
    For each model_name and K, produce two sets of figures:

      (A) columns = states,        hue = reward_group  — stats: Mann-Whitney U between RGs
      (B) columns = reward_groups, hue = state         — stats: Kruskal-Wallis across states

    Data are loaded once and reused for both figure types.
    """
    from plotting_utils import remove_top_right_frame, save_figure_to_files
    from scipy.stats import mannwhitneyu, kruskal

    # ------------------------------------------------------------------
    # 1. Load once
    # ------------------------------------------------------------------
    logger.info("  [3] Loading global weights …")
    wdf = _load_global_weights_long(cfg)
    print('Done loading global weights.')
    if wdf.empty:
        logger.warning("  [3] No global weight data found.")
        return

    rg_order   = [rg for rg in ["R+", "R-",  "R+proba"] if rg in wdf["reward_group"].unique()]
    rg_palette = {rg: _RG_COLOR[rg] for rg in rg_order}

    # ------------------------------------------------------------------
    # 2. Shared annotation helpers
    # ------------------------------------------------------------------
    def _annotate_kruskal(ax, data, feats, groups_col, groups_order):
        y_lo, y_hi = ax.get_ylim()
        y_step = (y_hi - y_lo) * 0.07
        y_step = -0.05
        for f_i, feat in enumerate(feats):
            groups = [
                data.loc[(data["feature"] == feat) & (data[groups_col] == g), "weight"]
                .dropna().values
                for g in groups_order
            ]
            groups = [g for g in groups if len(g) > 0]
            if len(groups) < 2:
                continue
            try:
                _, p = kruskal(*groups)
            except ValueError:
                continue
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            ax.text(f_i, y_hi + y_step, sig, ha="center", va="bottom", fontsize=7)

    def _annotate_mannwhitney(ax, data, feats, rg_order):
        if len(rg_order) < 2:
            return
        rg_a, rg_b = rg_order[0], rg_order[1]
        y_lo, y_hi = ax.get_ylim()
        y_step = (y_hi - y_lo) * 0.07
        y_step = -0.05
        for f_i, feat in enumerate(feats):
            a = data.loc[(data["feature"] == feat) & (data["reward_group"] == rg_a), "weight"].dropna().values
            b = data.loc[(data["feature"] == feat) & (data["reward_group"] == rg_b), "weight"].dropna().values
            if len(a) < 2 or len(b) < 2:
                continue
            try:
                _, p = mannwhitneyu(a, b, alternative="two-sided")
            except ValueError:
                continue
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            ax.text(f_i, y_hi + y_step, sig, ha="center", va="bottom", fontsize=7)

    # ------------------------------------------------------------------
    # 3. Loop over model × K and draw both figure types
    # ------------------------------------------------------------------
    for model_name in wdf["model_name"].unique():
        sub_m = wdf[wdf["model_name"] == model_name]
        feats = [f for f in cfg["features"] if f in sub_m["feature"].unique()]

        for n_states in sorted(sub_m["n_states"].unique()):
            sub = sub_m[sub_m["n_states"] == n_states].copy()

            state_order  = list(range(n_states))
            state_labels = [f"State {s+1}" for s in state_order]
            state_palette = list(plt.cm.tab10.colors[:n_states])
            sub["state_label"] = sub["state_idx"].map(dict(zip(state_order, state_labels)))

            n_hues_a = len(rg_order)
            n_hues_b = n_states

            # ── (A) columns = states │ hue = reward_group ──────────────
            fig_a, axs_a = plt.subplots(
                1, n_states,
                figsize=(3.5 * n_states, 3.5), dpi=250,
                constrained_layout=True, squeeze=False, sharey=True,
            )
            for s_i in range(n_states):
                #print('by reward', 'n states', n_states, 'state id', s_i)
                ax   = axs_a[0, s_i]
                data = sub[sub["state_idx"] == s_i]

                sns.pointplot(
                    data=data, x="feature", y="weight", order=feats,
                    hue="reward_group", hue_order=rg_order,
                    palette=[rg_palette[rg] for rg in rg_order],
                    estimator=np.mean, errorbar="se",
                    ax=ax, dodge=(n_hues_a > 1), legend=(s_i == 0),
                )
                ax.axhline(0, color="grey", lw=0.6, ls="--")
                _annotate_mannwhitney(ax, data, feats, rg_order)
                ax.set_title(f"State {s_i+1}", fontsize=8)
                ax.set_xlabel("")
                ax.set_ylabel("Weight" if s_i == 0 else "", fontsize=8)
                ax.set_xticklabels(feats, rotation=45, ha="right", fontsize=8)
                remove_top_right_frame(ax)
                ax.yaxis.set_tick_params(labelleft=True)
                if s_i == 0 and ax.legend_:
                    ax.legend(frameon=False, fontsize=7, title="Reward group", title_fontsize=7)
                elif ax.legend_:
                    ax.legend_.remove()

            fig_a.suptitle(f"Global weights | {model_name} | K={n_states}", fontsize=9)

            # ── (B) columns = reward_groups │ hue = state ──────────────

            if n_states < 2:
                continue

            fig_b, axs_b = plt.subplots(
                1, len(rg_order),
                figsize=(3.5 * len(rg_order), 4.0), dpi=250,
                constrained_layout=True, squeeze=False, sharey=True,
            )
            for rg_i, rg in enumerate(rg_order):
                #print('by state', 'n states', n_states, 'reward id', rg_i)
                ax   = axs_b[0, rg_i]
                data = sub[sub["reward_group"] == rg]
                sns.pointplot(
                    data=data, x="feature", y="weight", order=feats,
                    hue="state_label", hue_order=state_labels, palette=state_palette,
                    estimator=np.mean, errorbar="se",
                    ax=ax, #dodge=(n_hues_b > 1), legend=(rg_i == 0), #error otherwise
                )
                ax.axhline(0, color="grey", lw=0.6, ls="--")
                _annotate_kruskal(ax, data, feats, "state_idx", state_order)
                ax.set_title(rg, fontsize=9)
                ax.set_xlabel("")
                ax.set_ylabel("Weight" if rg_i == 0 else "", fontsize=8)
                ax.set_xticklabels(feats, rotation=45, ha="right", fontsize=8)
                remove_top_right_frame(ax)
                ax.yaxis.set_tick_params(labelleft=True)
                if rg_i == 0 and ax.legend_:
                    ax.legend(frameon=False, fontsize=7, title="State", title_fontsize=7)
                elif ax.legend_:
                    ax.legend_.remove()

            fig_b.suptitle(f"Global weights | {model_name} | K={n_states}", fontsize=9)

            # ── Save both ───────────────────────────────────────────────
            for suffix, fig, subdir in [
                ("by_rg",    fig_a, "global_weights_by_rg"),
                ("by_state", fig_b, "global_weights_by_state"),
            ]:
                out_dir = figure_path / subdir / model_name
                out_dir.mkdir(parents=True, exist_ok=True)
                save_figure_to_files(
                    fig=fig, save_path=str(out_dir),
                    file_name=f"global_weights_K{n_states}_{suffix}",
                    suffix=None, file_types=["pdf", "eps"], dpi=250,
                )
                plt.close(fig)

    logger.info(f"  [3] Global weights → {figure_path}")


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS 4 – Single-mouse weights per state, hue = reward_group
# ─────────────────────────────────────────────────────────────────────────────

def plot_single_weights_by_rg(cfg, figure_path: Path):
    """
    For each model_type and each K, plot one subplot per state showing the
    within-mouse-averaged weights (mean across splits) with hue = reward_group.

    Each dot is one mouse; the thick line is the group mean±SE.
    Statistical annotations (Mann-Whitney U / Kruskal-Wallis) are drawn above
    each feature comparing reward groups.
    One figure per (model_name, K).
    """
    from plotting_utils import remove_top_right_frame, save_figure_to_files

    out_base = figure_path / "single_weights_by_rg"

    logger.info("  [4] Loading single-mouse weights …")
    wdf = _load_single_weights_long(cfg)
    if wdf.empty:
        logger.warning("  [4] No single-mouse weight data found.")
        return

    # Average within mouse across splits/instances first
    wdf_mouse = (
        wdf.groupby(["mouse_id", "model_name", "reward_group", "n_states", "state_idx", "feature"])
        ["weight"].mean().reset_index()
    )

    rg_order   = [rg for rg in ["R+", "R-", "R+proba"] if rg in wdf_mouse["reward_group"].unique()]
    rg_palette = [_RG_COLOR[rg] for rg in rg_order]

    for model_name in wdf_mouse["model_name"].unique():
        sub_m = wdf_mouse[wdf_mouse["model_name"] == model_name]
        feats = [f for f in cfg["features"] if f in sub_m["feature"].unique()]

        for n_states in sorted(sub_m["n_states"].unique()):
            sub = sub_m[sub_m["n_states"] == n_states]

            fig, axs = plt.subplots(
                1, n_states,
                figsize=(3.5 * n_states, 4.0),   # slightly taller to fit annotations
                dpi=250, constrained_layout=True,
                squeeze=False, sharey=False
            )

            for s_i in range(n_states):
                ax   = axs[0, s_i]
                data = sub[sub["state_idx"] == s_i]

                # Individual mouse dots (strip)
                sns.stripplot(
                    data=data, x="feature", y="weight",
                    order=feats, hue="reward_group",
                    hue_order=rg_order, palette=rg_palette,
                    ax=ax, dodge=True, size=3, alpha=0.4, jitter=True,
                    legend=False,
                )
                # Group mean ± SE on top
                sns.pointplot(
                    data=data, x="feature", y="weight",
                    order=feats, hue="reward_group",
                    hue_order=rg_order, palette=rg_palette,
                    estimator=np.mean, errorbar="se",
                    ax=ax, dodge=True, markers="o",
                    markersize=3, lw=1.2,
                    legend=(s_i == 0),
                )

                ax.axhline(0, color="grey", lw=0.6, ls="--")

                # Statistical annotations — must come after both plots so
                # ax.get_ylim() already reflects the full data range
                _annotate_feature_stats(ax, data, feats, rg_order)

                ax.set_title(f"State {s_i+1}", fontsize=8)
                ax.set_xlabel("")
                ax.set_ylabel("Weight" if s_i == 0 else "", fontsize=8)
                ax.set_xticklabels(feats, rotation=45, ha="right", fontsize=6)
                remove_top_right_frame(ax)
                ax.yaxis.set_tick_params(labelleft=True)
                if s_i == 0 and ax.legend_:
                    ax.legend(frameon=False, fontsize=7, title_fontsize=7)
                elif ax.legend_:
                    ax.legend_.remove()

            fig.suptitle(
                f"Single-mouse weights | {model_name} | K={n_states}", fontsize=9
            )

            out_dir = out_base / model_name
            out_dir.mkdir(parents=True, exist_ok=True)
            save_figure_to_files(
                fig=fig, save_path=str(out_dir),
                file_name=f"single_weights_K{n_states}",
                suffix=None, file_types=["pdf", "eps"], dpi=250,
            )
            plt.close()

    logger.info(f"  [4] Single-mouse weights by reward group → {out_base}")

def plot_single_weights_by_state(cfg, figure_path: Path):
    """
    For each model_type and each K, plot one subplot per reward group showing
    within-mouse-averaged weights (mean across splits) with hue = state.

    Each dot is one mouse; the thick line is the state mean±SE.
    Statistical annotations (Kruskal-Wallis across states) are drawn above
    each feature within each reward group.
    One figure per (model_name, K).
    """
    from plotting_utils import remove_top_right_frame, save_figure_to_files
    from scipy.stats import kruskal

    out_base = figure_path / "single_weights_by_state"

    logger.info("  [4] Loading single-mouse weights …")
    wdf = _load_single_weights_long(cfg)
    if wdf.empty:
        logger.warning("  [4] No single-mouse weight data found.")
        return

    # Average within mouse across splits/instances first
    wdf_mouse = (
        wdf.groupby(["mouse_id", "model_name", "reward_group", "n_states", "state_idx", "feature"])
        ["weight"].mean().reset_index()
    )

    rg_order = [rg for rg in ["R+", "R-", "R+proba"] if rg in wdf_mouse["reward_group"].unique()]

    for model_name in wdf_mouse["model_name"].unique():
        sub_m = wdf_mouse[wdf_mouse["model_name"] == model_name]
        feats = [f for f in cfg["features"] if f in sub_m["feature"].unique()]

        for n_states in sorted(sub_m["n_states"].unique()):
            sub = sub_m[sub_m["n_states"] == n_states]

            # Build a state palette (one color per state)
            state_palette = plt.cm.tab10.colors[:n_states]
            state_order = list(range(n_states))
            state_labels = [f"State {s+1}" for s in state_order]

            n_rg = len(rg_order)
            fig, axs = plt.subplots(
                1, n_rg,
                figsize=(3.5 * n_rg, 4.0),
                dpi=250, constrained_layout=True,
                squeeze=False, sharey=False,
            )

            for rg_i, rg in enumerate(rg_order):
                ax   = axs[0, rg_i]
                data = sub[sub["reward_group"] == rg].copy()
                data["state_label"] = data["state_idx"].map(
                    {s: f"State {s+1}" for s in state_order}
                )

                # Individual mouse dots per state
                sns.stripplot(
                    data=data, x="feature", y="weight",
                    order=feats, hue="state_label",
                    hue_order=state_labels,
                    palette=state_palette,
                    ax=ax, dodge=True, size=3, alpha=0.4, jitter=True,
                    legend=False,
                )
                # State mean ± SE on top
                sns.pointplot(
                    data=data, x="feature", y="weight",
                    order=feats, hue="state_label",
                    hue_order=state_labels,
                    palette=state_palette,
                    estimator=np.mean, errorbar="se",
                    ax=ax, dodge=True, markers="o",
                    markersize=3, lw=1.2,
                    legend=(rg_i == 0),
                )

                ax.axhline(0, color="grey", lw=0.6, ls="--")

                # --- Statistical annotations: Kruskal-Wallis across states ---
                y_max = ax.get_ylim()[1]
                y_step = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.07

                for f_i, feat in enumerate(feats):
                    groups = [
                        data.loc[
                            (data["feature"] == feat) & (data["state_idx"] == s),
                            "weight"
                        ].dropna().values
                        for s in state_order
                    ]
                    # Need at least 2 groups with data
                    groups = [g for g in groups if len(g) > 0]
                    if len(groups) < 2:
                        continue

                    try:
                        stat, p = kruskal(*groups)
                    except ValueError:
                        continue

                    if p < 0.001:
                        sig_str = "***"
                    elif p < 0.01:
                        sig_str = "**"
                    elif p < 0.05:
                        sig_str = "*"
                    else:
                        sig_str = "ns"

                    y_ann = y_max + y_step
                    ax.text(
                        f_i, y_ann, sig_str,
                        ha="center", va="bottom",
                        fontsize=7, color="black",
                    )

                ax.set_title(rg, fontsize=9, fontweight="bold")
                ax.set_xlabel("")
                ax.set_ylabel("Weight" if rg_i == 0 else "", fontsize=8)
                ax.set_xticklabels(feats, rotation=45, ha="right", fontsize=6)
                remove_top_right_frame(ax)
                ax.yaxis.set_tick_params(labelleft=True)

                if rg_i == 0 and ax.legend_:
                    ax.legend(frameon=False, fontsize=7, title="State",
                              title_fontsize=7)
                elif ax.legend_:
                    ax.legend_.remove()

            fig.suptitle(
                f"Single-mouse weights | {model_name} | K={n_states}", fontsize=9
            )

            out_dir = out_base / model_name
            out_dir.mkdir(parents=True, exist_ok=True)
            save_figure_to_files(
                fig=fig, save_path=str(out_dir),
                file_name=f"single_weights_K{n_states}",
                suffix=None, file_types=["pdf", "eps"], dpi=250,
            )
            plt.close()

    logger.info(f"  [4] Single-mouse weights by state → {out_base}")

def plot_single_weights(cfg, figure_path: Path, wdf: pd.DataFrame):
    """
    For each model_name and K, produce two sets of figures:

      (A) columns = states,  hue = reward_group  — stats: Mann-Whitney U between RGs
      (B) columns = reward_groups, hue = state    — stats: Kruskal-Wallis across states

    Data are loaded and averaged across splits/instances once, then reused for
    both figure types.
    """
    from plotting_utils import remove_top_right_frame, save_figure_to_files
    from scipy.stats import mannwhitneyu, kruskal

    # ------------------------------------------------------------------
    # 1. Load & average once
    # ------------------------------------------------------------------

    if wdf.empty:
        logger.warning("  [4] No single-mouse weight data found.")
        return

    logger.info("  [4] Loading single-mouse weights …")
    #wdf = _load_single_weights_long_viterbi(cfg)

    #wdf = wdf[wdf.mouse_id.isin(['AB118','AB119'])]

    if wdf.empty:
        logger.warning("  [4] No single-mouse weight data found.")
        return

    wdf_mouse = (
        wdf.groupby(["mouse_id", "model_name", "reward_group", "n_states", "state_idx", "feature"])
        ["weight"].mean().reset_index()
    )

    rg_order   = [rg for rg in ["R+", "R-", "R+proba"] if rg in wdf_mouse["reward_group"].unique()]
    rg_palette = {rg: _RG_COLOR[rg] for rg in rg_order}

    # ------------------------------------------------------------------
    # 2. Shared annotation helpers
    # ------------------------------------------------------------------
    def _annotate_kruskal(ax, data, feats, groups_col, groups_order):
        """Kruskal-Wallis across `groups_order` levels, one bracket per feature."""
        y_lo, y_hi = ax.get_ylim()
        y_step = (y_hi - y_lo) * 0.02
        y_step = -0.05
        for f_i, feat in enumerate(feats):
            groups = [
                data.loc[(data["feature"] == feat) & (data[groups_col] == g), "weight"]
                .dropna().values
                for g in groups_order
            ]
            groups = [g for g in groups if len(g) > 0]
            if len(groups) < 2:
                continue
            try:
                _, p = kruskal(*groups)
            except ValueError:
                continue
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            ax.text(f_i, y_hi + y_step, sig, ha="center", va="bottom", fontsize=7)

    def _annotate_mannwhitney(ax, data, feats, rg_order):
        """Mann-Whitney U between the first two reward groups, one bracket per feature."""
        if len(rg_order) < 2:
            return
        rg_a, rg_b = rg_order[0], rg_order[1]
        y_lo, y_hi = ax.get_ylim()
        y_step = (y_hi - y_lo) * 0.07
        y_step = -0.05
        for f_i, feat in enumerate(feats):
            a = data.loc[(data["feature"] == feat) & (data["reward_group"] == rg_a), "weight"].dropna().values
            b = data.loc[(data["feature"] == feat) & (data["reward_group"] == rg_b), "weight"].dropna().values
            if len(a) < 2 or len(b) < 2:
                continue
            try:
                _, p = mannwhitneyu(a, b, alternative="two-sided")
            except ValueError:
                continue
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            ax.text(f_i, y_hi + y_step, sig, ha="center", va="bottom", fontsize=7)

    # ------------------------------------------------------------------
    # 3. Loop over model × K and draw both figure types
    # ------------------------------------------------------------------
    for model_name in wdf_mouse["model_name"].unique():
        sub_m = wdf_mouse[wdf_mouse["model_name"] == model_name]
        feats = [f for f in cfg["features"] if f in sub_m["feature"].unique()]

        for n_states in sorted(sub_m["n_states"].unique()):
            sub = sub_m[sub_m["n_states"] == n_states].copy()

            state_palette = list(plt.cm.tab10.colors[:n_states])
            state_palette = {f'State {s+1}':c for s,c in state_index_cmap.items()}
            state_order   = list(range(n_states))
            state_labels  = [f"State {s+1}" for s in state_order]
            sub["state_label"] = sub["state_idx"].map(dict(zip(state_order, state_labels)))

            # ── (A) columns = states │ hue = reward_group ──────────────
            fig_a, axs_a = plt.subplots(
                1, n_states,
                figsize=(3.5 * n_states, 4.0), dpi=250, squeeze=False, sharey=False)
            for s_i in range(n_states):
                ax   = axs_a[0, s_i]
                data = sub[sub["state_idx"] == s_i]

                sns.stripplot(
                    data=data, x="feature", y="weight", order=feats,
                    hue="reward_group", hue_order=rg_order,
                    palette=[rg_palette[rg] for rg in rg_order],
                    ax=ax, dodge=True, size=3, alpha=0.4, jitter=True, legend=False,
                )
                sns.pointplot(
                    data=data, x="feature", y="weight", order=feats,
                    hue="reward_group", hue_order=rg_order,
                    palette=[rg_palette[rg] for rg in rg_order],
                    estimator=np.mean, errorbar="sd",
                    ax=ax, dodge=True, markers="o", markersize=3, lw=1.2,
                    legend=(s_i == 0),
                )
                ax.axhline(0, color="grey", lw=0.6, ls="--")
                _annotate_mannwhitney(ax, data, feats, rg_order)
                ax.set_title(f"State {s_i+1}", fontsize=8, pad=-15)
                ax.set_xlabel("")
                ax.set_ylabel("Weight", fontsize=8)
                ax.set_xticklabels(feats, rotation=45, ha="right", fontsize=6)
                remove_top_right_frame(ax)
                ax.legend(frameon=False, fontsize=7, title='')

            fig_a.suptitle(f"Single-mouse weights | {model_name} | K={n_states}", fontsize=9)

            # ── (B) columns = reward_groups │ hue = state ──────────────

            fig_b, axs_b = plt.subplots(
                1, len(rg_order),
                figsize=(3.5 * len(rg_order), 4.0), dpi=250,
                constrained_layout=True, squeeze=False, sharey=False,
            )
            for rg_i, rg in enumerate(rg_order):
                ax   = axs_b[0, rg_i]
                data = sub[sub["reward_group"] == rg]

                sns.stripplot(
                    data=data, x="feature", y="weight", order=feats,
                    hue="state_label", hue_order=state_labels, palette=state_palette,
                    ax=ax, dodge=True, size=3, alpha=0.4, jitter=True, legend=False,
                )
                sns.pointplot(
                    data=data, x="feature", y="weight", order=feats,
                    hue="state_label", hue_order=state_labels, palette=state_palette,
                    estimator=np.mean, errorbar="sd",
                    ax=ax, dodge=True, markers="o", markersize=6, lw=1.2,
                    legend=(rg_i == 0),
                )
                ax.axhline(0, color="grey", lw=0.6, ls="--")
                _annotate_kruskal(ax, data, feats, "state_idx", state_order)
                ax.set_title(rg, fontsize=9, pad=-5)
                ax.set_xlabel("")
                ax.set_ylabel("Weight", fontsize=8)
                ax.set_xticklabels(feats, rotation=45, ha="right", fontsize=6)
                remove_top_right_frame(ax)
                ax.legend(frameon=False, fontsize=7, title_fontsize=7)

            fig_b.suptitle(f"Single-mouse weights | {model_name} | K={n_states}", fontsize=9)

            # ── Save both ───────────────────────────────────────────────
            for suffix, fig, subdir in [
                ("by_rg",    fig_a, "single_weights_by_rg"),
                ("by_state", fig_b, "single_weights_by_state"),
            ]:
                out_dir = figure_path / subdir / model_name
                out_dir.mkdir(parents=True, exist_ok=True)
                save_figure_to_files(
                    fig=fig, save_path=str(out_dir),
                    file_name=f"single_weights_K{n_states}_{suffix}",
                    suffix=None, file_types=["pdf", "eps"], dpi=250,
                )
                plt.close(fig)



    logger.info(f"  [4] Single-mouse weights → {figure_path}")


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS 5 – Lick rate per trial type × state, for each reward group
# ─────────────────────────────────────────────────────────────────────────────

def plot_lick_rate_per_state_old(cfg, figure_path: Path, model_name: str = "full"):
    """
    For each model_name and K, compute the mouse-average lick rate (lick_flag
    mean) broken down by (dominant state, trial type, reward_group).

    Dominant state = argmax of posterior_state_* for each trial.

    Layout: one figure per (model_name, K).
    Rows = reward groups,  cols = states.
    x-axis = trial type,   y-axis = lick rate,  each mouse shown as a thin line.

    A second figure per (model_name, K, day) shows state occupancy:
    fraction of trials assigned to each dominant state, broken down by
    trial type and reward group.
    Rows = reward groups, cols = trial types, x-axis = state, y-axis = fraction.
    """
    from plotting_utils import remove_top_right_frame, save_figure_to_files
    sns.set_style('ticks')

    out_base = figure_path / "lick_rate_per_state"

    tt_order  = ["auditory_trial", "whisker_trial", "no_stim_trial"]
    tt_labels = {"auditory_trial": "Auditory", "whisker_trial": "Whisker",
                 "no_stim_trial": "No stim"}

    feature_sets = build_feature_sets(cfg["features"], cfg.get("trial_types"))
    model_names  = ['full']

    for model_name in model_names:
        for n_states in sorted(cfg["n_states_list"]):

            if n_states < 2:
                continue

            trial_df = _load_single_trial_data_viterbi(cfg, model_name, n_states)

            trial_df["trial_type"] = "no_stim_trial"
            trial_df.loc[trial_df.whisker  == 1, "trial_type"] = "whisker_trial"
            trial_df.loc[trial_df.auditory == 1, "trial_type"] = "auditory_trial"

            lick_col    = "choice"
            rgs_present = ["R+", "R-"]
            base_df     = trial_df[trial_df["reward_group"].isin(rgs_present)]

            state_order = list(range(n_states))

            for day, day_df in base_df.groupby("day"):

                # ── shared per-mouse aggregate ────────────────────────────────
                mouse_means = (
                    day_df
                    .groupby(["reward_group", "dominant_state", "mouse_id", "trial_type"])[lick_col]
                    .mean()
                    .reset_index()
                )
                mouse_means["trial_type"] = pd.Categorical(
                    mouse_means["trial_type"], categories=tt_order, ordered=True
                )

                out_dir = out_base / model_name / f"K{n_states}"
                out_dir.mkdir(parents=True, exist_ok=True)

                # ── Figure 1: lick rate ───────────────────────────────────────
                # Determine if we should use a FacetGrid or single axes
                if cfg.get('trial_types') == 'whisker':
                    tt_order = ['whisker_trial']
                    #TODO: add other trial types, require getting all trials and placing them in trials
                    #TODO: or get by comparing with HMMs, you get false alarm rate...

                    # ── Single axes, states as hue ───────────────────────────────
                    fig, axs = plt.subplots(1,2,figsize=(3.2 * n_states, 3.0 * len(rgs_present)), dpi=250)

                    for rg_i, rg in enumerate(rgs_present):
                        color = _RG_COLOR.get(rg, "steelblue")
                        df_rg = mouse_means[mouse_means["reward_group"] == rg]
                        ax = axs[rg_i]

                        sns.barplot(
                            data=df_rg,
                            x="trial_type",
                            y=lick_col,
                            hue="dominant_state",
                            hue_order=state_order,
                            order=tt_order,
                            ax=ax,
                            errorbar="se",
                            err_kws={"linewidth": 2},
                            edgecolor="none",
                            palette=sns.color_palette("tab10", n_colors=n_states),
                        )

                        ax.set_ylabel("P(lick)", fontsize=8)
                        ax.set_xlabel("Trial type", fontsize=8)
                        ax.set_xticks(range(len(tt_order)))
                        ax.set_xticklabels([tt_labels.get(t, t) for t in tt_order], fontsize=8)
                        ax.set_ylim(0.0, 1.05)
                        remove_top_right_frame(ax)
                        ax.legend(title="State", fontsize=7, frameon=False)

                    fig.suptitle(f"Day {day}", fontsize=9, y=1.02)
                    fig.align_xlabels()
                    fig.align_ylabels()

                    save_figure_to_files(
                        fig=fig,
                        save_path=str(out_dir),
                        file_name=f"lick_rate_K{n_states}_day{day:1d}_whisker",
                        suffix=None,
                        file_types=["pdf", "eps"],
                        dpi=250,
                    )
                    plt.close(fig)

                else:
                    # ── FacetGrid version (original) ─────────────────────────────
                    g = sns.FacetGrid(
                        mouse_means,
                        row="reward_group", col="dominant_state",
                        row_order=rgs_present, col_order=state_order,
                        height=2.5, aspect=1, margin_titles=True,
                        sharex=False, sharey=False
                    )

                    def _draw_panel(data: pd.DataFrame, **kwargs) -> None:
                        ax = plt.gca()
                        rg = data["reward_group"].iloc[0] if not data.empty else None
                        color = _RG_COLOR.get(rg, "steelblue")

                        for _, mdf in data.groupby("mouse_id"):
                            mdf_ord = mdf.set_index("trial_type").reindex(tt_order).dropna()

                        sns.barplot(
                            data=data,
                            x="trial_type", y=lick_col,
                            order=tt_order,
                            color=color, errorbar="se",
                            err_kws={"linewidth": 2},
                            width=0.5, ax=ax,
                            edgecolor="none",
                        )

                    g.map_dataframe(_draw_panel)

                    for ax in g.axes.flat:
                        ax.set_ylabel("P(lick)", fontsize=8)
                        ax.set_xlabel("Trial type", fontsize=8)
                        ax.set_ylim(0.0, 1.05)
                        ax.set_xticklabels(
                            [tt_labels.get(t, t) for t in tt_order],
                            fontsize=8,
                        )
                        remove_top_right_frame(ax)

                    g.set_titles(col_template="State {col_name}", row_template="{row_name}")
                    g.figure.suptitle(f"Day {day}", fontsize=9, y=1.02)
                    g.figure.set_dpi(250)
                    g.figure.set_size_inches(3.2 * n_states, 3.0 * len(rgs_present))
                    g.figure.align_xlabels()
                    g.figure.align_ylabels()

                    save_figure_to_files(
                        fig=g.figure, save_path=str(out_dir),
                        file_name=f"lick_rate_K{n_states}_day{day:1d}",
                        suffix=None, file_types=["pdf", "eps"], dpi=250,
                    )
                    plt.close()

                # ── Figure 2: state occupancy ─────────────────────────────────
                # Fraction of all trials assigned to each dominant state, per mouse.
                occ_counts = (
                    day_df
                    .groupby(["reward_group", "mouse_id", "dominant_state"])
                    .size()
                    .reset_index(name="n_trials")
                )
                occ_totals = (
                    occ_counts
                    .groupby(["mouse_id"])["n_trials"]
                    .transform("sum")
                )
                occ_counts["occupancy"] = occ_counts["n_trials"] / occ_totals
                occ_counts["dominant_state"] = pd.Categorical(
                    occ_counts["dominant_state"], categories=state_order, ordered=True
                )

                fig2, axes2 = plt.subplots(
                    1, len(rgs_present),
                    figsize=(3 * len(rgs_present), 3),
                    dpi=250, constrained_layout=True,
                )
                axes2 = np.atleast_1d(axes2)

                for ax, rg in zip(axes2, rgs_present):
                    data = occ_counts[occ_counts["reward_group"] == rg]
                    color = _RG_COLOR.get(rg, "steelblue")

                    #for _, mdf in data.groupby("mouse_id"):
                    #    mdf_ord = (
                    #        mdf.set_index("dominant_state")
                    #        .reindex(state_order)["occupancy"]
                    #    )
                    #    ax.plot(
                    #        state_order, mdf_ord.values,
                    #        color=color, linewidth=0.6, alpha=0.4,
                    #        marker="o", markersize=2, zorder=3,
                    #    )

                    sns.barplot(
                        data=data,
                        x="dominant_state", y="occupancy",
                        order=state_order,
                        color=color, errorbar="se",
                         err_kws={'linewidth': 1},
                        width=0.5, ax=ax,
                        edgecolor="none", legend=False
                    )
                    ax.set_xticks(range(n_states))
                    ax.set_xticklabels([f"State {s}" for s in state_order], fontsize=7)
                    ax.set_ylabel("Fraction of trials", fontsize=8)
                    ax.set_xlabel("State", fontsize=8)
                    ax.set_ylim(0.0, 1.05)
                    ax.set_title(rg, fontsize=8)
                    remove_top_right_frame(ax)

                fig2.suptitle(
                    f"State occupancy | {model_name} K={n_states} | Day {day}",
                    fontsize=9,
                )
                save_figure_to_files(
                    fig=fig2, save_path=str(out_dir),
                    file_name=f"state_occupancy_K{n_states}_day{day:1d}",
                    suffix=None, file_types=["pdf", "eps"], dpi=250,
                )
                plt.close(fig2)

                # ── Figure 3: balanced accuracy per trial type ────────────────
                # Requires a 'pred' column (predicted choice) in trial_df.
                if "pred" not in day_df.columns:
                    logger.warning(
                        f"  'pred' column missing – skipping Fig 3 "
                        f"(K={n_states}, day={day})"
                    )
                else:
                    # Per-mouse balanced accuracy per trial type
                    records = []
                    for (rg, mouse_id, tt), grp in day_df.groupby(
                            ["reward_group", "mouse_id", "trial_type"]):
                        if grp["choice"].nunique() < 2:
                            # Can't compute balanced accuracy with a single class
                            continue
                        from sklearn.metrics import balanced_accuracy_score
                        ba = balanced_accuracy_score(grp["choice"], grp["pred"])
                        records.append({
                            "reward_group": rg,
                            "mouse_id": mouse_id,
                            "trial_type": tt,
                            "balanced_acc": ba,
                        })

                    if not records:
                        logger.warning(
                            f"  No valid per-trial-type accuracy rows "
                            f"(K={n_states}, day={day}) – skipping Fig 3"
                        )
                    else:
                        perf_df = pd.DataFrame(records)
                        perf_df["trial_type"] = pd.Categorical(
                            perf_df["trial_type"],
                            categories=tt_order, ordered=True,
                        )

                        fig3, axes3 = plt.subplots(
                            1, len(rgs_present),
                            figsize=(2.5 * len(tt_order), 3.2),
                            dpi=250,
                            sharey=False, sharex=False,
                            constrained_layout=True,
                        )
                        axes3 = np.atleast_1d(axes3)

                        for ax, rg in zip(axes3, rgs_present):
                            rg_data = perf_df[perf_df["reward_group"] == rg]
                            color = _RG_COLOR.get(rg, "steelblue")

                            if rg_data.empty:
                                ax.set_visible(False)
                                continue

                            # Bar: mean ± SE across mice
                            sns.barplot(
                                data=rg_data,
                                x="trial_type", y="balanced_acc",
                                order=tt_order,
                                color=color, errorbar="se",
                                err_kws={'linewidth': 2},
                                width=0.5, ax=ax,
                                edgecolor="none",
                            )

                            # Individual mouse dots
                            sns.stripplot(
                                data=rg_data,
                                x="trial_type", y="balanced_acc",
                                order=tt_order,
                                color='dimgrey', edgecolor='none',
                                linewidth=0.6, size=3,
                                jitter=True, ax=ax,
                            )

                            ax.axhline(0.5, color="grey", linestyle="--",
                                       lw=0.8, zorder=0, label="chance")
                            ax.set_ylim(0.0, 1.05)
                            ax.set_title(rg, fontsize=9)
                            ax.set_xlabel("Trial type", fontsize=8)
                            ax.set_ylabel("Balanced accuracy", fontsize=8)
                            ax.set_xticklabels(
                                [tt_labels.get(t, t) for t in tt_order],
                                fontsize=8,
                            )
                            remove_top_right_frame(ax)

                        fig3.suptitle(
                            f"Performance per trial type | "
                            f"{model_name} K={n_states} | Day {day}",
                            fontsize=9,
                        )
                        save_figure_to_files(
                            fig=fig3, save_path=str(out_dir),
                            file_name=f"perf_trialtype_K{n_states}_day{day:1d}",
                            suffix=None, file_types=["pdf", "eps"], dpi=250,
                        )
                        plt.close(fig3)

    logger.info(f"  [5] Lick rate per state → {out_base}")

def plot_lick_rate_per_state(cfg, figure_path: Path, trial_df: pd.DataFrame, model_name: str = "full"): #todo: add weight
    """
    For each model_name and K, compute the mouse-average lick rate broken down
    by (dominant state, trial type, reward_group).

    :param cfg:         pipeline config dict
    :param figure_path: root output directory for figures
    :param trial_df:    long-form trial DataFrame from _load_single_trial_data(cfg, model_name, all_perms)
                        must contain columns: mouse_id, reward_group, n_states, model_name,
                        split_idx, instance_idx, dominant_state, choice, trial_type, day
    :param model_name:  model name to plot (default: 'full')
    """
    from plotting_utils import remove_top_right_frame, save_figure_to_files
    sns.set_style('ticks')
    out_base = figure_path / "lick_rate_per_state"
    tt_order = ["auditory_trial", "whisker_trial", "no_stim_trial"]
    tt_labels = {"auditory_trial": "Auditory", "whisker_trial": "Whisker",
                 "no_stim_trial": "No stim"}

    if trial_df.empty:
        logger.warning("  plot_lick_rate_per_state: trial_df is empty, skipping.")
        return

    # Filter to requested model
    trial_df = trial_df[trial_df["model_name"] == model_name].copy()
    if trial_df.empty:
        logger.warning(f"  plot_lick_rate_per_state: no data for model_name='{model_name}'.")
        return

    # Ensure trial_type column exists
    if "trial_type" not in trial_df.columns:
        trial_df["trial_type"] = "no_stim_trial"
        trial_df.loc[trial_df["whisker"] == 1, "trial_type"] = "whisker_trial"
        trial_df.loc[trial_df["auditory"] == 1, "trial_type"] = "auditory_trial"

    lick_col = "choice"
    rgs_present = ["R+", "R-"]
    base_df = trial_df[trial_df["reward_group"].isin(rgs_present)]

    for n_states in sorted(cfg["n_states_list"]):
        if n_states < 2:
            continue

        sub = base_df[base_df["n_states"] == n_states]
        if sub.empty:
            continue

        state_order = list(range(n_states))
        out_dir = out_base / model_name / f"K{n_states}"
        out_dir.mkdir(parents=True, exist_ok=True)

        for day, day_df in sub.groupby("day"):

            # ── shared per-mouse aggregate ────────────────────────────────────
            mouse_means = (
                day_df
                .groupby(["reward_group", "dominant_state", "mouse_id", "trial_type"])[lick_col]
                .mean()
                .reset_index()
            )
            mouse_means["trial_type"] = pd.Categorical(
                mouse_means["trial_type"], categories=tt_order, ordered=True
            )

            # ── Figure 1: lick rate ───────────────────────────────────────────
            if cfg.get('trial_types') == 'whisker':
                tt_order_plot = ['whisker_trial']
                fig, axs = plt.subplots(
                    1, 2,
                    figsize=(3.0 * n_states, 3.0), dpi=250,
                )
                for rg_i, rg in enumerate(rgs_present):
                    df_rg = mouse_means[mouse_means["reward_group"] == rg]
                    ax = axs[rg_i]
                    sns.barplot(
                        data=df_rg,
                        x="trial_type", y=lick_col,
                        hue="dominant_state", hue_order=state_order,
                        order=tt_order_plot,
                        ax=ax, errorbar="se", err_kws={"linewidth": 2},
                        edgecolor="none", width = 0.5,
                        palette=state_index_cmap,
                    )
                    ax.set_ylabel("P(lick)", fontsize=8)
                    ax.set_xlabel("Trial type", fontsize=8)
                    ax.set_xticks(range(len(tt_order_plot)))
                    ax.set_xticklabels(
                        [tt_labels.get(t, t) for t in tt_order_plot], fontsize=8
                    )
                    ax.set_ylim(0.0, 1.05)
                    remove_top_right_frame(ax)
                    ax.legend(title="State", fontsize=7, frameon=False)
                fig.suptitle(f"Day {day}", fontsize=9, y=1.02)
                fig.align_xlabels()
                fig.align_ylabels()
                save_figure_to_files(
                    fig=fig, save_path=str(out_dir),
                    file_name=f"lick_rate_K{n_states}_day{day:1d}_whisker",
                    suffix=None, file_types=["pdf", "eps"], dpi=250,
                )
                plt.close(fig)

            else:
                g = sns.FacetGrid(
                    mouse_means,
                    row="reward_group", col="dominant_state",
                    row_order=rgs_present, col_order=state_order,
                    height=2.5, aspect=1, margin_titles=True,
                    sharex=False, sharey=True,
                )

                def _draw_panel(data: pd.DataFrame, **kwargs) -> None:
                    ax = plt.gca()
                    rg = data["reward_group"].iloc[0] if not data.empty else None
                    color = _RG_COLOR.get(rg, "steelblue")
                    sns.barplot(
                        data=data, x="trial_type", y=lick_col,
                        order=tt_order, color=color, errorbar="se",
                        err_kws={"linewidth": 2}, width=0.5, ax=ax,
                        edgecolor="none", palette = state_index_cmap,
                    )

                g.map_dataframe(_draw_panel)
                for ax in g.axes.flat:
                    ax.set_ylabel("P(lick)", fontsize=8)
                    ax.set_xlabel("Trial type", fontsize=8)
                    ax.set_ylim(0.0, 1.05)
                    ax.set_xticklabels(
                        [tt_labels.get(t, t) for t in tt_order], fontsize=8
                    )
                    remove_top_right_frame(ax)
                g.set_titles(col_template="State {col_name}", row_template="{row_name}")
                g.figure.suptitle(f"Day {day}", fontsize=9, y=1.02)
                g.figure.set_dpi(250)
                g.figure.set_size_inches(3.2 * n_states, 3.0 * len(rgs_present))
                g.figure.align_xlabels()
                g.figure.align_ylabels()
                save_figure_to_files(
                    fig=g.figure, save_path=str(out_dir),
                    file_name=f"lick_rate_K{n_states}_day{day:1d}",
                    suffix=None, file_types=["pdf", "eps"], dpi=250,
                )
                plt.close()

            # ── Figure 2: state occupancy ─────────────────────────────────────
            occ_counts = (
                day_df
                .groupby(["reward_group", "mouse_id", "dominant_state"])
                .size()
                .reset_index(name="n_trials")
            )
            occ_counts["occupancy"] = (
                    occ_counts["n_trials"]
                    / occ_counts.groupby("mouse_id")["n_trials"].transform("sum")
            )
            occ_counts["dominant_state"] = pd.Categorical(
                occ_counts["dominant_state"], categories=state_order, ordered=True
            )
            fig2, axes2 = plt.subplots(
                1, len(rgs_present),
                figsize=(3 * len(rgs_present), 3),
                dpi=250, constrained_layout=True,
            )
            axes2 = np.atleast_1d(axes2)
            for ax, rg in zip(axes2, rgs_present):
                data = occ_counts[occ_counts["reward_group"] == rg]
                color = _RG_COLOR.get(rg, "steelblue")
                sns.barplot(
                    data=data, x="dominant_state", y="occupancy",
                    hue='dominant_state', hue_order=state_order,
                    order=state_order,  errorbar="se",
                    err_kws={"linewidth": 2}, width=0.5, ax=ax,
                    edgecolor="none", palette= state_index_cmap,
                )
                ax.set_xticks(range(n_states))
                ax.set_xticklabels([f"State {s}" for s in state_order], fontsize=7)
                ax.set_ylabel("Fraction of trials", fontsize=8)
                ax.set_xlabel("State", fontsize=8)
                ax.set_ylim(0.0, 1.05)
                ax.set_title(rg, fontsize=8)
                fig2.set_dpi(250)
                fig2.set_size_inches(3.2 * n_states, 3.0 * len(rgs_present))
                fig2.align_xlabels()
                fig2.align_ylabels()
                remove_top_right_frame(ax)
            fig2.suptitle(
                f"State occupancy | {model_name} K={n_states} | Day {day}", fontsize=9
            )
            save_figure_to_files(
                fig=fig2, save_path=str(out_dir),
                file_name=f"state_occupancy_K{n_states}_day{day:1d}",
                suffix=None, file_types=["pdf", "eps"], dpi=250,
            )
            plt.close(fig2)

            # ── Figure 3: balanced accuracy per trial type ────────────────────
            if "pred" not in day_df.columns:
                logger.warning(
                    f"  'pred' column missing – skipping Fig 3 (K={n_states}, day={day})"
                )
            else:
                records = []
                for (rg, mouse_id, tt), grp in day_df.groupby(
                        ["reward_group", "mouse_id", "trial_type"]):
                    if grp["choice"].nunique() < 2:
                        continue
                    from sklearn.metrics import balanced_accuracy_score
                    ba = balanced_accuracy_score(grp["choice"], grp["pred"])
                    records.append({
                        "reward_group": rg, "mouse_id": mouse_id,
                        "trial_type": tt, "balanced_acc": ba,
                    })
                if not records:
                    logger.warning(
                        f"  No valid per-trial-type accuracy rows "
                        f"(K={n_states}, day={day}) – skipping Fig 3"
                    )
                else:
                    perf_df = pd.DataFrame(records)
                    perf_df["trial_type"] = pd.Categorical(
                        perf_df["trial_type"], categories=tt_order, ordered=True
                    )
                    fig3, axes3 = plt.subplots(
                        1, len(rgs_present),
                        figsize=(2.5 * len(tt_order), 3.2),
                        dpi=250, sharey=False, sharex=False,
                        constrained_layout=True,
                    )
                    axes3 = np.atleast_1d(axes3)
                    for ax, rg in zip(axes3, rgs_present):
                        rg_data = perf_df[perf_df["reward_group"] == rg]
                        color = _RG_COLOR.get(rg, "steelblue")
                        if rg_data.empty:
                            ax.set_visible(False)
                            continue
                        sns.barplot(
                            data=rg_data, x="trial_type", y="balanced_acc",
                            order=tt_order, color=color, errorbar="se",
                            err_kws={"linewidth": 2}, width=0.5, ax=ax,
                            edgecolor="none",
                        )
                        sns.stripplot(
                            data=rg_data, x="trial_type", y="balanced_acc",
                            order=tt_order, color="dimgrey", edgecolor="none",
                            linewidth=0.6, size=3, jitter=True, ax=ax,
                        )
                        ax.axhline(0.5, color="grey", linestyle="--",
                                   lw=0.8, zorder=0, label="chance")
                        ax.set_ylim(0.0, 1.05)
                        ax.set_title(rg, fontsize=9)
                        ax.set_xlabel("Trial type", fontsize=8)
                        ax.set_ylabel("Balanced accuracy", fontsize=8)
                        ax.set_xticklabels(
                            [tt_labels.get(t, t) for t in tt_order], fontsize=8
                        )
                        remove_top_right_frame(ax)
                    fig3.suptitle(
                        f"Performance per trial type | {model_name} K={n_states} | Day {day}",
                        fontsize=9,
                    )
                    save_figure_to_files(
                        fig=fig3, save_path=str(out_dir),
                        file_name=f"perf_trialtype_K{n_states}_day{day:1d}",
                        suffix=None, file_types=["pdf", "eps"], dpi=250,
                    )
                    plt.close(fig3)

    logger.info(f"  [5] Lick rate per state → {out_base}")

def plot_lick_rate_around_transitions(cfg, figure_path: Path, trial_df: pd.DataFrame,
                                      model_name: str = "full"):
    """
    For each K, plot trial-by-trial P(lick) in a ±TRANSITION_WINDOW trial window
    around every directed state transition (e.g. 0→1, 1→0).

    Produces two kinds of figures per (K, day):
      - "all":   average over every transition occurrence  (original figure)
      - "rank_N": average over only the N-th occurrence of each transition
                  within a session (1-indexed). One figure per rank up to the
                  maximum rank observed in the data.

    Rows = reward_group, columns = transition type.
    Each panel shows mean ± SE across mice (per-mouse average computed first).
    Trial 0 is the first trial of the new state.
    Transitions too close to session edges are discarded.
    """
    from plotting_utils import remove_top_right_frame, save_figure_to_files

    TRANSITION_WINDOW = 10
    sns.set_style("ticks")
    out_base    = figure_path / "lick_rate_per_transition"
    rgs_present = ["R+", "R-"]
    x_ticks     = list(range(-TRANSITION_WINDOW, TRANSITION_WINDOW + 1))

    if trial_df.empty:
        logger.warning("  plot_lick_rate_around_transitions: trial_df is empty, skipping.")
        return

    trial_df = trial_df[trial_df["model_name"] == model_name].copy()
    if trial_df.empty:
        logger.warning(
            f"  plot_lick_rate_around_transitions: no data for model_name='{model_name}'."
        )
        return

    base_df    = trial_df[trial_df["reward_group"].isin(rgs_present)]
    group_cols = ["mouse_id", "session_id", "split_idx", "instance_idx"]
    group_cols = [c for c in group_cols if c in base_df.columns]

    # ------------------------------------------------------------------
    # Helper: extract ±WINDOW choice traces around a directed transition.
    # Returns (windows, ranks) where both have length n_valid_transitions:
    #   windows : list of 1-D float arrays, each of length 2*window+1
    #   ranks   : list of ints (0-indexed occurrence count within this group)
    # ------------------------------------------------------------------
    def _extract_transition_windows(group_df, from_state, to_state, window):
        states  = group_df["dominant_state"].values
        choices = group_df["choice"].values
        n       = len(states)

        windows = []
        ranks   = []
        rank    = 0
        for i in range(1, n):
            if states[i - 1] == from_state and states[i] == to_state:
                lo, hi = i - window, i + window + 1
                if 0 <= lo and hi <= n:
                    windows.append(choices[lo:hi].astype(float))
                    ranks.append(rank)
                rank += 1          # count every detected transition, even clipped ones

        return windows, ranks

    # ------------------------------------------------------------------
    # Shared drawing routine — draws into an already-created axs grid.
    # `windows_by_key[(rg, s_from, s_to)]` = (mat, mouse_tags) after
    # optional rank filtering.
    # ------------------------------------------------------------------
    def _draw_figure(axs, rgs_present, transition_pairs, windows_by_key, x_axis):
        for rg_i, rg in enumerate(rgs_present):
            color = _RG_COLOR.get(rg, "steelblue")

            for t_i, (s_from, s_to) in enumerate(transition_pairs):
                ax = axs[rg_i, t_i]
                ax.axvline(0,   color="grey",      lw=0.8, ls="--", zorder=0)
                #ax.axhline(0.5, color="lightgrey", lw=0.6, ls=":",  zorder=0)
                ax.axvspan(-TRANSITION_WINDOW - 0.5, -0.5,
                           alpha=0.2, color=state_index_cmap[s_from], zorder=0)
                ax.axvspan(-0.5, TRANSITION_WINDOW + 0.5,
                           alpha=0.2, color=state_index_cmap[s_to],   zorder=0)

                key  = (rg, s_from, s_to)
                data = windows_by_key.get(key)

                if data is None or len(data[0]) == 0:
                    ax.set_title(f"State {s_from}→{s_to}\n(no events)", fontsize=7)
                    remove_top_right_frame(ax)
                    ax.set_xticks(x_ticks)
                    ax.set_xticklabels(x_ticks, fontsize=5, rotation=90)
                    continue

                mat, mouse_tags = data
                mat         = np.vstack(mat)
                mouse_tags  = np.array(mouse_tags)
                unique_mice = np.unique(mouse_tags)

                per_mouse_mean = np.array([
                    mat[mouse_tags == m].mean(axis=0) for m in unique_mice
                ])
                grand_mean = per_mouse_mean.mean(axis=0)
                grand_se   = per_mouse_mean.std(axis=0) / np.sqrt(len(unique_mice))

                ax.fill_between(x_axis, grand_mean - grand_se, grand_mean + grand_se,
                                alpha=0.25, color=color)
                ax.plot(x_axis, grand_mean, color=color, lw=1.5)

                ax.set_title(
                    f"State {s_from} → State {s_to}\n"
                    f"n={len(mat)} events, {len(unique_mice)} mice",
                    fontsize=7,
                )
                ax.set_xlim(-TRANSITION_WINDOW - 0.5, TRANSITION_WINDOW + 0.5)
                ax.set_ylim(0.0, 1.05)
                ax.set_xlabel("Trial relative to transition", fontsize=7)
                ax.set_xticks(x_ticks)
                ax.set_xticklabels(x_ticks, fontsize=5, rotation=90)
                ax.tick_params(labelsize=6)
                remove_top_right_frame(ax)

            axs[rg_i, 0].set_ylabel(f"{rg}\nP(lick)", fontsize=8)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    x_axis = np.arange(-TRANSITION_WINDOW, TRANSITION_WINDOW + 1)

    for n_states in sorted(cfg["n_states_list"]):
        if n_states < 2:
            continue

        sub = base_df[base_df["n_states"] == n_states]
        if sub.empty:
            continue

        state_order      = list(range(n_states))
        transition_pairs = [
            (s_from, s_to)
            for s_from in state_order
            for s_to   in state_order
            if s_from != s_to
        ]
        n_trans = len(transition_pairs)

        out_dir = out_base / model_name / f"K{n_states}"
        out_dir.mkdir(parents=True, exist_ok=True)

        for day, day_df in sub.groupby("day"):

            # --------------------------------------------------------------
            # Collect all windows once, keyed by (rg, s_from, s_to).
            # Also store per-event rank so we can filter later.
            # --------------------------------------------------------------
            # raw_data[(rg, s_from, s_to)] = list of (window_array, mouse_id, rank)
            raw_data   = {}
            max_rank   = 0    # track highest rank seen across all keys

            for rg in rgs_present:
                rg_df = day_df[day_df["reward_group"] == rg]
                for s_from, s_to in transition_pairs:
                    key     = (rg, s_from, s_to)
                    entries = []
                    for _, grp in rg_df.groupby(group_cols):
                        if "trial_id" in grp.columns:
                            grp = grp.sort_values("trial_id")
                        windows, ranks = _extract_transition_windows(
                            grp, s_from, s_to, TRANSITION_WINDOW
                        )
                        if windows:
                            mouse_id = grp["mouse_id"].iloc[0]
                            for w, r in zip(windows, ranks):
                                entries.append((w, mouse_id, r))
                                max_rank = max(max_rank, r)
                    raw_data[key] = entries

            # --------------------------------------------------------------
            # Figure A: all transitions pooled
            # --------------------------------------------------------------
            def _build_windows_by_key(rank_filter=None):
                """
                Collapse raw_data into the {key: (list_of_arrays, list_of_mouse_ids)}
                format expected by _draw_figure, optionally keeping only one rank.
                """
                out = {}
                for key, entries in raw_data.items():
                    if rank_filter is not None:
                        entries = [e for e in entries if e[2] == rank_filter]
                    if not entries:
                        out[key] = ([], [])
                        continue
                    arrays    = [e[0] for e in entries]
                    mouse_ids = [e[1] for e in entries]
                    out[key]  = (arrays, mouse_ids)
                return out

            fig_a, axs_a = plt.subplots(
                len(rgs_present), n_trans,
                figsize=(2.8 * n_trans, 2.8 * len(rgs_present)),
                dpi=250, constrained_layout=True,
                squeeze=False, sharey=True,
            )
            _draw_figure(axs_a, rgs_present, transition_pairs,
                         _build_windows_by_key(rank_filter=None), x_axis)
            fig_a.suptitle(
                f"P(lick) around state transitions — all | {model_name} K={n_states} | Day {day}",
                fontsize=9,
            )
            save_figure_to_files(
                fig=fig_a, save_path=str(out_dir),
                file_name=f"lick_transition_K{n_states}_day{day:1d}_all",
                suffix=None, file_types=["pdf", "eps"], dpi=250,
            )
            plt.close(fig_a)

            # --------------------------------------------------------------
            # Figures B1…BN: one per transition rank (0-indexed internally,
            # labelled 1-indexed in titles and filenames)
            # --------------------------------------------------------------
            for rank in range(max_rank + 1):
                fig_r, axs_r = plt.subplots(
                    len(rgs_present), n_trans,
                    figsize=(2.8 * n_trans, 2.8 * len(rgs_present)),
                    dpi=250, constrained_layout=True,
                    squeeze=False, sharey=True,
                )
                _draw_figure(axs_r, rgs_present, transition_pairs,
                             _build_windows_by_key(rank_filter=rank), x_axis)
                ordinal = f"{rank + 1}{'st' if rank == 0 else 'nd' if rank == 1 else 'rd' if rank == 2 else 'th'}"
                fig_r.suptitle(
                    f"P(lick) around state transitions — {ordinal} occurrence | "
                    f"{model_name} K={n_states} | Day {day}",
                    fontsize=9,
                )
                save_figure_to_files(
                    fig=fig_r, save_path=str(out_dir),
                    file_name=f"lick_transition_K{n_states}_day{day:1d}_rank{rank + 1}",
                    suffix=None, file_types=["pdf", "eps"], dpi=250,
                )
                plt.close(fig_r)

    logger.info(f"  plot_lick_rate_around_transitions → {out_base}")


def plot_weight_metric_correlation(cfg, trial_df, weight_df, figure_path,
                                   inflection_df=None):
    """
    For every feature in the model, correlate the per-mouse GLM-HMM weight
    difference (state high - state low) with performance metrics:
      - occupancy in high-performance state
      - whisker hit rate on a given day
      - first low→high transition in the Viterbi state sequence
      - inflection trial (optional, provided externally)

    One figure (1 row × n_metrics columns) is saved per (feature, reward_group).

    Parameters
    ----------
    cfg : dict with keys:
            'features'   : list[str], features to iterate over
            'high_state' : int, index of high-performance state (default 1)
            'low_state'  : int, index of low-performance state  (default 0)
            'day_val'    : int, day index for hit rate           (default 0)
    trial_df      : trial-level DataFrame, pre-filtered to K=2 mice, states
                    permuted; columns: mouse_id, reward_group, day, trial_id,
                    most_likely_state, choice, stimulus_type
    weight_df     : long-format weight DataFrame, same mouse subset;
                    columns: mouse_id, reward_group, feature, state_idx, weight
    figure_path   : str or Path, directory for figure output
    inflection_df : optional DataFrame with columns ['mouse_id', 'inflection_trial']
    """
    from plotting_utils import remove_top_right_frame, save_figure_to_files
    from scipy import stats

    out_base   = figure_path / "correlations"
    high_state = cfg.get('high_state', 1)
    low_state  = cfg.get('low_state',  0)
    day_val    = cfg.get('day_val',    0)
    features   = cfg.get('features',  weight_df['feature'].unique().tolist())

    # ------------------------------------------------------------------
    # 2. Scatter helper
    # ------------------------------------------------------------------
    def _scatter(ax, x, y, xlabel, ylabel, color='steelblue'):
        mask   = x.notna() & y.notna()
        xi, yi = x[mask].values.astype(float), y[mask].values.astype(float)
        if len(xi) < 3:
            ax.set_title('insufficient data', fontsize=8)
            ax.set_xlabel(xlabel, fontsize=9)
            ax.set_ylabel(ylabel, fontsize=9)
            remove_top_right_frame(ax)
            return
        ax.scatter(xi, yi, s=40, color=color, edgecolors='k',
                   linewidths=0.5, alpha=0.85)
        xr = np.linspace(xi.min(), xi.max(), 200)
        ax.plot(xr, np.polyval(np.polyfit(xi, yi, 1), xr), 'k--', lw=1)
        r, p = stats.spearmanr(xi, yi)
        sig  = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(f'r={r:+.2f},  p={p:.3f}  {sig}  n={mask.sum()}', fontsize=8)
        remove_top_right_frame(ax)

    # ------------------------------------------------------------------
    # 3. Iterate over reward groups, then features
    # ------------------------------------------------------------------
    reward_groups = sorted(trial_df['reward_group'].unique())
    rg_colors     = {'R+': 'steelblue', 'R-': 'tomato'}   # extend as needed

    print('=== Spearman correlations ===')

    for rg in reward_groups:

        rg_color   = rg_colors.get(rg, 'steelblue')
        rg_trial   = trial_df[trial_df['reward_group'] == rg]
        rg_weight  = weight_df[weight_df['reward_group'] == rg]
        mouse_ids  = rg_weight['mouse_id'].unique()

        if len(mouse_ids) == 0:
            logger.warning(f"  No mice for reward_group='{rg}', skipping.")
            continue

        # --------------------------------------------------------------
        # 1. Compute performance metrics — once per reward group
        # --------------------------------------------------------------

        # Occupancy in high-performance state
        occ = (rg_trial
               .groupby('mouse_id')['most_likely_state']
               .apply(lambda x: (x == high_state).mean())
               .rename('occupancy')
               .reindex(mouse_ids))

        # Whisker hit rate on day_val (whisker trials only)
        d0     = rg_trial[(rg_trial['day'] == day_val)]
        wh_hit = (d0.groupby('mouse_id')['choice']
                  .mean()
                  .rename('wh_hit')
                  .reindex(mouse_ids))

        # First low→high transition in the Viterbi sequence
        def _first_transition(grp):
            seq = grp.sort_values(['day', 'trial_id'])['most_likely_state'].values
            for i in range(len(seq) - 1):
                if seq[i] == low_state and seq[i + 1] == high_state:
                    return i + 1
            return np.nan

        first_trans = (rg_trial
                       .groupby('mouse_id')
                       .apply(_first_transition)
                       .rename('first_transition')
                       .reindex(mouse_ids))

        # Inflection trial (external)
        if inflection_df is not None:
            infl = (inflection_df
                    .set_index('mouse_id')['inflection_trial']
                    .reindex(mouse_ids))
        else:
            infl = pd.Series(np.nan, index=mouse_ids, name='inflection_trial')

        plot_pairs = [
            (f'State {high_state} occupancy',              occ),
            (f'Wh hit rate (day {day_val})',               wh_hit),
            (f'First {low_state}→{high_state} transition', first_trans),
            ('Inflection trial',                            infl),
        ]
        n_metrics = len(plot_pairs)

        print(f'\n── Reward group: {rg}  (n={len(mouse_ids)} mice) ──')

        # --------------------------------------------------------------
        # 4. Loop over features
        # --------------------------------------------------------------
        for feature in features:

            feat_df = rg_weight[rg_weight['feature'] == feature]
            if feat_df.empty:
                logger.warning(
                    f"  [{rg}] No weight data for feature '{feature}', skipping."
                )
                continue

            # Weight difference: high_state − low_state, mean across splits
            piv = (feat_df
                   .groupby(['mouse_id', 'state_idx'])['weight']
                   .mean()
                   .unstack('state_idx'))

            if high_state not in piv.columns or low_state not in piv.columns:
                logger.warning(
                    f"  [{rg}] States {low_state}/{high_state} not both present "
                    f"for feature '{feature}', skipping."
                )
                continue

            wd = (piv[high_state] - piv[low_state]).rename('weight_diff').reindex(mouse_ids)

            # Console report
            print(f'\n  Feature: {feature}')
            for ylabel, y in plot_pairs:
                mask = wd.notna() & y.notna()
                if mask.sum() < 3:
                    print(f'    {"weight_diff":30s} vs  {ylabel:35s}:  '
                          f'insufficient data (n={mask.sum()})')
                    continue
                r, p = stats.spearmanr(wd[mask].astype(float), y[mask].astype(float))
                print(f'    {"weight_diff":30s} vs  {ylabel:35s}:  '
                      f'r={r:+.3f},  p={p:.3f},  n={mask.sum()}')

            # Figure
            fig, axs = plt.subplots(
                1, n_metrics,
                figsize=(4 * n_metrics, 4), dpi=200,
                constrained_layout=True,
            )
            axs = np.atleast_1d(axs)

            for ax, (ylabel, y) in zip(axs, plot_pairs):
                _scatter(ax, wd, y,
                         xlabel=f'Δw {feature} (state {high_state} − state {low_state})',
                         ylabel=ylabel,
                         color=rg_color)

            fig.suptitle(
                f'{rg}  |  Δw {feature} (state {high_state} − state {low_state})'
                f'  vs. performance metrics',
                fontsize=11,
            )

            rg_out = out_base / rg
            rg_out.mkdir(parents=True, exist_ok=True)
            save_figure_to_files(
                fig, rg_out,
                f'weight_diff_correlations_{feature}',
                file_types=['pdf', 'eps'], dpi=250,
            )
            plt.close(fig)

    logger.info(f"  plot_weight_metric_correlation → {figure_path}")
# ─────────────────────────────────────────────────────────────────────────────
# STAGE  –  FIND PERMUTATIONS
# ─────────────────────────────────────────────────────────────────────────────

def stage_find_permutations_old(cfg):
    """
    Compute and save composed (within-mouse + cross-mouse) state permutations
    for all result files, per (reward_group, model_name, n_states).

    Permutations are saved as a pickle file at:
        cfg["result_path"] / "all_perms.pkl"

    Each permutation is an inv_perm array of shape (K,) such that:
        new_label = inv_perm[old_label]
    applicable to both integer state labels and posterior column indices.

    To reload for plotting:
        all_perms = load_permutations(cfg["result_path"] / "all_perms.pkl")
        inv_perm  = all_perms[(rg, model_name, n_states, mouse_id, si, inst)]

    :param cfg: pipeline config dict with keys:
                  features, trial_types, reward_groups, n_states_list,
                  single_path, result_path, n_splits, n_instances
    """
    feature_sets = build_feature_sets(cfg["features"], cfg.get("trial_types"))
    all_perms = {}

    for rg_int in cfg["reward_groups"]:
        rg = _RG_STR.get(int(rg_int), str(rg_int))

        for model_name in feature_sets:

            if model_name != 'full':
                continue

            feats = feature_sets[model_name]

            for n_states in cfg["n_states_list"]:

                if n_states not in [2, 3]:
                    logger.debug(f"Skipping K={n_states} for permutation finding ")
                    continue

                single_base = cfg["single_path"] / _rg_label(rg_int) / model_name
                if not single_base.exists():
                    continue

                logger.info(f"[{rg} | {model_name} | K={n_states}] "
                            f"Computing permutations...")

                # --------------------------------------------------------------
                # Stage 1: within-mouse alignment
                # For each mouse, align states across splits x instances using
                # Viterbi sequences (falling back to weights if unavailable).
                # --------------------------------------------------------------
                within_inv_perms: dict[str, dict[tuple, np.ndarray]] = {}
                mouse_mean_weights: dict[str, np.ndarray] = {}

                for mouse_dir in sorted(single_base.iterdir()):
                    mouse_id = mouse_dir.name

                    path_map = {
                        (si, inst): single_model_dir(
                            cfg, mouse_id, si, n_states, inst, model_name, rg_int
                        )
                        for si in range(cfg["n_splits"])
                        for inst in range(cfg["n_instances"])
                    }
                    valid_keys = {
                        k for k, p in path_map.items()
                        if (p / "fit_glmhmm_results.npz").exists()
                    }
                    if not valid_keys:
                        continue

                    # -- Viterbi sequences (preferred) -------------------------
                    viterbi_dict, sequences_ok = utils._load_viterbi_dict(
                        path_map, valid_keys, n_states
                    )
                    if sequences_ok and viterbi_dict:
                        viterbi_perms = utils.compute_permutations_from_viterbi(
                            viterbi_dict, n_states
                        )
                        logger.info(f"  [{mouse_id}] Viterbi-based alignment")
                    else:
                        viterbi_perms = None
                        logger.warning(f"  [{mouse_id}] Falling back to weight-based alignment")

                    # -- weight df for within-mouse alignment ------------------
                    weight_df = utils._build_weight_df(
                        path_map, valid_keys, mouse_id, rg, n_states, feats
                    )
                    aligned_weight_df, permut_ids = utils.align_weights_dataframe(
                        weight_df,
                        use_mean_reference=False,
                        permutations=viterbi_perms,
                    )

                    # -- store inv_perm per (si, inst) -------------------------
                    within_inv_perms[mouse_id] = {}
                    for (si, inst) in valid_keys:
                        perm = permut_ids.get((n_states, si, inst), np.arange(n_states))
                        inv = np.empty_like(perm)
                        inv[perm] = np.arange(len(perm))
                        within_inv_perms[mouse_id][(si, inst)] = inv

                    # -- mean weight matrix after within-mouse alignment -------
                    mouse_mean_weights[mouse_id] = utils._mean_weight_matrix(
                        aligned_weight_df, n_states, feats
                    )

                if not mouse_mean_weights:
                    continue

                # --------------------------------------------------------------
                # Stage 2: cross-mouse alignment
                # Align each mouse's mean weight matrix to the grand mean.
                # --------------------------------------------------------------
                ref = np.mean(list(mouse_mean_weights.values()), axis=0)  # (K, M)

                cross_inv_perms: dict[str, np.ndarray] = {}
                for mouse_id, w_mat in mouse_mean_weights.items():
                    D = np.linalg.norm(
                        ref[:, None, :] - w_mat[None, :, :], axis=-1
                    )  # (K, K)
                    _, col_ind = linear_sum_assignment(D)
                    inv = np.empty_like(col_ind)
                    inv[col_ind] = np.arange(len(col_ind))
                    cross_inv_perms[mouse_id] = inv
                    logger.info(f"  [{mouse_id}] Cross-mouse perm: {col_ind}")

                # --------------------------------------------------------------
                # Compose within + cross into a single inv_perm per result file:
                #   new_label = cross_inv[within_inv[old_label]]
                # --------------------------------------------------------------
                for mouse_id, si_inst_map in within_inv_perms.items():
                    cross_inv = cross_inv_perms[mouse_id]
                    for (si, inst), within_inv in si_inst_map.items():
                        composed = cross_inv[within_inv]  # (K,) int array
                        all_perms[(rg, model_name, n_states, mouse_id, si, inst)] = composed

    # --------------------------------------------------------------------------
    # Save
    # --------------------------------------------------------------------------
    if cfg['trial_types']=='whisker':
        perm_path = Path(cfg["global_path"]).parent / "figures_whisker" / "all_perms.pkl"
    else:
        perm_path = Path(cfg["global_path"]).parent / "figures" / "all_perms.pkl"
    utils.save_permutations(all_perms, perm_path)
    logger.info(f"stage_find_permutations done — {len(all_perms)} entries saved to {perm_path}")

import plotting_utils

def _debug_plot_whisker_lick_rate(all_perms, cfg, perm_path):
    """
    For each (model_name, n_states), plot whisker trial lick rate per aligned
    state, with R+ and R- mice shown separately as individual points.
    """
    rg_colors = {"R+": "forestgreen", "R-": "crimson"}
    feature_sets = build_feature_sets(cfg["features"], cfg.get("trial_types"))

    for model_name in feature_sets:
        for n_states in cfg["n_states_list"]:

            # {rg: {mouse_id: (K,) lick rate array}}
            group_lick_rates: dict[str, dict[str, np.ndarray]] = {}

            # Collect all keys relevant to this (model_name, n_states)
            relevant = {
                k: v for k, v in all_perms.items()
                if k[1] == model_name and k[2] == n_states
            }
            if not relevant:
                continue

            for (rg, mn, ns, mouse_id, si, inst), inv_perm in relevant.items():
                if rg not in group_lick_rates:
                    group_lick_rates[rg] = {}
                if mouse_id not in group_lick_rates[rg]:
                    group_lick_rates[rg][mouse_id] = {
                        "lick_counts":  np.zeros(n_states),
                        "trial_counts": np.zeros(n_states),
                    }

                rg_int = {v: k for k, v in _RG_STR.items()}.get(rg, 0)
                h5 = (single_model_dir(cfg, mouse_id, si, n_states, inst, model_name, rg_int)
                      / "data_preds.h5")
                if not h5.exists():
                    continue
                try:
                    df = pd.read_hdf(h5)
                except Exception as e:
                    logger.warning(f"  Could not read {h5}: {e}")
                    continue

                # Apply composed inv_perm
                if "most_likely_state" in df.columns:
                    state_labels = inv_perm[df["most_likely_state"].values]
                else:
                    post_cols    = sorted(c for c in df.columns if c.startswith("posterior_state_"))
                    state_labels = inv_perm[df[post_cols].values.argmax(axis=1)]

                df["_aligned_state"] = state_labels
                #whisker = df[df["trial_type"] == "whisker_trial"]
                whisker = df
                for s in range(n_states):
                    mask = whisker["_aligned_state"] == s
                    group_lick_rates[rg][mouse_id]["lick_counts"][s]  += whisker.loc[mask, "choice"].sum()
                    group_lick_rates[rg][mouse_id]["trial_counts"][s] += mask.sum()

            # Finalise lick rates
            for rg in group_lick_rates:
                for mouse_id in group_lick_rates[rg]:
                    lc = group_lick_rates[rg][mouse_id]["lick_counts"]
                    tc = group_lick_rates[rg][mouse_id]["trial_counts"]
                    with np.errstate(invalid="ignore"):
                        group_lick_rates[rg][mouse_id] = np.where(tc > 0, lc / tc, np.nan)

            # --- Plot ---
            rg_list  = sorted(group_lick_rates.keys())
            x_pos    = {rg: i for i, rg in enumerate(rg_list)}
            x_jitter = 0.08

            fig, axs = plt.subplots(
                1, n_states,
                figsize=(3 * n_states, 3.5),
                sharey=True, dpi=150, facecolor="w",
            )
            if n_states == 1:
                axs = [axs]

            for s, ax in enumerate(axs):
                for rg in rg_list:
                    rates = np.array(list(group_lick_rates[rg].values()))  # (n_mice, K)
                    col   = rg_colors.get(rg, "steelblue")
                    x     = x_pos[rg]
                    valid = rates[:, s][~np.isnan(rates[:, s])]

                    ax.scatter(
                        np.random.uniform(x - x_jitter, x + x_jitter, len(valid)),
                        valid,
                        color=col, s=30, alpha=0.7, zorder=3, label=rg,
                    )
                    if len(valid):
                        ax.hlines(
                            np.nanmean(valid), x - 0.2, x + 0.2,
                            colors=col, linewidths=2, zorder=4,
                        )

                ax.set_xlim(-0.5, len(rg_list) - 0.5)
                ax.set_xticks(range(len(rg_list)))
                ax.set_xticklabels(rg_list, fontsize=9)
                ax.set_ylim(-0.05, 1.05)
                ax.set_title(f"State {s}", fontsize=10)
                ax.axhline(0.5, color="grey", lw=0.8, ls="--", alpha=0.5)
                plotting_utils.remove_top_right_frame(ax)

            axs[0].set_ylabel("Whisker lick rate", fontsize=10)
            handles, labels = axs[-1].get_legend_handles_labels()
            seen = {}
            axs[-1].legend(
                [h for h, l in zip(handles, labels) if not (l in seen or seen.update({l: True}))],
                [l for l in labels if not seen.get(l)],
                frameon=False, fontsize=8,
            )
            fig.suptitle(f"{model_name} | K={n_states}", fontsize=10)
            fig.tight_layout()

            out_dir = Path(perm_path).parent / "debug_permutations"
            out_dir.mkdir(parents=True, exist_ok=True)
            plotting_utils.save_figure_to_files(
                fig, str(out_dir),
                f"whisker_lick_rate_{model_name}_K{n_states}",
                file_types=["png"], dpi=150,
            )
            plt.close(fig)
            logger.info(f"  Debug plot → {out_dir}")

def stage_find_permutations(cfg, cross_mouse_method: str = "weights") -> None:
    """
    Compute and save composed (within-mouse + cross-mouse) state permutations
    for all result files, per (reward_group, model_name, n_states).

    Permutations are saved as a pickle file at:
        cfg["result_path"] / "all_perms.pkl"

    Each permutation is an inv_perm array of shape (K,) such that:
        new_label = inv_perm[old_label]
    applicable to both integer state labels and posterior column indices.

    To reload for plotting:
        all_perms = load_permutations(cfg["result_path"] / "all_perms.pkl")
        inv_perm  = all_perms[(rg, model_name, n_states, mouse_id, si, inst)]

    :param cfg: pipeline config dict with keys:
                  features, trial_types, reward_groups, n_states_list,
                  single_path, result_path, n_splits, n_instances
    :param cross_mouse_method: how to align states across mice. One of:
                  "weights"   — align each mouse's mean weight matrix to the
                                grand mean (Hungarian, weight-space). Default.
                  "lick_rate" — sort each mouse's states by mean lick rate on
                                whisker trials (ascending: state 0 = low lick,
                                state K-1 = high lick). No reference needed.
    """
    logger.info("=" * 60)
    logger.info("STAGE – Find permutations")
    logger.info("=" * 60)

    assert cross_mouse_method in ("weights", "lick_rate"), (
        f"cross_mouse_method must be 'weights' or 'lick_rate', got '{cross_mouse_method}'"
    )

    feature_sets = build_feature_sets(cfg["features"], cfg.get("trial_types"))
    all_perms = {}

    for rg_int in cfg["reward_groups"]:
        rg = _RG_STR.get(int(rg_int), str(rg_int))

        for model_name in feature_sets:

            if model_name != 'full':
                logger.debug(f"Skipping model_name='{model_name}' for permutation finding")
                continue

            feats = feature_sets[model_name]

            for n_states in cfg["n_states_list"]:

                if n_states not in [2,3]:
                    logger.debug(f"Skipping K={n_states} for permutation finding ")
                    continue

                single_base = cfg["single_path"] / _rg_label(rg_int) / model_name
                if not single_base.exists():
                    continue

                logger.info(f"[{rg} | {model_name} | K={n_states}] "
                            f"Computing permutations (cross-mouse: {cross_mouse_method})...")

                # --------------------------------------------------------------
                # Stage 1: within-mouse alignment
                # For each mouse, align states across splits x instances using
                # Viterbi sequences (falling back to weights if unavailable).
                # --------------------------------------------------------------
                within_inv_perms:   dict[str, dict[tuple, np.ndarray]] = {}
                mouse_mean_weights: dict[str, np.ndarray]              = {}
                path_map_per_mouse: dict[str, dict[tuple, Path]]       = {}

                for mouse_dir in sorted(single_base.iterdir()):
                    mouse_id = mouse_dir.name

                    path_map = {
                        (si, inst): single_model_dir(
                            cfg, mouse_id, si, n_states, inst, model_name, rg_int
                        )
                        for si   in range(cfg["n_splits"])
                        for inst in range(cfg["n_instances"])
                    }
                    valid_keys = {
                        k for k, p in path_map.items()
                        if (p / "fit_glmhmm_results.npz").exists()
                    }
                    if not valid_keys:
                        continue

                    path_map_per_mouse[mouse_id] = path_map

                    # -- Viterbi sequences (preferred) -------------------------
                    viterbi_dict, sequences_ok = utils._load_viterbi_dict(
                        path_map, valid_keys, n_states
                    )
                    if sequences_ok and viterbi_dict:
                        viterbi_perms = utils.compute_permutations_from_viterbi(
                            viterbi_dict, n_states
                        )
                        logger.info(f"  [{mouse_id}] Viterbi-based alignment")
                    else:
                        viterbi_perms = None
                        logger.warning(f"  [{mouse_id}] Falling back to weight-based alignment")

                    # -- weight df for within-mouse alignment ------------------
                    weight_df = utils._build_weight_df(
                        path_map, valid_keys, mouse_id, rg, n_states, feats
                    )
                    aligned_weight_df, permut_ids = utils.align_weights_dataframe(
                        weight_df,
                        use_mean_reference=False,
                        permutations=viterbi_perms,
                    )

                    # -- store inv_perm per (si, inst) -------------------------
                    within_inv_perms[mouse_id] = {}
                    for (si, inst) in valid_keys:
                        perm = permut_ids.get((n_states, si, inst), np.arange(n_states))
                        inv       = np.empty_like(perm)
                        inv[perm] = np.arange(len(perm))
                        within_inv_perms[mouse_id][(si, inst)] = inv

                    # -- mean weight matrix after within-mouse alignment -------
                    mouse_mean_weights[mouse_id] = utils._mean_weight_matrix(
                        aligned_weight_df, n_states, feats
                    )

                if not mouse_mean_weights:
                    continue

                # --------------------------------------------------------------
                # Stage 2: cross-mouse alignment
                # --------------------------------------------------------------
                if cross_mouse_method == "weights":

                    ref = np.mean(list(mouse_mean_weights.values()), axis=0)  # (K, M)

                    cross_inv_perms: dict[str, np.ndarray] = {}
                    for mouse_id, w_mat in mouse_mean_weights.items():
                        D = np.linalg.norm(
                            ref[:, None, :] - w_mat[None, :, :], axis=-1
                        )  # (K, K)
                        _, col_ind        = linear_sum_assignment(D)
                        inv               = np.empty_like(col_ind)
                        inv[col_ind]      = np.arange(len(col_ind))
                        cross_inv_perms[mouse_id] = inv
                        logger.info(f"  [{mouse_id}] Cross-mouse perm (weights): {col_ind}")

                else:  # "lick_rate"

                    cross_inv_perms = utils._cross_mouse_alignment_from_lick_rate(
                        path_map_per_mouse=path_map_per_mouse,
                        within_inv_perms=within_inv_perms,
                        n_states=n_states,
                        trial_type="whisker_trial",
                    )

                # --------------------------------------------------------------
                # Compose within + cross into a single inv_perm per result file:
                #   new_label = cross_inv[within_inv[old_label]]
                # --------------------------------------------------------------
                for mouse_id, si_inst_map in within_inv_perms.items():
                    cross_inv = cross_inv_perms[mouse_id]
                    for (si, inst), within_inv in si_inst_map.items():
                        composed = cross_inv[within_inv]  # (K,) int array
                        all_perms[(rg, model_name, n_states, mouse_id, si, inst)] = composed


    # --------------------------------------------------------------------------
    # Save
    # --------------------------------------------------------------------------
    if cfg['trial_types'] == 'whisker':
        perm_path = Path(cfg["global_path"]).parent / "perm_whisker" / "all_perms.pkl"
    else:
        perm_path = Path(cfg["global_path"]).parent / "perm" / "all_perms.pkl"

    #for rg in cfg["reward_groups"]:
    #    for n_states in [2,3]:
#
    #        utils._debug_plot_whisker_lick_rate(
    #            path_map_per_mouse=path_map_per_mouse,
    #            within_inv_perms=within_inv_perms,
    #            cross_inv_perms=cross_inv_perms,
    #            n_states=n_states,
    #            rg=rg,
    #            model_name='full',
    #            perm_path=perm_path,
    #        )

    utils.save_permutations(all_perms, perm_path)

    _debug_plot_whisker_lick_rate(
        all_perms=all_perms,
        cfg=cfg,
        perm_path=perm_path,
    )


    logger.info(f"stage_find_permutations done — {len(all_perms)} entries saved to {perm_path}")


def list_dlc_bodyparts(nwb_file: str | Path) -> list[str]:
    """
    Open one NWB file and return all DLC bodypart names available under
    processing['behavior']['BehavioralTimeSeries'].

    Parameters
    ----------
    nwb_file : path to an NWB file

    Returns
    -------
    Sorted list of bodypart name strings,
    e.g. ['jaw_distance', 'nose_distance', 'pupil_area', 'whisker_angle']
    """
    from pynwb import NWBHDF5IO

    with NWBHDF5IO(str(nwb_file), 'r', load_namespaces=True) as io:
        nwb = io.read()
        beh = nwb.processing.get('behavior')
        if beh is None:
            raise KeyError("No 'behavior' processing module found in NWB file.")

        bts = beh.data_interfaces.get('BehavioralTimeSeries')
        if bts is None:
            raise KeyError(
                "No 'BehavioralTimeSeries' found under processing['behavior']."
            )

        bodyparts = sorted(bts.time_series.keys())
        logger.info(f"  Available DLC bodyparts: {bodyparts}")
        return bodyparts


def _get_dlc_timeseries(nwb) -> dict[str, object]:
    """
    Return {bodypart_name: TimeSeries} from an open NWB object via
    processing['behavior']['BehavioralTimeSeries'].
    Returns an empty dict if the interface is absent.
    """
    beh = nwb.processing.get('behavior')
    if beh is None:
        return {}
    bts = beh.data_interfaces.get('BehavioralTimeSeries')
    if bts is None:
        return {}
    return dict(bts.time_series)


def _extract_window(ts_data: np.ndarray,
                    ts_timestamps: np.ndarray,
                    start_time: float,
                    pre_time: float,
                    post_time: float,
                    n_samples: int) -> np.ndarray | None:
    """
    Extract a fixed-length window from a 1-D timeseries aligned to start_time.

    The raw segment is resampled to exactly n_samples points via np.interp so
    that all trials stack cleanly into a (n_trials, n_samples) matrix regardless
    of camera frame-rate variation across sessions.

    Parameters
    ----------
    ts_data       : 1-D float array of DLC values
    ts_timestamps : 1-D float array, same time base as start_time
    start_time    : alignment event time (seconds)
    pre_time      : seconds before start_time to include (positive number)
    post_time     : seconds after  start_time to include
    n_samples     : number of output samples

    Returns
    -------
    1-D array of length n_samples, or None if the window falls outside the
    recording.
    """
    t0 = start_time - pre_time
    t1 = start_time + post_time

    if t0 < ts_timestamps[0] or t1 > ts_timestamps[-1]:
        return None

    i0 = np.searchsorted(ts_timestamps, t0, side='left')
    i1 = np.searchsorted(ts_timestamps, t1, side='right')

    seg_t = ts_timestamps[i0:i1]
    seg_d = ts_data[i0:i1]

    if len(seg_t) < 2:
        return None

    t_uniform = np.linspace(t0, t1, n_samples)
    return np.interp(t_uniform, seg_t, seg_d)


def plot_dlc_traces_by_state(cfg: dict,
                             nwb_paths: dict[str, str | Path],
                             trial_df: pd.DataFrame,
                             figure_path: str | Path):
    """
    Plot trial-type-averaged DeepLabCut traces per bodypart, split by reward
    group and GLM-HMM state, aligned at trial start_time.

    DLC data is read from processing['behavior']['BehavioralTimeSeries'] in
    each NWB file.

    Layout (one figure per bodypart)
    ---------------------------------
    rows  = reward groups  (e.g. R+, R-)
    cols  = trial types    (wh, wm, ah, am, fa, cr — or a configurable subset)
    lines = states         (one per state index, coloured by state_index_cmap)

    Mean ± SEM is computed hierarchically: average within each mouse first,
    then compute the grand mean and inter-mouse SEM.

    Parameters
    ----------
    cfg : dict with keys:
        'pre_time'    : float, seconds before start_time (default 0.5)
        'post_time'   : float, seconds after  start_time (default 1.5)
        'n_samples'   : int,   samples per window after resampling (default 200)
        'trial_types' : list[str], subset of
                        ['wh','wm','ah','am','fa','cr'] (default: all six)
        'bodyparts'   : list[str] or None — None → discovered at runtime from
                        the first matching NWB file
        'day_val'     : int or None — restrict to a single training day

    nwb_paths : dict  {session_id: path_to_nwb_file}
                Keys must match the 'session_id' column in trial_df.

    trial_df  : trial-level DataFrame with columns:
                  mouse_id, session_id, day, trial_id, start_time,
                  stimulus_type, choice, most_likely_state, reward_group

    figure_path : root directory for figure output
    """
    from pynwb import NWBHDF5IO
    from plotting_utils import remove_top_right_frame, save_figure_to_files

    figure_path = Path(figure_path)
    out_base = figure_path / "dlc_traces_by_state"

    # ── cfg defaults ──────────────────────────────────────────────────
    pre_time = cfg.get('pre_time', 0.5)
    post_time = cfg.get('post_time', 1.5)
    n_samples = cfg.get('n_samples', 200)
    day_val = cfg.get('day_val', None)

    all_trial_types = ['wh', 'wm', 'ah', 'am', 'fa', 'cr']
    trial_types_req = cfg.get('trial_types', all_trial_types)
    trial_types_req = [t for t in all_trial_types if t in trial_types_req]  # preserve order

    tt_encode = {
        (1, 1): 'wh',
        (1, 0): 'wm',
        (-1, 1): 'ah',
        (-1, 0): 'am',
        (0, 1): 'fa',
        (0, 0): 'cr',
    }
    tt_label = {
        'wh': 'Whisker hit',
        'wm': 'Whisker miss',
        'ah': 'Auditory hit',
        'am': 'Auditory miss',
        'fa': 'False alarm',
        'cr': 'Correct rejection',
    }

    t_axis = np.linspace(-pre_time, post_time, n_samples)

    # ── Optional day filter ───────────────────────────────────────────
    if day_val is not None:
        trial_df = trial_df[trial_df['day'] == day_val].copy()
        if trial_df.empty:
            logger.warning(f"  plot_dlc_traces_by_state: no trials for day={day_val}.")
            return

    # ── Discover bodyparts from the first available NWB file ──────────
    bodyparts_req = cfg.get('bodyparts', None)
    if bodyparts_req is None:
        available_sessions = [s for s in trial_df['session_id'].unique()
                              if s in nwb_paths]
        if not available_sessions:
            logger.error("  plot_dlc_traces_by_state: no matching NWB paths found.")
            return
        bodyparts_req = list_dlc_bodyparts(nwb_paths[available_sessions[0]])

    if not bodyparts_req:
        logger.error("  plot_dlc_traces_by_state: no bodyparts found, aborting.")
        return

    logger.info(f"  Bodyparts to process: {bodyparts_req}")

    # ── Annotate trial_df with trial-type label ───────────────────────
    trial_df = trial_df.copy()
    trial_df['trial_type'] = trial_df.apply(
        lambda r: tt_encode.get((int(r['stimulus_type']), int(r['choice'])), 'unknown'),
        axis=1,
    )
    trial_df = trial_df[trial_df['trial_type'].isin(trial_types_req)]

    # ── Extraction loop — one NWB file at a time ──────────────────────
    records = []

    sessions_in_df = trial_df['session_id'].unique()
    sessions_with_nwb = [s for s in sessions_in_df if s in nwb_paths]
    n_missing = len(sessions_in_df) - len(sessions_with_nwb)
    if n_missing:
        logger.warning(f"  {n_missing} sessions in trial_df have no NWB path — skipped.")

    for session_id in sessions_with_nwb:
        sess_trials = trial_df[trial_df['session_id'] == session_id]
        if sess_trials.empty:
            continue

        nwb_file = nwb_paths[session_id]
        logger.info(f"  Loading DLC from {Path(nwb_file).name} "
                    f"({len(sess_trials)} trials)")

        try:
            with NWBHDF5IO(str(nwb_file), 'r', load_namespaces=True) as io:
                nwb = io.read()
                ts_dict = _get_dlc_timeseries(nwb)

                if not ts_dict:
                    logger.warning(
                        f"  No BehavioralTimeSeries in {Path(nwb_file).name}, skipping."
                    )
                    continue

                # Filter to requested bodyparts present in this file
                bodyparts_this = [b for b in bodyparts_req if b in ts_dict]
                missing_bp = [b for b in bodyparts_req if b not in ts_dict]
                if missing_bp:
                    logger.warning(
                        f"  Bodyparts absent from {Path(nwb_file).name}: {missing_bp}"
                    )

                # Load each bodypart's data array + timestamps once per file
                bp_arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
                for bp in bodyparts_this:
                    ts_obj = ts_dict[bp]
                    raw = np.array(ts_obj.data[:]).astype(float)
                    if raw.ndim == 2:  # (T, 2) x/y → take first column
                        raw = raw[:, 0]
                    if ts_obj.timestamps is not None:
                        ts_t = np.array(ts_obj.timestamps[:])
                    else:
                        rate = ts_obj.rate
                        t0_r = ts_obj.starting_time or 0.0
                        ts_t = t0_r + np.arange(len(raw)) / rate
                    bp_arrays[bp] = (raw, ts_t)

                # Extract one window per trial per bodypart
                for _, trial in sess_trials.iterrows():
                    start_t = float(trial['start_time'])
                    mouse_id = trial['mouse_id']
                    rg = trial['reward_group']
                    state = int(trial['most_likely_state'])
                    ttype = trial['trial_type']

                    for bp, (raw, ts_t) in bp_arrays.items():
                        window = _extract_window(
                            raw, ts_t, start_t, pre_time, post_time, n_samples
                        )
                        if window is None:
                            continue
                        records.append({
                            'mouse_id': mouse_id,
                            'session_id': session_id,
                            'trial_id': trial['trial_id'],
                            'reward_group': rg,
                            'state': state,
                            'trial_type': ttype,
                            'bodypart': bp,
                            'window': window,
                        })

        except Exception as e:
            logger.error(f"  Error loading {Path(nwb_file).name}: {e}")
            continue

    if not records:
        logger.warning("  plot_dlc_traces_by_state: no windows extracted, aborting.")
        return

    rec_df = pd.DataFrame(records)
    logger.info(
        f"  Extracted {len(rec_df)} trial windows across "
        f"{rec_df['mouse_id'].nunique()} mice."
    )

    # ── Plotting ──────────────────────────────────────────────────────
    reward_groups = sorted(rec_df['reward_group'].unique())
    n_rows = len(reward_groups)
    n_cols = len(trial_types_req)

    max_states = int(rec_df['state'].max()) + 1
    state_palette = sns.color_palette("tab10", max_states)

    for bp in bodyparts_req:
        bp_df = rec_df[rec_df['bodypart'] == bp]
        if bp_df.empty:
            logger.warning(f"  No data for bodypart '{bp}', skipping.")
            continue

        fig, axs = plt.subplots(
            n_rows, n_cols,
            figsize=(3.5 * n_cols, 3.0 * n_rows),
            dpi=200,
            constrained_layout=True,
            sharey='row',
        )
        axs = np.atleast_2d(axs)

        fig.suptitle(
            f"DLC — {bp}  |  aligned at trial start  "
            f"(n={bp_df['mouse_id'].nunique()} mice)",
            fontsize=11,
        )

        for r_idx, rg in enumerate(reward_groups):
            rg_df = bp_df[bp_df['reward_group'] == rg]
            rg_states = sorted(rg_df['state'].unique())

            for c_idx, ttype in enumerate(trial_types_req):
                ax = axs[r_idx, c_idx]
                sub_df = rg_df[rg_df['trial_type'] == ttype]

                remove_top_right_frame(ax)
                ax.axvline(0, color='k', lw=0.8, ls='--', alpha=0.4)
                ax.axhline(0, color='k', lw=0.5, alpha=0.2)

                if r_idx == 0:
                    ax.set_title(tt_label.get(ttype, ttype), fontsize=9)
                if c_idx == 0:
                    ax.set_ylabel(f"{rg}\n{bp}", fontsize=9)
                if r_idx == n_rows - 1:
                    ax.set_xlabel("Time from trial start (s)", fontsize=8)

                if sub_df.empty:
                    ax.text(0.5, 0.5, 'no trials', transform=ax.transAxes,
                            ha='center', va='center', fontsize=8, color='grey')
                    continue

                for state in rg_states:
                    state_df = sub_df[sub_df['state'] == state]
                    if state_df.empty:
                        continue

                    color = state_palette[state]

                    # Hierarchical average: within-mouse first, then across mice
                    mouse_means = []
                    for _, mouse_grp in state_df.groupby('mouse_id'):
                        windows = np.vstack(mouse_grp['window'].values)
                        mouse_means.append(windows.mean(axis=0))

                    mouse_means = np.array(mouse_means)  # (n_mice, n_samples)
                    n_mice = len(mouse_means)
                    grand_mean = mouse_means.mean(axis=0)
                    sem = (mouse_means.std(axis=0, ddof=1) / np.sqrt(n_mice)
                           if n_mice > 1 else np.zeros(n_samples))

                    n_trials = state_df['trial_id'].nunique()
                    label = f"State {state + 1}  (n={n_mice} mice, {n_trials} trials)"

                    ax.plot(t_axis, grand_mean,
                            color=color, lw=1.5, label=label)
                    ax.fill_between(t_axis,
                                    grand_mean - sem,
                                    grand_mean + sem,
                                    color=color, alpha=0.2)

                ax.legend(frameon=False, fontsize=7, loc='upper right')

        out_base.mkdir(parents=True, exist_ok=True)
        day_suffix = f"_day{day_val}" if day_val is not None else ""
        save_figure_to_files(
            fig, str(out_base),
            f"dlc_traces_{bp}{day_suffix}",
            suffix=None,
            file_types=['pdf', 'png'],
            dpi=200,
        )
        plt.close(fig)
        logger.info(f"  Saved DLC figure for bodypart '{bp}'")

    logger.info(f"  plot_dlc_traces_by_state → {out_base}")


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 4  –  PERFORMANCE FIGURES  (entry point)
# ─────────────────────────────────────────────────────────────────────────────

def stage_plot_performance(cfg, model_name_for_per_mouse: str = "full"):
    """
    Stage 4: load summary files from stages 2 & 3 and produce:
      • Global model performance panel (LL / acc / balanced-acc / bpt)
        for every model type (feature set), split by reward group.
      • Per-mouse performance across n_states for `model_name_for_per_mouse`.
    """
    logger.info("=" * 60)
    logger.info("STAGE 4 – Performance figures")
    logger.info("=" * 60)

    sns.set_context("paper")
    sns.set_style("ticks")
    plt.rcParams["font.size"] = 10

    # If fit only whisker trials, filter
    if cfg["trial_types"] == "whisker":
        figure_path = cfg.get("figure_path",
                              cfg["global_path"].parent / "figures_whisker")
    else:
        figure_path = cfg.get("figure_path",
                              cfg["global_path"].parent / "figures")
    figure_path = Path(figure_path)
    figure_path.mkdir(parents=True, exist_ok=True)

    # Load permutations computed in stage_find_permutations
    if cfg['trial_types'] == 'whisker':
        perm_path = Path(cfg["global_path"]).parent / "perm_whisker" / "all_perms.pkl"
    else:
        perm_path = Path(cfg["global_path"]).parent / "perm" / "all_perms.pkl"
    all_perms = utils.load_permutations(perm_path)
    # Pass to any loader that reads trial data or weights
    weight_df = _load_single_weights_long_permut(cfg, all_perms)
    print("Loaded weight_df with columns:", weight_df.columns)
    trial_df = _load_single_trial_data_permut(cfg, 'full', all_perms)
    # Keep model 0 iter 0
    trial_df = trial_df[(trial_df.split_idx==0) & (trial_df.instance_idx==0)]
    print('Loaded trial_df with columns:', trial_df.columns)

    # ── Global performance ────────────────────────────────────────────────────
    plot_global_perf = False
    if plot_global_perf:
        logger.info(f" Plotting global performance …")
        try:
            df_global = _prep_global_df(cfg)
            #df_global = _compute_bpt(df_global)

            # All feature sets together
            model_names = sorted(df_global["model_type"].unique(),
                                 key=lambda x: (x != "full", x))
            print('Global performance for model types:', model_names)
            plot_global_performance(df_global, figure_path, model_subset=model_names)


        except FileNotFoundError as e:
            logger.warning(f"  Skipping global performance plots: {e}")

    # ── Per-mouse performance ─────────────────────────────────────────────────
    plot_single_mouse_perf = False
    if plot_single_mouse_perf:
        logger.info(" Plotting per-mouse performance …")
        try:
            df_single = _prep_single_df(cfg)
            # Ensure model_name column exists (older runs may not have it)
            if "model_name" not in df_single.columns:
                df_single["model_name"] = "full"

            plot_per_mouse_performance(df_single, figure_path,
                                       model_name=model_name_for_per_mouse)

            # Also produce one panel per model_name that exists in the data
            for mname in df_single["model_name"].unique():
                if mname == model_name_for_per_mouse:
                    continue
                plot_per_mouse_performance(df_single, figure_path, model_name=mname)

        except FileNotFoundError as e:
            logger.warning(f"  Skipping per-mouse performance plots: {e}")

    ## ── Analysis 1 – state alignment diagnostics (run first, gates all downstream)
    feature_sets = build_feature_sets(cfg["features"], cfg.get("trial_types"))
    feature_sets = ['full']
    #for mname in feature_sets:
    ##    try:
    #        plot_state_alignment_diagnostics(cfg, figure_path, model_name=mname)
    #    except Exception as e:
    #        logger.warning(f"  [1] state alignment failed for {mname}: {e}")

    # ── Analysis 1 – per-mouse weight diagnostic grid ─────────────────────────
    plot_per_mouse_weight_grid = False
    if plot_per_mouse_weight_grid:
        logger.info(" Plotting per-mouse weight grid …")
        for mname in feature_sets:
            try:
                plot_mouse_weight_splits(cfg, figure_path, model_name=mname)
            except Exception as e:
                logger.warning(f"  [1] weight diagnostics failed for {mname}: {e}")

    # ── Analysis 2 – posterior curves by day ──────────────────────────────────
    plot_mean_posteriors = False
    if plot_mean_posteriors:
        logger.info(f" Plotting posterior curves by day …")
        for mname in feature_sets:
            try:
                plot_posterior_curves_by_day(cfg, figure_path, trial_df, model_name=mname, max_trials=50) #use trial_df #todo: interpolate to compare sesions?
            except Exception as e:
                logger.warning(f"  [2] posterior curves failed for {mname}: {e}")

    # ── Analysis 3 – global weights, hue = reward_group ───────────────────────
    plot_global_weights_per_group = False
    if plot_global_weights_per_group:
        logger.info(f" Plotting global weights by reward group …")
        try:
            #plot_global_weights_by_rg(cfg, figure_path)
            plot_global_weights(cfg, figure_path)
        except Exception as e:
            logger.warning(f"  [3] global weights by rg failed: {e}")

    # ── Analysis 4 – single-mouse weights, hue = reward_group ─────────────────
    plot_single_weights_per_group= False #TODO: add states between mice cohorts
    if plot_single_weights_per_group: #todo: add transition matrix
        logger.info(f" Plotting single-mouse weights by state/reward group …")
        try:
            #plot_single_weights_by_rg(cfg, figure_path)
            #plot_single_weights_by_state(cfg, figure_path)
            #plot_single_weights(cfg, figure_path) # combined plot
            plot_single_weights(cfg, figure_path, weight_df) # combined plot
        except Exception as e:
            logger.warning(f"  [4] single weights plots failed: {e}")

    # ── Analysis 5 – lick rate per trial type × state ─────────────────────────
    plot_beh_perf_in_states=False #todo: add model performance per trial_type, resolve for no stim,auditory trials
    if plot_beh_perf_in_states:
        logger.info(f" Plotting trial-type performance per state …")

        for mname in feature_sets:
            try:
                plot_lick_rate_per_state(cfg, figure_path, trial_df, model_name=mname)
            except Exception as e:
                logger.warning(f"  [5] lick rate per state failed for {mname}: {e}")


    # ── Analysis – lick rate around state transition ─────────────────────────
    plot_beh_perf_at_state_transition=False
    if plot_beh_perf_at_state_transition:
        logger.info(f" Plotting trial-type performance around state transitions …")

        for mname in feature_sets:
            try:
                plot_lick_rate_around_transitions(cfg, figure_path, trial_df, model_name=mname)
            except Exception as e:
                logger.warning(f"  [5] lick rate around state transition failed for {mname}: {e}")



    # ── Analysis – weight perf metric correlations  ─────────────────────────
    plot_weight_beh_metric_correlation=False
    if plot_weight_beh_metric_correlation:
        logger.info(f" Plotting correlation between state weights and perf metrics …")

        for mname in feature_sets:
            try:
                plot_weight_metric_correlation(cfg, trial_df, weight_df, figure_path, inflection_df=None)
            except Exception as e:
                logger.warning(f"  [6] correlating weight and perf failed for {mname}: {e}")



    # Analysis - mean movement DLC by state
    plot_state_dlc_curves = True
    if plot_state_dlc_curves:
        logger.info(f" Plotting DLC mean cruves state weights and perf metrics …")

        base_path = r"M:\analysis\Axel_Bisi\NWB_combined"

        nwb_list = [os.path.join(base_path, f)for f in os.listdir(base_path) if 'AB141_20241129_130221.nwb' in f]
        print(nwb_list)
        # Make dict sess_id:path
        nwb_paths = {NWB_read.get_session_id(f): f for f in nwb_list}
        print(nwb_paths)
        print(list_dlc_bodyparts(nwb_list[0]))

        dlc_plot_cfg = {
            'pre_time': 0.5,
            'post_time': 1.5,
            'n_samples': 200,
            'trial_types': ['wh', 'wm', 'ah', 'am', 'fa', 'cr'],
            'bodyparts': ['jaw_angle', 'jaw_distance', 'nose_distance', 'pupil_area', 'whisker_angle'],
            'day_val': 0,  # None → all days pooled
        }
        try:
            plot_dlc_traces_by_state(dlc_plot_cfg, nwb_paths, trial_df, figure_path)
        except Exception as e:
            logger.warning(f"  [7] mean DLC per state failed for: {e}")

    logger.info(f"Stage 4 complete. Figures → {figure_path}\n")
    return


# TODO: analysis across learning days, figures across days 0,1,2 , not the priority for now focus on day0

# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="GLM-HMM pipeline")
    p.add_argument(
        "--stages", nargs="+",
        choices=["dataset", "global", "single", "plot", "all"],
        default=["all"],
    )
    p.add_argument("--n_states",      type=int,   nargs="+", default=None)
    p.add_argument("--n_splits",      type=int,              default=None)
    p.add_argument("--n_instances",   type=int,              default=None)
    p.add_argument("--kappa",         type=float,            default=None)
    p.add_argument("--reward_groups", type=int,   nargs="+", default=None,
                   help="e.g. --reward_groups 1 0")
    p.add_argument("--single_n_states", type=int,            default=None,
                   help="Fit single-mouse for this K only")
    p.add_argument("--n_workers",     type=int,              default=None)
    p.add_argument("--plot_model",    type=str,              default="full",
                   help="model_name (feature set) to use for per-mouse performance plot")
    p.add_argument("--figure_path",   type=str,              default=None,
                   help="Override output directory for figures")
    return p.parse_args()


def main():
    args = parse_args()

    cfg = dict(CFG)
    if args.n_states        is not None: cfg["n_states_list"]  = args.n_states
    if args.n_splits        is not None: cfg["n_splits"]        = args.n_splits
    if args.n_instances     is not None: cfg["n_instances"]     = args.n_instances
    if args.kappa           is not None: cfg["kappa"]           = args.kappa
    if args.reward_groups   is not None: cfg["reward_groups"]   = args.reward_groups
    if args.n_workers       is not None: cfg["n_workers"]       = args.n_workers
    if args.figure_path     is not None: cfg["figure_path"]     = Path(args.figure_path)

    run_all     = "all" in args.stages
    run_dataset = (run_all or "dataset" in args.stages) and cfg["run_dataset"]
    run_global  = (run_all or "global"  in args.stages) and cfg["run_global"]
    run_single  = (run_all or "single"  in args.stages) and cfg["run_single"]
    run_permute  = (run_all or "permute"  in args.stages) and cfg["run_permute"]
    run_plot    = (run_all or "plot"    in args.stages)

    feature_sets = build_feature_sets(cfg["features"], cfg.get("trial_types"))

    logger.info("GLM-HMM pipeline starting")
    logger.info(f"  Stages        : dataset={run_dataset}  global={run_global}  single={run_single}  plot={run_plot}")
    logger.info(f"  n_states      : {cfg['n_states_list']}")
    logger.info(f"  n_splits      : {cfg['n_splits']}")
    logger.info(f"  n_instances   : {cfg['n_instances']}")
    logger.info(f"  kappa         : {cfg['kappa']}")
    logger.info(f"  reward_groups : {cfg['reward_groups']}")
    logger.info(f"  n_workers     : {cfg['n_workers']}")
    logger.info(f"  feature_sets  : {list(feature_sets.keys())}")

    t0 = time.time()
    #if run_dataset: stage_create_dataset(cfg)
    #if run_global:  stage_fit_global(cfg)
    #if run_single: stage_fit_single(cfg, n_states_to_fit=args.single_n_states)
    #if run_single: stage_find_permutations(cfg, cross_mouse_method='lick_rate')
    if run_plot:   stage_plot_performance(cfg, model_name_for_per_mouse=args.plot_model)
    logger.info(f"Pipeline finished in {time.time() - t0:.1f} s")

if __name__ == "__main__":
    main()
