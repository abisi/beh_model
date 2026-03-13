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

# ── project ───────────────────────────────────────────────────────────────────
from create_behaviour_dataset import create_behavior_dataset, split_dataset
from data_utils import create_data_lists
from plotting_utils import (
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
    run_dataset   = False,
    run_global    = True,
    run_single    = True,

    # ── dataset params ─────────────────────────────────────────────────────────
    n_splits          = 2, #cv across splits
    fraction_training = 0.8,
    n_trials_max      = 10000,        # set to int to cap trials per session

    # ── model params ───────────────────────────────────────────────────────────
    n_states_list = [1, 2, 3, 4, 5, 6],
    n_instances   = 3, #todo: to 5
    n_train_iters = 300,
    tolerance     = 1e-4,
    prior_sigma   = 2.0,
    prior_alpha   = 2.0,
    kappa         = 0.0,             # sticky HMM self-transition bias (0 = no stickiness)
    noise_level = 0.1,              # std dev of noise added to weights for random restarts

    reward_groups = [1,0],
    # ── features ──────────────────────────────────────────────────────────────
    features = [
        "bias",
        "whisker",
        "auditory",
        "time_since_last_auditory_reward",
        "time_since_last_whisker_reward",
        #"time_since_last_auditory_lick",
        #"time_since_last_whisker_lick",
        'jaw_distance',
        'whisker_angle',
        'pupil_area',
    ],
    # Feature names that correspond to trial types for block-LOO sets.
    # Set to None to skip block-level leave-one-out.
    trial_types = "all_trials", #"all_trials", or "whisker"

    # ── parallelism ───────────────────────────────────────────────────────────
    n_workers = max(1, os.cpu_count() - 2),
)


# ─────────────────────────────────────────────────────────────────────────────
# PATH HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _rg_label(reward_group) -> str:
    """Map numeric reward-group code to a short directory-safe string."""
    return {1: "Rplus", 0: "Rminus", 2: "Rplus_proba"}.get(int(reward_group), f"rg{reward_group}")


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


def split_data_dir(cfg, split_idx) -> Path:
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
    ll_null_test = _null_log_likelihood(output_test, p_lick=p_lick_train)
    n_train = sum(len(o) for o in output_train)
    n_test = sum(len(o) for o in output_test)
    bpt_train = _bits_per_trial(ll_train, ll_null_train, n_train)
    bpt_test = _bits_per_trial(ll_test, ll_null_test, n_test)

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
    #valid_mice = ['AB131','AB085']

    nwb_list = []
    for mouse in valid_mice:
        for name in all_nwb:
            if mouse not in name:
                continue
            fpath = str(cfg["nwb_root"] / name)
            try:
                btype, _ = NWB_read.get_bhv_type_and_training_day_index(fpath)
                if btype == "whisker":
                    nwb_list.append(fpath)
            except Exception as exc:
                logger.warning(f"  Skipping {name}: {exc}")

    logger.info(f"Found {len(nwb_list)} whisker NWB files from {len(valid_mice)} mice.")

    # ── Build dataset ─────────────────────────────────────────────────────────
    params = dict(n_trials_max=cfg["n_trials_max"])
    dataset = create_behavior_dataset(nwb_list, mouse_info_df, cfg["n_trials_max"])
    print('Dataset created')

    # TODO: remove
    #dataset = dataset[dataset.whisker==1]

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

        k1_file = (global_model_dir(cfg, split_idx, 1, best_inst, model_name, reward_group)
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
    print('SUMMARY df', summary_df.columns, summary_df.split_idx.unique(), summary_df.n_states.unique(), summary_df.model_name.unique(), summary_df.reward_group.unique(), summary_df.instance_idx.unique())
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
    (mouse_id, split_idx, n_states,
     model_name, features, cfg, reward_group) = args

    # ── Early exit: skip entirely if this mouse has no data for this reward group
    # This prevents result folders being created for the wrong reward group.
    data_train_all, data_test_all = load_split(cfg, split_idx)
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

    # ── Best global instance for this (split, K, model, rg) ──────────────────
    best_ll, best_inst = -np.inf, 0
    for inst in range(cfg["n_instances"]):
        f = (global_model_dir(cfg, split_idx, n_states, inst, model_name, reward_group)
             / "global_fit_glmhmm_results.npz")
        if split_idx != 0:
            print('Loading split', split_idx, 'from', f)
            # Assert it exists
            assert f.exists(), f"Expected global result file not found: {f}"
        #if not f.exists():
        #    continue
        res_ll = np.load(f, allow_pickle=True)["arr_0"].item().get("ll_test", -np.inf)
        if res_ll > best_ll:
            best_ll, best_inst = res_ll, inst

    global_file = (global_model_dir(cfg, split_idx, n_states, best_inst, model_name, reward_group)
                   / "global_fit_glmhmm_results.npz")

    if not global_file.exists():
        logger.warning(f"  [skip – no global result] {tag}")
        return None

    global_w = np.load(global_file, allow_pickle=True)["arr_0"].item()["weights"]

    out_dir = single_model_dir(cfg, mouse_id, split_idx, n_states,
                               best_inst, model_name, reward_group)
    result_file = out_dir / "fit_glmhmm_results.npz"

    #if result_file.exists():
    #    logger.info(f"  [skip] {tag} inst={best_inst}")
    #    return np.load(result_file, allow_pickle=True)["arr_0"].item()

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Mouse + reward-group filtered data ────────────────────────────────────
    data_train_all, data_test_all = load_split(cfg, split_idx)
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
    data.to_hdf(out_dir / "data_preds.h5", key="data", mode="w")

    result_dict = dict(
        mouse_id=mouse_id,
        split_idx=split_idx,
        n_states=n_states,
        instance_idx=best_inst,
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
        f"  {tag} inst={best_inst}  "
        f"acc_train={metrics['acc_train']:.3f} (bal={metrics['balanced_acc_train']:.3f})  "
        f"acc_test={metrics['acc_test']:.3f} (bal={metrics['balanced_acc_test']:.3f})"
    )
    return result_dict


def _make_single_tasks(cfg, k_states, feature_sets, mouse_ids):
    return [
        (mouse_id, split_idx, k_state, model_name, features, cfg, reward_group)
        for k_state, split_idx, mouse_id, (model_name, features), reward_group
        in product(
            k_states,
            range(cfg["n_splits"]),
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

    all_k = sorted(
        [n_states_to_fit] if n_states_to_fit is not None else cfg["n_states_list"]
    )

    data_train, _ = load_split(cfg, 0)
    mouse_ids = data_train["mouse_id"].unique().tolist()

    mouse_ids = ['AB130', 'AB131']

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
    print(summary_df.groupby(["split_idx", "n_states", "mouse_id"]).size())

    print('SUMMARY SINGLE df', summary_df.columns)
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
    print('HERE 1', df.columns)

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
    row_metrics = [#todo add bpt if available
        ("ll_train_final",                "ll_test",                    "Log-likelihood"),
        ("predictive_acc_train",          "predictive_acc_test",        "Predictive accuracy"),
        ("balanced_predictive_acc_train", "balanced_predictive_acc_test","Balanced accuracy"),
        #("bpt_train", "bpt_test", "Bits / trial"),
    ]
    if has_bpt:
        row_metrics.append(("bpt_train", "bpt_test", "Bits / trial"))

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
             "balanced_predictive_acc_train", "balanced_predictive_acc_test"]
        ]
        .mean()
        .reset_index()
    )

    k_vals = sorted(agg["n_states"].unique())

    metrics = [
        ("ll_test",                       "Test log-likelihood"),
        ("predictive_acc_test",           "Predictive accuracy (test)"),
        ("balanced_predictive_acc_test",  "Balanced accuracy (test)"),
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

            # Group mean ± SD
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
    """
    rows = []
    for rg_int in cfg["reward_groups"]:
        rg = _RG_STR.get(int(rg_int), str(rg_int))
        feature_sets = build_feature_sets(cfg["features"], cfg.get("trial_types"))
        for model_name in feature_sets:
            for n_states in cfg["n_states_list"]:
                for split_idx in range(cfg["n_splits"]):
                    for instance_idx in range(cfg["n_instances"]):
                        f = (global_model_dir(cfg, split_idx, n_states,
                                              instance_idx, model_name, rg_int)
                             / "global_fit_glmhmm_results.npz")
                        if not f.exists():
                            continue
                        res = np.load(f, allow_pickle=True)["arr_0"].item()
                        w       = np.array(res["weights"])   # (K, C-1, M)
                        feats   = list(res.get("features", cfg["features"]))
                        for s in range(w.shape[0]):
                            for fi, feat in enumerate(feats):
                                rows.append(dict(
                                    model_name=model_name,
                                    reward_group=rg,
                                    n_states=n_states,
                                    split_idx=split_idx,
                                    instance_idx=instance_idx,
                                    state_idx=s,
                                    feature=feat,
                                    weight=float(w[s, 0, fi]),
                                ))
    return pd.DataFrame(rows)


def _load_single_weights_long(cfg) -> pd.DataFrame:
    """
    Walk every single-mouse NPZ result file and return a long-form DataFrame
    with the same schema as _load_global_weights_long plus mouse_id.
    Only one fit per (mouse, split, model, rg, K) is stored (best global inst).
    """
    rows = []
    feature_sets = build_feature_sets(cfg["features"], cfg.get("trial_types"))
    for rg_int in cfg["reward_groups"]:
        rg = _RG_STR.get(int(rg_int), str(rg_int))
        for model_name in feature_sets:
            for n_states in cfg["n_states_list"]:
                for split_idx in range(cfg["n_splits"]):
                    # find which instance was used (best global)
                    best_ll, best_inst = -np.inf, 0
                    for inst in range(cfg["n_instances"]):
                        gf = (global_model_dir(cfg, split_idx, n_states,
                                               inst, model_name, rg_int)
                              / "global_fit_glmhmm_results.npz")
                        if not gf.exists():
                            continue
                        ll = np.load(gf, allow_pickle=True)["arr_0"].item().get("ll_test", -np.inf)
                        if ll > best_ll:
                            best_ll, best_inst = ll, inst

                    # now iterate mice
                    single_base = cfg["single_path"] / _rg_label(rg_int) / model_name
                    if not single_base.exists():
                        continue
                    for mouse_dir in single_base.iterdir():
                        mouse_id = mouse_dir.name
                        f = (single_model_dir(cfg, mouse_id, split_idx, n_states,
                                              best_inst, model_name, rg_int)
                             / "fit_glmhmm_results.npz")
                        if not f.exists():
                            continue
                        res   = np.load(f, allow_pickle=True)["arr_0"].item()
                        w     = np.array(res["weights"])   # (K, C-1, M)
                        feats = list(res.get("features", cfg["features"]))
                        for s in range(w.shape[0]):
                            for fi, feat in enumerate(feats):
                                rows.append(dict(
                                    mouse_id=mouse_id,
                                    model_name=model_name,
                                    reward_group=rg,
                                    n_states=n_states,
                                    split_idx=split_idx,
                                    instance_idx=best_inst,
                                    state_idx=s,
                                    feature=feat,
                                    weight=float(w[s, 0, fi]),
                                ))
    return pd.DataFrame(rows)


def _load_single_trial_data(cfg, model_name: str, n_states: int) -> pd.DataFrame:
    """
    Concatenate all data_preds.h5 files for a given (model_name, n_states),
    choosing the best split (highest test LL) for each mouse.
    Returns a DataFrame with trial-level data including posterior_state_* cols.
    """
    feature_sets = build_feature_sets(cfg["features"], cfg.get("trial_types"))
    dfs = []
    for rg_int in cfg["reward_groups"]:
        rg = _RG_STR.get(int(rg_int), str(rg_int))
        single_base = cfg["single_path"] / _rg_label(rg_int) / model_name
        if not single_base.exists():
            continue
        for mouse_dir in single_base.iterdir():
            mouse_id = mouse_dir.name
            # Find split with highest test LL for this mouse/model/rg/K
            best_ll, best_split, best_inst = -np.inf, 0, 0
            for split_idx in range(cfg["n_splits"]):
                for inst in range(cfg["n_instances"]):
                    f = (single_model_dir(cfg, mouse_id, split_idx, n_states,
                                         inst, model_name, rg_int)
                         / "fit_glmhmm_results.npz")
                    if not f.exists():
                        continue
                    ll = np.load(f, allow_pickle=True)["arr_0"].item().get("ll_test", -np.inf)
                    if ll > best_ll:
                        best_ll, best_split, best_inst = ll, split_idx, inst

            h5 = (single_model_dir(cfg, mouse_id, best_split, n_states,
                                   best_inst, model_name, rg_int)
                  / "data_preds.h5")
            if not h5.exists():
                continue
            try:
                df = pd.read_hdf(h5)
                df["mouse_id"]     = mouse_id
                df["reward_group"] = rg
                df["n_states"]     = n_states
                df["model_name"]   = model_name
                # dominant state: argmax of posterior columns
                post_cols = [c for c in df.columns if c.startswith("posterior_state_")]
                if post_cols:
                    df["dominant_state"] = df[post_cols].values.argmax(axis=1)
                dfs.append(df)
            except Exception as e:
                logger.warning(f"  Could not load {h5}: {e}")
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS 1 – Per-mouse weight diagnostic: splits × states grid
# ─────────────────────────────────────────────────────────────────────────────

def plot_mouse_weight_splits(cfg, figure_path: Path, model_name: str = "full"):
    """
    For each mouse, one figure per (model_name, n_states, reward_group) showing
    a grid of weight plots:  rows = data splits,  cols = latent states.
    This lets you visually verify that weights are consistent across splits.
    """
    from plotting_utils import remove_top_right_frame, save_figure_to_files

    out_base = figure_path / "weight_diagnostics" / model_name
    feature_sets = build_feature_sets(cfg["features"], cfg.get("trial_types"))
    feats = feature_sets.get(model_name, cfg["features"])

    for rg_int in cfg["reward_groups"]:
        rg    = _RG_STR.get(int(rg_int), str(rg_int))
        color = _RG_COLOR.get(rg, "k")

        single_base = cfg["single_path"] / _rg_label(rg_int) / model_name
        if not single_base.exists():
            continue

        for mouse_dir in sorted(single_base.iterdir()):
            mouse_id = mouse_dir.name

            for n_states in sorted(cfg["n_states_list"]):
                # collect one weight array per split
                split_weights = {}   # split_idx → ndarray (K, M)
                for split_idx in range(cfg["n_splits"]):
                    # find best instance for this split
                    best_ll, best_inst = -np.inf, 0
                    for inst in range(cfg["n_instances"]):
                        gf = (global_model_dir(cfg, split_idx, n_states,
                                               inst, model_name, rg_int)
                              / "global_fit_glmhmm_results.npz")
                        if not gf.exists():
                            continue

                        ll = np.load(gf, allow_pickle=True)["arr_0"].item().get("ll_test", -np.inf)
                        if ll > best_ll:
                            best_ll, best_inst = ll, inst

                    # Single model
                    f = (single_model_dir(cfg, mouse_id, split_idx, n_states,
                                         best_inst, model_name, rg_int)
                         / "fit_glmhmm_results.npz")
                    if not f.exists():
                        print(f"  No weights for {mouse_id} split {split_idx} (file not found: {f})")
                        continue
                    res = np.load(f, allow_pickle=True)["arr_0"].item()
                    split_weights[split_idx] = np.array(res["weights"])[:, 0, :]  # (K, M)

                if not split_weights:
                    continue

                splits_present = sorted(split_weights.keys())
                n_rows = len(splits_present)
                n_cols = n_states

                fig, axs = plt.subplots( n_rows, n_cols, figsize=(2.5 * n_cols, 2.5 * n_rows), dpi=200,
                                         constrained_layout=True, squeeze=False)

                for row_i, split_idx in enumerate(splits_present):
                    w = split_weights[split_idx]   # (K, M)
                    for col_i in range(n_states):
                        ax = axs[row_i, col_i]
                        print("feats:", len(feats))
                        print("weights:", w.shape)
                        ax.bar(range(len(feats)), w[col_i], color=color, alpha=0.75, width=0.6)
                        ax.axhline(0, color="k", lw=0.6, ls="--")
                        ax.set_xticks(range(len(feats)))
                        ax.set_xticklabels(feats, rotation=45, ha="right", fontsize=6)
                        ax.set_ylabel("Weight" if col_i == 0 else "", fontsize=7)
                        ax.set_title(
                            f"Split {split_idx} – State {col_i+1}" if row_i == 0
                            else f"Split {split_idx} – St.{col_i+1}",
                            fontsize=7,
                        )
                        remove_top_right_frame(ax)

                fig.suptitle(
                    f"{mouse_id} | {model_name} | K={n_states} | {rg}",
                    fontsize=9,
                )

                out_dir = out_base / rg / mouse_id
                out_dir.mkdir(parents=True, exist_ok=True)
                save_figure_to_files(
                    fig=fig, save_path=str(out_dir),
                    file_name=f"weight_splits_K{n_states}",
                    suffix=None, file_types=["pdf", "png"], dpi=200,
                )
                plt.close()

    logger.info(f"  [1] Per-mouse weight diagnostics → {out_base}")


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS 2 – Average posterior probability curve across trials, per day
# ─────────────────────────────────────────────────────────────────────────────

def plot_posterior_curves_by_day(cfg, figure_path: Path, model_name: str = "full",
                                 max_trials: int = 300):
    """
    For each (reward_group, training day) and each K in n_states_list, plot the
    trial-averaged posterior probability of every state on a single axis, with
    the inter-mouse standard deviation shown as a shaded band.

    Each session is aligned to trial index 0 independently (cumcount per
    session_id), and only the first `max_trials` trials are shown.

    Layout: one figure per (reward_group, day, K).
    All states overlaid on one axis; mean = solid line, ±1 SD = shaded band.
    """
    from plotting_utils import remove_top_right_frame, save_figure_to_files

    out_base = figure_path / "posterior_curves" / model_name

    for n_states in sorted(cfg["n_states_list"]):
        trial_df = _load_single_trial_data(cfg, model_name, n_states)
        if trial_df.empty:
            continue

        post_cols = sorted(
            [c for c in trial_df.columns if c.startswith("posterior_state_")],
            key=lambda c: int(c.split("_")[-1]),
        )
        if not post_cols:
            continue

        state_colors = sns.color_palette("tab10", n_states)

        # ── Align every session independently to trial index 0, cap at max_trials
        trial_df["trial_in_session"] = (
            trial_df.groupby(["mouse_id", "session_id"]).cumcount()
        )
        trial_df = trial_df[trial_df["trial_in_session"] < max_trials]

        rgs = sorted(trial_df["reward_group"].unique())
        days = sorted(trial_df["day"].unique()) if "day" in trial_df.columns else [None]

        for rg in rgs:
            for day in days:
                sub = trial_df[trial_df["reward_group"] == rg]
                if day is not None:
                    sub = sub[sub["day"] == day]
                if sub.empty:
                    continue

                # ── Per-mouse mean at each trial position, then group stats
                # Step 1: average within each mouse across sessions of the same day
                mouse_avg = (
                    sub.groupby(["mouse_id", "trial_in_session"])[post_cols]
                    .mean()
                    .reset_index()
                )
                # Step 2: mean and std across mice at each trial position
                grp = mouse_avg.groupby("trial_in_session")[post_cols]
                means = grp.mean()
                stds = grp.std(ddof=1).fillna(0)
                sems = grp.sem(ddof=1).fillna(0)

                trial_idx = means.index.values

                # ── Single-axis figure, all states overlaid
                fig, ax = plt.subplots(figsize=(7, 3.5), dpi=200,
                                       constrained_layout=True)

                # Plot uniform prior baseline
                ax.axhline(1 / n_states, color="grey", lw=0.6, ls="--", label="uniform prior")

                for s_i, pcol in enumerate(post_cols):
                    color = state_colors[s_i]
                    mean = means[pcol].values
                    err = sems[pcol].values
                    ax.plot(trial_idx, mean,
                            color=color, lw=1.8, label=f"State {s_i + 1}")
                    ax.fill_between(trial_idx,
                                    mean - err, mean + err,
                                    color=color, alpha=0.15)

                ax.set_xlim(0, max_trials - 1)
                ax.set_ylim(-0.02, 1.02)
                ax.set_xlabel("Trial in session", fontsize=9)
                ax.set_ylabel("p(state)", fontsize=9)
                ax.legend(frameon=False, fontsize=7, loc="upper right")
                remove_top_right_frame(ax)

                day_str = f"day{day}" if day is not None else "alldays"
                n_mice = mouse_avg["mouse_id"].nunique()
                fig.suptitle(
                    f"Posterior | {model_name} K={n_states} | {rg} | {day_str}"
                    f"  (n={n_mice} mice, first {max_trials} trials)",
                    fontsize=9,
                )

                out_dir = out_base / rg / day_str
                out_dir.mkdir(parents=True, exist_ok=True)
                save_figure_to_files(
                    fig=fig, save_path=str(out_dir),
                    file_name=f"posterior_curve_K{n_states}",
                    suffix=None, file_types=["pdf", "png"], dpi=200,
                )
                plt.close()

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
                squeeze=False,
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
                ax.set_xticklabels(feats, rotation=45, ha="right", fontsize=6)
                remove_top_right_frame(ax)
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


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS 4 – Single-mouse weights per state, hue = reward_group
# ─────────────────────────────────────────────────────────────────────────────

def plot_single_weights_by_rg(cfg, figure_path: Path):
    """
    For each model_type and each K, plot one subplot per state showing the
    within-mouse-averaged weights (mean across splits) with hue = reward_group.

    Each dot is one mouse; the thick line is the group mean±SE.
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
                figsize=(3.5 * n_states, 3.5),
                dpi=250, constrained_layout=True,
                squeeze=False,
            )

            for s_i in range(n_states):
                ax   = axs[0, s_i]
                data = sub[sub["state_idx"] == s_i]

                # individual mouse dots (strip)
                sns.stripplot(
                    data=data, x="feature", y="weight",
                    order=feats, hue="reward_group",
                    hue_order=rg_order, palette=rg_palette,
                    ax=ax, dodge=True, size=3, alpha=0.4, jitter=True,
                    legend=False,
                )
                # group mean ± SE on top
                sns.pointplot(
                    data=data, x="feature", y="weight",
                    order=feats, hue="reward_group",
                    hue_order=rg_order, palette=rg_palette,
                    estimator=np.mean, errorbar="se",
                    ax=ax, dodge=True, markers="D",
                    markersize=4, lw=1.2,
                    legend=(s_i == 0),
                )

                ax.axhline(0, color="grey", lw=0.6, ls="--")
                ax.set_title(f"State {s_i+1}", fontsize=8)
                ax.set_xlabel("")
                ax.set_ylabel("Weight" if s_i == 0 else "", fontsize=8)
                ax.set_xticklabels(feats, rotation=45, ha="right", fontsize=6)
                remove_top_right_frame(ax)
                if s_i == 0 and ax.legend_:
                    ax.legend(frameon=False, fontsize=7, title="Reward group",
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

    logger.info(f"  [4] Single-mouse weights by reward group → {out_base}")


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS 5 – Lick rate per trial type × state, for each reward group
# ─────────────────────────────────────────────────────────────────────────────

def plot_lick_rate_per_state(cfg, figure_path: Path, model_name: str = "full"):
    """
    For each model_name and K, compute the mouse-average lick rate (lick_flag
    mean) broken down by (dominant state, trial type, reward_group).

    Dominant state = argmax of posterior_state_* for each trial.

    Layout: one figure per (model_name, K).
    Rows = reward groups,  cols = states.
    x-axis = trial type,   y-axis = lick rate,  each mouse shown as a thin line.
    """
    from plotting_utils import remove_top_right_frame, save_figure_to_files

    out_base = figure_path / "lick_rate_per_state"

    # Trial-type display order and labels
    tt_map   = {-1: "auditory", 0: "no-stim", 1: "whisker"}
    tt_order = ["auditory", "whisker", "no-stim"]

    feature_sets = build_feature_sets(cfg["features"], cfg.get("trial_types"))
    model_names  = list(feature_sets.keys())

    for model_name in model_names:
        for n_states in sorted(cfg["n_states_list"]):
            trial_df = _load_single_trial_data(cfg, model_name, n_states)
            if trial_df.empty:
                continue

            # map numeric stimulus_type to label
            if "stimulus_type" in trial_df.columns:
                trial_df["trial_type_str"] = trial_df["stimulus_type"].map(tt_map).fillna("other")
            else:
                continue

            # choice column may be called "choice" or "lick_flag"
            lick_col = "lick_flag" if "lick_flag" in trial_df.columns else "choice"

            rgs_present = [rg for rg in ["R+", "R-", "R+proba"]
                           if rg in trial_df["reward_group"].unique()]

            if not rgs_present:
                continue

            fig, axs = plt.subplots(
                len(rgs_present), n_states,
                figsize=(3.2 * n_states, 3.0 * len(rgs_present)),
                dpi=250, constrained_layout=True,
                squeeze=False,
            )

            for row_i, rg in enumerate(rgs_present):
                rg_color = _RG_COLOR.get(rg, "steelblue")
                sub_rg   = trial_df[trial_df["reward_group"] == rg]

                for col_i in range(n_states):
                    ax      = axs[row_i, col_i]
                    sub_s   = sub_rg[sub_rg["dominant_state"] == col_i]

                    if sub_s.empty:
                        ax.set_visible(False)
                        continue

                    # per-mouse mean lick rate per trial type in this state
                    mouse_means = (
                        sub_s.groupby(["mouse_id", "trial_type_str"])[lick_col]
                        .mean()
                        .reset_index()
                    )

                    # individual mouse lines (thin)
                    for _, mdf in mouse_means.groupby("mouse_id"):
                        mdf_ord = (
                            mdf.set_index("trial_type_str")
                            .reindex(tt_order)
                            .dropna()
                        )
                        ax.plot(
                            range(len(mdf_ord)),
                            mdf_ord[lick_col].values,
                            color=rg_color, alpha=0.25, lw=0.8,
                        )

                    # group mean ± SE
                    grp = (
                        mouse_means.groupby("trial_type_str")[lick_col]
                        .agg(["mean", "sem"])
                        .reindex(tt_order)
                        .dropna()
                        .reset_index()
                    )
                    ax.errorbar(
                        range(len(grp)),
                        grp["mean"], yerr=grp["sem"],
                        color=rg_color, lw=2, marker="o", capsize=3,
                        markeredgecolor="white", zorder=5,
                    )

                    ax.set_xticks(range(len(grp)))
                    ax.set_xticklabels(grp["trial_type_str"], rotation=30,
                                       ha="right", fontsize=7)
                    ax.set_ylim(-0.05, 1.05)
                    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
                    ax.set_ylabel("Lick rate" if col_i == 0 else "", fontsize=8)
                    ax.set_title(
                        f"{rg} – State {col_i+1}" if row_i == 0
                        else f"State {col_i+1}",
                        fontsize=8,
                    )
                    ax.axhline(0.5, color="grey", lw=0.6, ls="--")
                    remove_top_right_frame(ax)

            fig.suptitle(
                f"Lick rate per trial type & state | {model_name} | K={n_states}",
                fontsize=9,
            )

            out_dir = out_base / model_name
            out_dir.mkdir(parents=True, exist_ok=True)
            save_figure_to_files(
                fig=fig, save_path=str(out_dir),
                file_name=f"lick_rate_K{n_states}",
                suffix=None, file_types=["pdf", "eps"], dpi=250,
            )
            plt.close()

    logger.info(f"  [5] Lick rate per state → {out_base}")


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

    figure_path = cfg.get("figure_path",
                          cfg["global_path"].parent / "figures")
    figure_path = Path(figure_path)
    figure_path.mkdir(parents=True, exist_ok=True)


    # ── Global performance ────────────────────────────────────────────────────
    plot_global_perf = False
    if plot_global_perf:
        try:
            df_global = _prep_global_df(cfg)
            df_global = _compute_bpt(df_global)

            # All feature sets together
            model_names = sorted(df_global["model_type"].unique(),
                                 key=lambda x: (x != "full", x))
            plot_global_performance(df_global, figure_path, model_subset=model_names)

            # LOO subsets: full + each individual LOO model
            #loo_models = [m for m in model_names if m.startswith("loo_") and "_block" not in m]
            #if loo_models:
            #    plot_global_performance(df_global, figure_path / "loo_single",
            #                            model_subset=["full"] + loo_models)

            #block_models = [m for m in model_names if m.startswith("loo_") and "_block" in m]
            #if block_models:
            ##    plot_global_performance(df_global, figure_path / "loo_block",
            #                            model_subset=["full"] + block_models)

        except FileNotFoundError as e:
            logger.warning(f"  Skipping global performance plots: {e}")

    # ── Per-mouse performance ─────────────────────────────────────────────────
    plot_single_mouse_perf = False
    if plot_single_mouse_perf:
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

    feature_sets = build_feature_sets(cfg["features"], cfg.get("trial_types"))

    # ── Analysis 1 – per-mouse weight diagnostic grid ─────────────────────────
    plot_per_mouse_weight_grid = True
    if plot_per_mouse_weight_grid:
        for mname in feature_sets:
            try:
                plot_mouse_weight_splits(cfg, figure_path, model_name=mname)
            except Exception as e:
                logger.warning(f"  [1] weight diagnostics failed for {mname}: {e}")

    # ── Analysis 2 – posterior curves by day ──────────────────────────────────
    plot_mean_posteriors = False
    if plot_mean_posteriors:
        for mname in feature_sets:
            try:
                plot_posterior_curves_by_day(cfg, figure_path, model_name=mname)
            except Exception as e:
                logger.warning(f"  [2] posterior curves failed for {mname}: {e}")

    # ── Analysis 3 – global weights, hue = reward_group ───────────────────────
    plot_global_weights_per_group = False
    if plot_global_weights_per_group:
        try:
            plot_global_weights_by_rg(cfg, figure_path)
        except Exception as e:
            logger.warning(f"  [3] global weights by rg failed: {e}")

    # ── Analysis 4 – single-mouse weights, hue = reward_group ─────────────────
    plot_single_weights_per_group=False
    if plot_single_weights_per_group:
        try:
            plot_single_weights_by_rg(cfg, figure_path)
        except Exception as e:
            logger.warning(f"  [4] single weights by rg failed: {e}")

    # ── Analysis 5 – lick rate per trial type × state ─────────────────────────
    plot_beh_perf_in_states=False
    if plot_beh_perf_in_states:
        for mname in feature_sets:
            try:
                plot_lick_rate_per_state(cfg, figure_path, model_name=mname)
            except Exception as e:
                logger.warning(f"  [5] lick rate per state failed for {mname}: {e}")

    logger.info(f"Stage 4 complete. Figures → {figure_path}\n")


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
    if run_global:  stage_fit_global(cfg)
    if run_single:  stage_fit_single(cfg, n_states_to_fit=args.single_n_states)
    if run_plot:    stage_plot_performance(cfg, model_name_for_per_mouse=args.plot_model)
    logger.info(f"Pipeline finished in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
