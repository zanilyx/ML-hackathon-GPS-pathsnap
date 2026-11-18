"""Optimised PathSnap XGBoost + HMM hyperparameter trainer.

Run from the repository root after make_features.py:
    python -u train_tune_hmm_optimized.py

Optional CMD settings:
    set PATHSNAP_OPTUNA_TRIALS=10
    set PATHSNAP_DEVICE=cuda
    set PATHSNAP_N_JOBS=8
    set PATHSNAP_MAX_HOPS=5
    set PATHSNAP_GPU_PREDICT=1

The student-test split is reported only. Model selection uses grouped CV.
"""

from __future__ import annotations

import json
import os
import pickle
import time
import warnings
import uuid
from datetime import datetime, timezone
from collections import deque
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import GroupKFold

from common import FEATURES, SEED, route_continuity, top1_and_miss

# ----------------------------- Configuration -----------------------------
OUT = Path("outputs")
MODELS = OUT / "models"
OUT.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

FEATURES_FILE = OUT / "features.pkl"
if not FEATURES_FILE.exists():
    raise FileNotFoundError("outputs/features.pkl not found. Run: python make_features.py")

with FEATURES_FILE.open("rb") as f:
    CACHE = pickle.load(f)

TRAIN = CACHE["train"].copy()
STUDENT_TEST = CACHE["student_test"].copy()
NET = CACHE["net"]
X = TRAIN[FEATURES].to_numpy(dtype=np.float32, copy=True)
Y = TRAIN["label"].to_numpy(dtype=np.int32, copy=True)
GROUPS = TRAIN["trip"].to_numpy()
SPLITS = list(GroupKFold(n_splits=5).split(X, Y, GROUPS))

N_JOBS = max(1, int(os.environ.get("PATHSNAP_N_JOBS", "8")))
# Every launch gets a new run by default. Set PATHSNAP_RUN_ID to resume that run.
RUNS_DIR = OUT / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)
RUN_ID = os.environ.get("PATHSNAP_RUN_ID") or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
RUN_DIR = RUNS_DIR / RUN_ID
RUN_DIR.mkdir(parents=True, exist_ok=True)
STATUS_FILE = RUN_DIR / "status.json"
DB_FILE = RUN_DIR / "optuna.sqlite3"
TRIALS_FILE = RUN_DIR / "trials.csv"
EVENTS_FILE = RUN_DIR / "events.jsonl"
RUN_MODEL_FILE = RUN_DIR / "selected_model.joblib"
REQUESTED_DEVICE = os.environ.get("PATHSNAP_DEVICE", "cuda").lower()
MAX_HOPS_DEFAULT = max(1, int(os.environ.get("PATHSNAP_MAX_HOPS", "5")))
GPU_PREDICT = os.environ.get("PATHSNAP_GPU_PREDICT", "1") == "1"

BASELINE = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.1,
    "tree_method": "hist",
    "device": REQUESTED_DEVICE,
    "eval_metric": "logloss",
    "random_state": SEED,
    "n_jobs": N_JOBS,
}

# Optional CuPy path: used only for prediction. Training remains XGBoost CUDA.
cp = None
if GPU_PREDICT:
    try:
        import cupy as _cp
        cp = _cp
    except Exception:
        cp = None

DEVICE = "cpu"


def choose_device() -> None:
    """Test the installed XGBoost CUDA build before committing to CUDA."""
    global DEVICE, BASELINE
    if REQUESTED_DEVICE != "cuda":
        DEVICE = "cpu"
    else:
        try:
            probe = xgb.XGBClassifier(
                n_estimators=2,
                max_depth=2,
                tree_method="hist",
                device="cuda",
                eval_metric="logloss",
                n_jobs=1,
                random_state=SEED,
            )
            probe.fit(X[: min(512, len(X))], Y[: min(512, len(Y))], verbose=False)
            DEVICE = "cuda"
        except Exception as exc:
            print(f"[warning] CUDA unavailable; using CPU: {exc}", flush=True)
            DEVICE = "cpu"
    BASELINE["device"] = DEVICE
    print(f"XGBoost device: {DEVICE}; CPU threads: {N_JOBS}", flush=True)
    if DEVICE == "cuda" and cp is None:
        print("GPU prediction: disabled (CuPy not installed); XGBoost prediction will use NumPy inputs.", flush=True)
    elif DEVICE == "cuda":
        try:
            cp.cuda.runtime.getDeviceCount()
            print("GPU prediction: enabled with CuPy", flush=True)
        except Exception as exc:
            print(f"[warning] CuPy CUDA unavailable; using NumPy prediction: {exc}", flush=True)
            globals()["cp"] = None


def predict_positive_probability(model: xgb.XGBClassifier, features: np.ndarray) -> np.ndarray:
    """Predict class-1 probabilities, using GPU-resident input when available."""
    if DEVICE == "cuda" and cp is not None:
        try:
            gpu_x = cp.asarray(features, dtype=cp.float32)
            gpu_p = model.get_booster().inplace_predict(gpu_x)
            return cp.asnumpy(gpu_p).astype(np.float64, copy=False)
        except Exception as exc:
            warnings.warn(f"GPU inplace prediction failed; using normal predict_proba: {exc}")
    return model.predict_proba(features)[:, 1].astype(np.float64, copy=False)


# --------------------------- HMM graph acceleration -----------------------
# Road network adjacency is immutable during this run. Build it once per trip.
_ADJ_CACHE: dict[str, list[list[int]]] = {}
# Cache bounded-hop BFS results by (trip, source-node, max-hops).
_HOP_CACHE: dict[tuple[str, int, int], dict[int, int]] = {}


def get_adjacency(trip_id: str, arcs: np.ndarray) -> list[list[int]]:
    key = str(trip_id)
    if key not in _ADJ_CACHE:
        if len(arcs) == 0:
            _ADJ_CACHE[key] = []
        else:
            max_node = int(np.max(arcs[:, :2]))
            adjacency = [[] for _ in range(max_node + 1)]
            for u, v in arcs[:, :2]:
                u, v = int(u), int(v)
                if 0 <= u < len(adjacency):
                    adjacency[u].append(v)
            _ADJ_CACHE[key] = adjacency
    return _ADJ_CACHE[key]


def reachable_within_hops(
    trip_id: str, source: int, adjacency: list[list[int]], max_hops: int
) -> dict[int, int]:
    """Return {node: shortest directed edge count} up to max_hops; cached."""
    key = (str(trip_id), int(source), int(max_hops))
    cached = _HOP_CACHE.get(key)
    if cached is not None:
        return cached

    distances = {int(source): 0}
    queue = deque([int(source)])
    while queue:
        node = queue.popleft()
        depth = distances[node]
        if depth >= max_hops or node < 0 or node >= len(adjacency):
            continue
        for nxt in adjacency[node]:
            if nxt not in distances:
                distances[nxt] = depth + 1
                queue.append(nxt)
    _HOP_CACHE[key] = distances
    return distances


def build_transition_matrix(
    prev_arc_ids: np.ndarray,
    curr_arc_ids: np.ndarray,
    arcs: np.ndarray,
    trip_id: str,
    adjacency: list[list[int]],
    hop_penalty: float,
    disconnected_penalty: float,
    max_hops: int,
) -> np.ndarray:
    """Build transitions with one cached BFS per unique previous-end node.

    Arc IDs are row indices into the trip's directed .arcs array. A transition
    goes from previous arc's end node to current arc's start node.
    """
    prev_ids = np.asarray(prev_arc_ids, dtype=np.int64)
    curr_ids = np.asarray(curr_arc_ids, dtype=np.int64)
    n_prev, n_curr = len(prev_ids), len(curr_ids)
    trans = np.full((n_prev, n_curr), -float(disconnected_penalty), dtype=np.float64)

    valid_prev = (prev_ids >= 0) & (prev_ids < len(arcs))
    valid_curr = (curr_ids >= 0) & (curr_ids < len(arcs))
    if not valid_prev.any() or not valid_curr.any():
        trans[prev_ids[:, None] == curr_ids[None, :]] = 0.0
        return trans

    prev_end = np.full(n_prev, -1, dtype=np.int64)
    curr_start = np.full(n_curr, -2, dtype=np.int64)
    prev_end[valid_prev] = arcs[prev_ids[valid_prev], 1].astype(np.int64)
    curr_start[valid_curr] = arcs[curr_ids[valid_curr], 0].astype(np.int64)

    # Group rows by source node so BFS is performed once per unique source.
    for source in np.unique(prev_end[valid_prev]):
        row_idx = np.flatnonzero(prev_end == source)
        distances = reachable_within_hops(trip_id, int(source), adjacency, max_hops)
        # Candidate target nodes are vectorised; missing paths keep the fallback.
        hops = np.fromiter(
            (distances.get(int(target), -1) for target in curr_start),
            dtype=np.int32,
            count=n_curr,
        )
        reachable = hops >= 0
        trans[np.ix_(row_idx, np.flatnonzero(reachable))] = (
            -float(hop_penalty) * (hops[reachable][None, :] + 1)
        )

    # Staying on the same road arc has no transition penalty.
    trans[prev_ids[:, None] == curr_ids[None, :]] = 0.0
    return trans


def observation_groups(frame: pd.DataFrame, scores: np.ndarray):
    work = frame[["trip", "obs", "arc_id", "label"]].copy()
    work["score"] = np.asarray(scores, dtype=np.float64)
    work = work.sort_values(["trip", "obs", "arc_id"], kind="mergesort")
    groups = []
    for (trip_id, obs), group in work.groupby(["trip", "obs"], sort=False):
        groups.append((
            str(trip_id),
            int(obs),
            group["arc_id"].to_numpy(dtype=np.int64),
            group["label"].to_numpy(dtype=np.int8),
            group["score"].to_numpy(dtype=np.float64),
        ))
    return groups


def viterbi_decode(
    frame: pd.DataFrame,
    scores: np.ndarray,
    net: dict,
    hop_penalty: float = 0.7,
    disconnected_penalty: float = 12.0,
    max_hops: int = MAX_HOPS_DEFAULT,
) -> pd.DataFrame:
    groups = observation_groups(frame, scores)
    by_trip: dict[str, list] = {}
    for item in groups:
        by_trip.setdefault(item[0], []).append(item)

    chosen_rows = []
    for trip_id, obs_groups in by_trip.items():
        obs_groups.sort(key=lambda item: item[1])
        arcs_net = np.asarray(net[trip_id])
        adjacency = get_adjacency(trip_id, arcs_net)

        emissions = []
        candidate_arcs = []
        labels = []
        for _, _, arc_ids, y, raw_scores in obs_groups:
            p = np.clip(raw_scores, 1e-7, 1.0 - 1e-7)
            p /= max(float(p.sum()), 1e-12)
            emissions.append(np.log(np.clip(p, 1e-12, 1.0)))
            candidate_arcs.append(arc_ids)
            labels.append(y)

        if not emissions:
            continue
        dp = emissions[0].copy()
        backpointers = []
        for t in range(1, len(obs_groups)):
            trans = build_transition_matrix(
                candidate_arcs[t - 1], candidate_arcs[t], arcs_net, trip_id,
                adjacency, hop_penalty, disconnected_penalty, max_hops,
            )
            values = dp[:, None] + trans
            best_prev = np.argmax(values, axis=0)
            dp = values[best_prev, np.arange(len(candidate_arcs[t]))] + emissions[t]
            backpointers.append(best_prev)

        state = int(np.argmax(dp))
        path = [state]
        for backpointer in reversed(backpointers):
            state = int(backpointer[state])
            path.append(state)
        path.reverse()

        for group, state_idx, y in zip(obs_groups, path, labels):
            _, obs, arc_ids, _, _ = group
            chosen_rows.append({
                "trip": trip_id,
                "obs": obs,
                "arc_id": int(arc_ids[state_idx]),
                "label": int(y[state_idx]),
            })

    if not chosen_rows:
        return pd.DataFrame(columns=["trip", "obs", "arc_id", "label"])
    return pd.DataFrame(chosen_rows).sort_values(["trip", "obs"]).reset_index(drop=True)


def hmm_top1(frame, scores, net, **hmm_params):
    decoded = viterbi_decode(frame, scores, net, **hmm_params)
    score = float(decoded["label"].mean()) if len(decoded) else float("nan")
    return score, decoded


# ------------------------- UI / run-state reporting -------------------------
def write_status(**updates):
    """Atomically update a small JSON status document for a future UI to poll."""
    current = {}
    if STATUS_FILE.exists():
        try:
            current = json.loads(STATUS_FILE.read_text(encoding="utf-8"))
        except Exception:
            current = {}
    current.update(updates)
    current.update({"run_id": RUN_ID, "updated_at": datetime.now(timezone.utc).isoformat()})
    tmp = STATUS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(current, indent=2, default=str), encoding="utf-8")
    tmp.replace(STATUS_FILE)


def log_event(event, **payload):
    record = {"event": event, "time": datetime.now(timezone.utc).isoformat(),
              "run_id": RUN_ID, **payload}
    with EVENTS_FILE.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str) + "\n")


def export_trials(study):
    study.trials_dataframe().to_csv(TRIALS_FILE, index=False)


def optuna_callback(study, trial):
    export_trials(study)
    completed = sum(t.state.name == "COMPLETE" for t in study.trials)
    write_status(
        phase="optimizing", trial_number=trial.number + 1,
        total_trials=study.user_attrs.get("target_trials"),
        completed_trials=completed, best_value=study.best_value if completed else None,
        best_trial=study.best_trial.number if completed else None,
        last_trial_state=trial.state.name,
        last_trial_value=trial.value,
    )
    log_event("trial_finished", trial_number=trial.number, state=trial.state.name,
              value=trial.value, best_value=study.best_value if completed else None)


# ------------------------------- Evaluation -------------------------------
def classifier_cv_scores(params, include_metrics=False, hmm_params=None, label="CV"):
    hmm_params = hmm_params or {
        "hop_penalty": 0.7,
        "disconnected_penalty": 12.0,
        "max_hops": MAX_HOPS_DEFAULT,
    }
    fold_raw, fold_hmm, fold_f1, fold_loss, fold_cont = [], [], [], [], []

    for fold, (train_idx, valid_idx) in enumerate(SPLITS, start=1):
        started = time.perf_counter()
        write_status(phase=label, fold=fold, total_folds=5, detail="fitting XGBoost")
        print(f"    {label}: fold {fold}/5 — fitting XGBoost...", flush=True)
        model = xgb.XGBClassifier(**params)
        model.fit(X[train_idx], Y[train_idx], verbose=False)
        fit_seconds = time.perf_counter() - started

        pred_started = time.perf_counter()
        print(f"    {label}: fold {fold}/5 — predicting + HMM...", flush=True)
        probabilities = predict_positive_probability(model, X[valid_idx])
        valid = TRAIN.iloc[valid_idx].copy()
        raw_top1, _ = top1_and_miss(valid, probabilities)
        decoded_top1, decoded = hmm_top1(valid, probabilities, NET, **hmm_params)
        fold_raw.append(float(raw_top1))
        fold_hmm.append(float(decoded_top1))
        fold_f1.append(float(f1_score(Y[valid_idx], probabilities >= 0.5,
                                      average="macro", zero_division=0)))
        fold_loss.append(float(log_loss(Y[valid_idx],
                                        np.clip(probabilities, 1e-7, 1 - 1e-7),
                                        labels=[0, 1])))
        fold_cont.append(float(route_continuity(decoded, NET)))
        elapsed = time.perf_counter() - pred_started
        print(
            f"    {label}: fold {fold}/5 done | raw={raw_top1:.4f} "
            f"HMM={decoded_top1:.4f} | fit={fit_seconds:.1f}s "
            f"predict+HMM={elapsed:.1f}s",
            flush=True,
        )
        write_status(phase=label, fold=fold, total_folds=5, detail="fold complete",
                     latest_raw_top1=float(raw_top1), latest_hmm_top1=float(decoded_top1),
                     fold_seconds=round(fit_seconds + elapsed, 2))

    if not include_metrics:
        return float(np.mean(fold_hmm))
    return {
        "cv_raw_top1": float(np.mean(fold_raw)),
        "cv_hmm_top1": float(np.mean(fold_hmm)),
        "cv_macro_f1": float(np.mean(fold_f1)),
        "cv_log_loss": float(np.mean(fold_loss)),
        "cv_route_continuity": float(np.nanmean(fold_cont)),
        "fold_hmm_top1": fold_hmm,
    }


def make_params(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 150, 700),
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.25, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.65, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
        "tree_method": "hist",
        "device": DEVICE,
        "eval_metric": "logloss",
        "random_state": SEED,
        "n_jobs": N_JOBS,
    }


def evaluate_test(model, hmm_params=None):
    hmm_params = hmm_params or {
        "hop_penalty": 0.7,
        "disconnected_penalty": 12.0,
        "max_hops": MAX_HOPS_DEFAULT,
    }
    test = STUDENT_TEST.copy()
    test_x = test[FEATURES].to_numpy(dtype=np.float32, copy=False)
    probabilities = predict_positive_probability(model, test_x)
    raw_top1, miss = top1_and_miss(test, probabilities)
    decoded_top1, decoded = hmm_top1(test, probabilities, NET, **hmm_params)
    return {
        "student_test_raw_top1": float(raw_top1),
        "student_test_hmm_top1": float(decoded_top1),
        "student_test_candidate_miss_rate": float(miss),
        "student_test_macro_f1": float(f1_score(
            test.label.to_numpy(), probabilities >= 0.5,
            average="macro", zero_division=0,
        )),
        "student_test_log_loss": float(log_loss(
            test.label.to_numpy(), np.clip(probabilities, 1e-7, 1 - 1e-7), labels=[0, 1]
        )),
        "student_test_route_continuity": float(route_continuity(decoded, NET)),
    }


# ---------------------------------- Main -----------------------------------
def main():
    choose_device()
    trials = max(1, int(os.environ.get("PATHSNAP_OPTUNA_TRIALS", "50")))
    write_status(phase="starting", target_trials=trials, completed_trials=0,
                 device=DEVICE, cpu_threads=N_JOBS, dataset_rows=len(TRAIN),
                 train_trips=int(TRAIN.trip.nunique()), run_dir=str(RUN_DIR))
    log_event("run_started", target_trials=trials, device=DEVICE, cpu_threads=N_JOBS)
    (OUT / "latest_run.txt").write_text(RUN_ID + "\n", encoding="utf-8")
    print(
        f"Training rows={len(TRAIN):,}; trips={TRAIN.trip.nunique()}; "
        f"folds=5; trials={trials}; n_jobs={N_JOBS}",
        flush=True,
    )
    print(f"Run ID: {RUN_ID}", flush=True)
    print(f"Live status: {STATUS_FILE}", flush=True)
    print(f"Persistent Optuna database: {DB_FILE}", flush=True)

    baseline_params = dict(BASELINE)
    baseline_hmm_params = {
        "hop_penalty": 0.7,
        "disconnected_penalty": 12.0,
        "max_hops": MAX_HOPS_DEFAULT,
    }

    print("\nEvaluating baseline XGBoost + HMM...", flush=True)
    baseline_cv = classifier_cv_scores(
        baseline_params, include_metrics=True,
        hmm_params=baseline_hmm_params, label="Baseline",
    )
    write_status(phase="baseline_complete", baseline_cv_hmm_top1=baseline_cv["cv_hmm_top1"])
    log_event("baseline_complete", cv_hmm_top1=baseline_cv["cv_hmm_top1"])
    print(f"Baseline grouped-CV HMM Top-1: {baseline_cv['cv_hmm_top1']:.6f}", flush=True)

    def objective(trial):
        trial_started = time.perf_counter()
        params = make_params(trial)
        hmm_params = {
            "hop_penalty": trial.suggest_float("hmm_hop_penalty", 0.05, 3.0),
            "disconnected_penalty": trial.suggest_float("hmm_disconnected_penalty", 4.0, 25.0),
            "max_hops": trial.suggest_int("hmm_max_hops", 1, 8),
        }
        print(
            f"\n[Trial {trial.number + 1}/{trials}] Starting | "
            f"depth={params['max_depth']} trees={params['n_estimators']} "
            f"lr={params['learning_rate']:.4f} | HMM={hmm_params}",
            flush=True,
        )
        score = classifier_cv_scores(
            params, include_metrics=False, hmm_params=hmm_params,
            label=f"Trial {trial.number + 1}",
        )
        trial.set_user_attr("device", DEVICE)
        trial.set_user_attr("seconds", round(time.perf_counter() - trial_started, 2))
        print(
            f"[Trial {trial.number + 1}/{trials}] HMM CV Top-1={score:.6f} "
            f"| elapsed={time.perf_counter() - trial_started:.1f}s",
            flush=True,
        )
        return score

    study_name = f"pathsnap_{RUN_ID}"
    storage_url = f"sqlite:///{DB_FILE.as_posix()}"
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        study_name=study_name,
        storage=storage_url,
        load_if_exists=True,
    )
    study.set_user_attr("target_trials", trials)
    already_done = sum(t.state.name in ("COMPLETE", "PRUNED", "FAIL") for t in study.trials)
    remaining = max(0, trials - already_done)
    write_status(phase="optimizing", completed_trials=sum(t.state.name == "COMPLETE" for t in study.trials),
                 target_trials=trials, total_trials=trials, study_name=study_name)
    print(f"\nStarting/resuming Optuna optimisation: {already_done}/{trials} trials already recorded", flush=True)
    study.optimize(objective, n_trials=remaining, n_jobs=1, show_progress_bar=True,
                   callbacks=[optuna_callback])

    best_trial_params = dict(study.best_params)
    best_hmm_params = {
        "hop_penalty": best_trial_params.pop("hmm_hop_penalty"),
        "disconnected_penalty": best_trial_params.pop("hmm_disconnected_penalty"),
        "max_hops": best_trial_params.pop("hmm_max_hops"),
    }
    best_params = dict(best_trial_params)
    best_params.update({
        "tree_method": "hist",
        "device": DEVICE,
        "eval_metric": "logloss",
        "random_state": SEED,
        "n_jobs": N_JOBS,
    })

    print("\nBest Optuna parameters:", best_params, flush=True)
    print("Evaluating best parameters with full grouped CV...", flush=True)
    tuned_cv = classifier_cv_scores(
        best_params, include_metrics=True,
        hmm_params=best_hmm_params, label="Tuned final CV",
    )

    print("\nFitting baseline model on all training trips...", flush=True)
    baseline_model = xgb.XGBClassifier(**baseline_params)
    started = time.perf_counter()
    baseline_model.fit(X, Y, verbose=False)
    baseline_seconds = time.perf_counter() - started

    print("Fitting tuned model on all training trips...", flush=True)
    tuned_model = xgb.XGBClassifier(**best_params)
    started = time.perf_counter()
    tuned_model.fit(X, Y, verbose=False)
    tuned_seconds = time.perf_counter() - started

    print("Evaluating student-test split (reporting only)...", flush=True)
    baseline_test = evaluate_test(baseline_model, baseline_hmm_params)
    tuned_test = evaluate_test(tuned_model, best_hmm_params)

    # Select exclusively on grouped CV; do not use student-test for selection.
    use_tuned = tuned_cv["cv_hmm_top1"] >= baseline_cv["cv_hmm_top1"]
    selected = tuned_model if use_tuned else baseline_model
    selected_name = "XGBoost tuned + HMM" if use_tuned else "XGBoost baseline + HMM"
    model_path = RUN_MODEL_FILE
    joblib.dump(selected, model_path)
    # Keep a run-specific result; do not overwrite any older model automatically.

    rows = []
    for name, cv, test, seconds in [
        ("XGBoost baseline + HMM", baseline_cv, baseline_test, baseline_seconds),
        ("XGBoost tuned + HMM", tuned_cv, tuned_test, tuned_seconds),
    ]:
        row = {
            "model": name,
            "selected_for_final_fit": name == selected_name,
            "train_seconds": float(seconds),
        }
        row.update({k: v for k, v in cv.items() if not k.startswith("fold_")})
        row.update(test)
        rows.append(row)

    comparison = pd.DataFrame(rows)
    comparison_path = RUN_DIR / "comparison.csv"
    trials_path = TRIALS_FILE
    config_path = RUN_DIR / "best_config.json"
    comparison.to_csv(comparison_path, index=False)
    study.trials_dataframe().to_csv(trials_path, index=False)

    config = {
        "run_id": RUN_ID,
        "selected_model": selected_name,
        "selection_metric": "5-fold grouped-CV HMM Top-1 accuracy",
        "device": DEVICE,
        "cpu_threads": N_JOBS,
        "gpu_prediction_enabled": bool(DEVICE == "cuda" and cp is not None),
        "n_trials": trials,
        "best_trial_number": int(study.best_trial.number),
        "best_trial_cv_hmm_top1": float(study.best_value),
        "baseline_params": baseline_params,
        "best_optuna_params": best_params,
        "baseline_cv": {k: v for k, v in baseline_cv.items() if not k.startswith("fold_")},
        "tuned_cv": {k: v for k, v in tuned_cv.items() if not k.startswith("fold_")},
        "baseline_student_test": baseline_test,
        "tuned_student_test": tuned_test,
        "hmm": {
            "max_directed_intermediate_hops": int(best_hmm_params["max_hops"]),
            "hop_penalty": float(best_hmm_params["hop_penalty"]),
            "disconnected_transition_penalty": float(best_hmm_params["disconnected_penalty"]),
            "emission": "within-observation-normalized XGBoost positive-class probabilities",
            "transition": "directed shortest-hop connectivity between previous arc end and next arc start",
        },
    }
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    write_status(phase="completed", completed_trials=len(study.trials), total_trials=trials,
                 selected_model=selected_name, best_cv_hmm_top1=float(study.best_value),
                 model_path=str(model_path), comparison_path=str(comparison_path),
                 config_path=str(config_path))
    log_event("run_completed", selected_model=selected_name, best_value=float(study.best_value))
    print("\n================ FINAL COMPARISON ================", flush=True)
    print(comparison.to_string(index=False), flush=True)
    print(f"\nSelected: {selected_name}", flush=True)
    print(f"Saved model: {model_path}", flush=True)
    print(f"Saved comparison: {comparison_path}", flush=True)
    print(f"Saved configuration: {config_path}", flush=True)
    print(f"Saved Optuna trials: {trials_path}", flush=True)


if __name__ == "__main__":
    main()
