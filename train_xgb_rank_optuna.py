"""Compare XGBoost classification with pairwise ranking for per-observation map matching."""
import json
import os
import pickle
import time
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import GroupKFold

from common import FEATURES, SEED, predicted_arcs, route_continuity, top1_and_miss

OUT = Path("outputs")
MODELS = OUT / "models"
MODELS.mkdir(parents=True, exist_ok=True)
with (OUT / "features.pkl").open("rb") as f:
    CACHE = pickle.load(f)
TRAIN, STUDENT_TEST = CACHE["train"], CACHE["student_test"]
NET = CACHE["net"]
X = TRAIN[FEATURES].values
Y = TRAIN.label.values
GROUPS = TRAIN.trip.values
SPLITS = list(GroupKFold(5).split(X, Y, GROUPS))


def select_device():
    try:
        probe_x = np.random.default_rng(SEED).normal(size=(32, len(FEATURES))).astype(np.float32)
        probe_y = np.tile(np.array([0, 1], dtype=np.int32), 16)
        xgb.XGBClassifier(n_estimators=2, tree_method="hist", device="cuda",
                          eval_metric="logloss").fit(probe_x, probe_y)
        return "cuda"
    except Exception as exc:
        print(f"CUDA probe failed; using CPU: {exc}", flush=True)
        return "cpu"


DEVICE = select_device()
print(f"XGBoost device: {DEVICE}", flush=True)
BASELINE = dict(n_estimators=300, max_depth=6, learning_rate=0.1,
                tree_method="hist", device=DEVICE, eval_metric="logloss", random_state=SEED)


def rank_arrays(frame):
    """Sort candidate rows into query groups; one GPS observation is one ranking query."""
    ordered = frame.sort_values(["trip", "segment", "obs"], kind="mergesort")
    sizes = ordered.groupby(["trip", "segment", "obs"], sort=False).size().to_numpy()
    return ordered, ordered[FEATURES].to_numpy(), ordered.label.to_numpy(), sizes


def make_ranker(params):
    return xgb.XGBRanker(objective="rank:pairwise", eval_metric="ndcg",
                         tree_method="hist", device=DEVICE, random_state=SEED, **params)


def fit_predict_ranker(train_frame, valid_frame, params):
    ordered_train, xr, yr, sizes = rank_arrays(train_frame)
    model = make_ranker(params)
    model.fit(xr, yr, group=sizes, verbose=False)
    ordered_valid, xv, _, _ = rank_arrays(valid_frame)
    scores = model.predict(xv)
    return model, ordered_valid, scores


def pooled_top1(frames, scores):
    combined = pd.concat(frames, ignore_index=True)
    return top1_and_miss(combined, np.concatenate(scores))[0]


def classifier_cv_top1(params):
    frames, scores = [], []
    for a, b in SPLITS:
        model = xgb.XGBClassifier(**params).fit(X[a], Y[a])
        frames.append(TRAIN.iloc[b][["trip", "obs", "label"]].copy())
        scores.append(model.predict_proba(X[b])[:, 1])
    return pooled_top1(frames, scores)


def ranker_cv_top1(params):
    frames, scores = [], []
    for a, b in SPLITS:
        _, valid, pred = fit_predict_ranker(TRAIN.iloc[a], TRAIN.iloc[b], params)
        frames.append(valid[["trip", "obs", "label"]])
        scores.append(pred)
    return pooled_top1(frames, scores)


def classifier_cv_metrics(params):
    results = []
    for a, b in SPLITS:
        model = xgb.XGBClassifier(**params).fit(X[a], Y[a])
        p = model.predict_proba(X[b])[:, 1]
        frame = TRAIN.iloc[b]
        pred = predicted_arcs(frame, p)
        results.append(dict(
            macro_f1=float(f1_score(Y[b], p >= 0.5, average="macro")),
            log_loss=float(log_loss(Y[b], np.clip(p, 1e-7, 1 - 1e-7))),
            route_continuity=route_continuity(pred, NET)))
    return results


def ranker_cv_continuity(params):
    results = []
    for a, b in SPLITS:
        _, valid, pred = fit_predict_ranker(TRAIN.iloc[a], TRAIN.iloc[b], params)
        arcs = predicted_arcs(valid, pred)
        results.append(route_continuity(arcs, NET))
    return results


def objective(trial):
    params = dict(
        n_estimators=trial.suggest_int("n_estimators", 150, 700),
        max_depth=trial.suggest_int("max_depth", 3, 10),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        min_child_weight=trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
        gamma=trial.suggest_float("gamma", 0.0, 5.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True))
    return ranker_cv_top1(params)


def classifier_test_metrics(model):
    test = STUDENT_TEST
    p = model.predict_proba(test[FEATURES].values)[:, 1]
    pred = predicted_arcs(test, p)
    top1, miss = top1_and_miss(test, p)
    return dict(macro_f1=float(f1_score(test.label.values, p >= 0.5, average="macro")),
                log_loss=float(log_loss(test.label.values, np.clip(p, 1e-7, 1 - 1e-7))),
                top1_accuracy=top1, candidate_miss_rate=miss,
                route_continuity=route_continuity(pred, NET))


def ranker_test_metrics(model):
    ordered, xr, _, _ = rank_arrays(STUDENT_TEST)
    score = model.predict(xr)
    pred = predicted_arcs(ordered, score)
    top1, miss = top1_and_miss(ordered, score)
    return dict(top1_accuracy=top1, candidate_miss_rate=miss,
                route_continuity=route_continuity(pred, NET))


if __name__ == "__main__":
    trials = int(os.environ.get("PATHSNAP_OPTUNA_TRIALS", "100"))
    print("Baseline XGBClassifier pooled 5-fold grouped-CV Top-1:",
          f"{classifier_cv_top1(BASELINE):.6f}", flush=True)
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=trials)
    ranker_params = dict(study.best_params)
    optimized_cv_top1 = ranker_cv_top1(ranker_params)
    classifier_cv1 = classifier_cv_top1(BASELINE)
    selected_kind = "ranker" if optimized_cv_top1 > classifier_cv1 else "classifier"

    classifier = xgb.XGBClassifier(**BASELINE)
    started = time.perf_counter(); classifier.fit(X, Y)
    classifier_train_seconds = time.perf_counter() - started
    rank_train, rank_x, rank_y, rank_group = rank_arrays(TRAIN)
    ranker = make_ranker(ranker_params)
    started = time.perf_counter(); ranker.fit(rank_x, rank_y, group=rank_group, verbose=False)
    ranker_train_seconds = time.perf_counter() - started

    clf_fold = classifier_cv_metrics(BASELINE)
    rank_cont = ranker_cv_continuity(ranker_params)
    rows = [
        dict(model="XGBClassifier baseline", selected_for_final_fit=selected_kind == "classifier",
             cv_top1_accuracy=classifier_cv1,
             cv_macro_f1=float(np.mean([r["macro_f1"] for r in clf_fold])),
             cv_log_loss=float(np.mean([r["log_loss"] for r in clf_fold])),
             cv_route_continuity=float(np.nanmean([r["route_continuity"] for r in clf_fold])),
             **{f"student_test_{k}": v for k, v in classifier_test_metrics(classifier).items()},
             train_seconds=classifier_train_seconds),
        dict(model="XGBRanker optimized", selected_for_final_fit=selected_kind == "ranker",
             cv_top1_accuracy=optimized_cv_top1, cv_macro_f1=np.nan, cv_log_loss=np.nan,
             cv_route_continuity=float(np.nanmean(rank_cont)),
             **{f"student_test_{k}": v for k, v in ranker_test_metrics(ranker).items()},
             train_seconds=ranker_train_seconds),
    ]
    selected = ranker if selected_kind == "ranker" else classifier
    joblib.dump(selected, MODELS / "XGBoost_selected.joblib")
    pd.DataFrame(rows).to_csv(OUT / "xgb_approach_comparison.csv", index=False)
    with (OUT / "xgb_best_hyperparameters.json").open("w") as f:
        json.dump(dict(model="XGBRanker", objective="rank:pairwise", device=DEVICE,
                       n_trials=trials, best_cv_top1_accuracy=optimized_cv_top1,
                       selected_model=selected_kind, baseline_classifier=BASELINE,
                       ranker_params=ranker_params), f, indent=2)
    study.trials_dataframe().to_csv(OUT / "xgb_rank_optuna_trials.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
