"""Bayesian hyperparameter optimization and final training for selected XGBoost model."""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import GroupKFold

from common import FEATURES, SEED, predicted_arcs, route_continuity, top1_and_miss

OUT = Path("outputs")
MODELS = OUT / "models"
MODELS.mkdir(parents=True, exist_ok=True)
import pickle
with open(OUT / "features.pkl", "rb") as f:
    C = pickle.load(f)
train, test = C["train"], C["student_test"]
net = C["net"]
X, y, groups = train[FEATURES].values, train.label.values, train.trip.values
Xt, yt = test[FEATURES].values, test.label.values
device = "cpu"
try:
    xgb.XGBClassifier(n_estimators=2, device="cuda", tree_method="hist", eval_metric="logloss").fit(
        X[: min(20, len(X))], y[: min(20, len(y))])
except Exception:
    device = "cpu"

BASELINE = dict(n_estimators=300, max_depth=6, learning_rate=0.1,
                tree_method="hist", device=device, eval_metric="logloss", random_state=SEED)
SPLITS = list(GroupKFold(5).split(X, y, groups))


def score(model, frame, xx, yy):
    p = model.predict_proba(xx)[:, 1]
    top1, _ = top1_and_miss(frame, p)
    pred = predicted_arcs(frame, p)
    return dict(macro_f1=float(f1_score(yy, p >= .5, average="macro")),
                log_loss=float(log_loss(yy, np.clip(p, 1e-7, 1 - 1e-7))),
                top1_accuracy=top1, route_continuity=route_continuity(pred, net))


def cv_eval(params):
    rows = []
    for a, b in SPLITS:
        model = xgb.XGBClassifier(**params)
        model.fit(X[a], y[a])
        p = model.predict_proba(X[b])[:, 1]
        rows.append(float(log_loss(y[b], np.clip(p, 1e-7, 1 - 1e-7))))
    return float(np.mean(rows))


def cv_top1(params):
    frames, predictions = [], []
    for a, b in SPLITS:
        model = xgb.XGBClassifier(**params)
        model.fit(X[a], y[a])
        frames.append(train.iloc[b][["trip", "obs", "label"]].copy())
        predictions.append(model.predict_proba(X[b])[:, 1])
    # Pool all out-of-fold predictions so each observation has equal weight,
    # matching top1_and_miss's student-test metric instead of averaging folds.
    oof = pd.concat(frames, ignore_index=True)
    return top1_and_miss(oof, np.concatenate(predictions))[0]


def cv_metrics(params):
    rows = []
    for a, b in SPLITS:
        model = xgb.XGBClassifier(**params).fit(X[a], y[a])
        rows.append(score(model, train.iloc[b], X[b], y[b]))
    return rows


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
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
        tree_method="hist", device=device, eval_metric="logloss", random_state=SEED)
    return cv_top1(params)


if __name__ == "__main__":
    trials = int(__import__("os").environ.get("PATHSNAP_OPTUNA_TRIALS", "40"))
    baseline_params = BASELINE
    print(f"Baseline 5-fold grouped CV log loss: {cv_eval(baseline_params):.6f}", flush=True)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=trials)
    best_params = {**study.best_params, "tree_method": "hist", "device": device,
                   "eval_metric": "logloss", "random_state": SEED}

    results = []
    candidates = [("baseline", baseline_params), ("optimized", best_params)]
    cv_losses = {label: cv_eval(params) for label, params in candidates}
    cv_top1_scores = {label: cv_top1(params) for label, params in candidates}
    selected_label, selected_params = max(candidates, key=lambda item: cv_top1_scores[item[0]])
    for label, params in candidates:
        folds = cv_metrics(params)
        model = xgb.XGBClassifier(**params)
        t0 = time.perf_counter(); model.fit(X, y); train_seconds = time.perf_counter() - t0
        metrics = score(model, test, Xt, yt)
        results.append(dict(model=label, selected_for_final_fit=(label == selected_label),
                            cv_macro_f1=float(np.mean([r['macro_f1'] for r in folds])),
                            cv_log_loss=cv_losses[label],
                            cv_top1_accuracy=cv_top1_scores[label],
                            cv_route_continuity=float(np.nanmean([r['route_continuity'] for r in folds])),
                            **{f"student_test_{k}": v for k, v in metrics.items()},
                            train_seconds=train_seconds))
        if label == selected_label:
            joblib.dump(model, MODELS / "XGBoost_optimized.joblib")
    pd.DataFrame(results).to_csv(OUT / "xgb_optimization_comparison.csv", index=False)
    best = dict(best_params)
    best["n_trials"] = trials
    best["best_cv_top1_accuracy"] = study.best_value
    best["baseline"] = baseline_params
    best["selected_model"] = selected_label
    best["device"] = device
    with open(OUT / "xgb_best_hyperparameters.json", "w") as f:
        json.dump(best, f, indent=2)
    study.trials_dataframe().to_csv(OUT / "xgb_optuna_trials.csv", index=False)
    print(pd.DataFrame(results).to_string(index=False), flush=True)
    print("Best hyperparameters:", json.dumps(best, indent=2), flush=True)
