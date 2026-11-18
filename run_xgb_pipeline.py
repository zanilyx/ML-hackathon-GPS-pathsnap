"""One-command EDA, segmented features, and XGBoost-only optimization run."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT / "Student folder"
GAP_SECONDS = sys.argv[2] if len(sys.argv) > 2 else "120"

if not (DATA / "train").is_dir() or not (DATA / "student_test").is_dir():
    raise SystemExit(f"Expected train/ and student_test/ directories under: {DATA}")

env = os.environ.copy()
env.setdefault("PATHSNAP_OPTUNA_TRIALS", "100")
commands = [
    [sys.executable, "eda.py", str(DATA), GAP_SECONDS],
    [sys.executable, "make_features.py", str(DATA), GAP_SECONDS],
    [sys.executable, "train_xgb_rank_optuna.py"],
]
for command in commands:
    print("\n$", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)

print("\nDone. See outputs/eda/, outputs/xgb_optimization_comparison.csv,")
print("outputs/xgb_best_hyperparameters.json, and outputs/models/XGBoost_optimized.joblib")
