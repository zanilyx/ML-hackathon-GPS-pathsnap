"""Which TRAIN trips have candidate misses, and why? Run after make_features.py:  py diagnose_misses.py "../Student folder" """
import sys, pickle
from pathlib import Path
from common import *

root = Path(sys.argv[1] if len(sys.argv) > 1 else "Student folder")
C = pickle.load(open("outputs/features.pkl", "rb"))
rows = []
for split in ["train", "student_test"]:
    d = C[split]
    has_pos = d.groupby(["trip", "obs"]).label.max().reset_index()
    for tid, g in has_pos.groupby("trip"):
        rows.append(dict(split=split, trip=tid, n_obs=len(g), miss_rate=1 - g.label.mean()))
R = pd.DataFrame(rows).sort_values("miss_rate", ascending=False)
print("Worst trips by candidate miss rate:\n", R.head(12).to_string(index=False))

print("\nDistance (m) from each GPS point to the NEAREST REFERENCE-ROUTE ARC, for the worst trips:")
for _, r in R.head(6).iterrows():
    if r.miss_rate <= 0.01: break
    trip = load_trip(root / r.split, r.trip)
    N0 = trip["nodes"]; proj = Projector(N0[:, 0].mean(), N0[:, 1].mean())
    P = proj(trip["track"][:, 0], trip["track"][:, 1]); N = proj(N0[:, 0], N0[:, 1])
    ra = trip["arcs"][trip["route"]]
    D, _ = seg_dist(P, N[ra[:, 0]], N[ra[:, 1]])
    dmin = D.min(1); t = trip["track"][:, 2]; dt = np.r_[0, np.diff(t)]
    far = dmin > 150
    print(f"  trip {r.trip}: miss={r.miss_rate:.1%} | dist to route: median {np.median(dmin):.0f} m, p90 {np.percentile(dmin,90):.0f} m, "
          f"max {dmin.max():.0f} m | {far.mean():.1%} of points >150 m from route | big time gaps (>{DEFAULT_GAP_THRESHOLD_S:g} s): {(dt>DEFAULT_GAP_THRESHOLD_S).sum()}"
          f" | far points right after a gap: {(far & (dt>DEFAULT_GAP_THRESHOLD_S)).sum()}")
