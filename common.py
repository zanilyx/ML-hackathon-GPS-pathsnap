"""Shared loading, projection, candidate generation and metrics for PathSnap."""
from pathlib import Path
import numpy as np, pandas as pd

EARTH_R = 6371000.0
SEED = 42
DEFAULT_GAP_THRESHOLD_S = 120.0


def list_trips(split_dir):
    return sorted(p.name for p in Path(split_dir).iterdir() if p.is_dir())


def _read(path, ncols, dtype=float):
    try:
        df = pd.read_csv(path, sep=r"\s+", header=None, dtype=dtype)
        return df.values
    except (pd.errors.EmptyDataError, FileNotFoundError):
        return np.zeros((0, ncols))


def load_trip(split_dir, tid):
    d = Path(split_dir) / tid
    return dict(
        tid=tid,
        track=_read(d / f"{tid}.track", 3),                 # lon, lat, t
        nodes=_read(d / f"{tid}.nodes", 2),                 # lon, lat
        arcs=_read(d / f"{tid}.arcs", 2, int).astype(int),  # from, to ; row idx = arc_id
        route=_read(d / f"{tid}.route", 1, int).astype(int).ravel(),
    )


def segment_track(track, gap_threshold_s=DEFAULT_GAP_THRESHOLD_S):
    """Return segment IDs and contiguous index ranges split at large timestamp gaps.

    Every input observation is retained. The threshold applies to positive time
    differences only; duplicate or decreasing timestamps remain in the segment.
    """
    track = np.asarray(track)
    n = len(track)
    if n == 0:
        return np.zeros(0, dtype=int), []
    if gap_threshold_s <= 0:
        raise ValueError("gap_threshold_s must be positive")
    dt = np.diff(track[:, 2])
    cuts = np.flatnonzero(dt > gap_threshold_s) + 1
    bounds = np.r_[0, cuts, n]
    ranges = [(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:])]
    seg_id = np.empty(n, dtype=int)
    for sid, (a, b) in enumerate(ranges):
        seg_id[a:b] = sid
    return seg_id, ranges


class Projector:
    """Local equirectangular projection (metres). Fine at city scale, no extra deps."""
    def __init__(self, lon0, lat0):
        self.lon0, self.lat0 = lon0, lat0
        self.cos0 = np.cos(np.radians(lat0))

    def __call__(self, lon, lat):
        x = EARTH_R * np.radians(np.asarray(lon) - self.lon0) * self.cos0
        y = EARTH_R * np.radians(np.asarray(lat) - self.lat0)
        return np.column_stack([x, y])


def seg_dist(P, A, B, chunk=100):
    """Distance (n,m) and projection fraction (n,m) from points P to segments A->B."""
    AB = B - A
    L2 = np.maximum((AB ** 2).sum(1), 1e-9)
    n, m = len(P), len(A)
    D = np.empty((n, m), np.float32)
    T = np.empty((n, m), np.float32)
    for s in range(0, n, chunk):
        p = P[s:s + chunk]
        AP = p[:, None, :] - A[None]
        t = np.clip((AP * AB[None]).sum(2) / L2[None], 0, 1)
        proj = A[None] + t[..., None] * AB[None]
        D[s:s + chunk] = np.linalg.norm(p[:, None, :] - proj, axis=2)
        T[s:s + chunk] = t
    return D, T


FEATURES = ["dist", "frac", "arc_len", "rank", "dist_minus_min", "dist_ratio_min", "n_cand",
            "cos_diff", "abs_cos", "heading_valid", "speed", "dt_prev",
            "dist_prev", "dist_next", "deg_from", "deg_to"]


def build_candidates_slow(trip, K=10, radius=150.0, gap_threshold_s=DEFAULT_GAP_THRESHOLD_S):
    """One row per (GPS observation, candidate arc). Label = arc is in the reference route."""
    tr, nodes, arcs = trip["track"], trip["nodes"], trip["arcs"]
    if len(tr) < 2 or len(nodes) == 0 or len(arcs) == 0:
        return None
    lon0, lat0 = nodes[:, 0].mean(), nodes[:, 1].mean()
    proj = Projector(lon0, lat0)
    P = proj(tr[:, 0], tr[:, 1])
    N = proj(nodes[:, 0], nodes[:, 1])
    A, B = N[arcs[:, 0]], N[arcs[:, 1]]
    arc_len = np.linalg.norm(B - A, axis=1)
    arc_bear = np.arctan2((B - A)[:, 1], (B - A)[:, 0])
    deg = np.bincount(np.concatenate([arcs[:, 0], arcs[:, 1]]), minlength=len(nodes))

    D, T = seg_dist(P, A, B)
    n, m = D.shape
    k = min(K, m)
    idx = np.argpartition(D, k - 1, axis=1)[:, :k]
    # Temporal features only use neighbours within the same segment.
    seg_id, ranges = segment_track(tr, gap_threshold_s)
    prv, nxt = np.arange(n), np.arange(n)
    for a0, b0 in ranges:
        ix = np.arange(a0, b0)
        prv[ix] = np.maximum(ix - 1, a0)
        nxt[ix] = np.minimum(ix + 1, b0 - 1)
    t = tr[:, 2]
    disp = P[nxt] - P[prv]
    dt_span = np.maximum(t[nxt] - t[prv], 1e-3)
    speed = np.linalg.norm(disp, axis=1) / dt_span
    heading = np.arctan2(disp[:, 1], disp[:, 0])
    hvalid = (np.linalg.norm(disp, axis=1) > 1.0).astype(float)
    dt_prev = np.zeros(n)
    for a0, b0 in ranges:
        dt_prev[a0:b0] = np.r_[0.0, np.diff(t[a0:b0])]

    route_set = set(trip["route"].tolist())
    rows = []
    for i in range(n):
        cand = idx[i]
        d = D[i, cand]
        order = np.argsort(d)
        cand, d = cand[order], d[order]
        keep = d <= radius
        keep[0] = True  # always keep nearest
        cand, d = cand[keep], d[keep]
        dmin = d[0]
        for r, (a, dd) in enumerate(zip(cand, d)):
            cd = np.cos(heading[i] - arc_bear[a]) * hvalid[i]
            rows.append((trip["tid"], int(seg_id[i]), i, a, int(a in route_set), dd, T[i, a], arc_len[a], r,
                         dd - dmin, dd / (dmin + 1.0), len(cand), cd, abs(cd), hvalid[i],
                         speed[i], dt_prev[i], D[prv[i], a], D[nxt[i], a],
                         deg[arcs[a, 0]], deg[arcs[a, 1]]))
    cols = ["trip", "segment", "obs", "arc_id", "label"] + FEATURES
    return pd.DataFrame(rows, columns=cols)


def _pair_dist(p, a, b):
    """Distance from point p[i] to segment a[i]->b[i] (row-wise, no full matrix)."""
    ab = b - a
    t = np.clip(((p - a) * ab).sum(1) / np.maximum((ab ** 2).sum(1), 1e-9), 0, 1)
    return np.linalg.norm(p - (a + t[:, None] * ab), axis=1), t


def build_candidates(trip, K=10, radius=150.0, chunk=256, gap_threshold_s=DEFAULT_GAP_THRESHOLD_S):
    """Memory-safe: GPS points are processed in chunks of `chunk`, each against only the arcs
    near that chunk, so no (n_obs x n_arcs) matrix is ever built."""
    tr, nodes, arcs = trip["track"], trip["nodes"], trip["arcs"]
    if len(tr) < 2 or len(nodes) == 0 or len(arcs) == 0:
        return None
    proj = Projector(nodes[:, 0].mean(), nodes[:, 1].mean())
    P = proj(tr[:, 0], tr[:, 1])
    N = proj(nodes[:, 0], nodes[:, 1])
    A, B = N[arcs[:, 0]], N[arcs[:, 1]]
    amin, amax = np.minimum(A, B), np.maximum(A, B)
    arc_len = np.linalg.norm(B - A, axis=1)
    arc_bear = np.arctan2((B - A)[:, 1], (B - A)[:, 0])
    deg = np.bincount(arcs.ravel(), minlength=len(nodes))
    n, M = len(P), len(arcs)
    pad = max(radius, 300.0)

    def topk(obs, ids):
        d, t = seg_dist(P[obs], A[ids], B[ids])
        k = min(K, d.shape[1])
        ix = np.argpartition(d, k - 1, axis=1)[:, :k]
        ds, ts, ai = np.take_along_axis(d, ix, 1), np.take_along_axis(t, ix, 1), ids[ix]
        if k < K:                                           # pad to fixed width
            w = K - k
            ds = np.hstack([ds, np.full((len(obs), w), np.inf, np.float32)])
            ts = np.hstack([ts, np.zeros((len(obs), w), np.float32)])
            ai = np.hstack([ai, np.zeros((len(obs), w), int)])
        return ai, ds, ts

    def local_ids(Pc, factor=1):
        lo, hi = Pc.min(0) - factor * pad, Pc.max(0) + factor * pad
        return np.flatnonzero((amax[:, 0] >= lo[0]) & (amin[:, 0] <= hi[0]) &
                              (amax[:, 1] >= lo[1]) & (amin[:, 1] <= hi[1]))

    AI, DS, TS = [], [], []
    for s in range(0, n, chunk):
        obs = np.arange(s, min(s + chunk, n))
        ids = local_ids(P[obs])
        if len(ids) == 0:                                   # chunk is far from the network: widen gradually
            for f in (4, 16, 64):
                ids = local_ids(P[obs], f)
                if len(ids):
                    break
        if len(ids) == 0:
            ids = np.arange(M)
        ai, ds, ts = topk(obs, ids)
        if len(ids) < M:                                    # off-network points: retry with a 4x wider LOCAL box
            bad = np.flatnonzero(ds.min(1) > pad)
            if len(bad):
                ids2 = local_ids(P[obs[bad]], 4)
                if 0 < len(ids2) <= 20000:
                    ai[bad], ds[bad], ts[bad] = topk(obs[bad], ids2)
        AI.append(ai); DS.append(ds); TS.append(ts)
    AI, DS, TS = np.vstack(AI), np.vstack(DS), np.vstack(TS)
    o = np.argsort(DS, axis=1)
    AI, DS, TS = (np.take_along_axis(x, o, 1) for x in (AI, DS, TS))
    mask = np.isfinite(DS) & (DS <= radius)
    mask[:, 0] = True                                       # nearest always kept

    t = tr[:, 2]
    seg_id, ranges = segment_track(tr, gap_threshold_s)
    prv, nxt = np.arange(n), np.arange(n)
    for a0, b0 in ranges:
        ix = np.arange(a0, b0)
        prv[ix] = np.maximum(ix - 1, a0)
        nxt[ix] = np.minimum(ix + 1, b0 - 1)
    disp = P[nxt] - P[prv]
    speed = np.linalg.norm(disp, axis=1) / np.maximum(t[nxt] - t[prv], 1e-3)
    heading = np.arctan2(disp[:, 1], disp[:, 0])
    hvalid = (np.linalg.norm(disp, axis=1) > 1.0).astype(float)
    dt_prev = np.zeros(n)
    for a0, b0 in ranges:
        dt_prev[a0:b0] = np.r_[0.0, np.diff(t[a0:b0])]

    i, c = np.nonzero(mask)
    a = AI[i, c]
    dist, dmin = DS[i, c], DS[i, 0]
    cd = np.cos(heading[i] - arc_bear[a]) * hvalid[i]
    d_prev, _ = _pair_dist(P[prv[i]], A[a], B[a])
    d_next, _ = _pair_dist(P[nxt[i]], A[a], B[a])
    route_set = np.zeros(M, bool)
    r = trip["route"]
    if len(r) and r.max() < M:
        route_set[r] = True
    return pd.DataFrame({
        "trip": trip["tid"], "segment": seg_id[i], "obs": i, "arc_id": a, "label": route_set[a].astype(int),
        "dist": dist, "frac": TS[i, c], "arc_len": arc_len[a], "rank": c, "dist_minus_min": dist - dmin,
        "dist_ratio_min": dist / (dmin + 1.0), "n_cand": mask.sum(1)[i], "cos_diff": cd, "abs_cos": np.abs(cd),
        "heading_valid": hvalid[i], "speed": speed[i], "dt_prev": dt_prev[i],
        "dist_prev": d_prev, "dist_next": d_next, "deg_from": deg[arcs[a, 0]], "deg_to": deg[arcs[a, 1]]})


# ---------------- metrics ----------------
def top1_and_miss(df, score):
    d = df[["trip", "obs", "label"]].copy()
    d["score"] = score
    top = d.loc[d.groupby(["trip", "obs"])["score"].idxmax()]
    has_pos = d.groupby(["trip", "obs"])["label"].max()
    return float(top["label"].mean()), float(1 - has_pos.mean())


def predicted_arcs(df, score):
    d = df[["trip", "segment", "obs", "arc_id"]].copy()
    d["score"] = score
    return d.loc[d.groupby(["trip", "segment", "obs"])["score"].idxmax()].sort_values(["trip", "segment", "obs"])


def route_continuity(pred, net):
    """Share of consecutive distinct predicted arcs that are topologically connected
    (to_node of previous == from_node of next). net = {trip: arcs array}."""
    ok = tot = 0
    keys = ["trip", "segment"] if "segment" in pred.columns else ["trip"]
    grouped = pred.groupby(keys)
    for key, g in grouped:
        trip = key[0] if isinstance(key, tuple) else key
        seq = g["arc_id"].values
        seq = seq[np.r_[True, seq[1:] != seq[:-1]]]
        arcs = net[trip]
        if len(seq) < 2:
            continue
        ok += int((arcs[seq[:-1], 1] == arcs[seq[1:], 0]).sum())
        tot += len(seq) - 1
    return ok / tot if tot else float("nan")
