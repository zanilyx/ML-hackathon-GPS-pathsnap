"""PathSnap dashboard and ZIP upload inference endpoint."""
from io import BytesIO
import os
from pathlib import Path
from posixpath import basename, dirname, splitext
from zipfile import BadZipFile, ZipFile, ZIP_DEFLATED

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from common import (DEFAULT_GAP_THRESHOLD_S, FEATURES, build_candidates,
                    predicted_arcs, route_continuity, segment_track)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
MODELS = OUT / "models"
GAP_THRESHOLD_S = DEFAULT_GAP_THRESHOLD_S
EVALUATOR_SOURCE = Path(os.environ.get("PATHSNAP_EVALUATOR_DATA", os.environ.get("PATHSNAP_EVALUATOR_DIR", ROOT / "Student folder" / "evaluator_test"))).resolve()
FRONTEND = ROOT / "frontend"

app = FastAPI(title="PathSnap Dashboard", docs_url=None, redoc_url=None)

@app.get("/")
def home():
    return FileResponse(FRONTEND / "index.html")


def _local_network_lines(track, nodes, arcs, max_lines=2500):
    """Return a bounded road-network context around a trip, in lon/lat order."""
    # About 500 m of context around the trace, with a payload cap for browser rendering.
    lon_pad, lat_pad = 0.005, 0.005
    min_lon, max_lon = float(track[:, 0].min() - lon_pad), float(track[:, 0].max() + lon_pad)
    min_lat, max_lat = float(track[:, 1].min() - lat_pad), float(track[:, 1].max() + lat_pad)
    endpoints = nodes[arcs]
    keep = ((endpoints[:, :, 0].max(axis=1) >= min_lon) &
            (endpoints[:, :, 0].min(axis=1) <= max_lon) &
            (endpoints[:, :, 1].max(axis=1) >= min_lat) &
            (endpoints[:, :, 1].min(axis=1) <= max_lat))
    selected = np.flatnonzero(keep)
    if len(selected) > max_lines:
        selected = selected[np.linspace(0, len(selected) - 1, max_lines, dtype=int)]
    return [[nodes[int(u), :2].tolist(), nodes[int(v), :2].tolist()]
            for u, v in arcs[selected]]

def _read_zip_array(archive, name, ncols, integer=False):
    try:
        frame = pd.read_csv(BytesIO(archive.read(name)), sep=r"\s+", header=None)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read {name}: {exc}")
    if frame.shape[1] != ncols:
        raise HTTPException(status_code=400, detail=f"{name} should have {ncols} columns; found {frame.shape[1]}.")
    values = frame.to_numpy()
    if not np.isfinite(values).all():
        raise HTTPException(status_code=400, detail=f"{name} contains missing or non-finite values.")
    if integer:
        if not np.equal(values, np.floor(values)).all():
            raise HTTPException(status_code=400, detail=f"{name} must contain integer IDs.")
        values = values.astype(np.int64)
    return values


def _as_json_number(value):
    value = float(value)
    return value if np.isfinite(value) else None


def _evaluate_upload_payload(payload):
    """Validate a trip ZIP and run inference; never train or tune on it."""
    if len(payload) > 100 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="ZIP is too large (limit is 100 MB).")
    try:
        archive = ZipFile(BytesIO(payload))
    except BadZipFile:
        raise HTTPException(status_code=400, detail="Upload a valid ZIP archive.")

    with archive:
        infos = [info for info in archive.infolist()
                 if not info.is_dir() and "__MACOSX" not in info.filename and not basename(info.filename).startswith("._")]
        if sum(info.file_size for info in infos) > 500 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Uncompressed ZIP contents exceed the 500 MB limit.")
        trip_files = {}
        for info in infos:
            filename = basename(info.filename)
            stem, ext = splitext(filename)
            trip_id = basename(dirname(info.filename))
            if trip_id and stem == trip_id and ext in {".track", ".nodes", ".arcs", ".route"}:
                trip_files.setdefault(trip_id, {})[ext] = info.filename
        if not trip_files:
            raise HTTPException(status_code=400, detail="No trip folders found. Expected trip-id/trip-id.track files.")
        if len(trip_files) > 100:
            raise HTTPException(status_code=400, detail="ZIP contains more than 100 trips.")

        required = {".track", ".nodes", ".arcs"}
        for trip_id, files in trip_files.items():
            missing = required - set(files)
            if missing:
                raise HTTPException(status_code=400, detail=f"Trip {trip_id} is missing: {', '.join(sorted(missing))}.")

        model_path = MODELS / "XGBoost_selected.joblib"
        model_warning = ""
        if not model_path.exists():
            model_path = MODELS / "XGBoost.joblib"
            model_warning = "Using saved baseline model; its original evaluation predates the 120-second feature refresh."
        if not model_path.exists():
            raise HTTPException(status_code=503, detail="No XGBoost model artifact found under outputs/models/.")
        try:
            model = joblib.load(model_path)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Could not load model artifact: {exc}")

        results = []
        total_observations = total_segments = 0
        for trip_id in sorted(trip_files):
            files = trip_files[trip_id]
            track = _read_zip_array(archive, files[".track"], 3)
            nodes = _read_zip_array(archive, files[".nodes"], 2)
            arcs = _read_zip_array(archive, files[".arcs"], 2, integer=True)
            route = (_read_zip_array(archive, files[".route"], 1, integer=True).ravel()
                     if ".route" in files else np.zeros(0, dtype=np.int64))
            if len(track) < 2 or len(nodes) == 0 or len(arcs) == 0:
                raise HTTPException(status_code=400, detail=f"Trip {trip_id} needs at least 2 fixes and a non-empty network.")
            if ((track[:, 0] < -180) | (track[:, 0] > 180) |
                    (track[:, 1] < -90) | (track[:, 1] > 90)).any():
                raise HTTPException(status_code=400, detail=f"Trip {trip_id} has coordinates outside lon/lat ranges.")
            if ((arcs < 0) | (arcs >= len(nodes))).any():
                raise HTTPException(status_code=400, detail=f"Trip {trip_id} has arc endpoints outside its node table.")

            # Deliberately exclude the reference route from features and inference.
            trip = {"tid": trip_id, "track": track, "nodes": nodes, "arcs": arcs,
                    "route": np.zeros(0, dtype=np.int64)}
            candidates = build_candidates(trip, gap_threshold_s=GAP_THRESHOLD_S)
            if candidates is None or candidates.empty:
                raise HTTPException(status_code=400, detail=f"Could not generate candidates for trip {trip_id}.")

            if hasattr(model, "predict_proba"):
                scores = model.predict_proba(candidates[FEATURES].to_numpy())[:, 1]
            else:
                ordered = candidates.sort_values(["trip", "segment", "obs"], kind="mergesort")
                ranked = model.predict(ordered[FEATURES].to_numpy())
                scores = pd.Series(ranked, index=ordered.index).reindex(candidates.index).to_numpy()
            predictions = predicted_arcs(candidates, scores).sort_values(["segment", "obs"])

            has_route = len(route) > 0 and int(route.min()) >= 0 and int(route.max()) < len(arcs)
            if has_route:
                route_set = set(route.tolist())
                top1 = float(np.isin(predictions.arc_id.to_numpy(), route).mean())
                candidate_labels = candidates.arc_id.isin(route_set)
                grouped = candidates.assign(_positive=candidate_labels).groupby(["trip", "segment", "obs"])["_positive"].max()
                miss_rate = float(1 - grouped.mean())
            else:
                route_set, top1, miss_rate = set(), None, None

            continuity = route_continuity(predictions, {trip_id: arcs})
            predicted_lines, reference_lines = [], []
            last = None
            for row in predictions.itertuples(index=False):
                arc_id = int(row.arc_id)
                key = (int(row.segment), arc_id)
                if key != last:
                    u, v = arcs[arc_id]
                    predicted_lines.append([nodes[int(u), :2].tolist(), nodes[int(v), :2].tolist()])
                last = key
            if has_route:
                for arc_id in route:
                    u, v = arcs[int(arc_id)]
                    reference_lines.append([nodes[int(u), :2].tolist(), nodes[int(v), :2].tolist()])

            _, segment_ranges = segment_track(track, GAP_THRESHOLD_S)
            network_lines = _local_network_lines(track, nodes, arcs)
            prediction_rows = [{"segment": int(row.segment), "obs": int(row.obs), "arc_id": int(row.arc_id)}
                               for row in predictions.itertuples(index=False)]
            results.append(dict(
                trip_id=trip_id, observations=int(len(track)), segment_count=len(segment_ranges),
                top1_accuracy=top1, candidate_miss_rate=miss_rate,
                route_continuity=_as_json_number(continuity),
                track=track[:, :2].tolist(), network_lines=network_lines,
                predicted_lines=predicted_lines,
                reference_lines=reference_lines, predictions=prediction_rows))
            total_observations += len(track)
            total_segments += len(segment_ranges)

    return dict(trip_count=len(results), observation_count=total_observations,
                segment_count=total_segments, model_name=type(model).__name__,
                model_warning=model_warning, trips=results)



@app.post("/api/evaluate-upload")
async def evaluate_upload(request: Request):
    """Backward-compatible ZIP inference endpoint."""
    return _evaluate_upload_payload(await request.body())


def _zip_trip_members(archive):
    """Index complete trip folders inside an evaluator ZIP, ignoring OS metadata."""
    members = {}
    for name in archive.namelist():
        if name.endswith("/") or "__MACOSX" in name or basename(name).startswith("._"):
            continue
        filename = basename(name)
        stem, suffix = splitext(filename)
        trip_id = basename(dirname(name))
        if trip_id and stem == trip_id and suffix in {".track", ".nodes", ".arcs"}:
            members.setdefault(trip_id, {})[suffix] = name
    return {trip_id: files for trip_id, files in members.items()
            if all(suffix in files for suffix in (".track", ".nodes", ".arcs"))}


def _configured_trip_ids():
    if EVALUATOR_SOURCE.is_file() and EVALUATOR_SOURCE.suffix.lower() == ".zip":
        try:
            with ZipFile(EVALUATOR_SOURCE) as archive:
                return sorted(_zip_trip_members(archive))
        except BadZipFile:
            return []
    if not EVALUATOR_SOURCE.is_dir():
        return []
    return sorted(child.name for child in EVALUATOR_SOURCE.iterdir() if child.is_dir()
                  and all((child / f"{child.name}{suffix}").is_file()
                          for suffix in (".track", ".nodes", ".arcs")))


@app.get("/api/trips")
def evaluator_trips():
    """List valid evaluator trips configured on the model/API host."""
    ids = _configured_trip_ids()
    message = None if ids else (
        f"Set PATHSNAP_EVALUATOR_DATA to the evaluator ZIP or extracted trip folder. Current path: {EVALUATOR_SOURCE}"
    )
    return {"trips": ids, "message": message}


@app.post("/api/map-match/{trip_id}")
def map_match_evaluator_trip(trip_id: str):
    """Run inference for one selected evaluator trip, excluding any .route file."""
    if trip_id not in _configured_trip_ids():
        raise HTTPException(status_code=404, detail="Unknown trip. Reload the evaluator trip list.")
    payload = BytesIO()
    with ZipFile(payload, "w", ZIP_DEFLATED) as request_zip:
        if EVALUATOR_SOURCE.is_file() and EVALUATOR_SOURCE.suffix.lower() == ".zip":
            with ZipFile(EVALUATOR_SOURCE) as source_zip:
                members = _zip_trip_members(source_zip)[trip_id]
                for suffix in (".track", ".nodes", ".arcs"):
                    request_zip.writestr(f"{trip_id}/{trip_id}{suffix}", source_zip.read(members[suffix]))
        else:
            folder = EVALUATOR_SOURCE / trip_id
            for suffix in (".track", ".nodes", ".arcs"):
                request_zip.write(folder / f"{trip_id}{suffix}", f"{trip_id}/{trip_id}{suffix}")
    result = _evaluate_upload_payload(payload.getvalue())
    trip_result = result["trips"][0]
    trip_result["model_name"] = result["model_name"]
    trip_result["model_warning"] = result["model_warning"]
    return trip_result
