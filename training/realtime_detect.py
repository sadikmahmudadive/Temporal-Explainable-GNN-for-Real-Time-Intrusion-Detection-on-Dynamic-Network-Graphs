"""Near-real-time detection runner for trained TemporalGNN checkpoints.

This script tails an appending CSV of flow records, applies saved deployment
preprocessing parameters, builds temporal graph snapshots, and emits attack
probability alerts per completed snapshot window.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from training import pipeline_core as core


def log(msg: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


def load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(payload).__name__}")
    return payload


def resolve_existing_path(base_file: Path, raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute() and candidate.exists():
        return candidate

    probes = [
        Path.cwd() / candidate,
        base_file.parent / candidate,
        base_file.parent.parent / candidate,
    ]
    for p in probes:
        if p.exists():
            return p.resolve()
    return probes[0].resolve()


def choose_device(pref: str) -> torch.device:
    pref = pref.lower()
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device='cuda' requested but CUDA unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CSVTailReader:
    """Tails a CSV file that is being appended to."""

    def __init__(self, path: Path):
        self.path = path
        self._offset = 0
        self._header_line: Optional[str] = None

    def _reset_if_rotated(self) -> None:
        if self.path.exists():
            size = self.path.stat().st_size
            if size < self._offset:
                self._offset = 0
                self._header_line = None

    def read_new_rows(self, max_rows: int) -> pd.DataFrame:
        self._reset_if_rotated()
        if not self.path.exists():
            return pd.DataFrame()
        with self.path.open("r", encoding="utf-8", errors="ignore") as f:
            if self._header_line is None:
                header = f.readline()
                if not header:
                    return pd.DataFrame()
                self._header_line = header
                self._offset = f.tell()
            else:
                f.seek(self._offset)

            lines: List[str] = []
            for _ in range(max_rows):
                line = f.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                if self._header_line is not None and line.strip() == self._header_line.strip():
                    continue
                lines.append(line)
            self._offset = f.tell()

        if not lines or self._header_line is None:
            return pd.DataFrame()
        csv_text = self._header_line + "".join(lines)
        return pd.read_csv(io.StringIO(csv_text), low_memory=False)


def _transform_series(series: pd.Series, low: float, high: float, non_negative: bool) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(np.float64)
    s = s.clip(lower=low, upper=high)
    if non_negative:
        s = np.log1p(s)
    else:
        s = np.sign(s) * np.log1p(np.abs(s))
    return s.astype(np.float32)


def align_schema(df: pd.DataFrame, edge_feature_cols: List[str]) -> pd.DataFrame:
    out = core.standardize_columns(df.copy())

    # Resolve edge feature columns (handles minor naming variance).
    rename_map: Dict[str, str] = {}
    for feature in edge_feature_cols:
        found = core.locate_column(out, [feature])
        if not found:
            raise ValueError(f"Missing required edge feature column in input stream: {feature}")
        if found != feature:
            rename_map[found] = feature

    src_col = core.locate_column(out, ["Source IP", "Src IP", "source_ip"])
    dst_col = core.locate_column(out, ["Destination IP", "Dst IP", "destination_ip"])
    if src_col:
        rename_map[src_col] = "Source IP"
    if dst_col:
        rename_map[dst_col] = "Destination IP"

    if rename_map:
        out = out.rename(columns=rename_map)
    return out


def parse_event_time(df: pd.DataFrame) -> pd.Series:
    ts_col = core.locate_column(df, ["event_time", "Timestamp", "Flow Start", "Date"])
    if ts_col:
        ts = pd.to_datetime(df[ts_col], errors="coerce")
    else:
        now = pd.Timestamp.utcnow().tz_localize(None)
        ts = pd.Series([now] * len(df), index=df.index, dtype="datetime64[ns]")
    # Fallback for unparseable timestamps.
    if ts.isna().all():
        now = pd.Timestamp.utcnow().tz_localize(None)
        ts = pd.Series([now] * len(df), index=df.index, dtype="datetime64[ns]")
    else:
        ts = ts.fillna(method="ffill").fillna(method="bfill")
    return ts


class RealtimeTemporalDetector:
    def __init__(
        self,
        deployment: Dict[str, Any],
        deployment_path: Path,
        device: torch.device,
        context_size: int,
        emit_open_window: bool,
        output_csv: Optional[Path],
    ) -> None:
        self.deployment = deployment
        self.deployment_path = deployment_path
        self.device = device
        self.context_size = max(1, int(context_size))
        self.emit_open_window = emit_open_window
        self.output_csv = output_csv

        self.threshold = float(deployment["threshold"])
        self.edge_feature_cols = [str(c) for c in deployment["edge_feature_cols"]]
        self.transform_params = deployment["transform_params"]
        self.scaler_mean = np.asarray(deployment["scaler_mean"], dtype=np.float64)
        self.scaler_scale = np.asarray(deployment["scaler_scale"], dtype=np.float64)
        self.scaler_scale = np.where(self.scaler_scale == 0.0, 1.0, self.scaler_scale)

        self.window_seconds = int(pd.to_timedelta(str(deployment["window_size"])).total_seconds())
        self.max_edges_per_snapshot = int(deployment.get("max_edges_per_snapshot", 3000))
        core.CONFIG["max_edges_per_snapshot"] = self.max_edges_per_snapshot
        core.CONFIG["seed"] = int(deployment.get("seed", 42))
        core.CONFIG["verbose"] = False
        core.CONFIG["compact_progress"] = True

        state_dict_path = resolve_existing_path(
            deployment_path, str(deployment["model_state_dict_path"])
        )
        model_kwargs = dict(deployment["model_kwargs"])
        self.model = core.TemporalEdgeGNN(**model_kwargs).to(self.device)
        state = torch.load(state_dict_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

        self.origin_time: Optional[pd.Timestamp] = None
        self.pending_rows = pd.DataFrame()
        self.context: Deque[core.Data] = deque(maxlen=self.context_size)
        self.processed_snapshot_ids: set[int] = set()

        if self.output_csv is not None and not self.output_csv.exists():
            self.output_csv.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                columns=[
                    "timestamp_utc",
                    "snapshot_id",
                    "num_edges",
                    "mean_attack_probability",
                    "max_attack_probability",
                    "predicted_attack_snapshot",
                    "threshold",
                    "context_size",
                ]
            ).to_csv(self.output_csv, index=False)

    def _apply_feature_transforms(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col in self.edge_feature_cols:
            p = self.transform_params[col]
            out[col] = _transform_series(
                out[col],
                low=float(p["low"]),
                high=float(p["high"]),
                non_negative=bool(p["non_negative"]),
            )
        arr = out[self.edge_feature_cols].to_numpy(dtype=np.float64)
        arr = (arr - self.scaler_mean) / self.scaler_scale
        out[self.edge_feature_cols] = arr.astype(np.float32)
        return out

    def _prepare_rows(self, incoming: pd.DataFrame) -> pd.DataFrame:
        if incoming.empty:
            return incoming
        df = align_schema(incoming, edge_feature_cols=self.edge_feature_cols)
        if "Source IP" not in df.columns or "Destination IP" not in df.columns:
            raise ValueError("Input stream must contain Source/Destination endpoint columns.")

        df["event_time"] = parse_event_time(df)
        df = df.dropna(subset=["event_time"]).copy()
        if df.empty:
            return df

        if self.origin_time is None:
            self.origin_time = pd.Timestamp(df["event_time"].min())
        sid = ((df["event_time"] - self.origin_time).dt.total_seconds() // self.window_seconds).astype(int)
        df["snapshot_id"] = sid
        df["binary_label"] = 0
        df["day_file"] = "realtime"
        df["Source IP"] = df["Source IP"].astype(str)
        df["Destination IP"] = df["Destination IP"].astype(str)
        df = self._apply_feature_transforms(df)

        keep_cols = ["event_time", "snapshot_id", "Source IP", "Destination IP", "binary_label", "day_file"] + self.edge_feature_cols
        return df[keep_cols]

    def _extract_ready_snapshots(self) -> List[core.Data]:
        if self.pending_rows.empty:
            return []
        unique_ids = sorted(self.pending_rows["snapshot_id"].astype(int).unique().tolist())
        if not unique_ids:
            return []
        if self.emit_open_window:
            ready_ids = unique_ids
        else:
            ready_ids = unique_ids[:-1]
        ready_ids = [sid for sid in ready_ids if sid not in self.processed_snapshot_ids]
        if not ready_ids:
            return []

        ready_df = self.pending_rows[self.pending_rows["snapshot_id"].isin(ready_ids)].copy()
        self.pending_rows = self.pending_rows[~self.pending_rows["snapshot_id"].isin(ready_ids)].copy()
        if ready_df.empty:
            return []
        snaps = core.build_snapshot_graphs(ready_df, self.edge_feature_cols)
        snaps = sorted(snaps, key=lambda d: int(d.snapshot_id))
        return snaps

    def _run_model_for_snapshot(self, snap: core.Data) -> Dict[str, Any]:
        self.context.append(snap)
        seq = [d.to(self.device) for d in list(self.context)]
        with torch.no_grad():
            logits = self.model(seq)
            latest_logits = logits[-1].reshape(-1)
            probs = torch.sigmoid(latest_logits).detach().cpu().numpy()

        mean_prob = float(np.mean(probs)) if probs.size else 0.0
        max_prob = float(np.max(probs)) if probs.size else 0.0
        pred = int(mean_prob >= self.threshold)
        return {
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "snapshot_id": int(getattr(snap, "snapshot_id", -1)),
            "num_edges": int(snap.edge_index.shape[1]),
            "mean_attack_probability": mean_prob,
            "max_attack_probability": max_prob,
            "predicted_attack_snapshot": pred,
            "threshold": float(self.threshold),
            "context_size": int(len(self.context)),
        }

    def ingest_rows(self, incoming: pd.DataFrame) -> List[Dict[str, Any]]:
        prepared = self._prepare_rows(incoming)
        if prepared.empty:
            return []
        if self.pending_rows.empty:
            self.pending_rows = prepared
        else:
            self.pending_rows = pd.concat([self.pending_rows, prepared], axis=0, ignore_index=True)

        alerts: List[Dict[str, Any]] = []
        for snap in self._extract_ready_snapshots():
            sid = int(getattr(snap, "snapshot_id", -1))
            if sid in self.processed_snapshot_ids:
                continue
            self.processed_snapshot_ids.add(sid)
            alerts.append(self._run_model_for_snapshot(snap))

        if alerts and self.output_csv is not None:
            pd.DataFrame(alerts).to_csv(self.output_csv, mode="a", header=False, index=False)
        return alerts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tail flow CSV and emit TemporalGNN realtime alerts.")
    parser.add_argument(
        "--deployment",
        type=str,
        default="evaluation/deployment_artifacts.json",
        help="Path to deployment_artifacts.json generated by training.run_training.",
    )
    parser.add_argument("--input-csv", type=str, required=True, help="Path to appending flow CSV stream.")
    parser.add_argument("--poll-seconds", type=float, default=2.0, help="Polling interval for new lines.")
    parser.add_argument("--max-rows-per-read", type=int, default=5000, help="Max new rows consumed per poll.")
    parser.add_argument("--context-size", type=int, default=None, help="Override temporal context length.")
    parser.add_argument("--emit-open-window", action="store_true", help="Emit scores for latest incomplete window.")
    parser.add_argument("--output-csv", type=str, default="evaluation/realtime_alerts.csv", help="Alert log CSV path.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Inference device.")
    parser.add_argument("--once", action="store_true", help="Process currently available rows and exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    deployment_path = Path(args.deployment)
    if not deployment_path.exists():
        raise FileNotFoundError(f"Deployment artifact not found: {deployment_path}")

    deployment = load_json(deployment_path)
    context_size = int(args.context_size or deployment.get("snapshot_chunk_size", 4))
    output_csv = Path(args.output_csv) if args.output_csv else None
    device = choose_device(args.device)
    log(f"Using device: {device}")

    detector = RealtimeTemporalDetector(
        deployment=deployment,
        deployment_path=deployment_path,
        device=device,
        context_size=context_size,
        emit_open_window=bool(args.emit_open_window),
        output_csv=output_csv,
    )

    input_csv = Path(args.input_csv)
    tail = CSVTailReader(input_csv)
    log(f"Tailing stream file: {input_csv.resolve()}")
    if output_csv is not None:
        log(f"Alert output file: {output_csv.resolve()}")

    while True:
        rows = tail.read_new_rows(max_rows=int(args.max_rows_per_read))
        alerts = detector.ingest_rows(rows) if not rows.empty else []
        for alert in alerts:
            print(json.dumps(alert), flush=True)

        if args.once:
            break
        if rows.empty:
            time.sleep(float(args.poll_seconds))


if __name__ == "__main__":
    main()

