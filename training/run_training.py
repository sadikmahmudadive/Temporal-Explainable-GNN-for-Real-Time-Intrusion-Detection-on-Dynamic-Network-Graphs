"""Reproducible training/evaluation entrypoint for IEEE-style reporting.

Key guarantees implemented here:
- No feature-scaling leakage (fit on train split only).
- Explicit split policy (default: strict per-day temporal split).
- Baseline + ablation outputs written to evaluation artifacts.
- Optional multi-seed significance testing.
"""

from __future__ import annotations

import argparse
import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import ttest_rel, wilcoxon
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from torch.optim.lr_scheduler import ReduceLROnPlateau

from training import pipeline_core as core


DEFAULT_CONFIG: Dict[str, Any] = {
    "seed": 42,
    "seeds_for_significance": [42, 52, 62],
    "data_dir": "data",
    "evaluation_dir": "evaluation",
    "models_dir": "models",
    "window_size": "300s",
    "max_rows_per_file": 200000,
    "max_edges_per_snapshot": 3000,
    "split_mode": "per_day_temporal",
    "split_train": 0.70,
    "split_val": 0.15,
    "split_test": 0.15,
    "num_epochs": 40,
    "significance_epochs": 20,
    "patience": 8,
    "snapshot_chunk_size": 4,
    "lr": 8e-4,
    "weight_decay": 1e-4,
    "dropout": 0.2,
    "hidden_dim": 64,
    "lstm_hidden": 64,
    "focal_gamma": 1.5,
    "grad_clip_norm": 1.0,
    "threshold_far_target": 0.30,
    "threshold_min": 0.05,
    "threshold_max": 0.95,
    "threshold_step": 0.01,
    "max_train_snapshots": 500,
    "max_val_snapshots": 150,
    "max_test_snapshots": 150,
    "verbose": True,
    "compact_progress": True,
    "chunk_log_interval": 1,
    "device": "auto",
}


def log(msg: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


def load_config(path: Path) -> Dict[str, Any]:
    cfg = deepcopy(DEFAULT_CONFIG)
    if path.exists():
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Expected a mapping in {path}, got {type(loaded).__name__}")
        cfg.update(loaded)
    return cfg


def parse_seed_list(raw: Any) -> List[int]:
    if raw is None:
        return []
    if isinstance(raw, int):
        return [int(raw)]
    if isinstance(raw, str):
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        return [int(p) for p in parts]
    if isinstance(raw, Sequence):
        return [int(v) for v in raw]
    raise ValueError(f"Unsupported seed list value: {raw!r}")


def ensure_dirs(cfg: Dict[str, Any]) -> None:
    Path(cfg["evaluation_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["models_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["evaluation_dir"], "plots").mkdir(parents=True, exist_ok=True)


def sync_core_config(cfg: Dict[str, Any]) -> None:
    core.CONFIG["seed"] = int(cfg["seed"])
    core.CONFIG["max_edges_per_snapshot"] = int(cfg["max_edges_per_snapshot"])
    core.CONFIG["verbose"] = bool(cfg["verbose"])
    core.CONFIG["compact_progress"] = bool(cfg["compact_progress"])
    core.CONFIG["chunk_log_interval"] = int(cfg["chunk_log_interval"])
    core.CONFIG["threshold_min"] = float(cfg["threshold_min"])
    core.CONFIG["threshold_max"] = float(cfg["threshold_max"])
    core.CONFIG["threshold_step"] = float(cfg["threshold_step"])


def _split_ids(ids: List[int], train: float, val: float, test: float) -> Tuple[List[int], List[int], List[int]]:
    n = len(ids)
    if n < 3:
        return (ids[:1], ids[1:2], ids[2:])
    n_train = max(1, int(n * train))
    n_val = max(1, int(n * val))
    n_test = n - n_train - n_val
    if n_test < 1:
        n_test = 1
        if n_train > n_val:
            n_train = max(1, n_train - 1)
        else:
            n_val = max(1, n_val - 1)
    train_ids = ids[:n_train]
    val_ids = ids[n_train : n_train + n_val]
    test_ids = ids[n_train + n_val :]
    return (train_ids, val_ids, test_ids)


def split_snapshot_ids(df: pd.DataFrame, cfg: Dict[str, Any]) -> Dict[str, Any]:
    split_mode = str(cfg["split_mode"])
    train_ratio = float(cfg["split_train"])
    val_ratio = float(cfg["split_val"])
    test_ratio = float(cfg["split_test"])
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-8:
        raise ValueError("split_train + split_val + split_test must sum to 1.0")

    meta = (
        df.groupby("snapshot_id", as_index=False)
        .agg(
            first_time=("event_time", "min"),
            day_file=("day_file", "first"),
            attack_ratio=("binary_label", "mean"),
        )
        .sort_values("first_time")
        .reset_index(drop=True)
    )
    if len(meta) < 3:
        raise ValueError("Need at least 3 snapshots for train/val/test split.")

    if split_mode == "chronological":
        ids = meta["snapshot_id"].astype(int).tolist()
        train_ids, val_ids, test_ids = _split_ids(ids, train_ratio, val_ratio, test_ratio)
    elif split_mode == "per_day_temporal":
        train_ids, val_ids, test_ids = [], [], []
        for _, day_meta in meta.groupby("day_file", sort=True):
            ids = day_meta.sort_values("first_time")["snapshot_id"].astype(int).tolist()
            tr, va, te = _split_ids(ids, train_ratio, val_ratio, test_ratio)
            train_ids.extend(tr)
            val_ids.extend(va)
            test_ids.extend(te)
        if not train_ids or not val_ids or not test_ids:
            ids = meta["snapshot_id"].astype(int).tolist()
            train_ids, val_ids, test_ids = _split_ids(ids, train_ratio, val_ratio, test_ratio)
    elif split_mode == "stratified_snapshot":
        # Included for experimentation; not a strict real-time split.
        idx_all = np.arange(len(meta))
        strata = (meta["attack_ratio"] >= meta["attack_ratio"].median()).astype(int).to_numpy()
        if np.unique(strata).size < 2:
            strata = None
        idx_train, idx_temp = train_test_split(
            idx_all,
            test_size=(1.0 - train_ratio),
            random_state=int(cfg["seed"]),
            stratify=strata,
            shuffle=True,
        )
        temp_val_frac = val_ratio / max(val_ratio + test_ratio, 1e-12)
        temp_strata = strata[idx_temp] if strata is not None else None
        if temp_strata is not None and np.unique(temp_strata).size < 2:
            temp_strata = None
        idx_val, idx_test = train_test_split(
            idx_temp,
            test_size=(1.0 - temp_val_frac),
            random_state=int(cfg["seed"]),
            stratify=temp_strata,
            shuffle=True,
        )
        train_ids = meta.iloc[sorted(idx_train)]["snapshot_id"].astype(int).tolist()
        val_ids = meta.iloc[sorted(idx_val)]["snapshot_id"].astype(int).tolist()
        test_ids = meta.iloc[sorted(idx_test)]["snapshot_id"].astype(int).tolist()
    else:
        raise ValueError(f"Unsupported split_mode: {split_mode}")

    overlap = (set(train_ids) & set(val_ids)) | (set(train_ids) & set(test_ids)) | (set(val_ids) & set(test_ids))
    if overlap:
        raise RuntimeError(f"Split leakage: overlapping snapshot IDs found: {sorted(list(overlap))[:5]}")

    return {
        "split_mode": split_mode,
        "train_ids": sorted(train_ids),
        "val_ids": sorted(val_ids),
        "test_ids": sorted(test_ids),
        "num_snapshots_total": int(len(meta)),
        "num_snapshots_train": int(len(train_ids)),
        "num_snapshots_val": int(len(val_ids)),
        "num_snapshots_test": int(len(test_ids)),
    }


def add_snapshot_ids(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    t0_min = out["event_time"].min()
    window_seconds = int(pd.to_timedelta(cfg["window_size"]).total_seconds())
    out["snapshot_id"] = ((out["event_time"] - t0_min).dt.total_seconds() // window_seconds).astype(int)
    return out


def resolve_edge_features(df: pd.DataFrame) -> List[str]:
    edge_cols: List[str] = []
    for c in core.EDGE_FEATURE_CANDIDATES:
        found = core.locate_column(df, [c])
        if not found:
            raise ValueError(f"Missing required edge feature column: {c}")
        edge_cols.append(found)
    return edge_cols


def _transform_series(series: pd.Series, low: float, high: float, non_negative: bool) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(np.float64)
    s = s.clip(lower=low, upper=high)
    if non_negative:
        s = np.log1p(s)
    else:
        s = np.sign(s) * np.log1p(np.abs(s))
    return s.astype(np.float32)


def fit_train_only_edge_transform(
    train_df: pd.DataFrame, edge_cols: List[str]
) -> Tuple[Dict[str, Dict[str, Any]], core.StandardScaler]:
    if train_df.empty:
        raise ValueError("Train dataframe is empty; cannot fit transforms.")
    params: Dict[str, Dict[str, Any]] = {}
    transformed = train_df.copy()
    for col in edge_cols:
        s = pd.to_numeric(transformed[col], errors="coerce").fillna(0.0).astype(np.float64)
        low = float(s.quantile(0.001))
        high = float(s.quantile(0.999))
        non_negative = bool(float(s.min()) >= 0.0)
        transformed[col] = _transform_series(transformed[col], low=low, high=high, non_negative=non_negative)
        params[col] = {"low": low, "high": high, "non_negative": non_negative}
    scaler = core.StandardScaler()
    scaler.fit(transformed[edge_cols])
    return (params, scaler)


def apply_train_only_edge_transform(
    df: pd.DataFrame,
    edge_cols: List[str],
    params: Dict[str, Dict[str, Any]],
    scaler: core.StandardScaler,
) -> pd.DataFrame:
    out = df.copy()
    for col in edge_cols:
        p = params[col]
        out[col] = _transform_series(
            out[col],
            low=float(p["low"]),
            high=float(p["high"]),
            non_negative=bool(p["non_negative"]),
        )
    out[edge_cols] = scaler.transform(out[edge_cols])
    return out


def cap_for_split(snaps: List[core.Data], split: str, cfg: Dict[str, Any]) -> List[core.Data]:
    max_key = f"max_{split}_snapshots"
    return core.cap_snapshots(snaps, int(cfg.get(max_key, 0)))


def build_split_snapshots(
    split_frames: Dict[str, pd.DataFrame], edge_cols: List[str], cfg: Dict[str, Any]
) -> Dict[str, List[core.Data]]:
    sync_core_config(cfg)
    out: Dict[str, List[core.Data]] = {}
    for split_name, frame in split_frames.items():
        snaps = core.build_snapshot_graphs(frame, edge_cols)
        snaps = cap_for_split(snaps, split_name, cfg)
        out[split_name] = snaps
    return out


def chunk_sequences(
    snapshots: List[core.Data], chunk_size: int, temporal_context: bool
) -> Iterable[List[core.Data]]:
    if temporal_context:
        for i in range(0, len(snapshots), chunk_size):
            seq = snapshots[i : i + chunk_size]
            if seq:
                yield seq
    else:
        for snap in snapshots:
            yield [snap]


def choose_device(cfg: Dict[str, Any]) -> torch.device:
    pref = str(cfg.get("device", "auto")).lower()
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device='cuda' requested but CUDA is unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict_probs(
    model: torch.nn.Module,
    snapshots: List[core.Data],
    chunk_size: int,
    temporal_context: bool,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true_parts: List[np.ndarray] = []
    y_prob_parts: List[np.ndarray] = []
    with torch.no_grad():
        for seq in chunk_sequences(snapshots, chunk_size=chunk_size, temporal_context=temporal_context):
            seq_dev = [d.to(device) for d in seq]
            logits = model(seq_dev)
            edge_logits = core.concat_logits(logits)
            probs = torch.sigmoid(edge_logits).detach().cpu().numpy()
            targets = core.concat_targets(seq_dev).detach().cpu().numpy().astype(int)
            y_prob_parts.append(probs)
            y_true_parts.append(targets)
    if not y_true_parts:
        return (np.array([], dtype=np.int64), np.array([], dtype=np.float32))
    return (np.concatenate(y_true_parts, axis=0), np.concatenate(y_prob_parts, axis=0))


def select_threshold_with_far(
    y_true: np.ndarray, y_prob: np.ndarray, far_target: float, cfg: Dict[str, Any]
) -> Tuple[float, Dict[str, float]]:
    rows: List[Tuple[float, Dict[str, float]]] = []
    for t in np.arange(
        float(cfg["threshold_min"]),
        float(cfg["threshold_max"]) + 1e-12,
        float(cfg["threshold_step"]),
    ):
        m = core.compute_metrics(y_true, y_prob, threshold=float(t))
        rows.append((float(t), m))
    feasible = [(t, m) for t, m in rows if m["FAR"] <= far_target]
    if feasible:
        t_best, m_best = max(feasible, key=lambda x: (x[1]["f1"], x[1]["recall"]))
        return (float(t_best), m_best)
    t_best, m_best = max(rows, key=lambda x: x[1]["recall"] * max(0.0, 1.0 - x[1]["FAR"]))
    return (float(t_best), m_best)


def train_temporal_gnn(
    train_snaps: List[core.Data],
    val_snaps: List[core.Data],
    test_snaps: List[core.Data],
    cfg: Dict[str, Any],
    temporal_context: bool,
    run_name: str,
) -> Dict[str, Any]:
    if not train_snaps or not val_snaps or not test_snaps:
        raise ValueError(f"{run_name}: one or more snapshot splits are empty")

    device = choose_device(cfg)
    node_in = int(train_snaps[0].x.shape[1])
    edge_in = int(train_snaps[0].edge_attr.shape[1])
    model = core.TemporalEdgeGNN(
        node_in=node_in,
        edge_in=edge_in,
        hidden_dim=int(cfg["hidden_dim"]),
        lstm_hidden=int(cfg["lstm_hidden"]),
        dropout=float(cfg["dropout"]),
    ).to(device)

    y_train = np.concatenate([d.y.detach().cpu().numpy().astype(np.float32) for d in train_snaps], axis=0)
    pos = float(y_train.sum())
    neg = float(max(len(y_train) - pos, 1.0))
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device)
    criterion = core.FocalBCEWithLogits(pos_weight=pos_weight, gamma=float(cfg["focal_gamma"]))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    best_state = deepcopy(model.state_dict())
    best_val_f1 = -1.0
    best_epoch = 0
    bad_epochs = 0
    history: List[Dict[str, float]] = []
    chunk_size = int(cfg["snapshot_chunk_size"]) if temporal_context else 1

    for epoch in range(1, int(cfg["num_epochs"]) + 1):
        model.train()
        batch_losses: List[float] = []
        for seq in chunk_sequences(train_snaps, chunk_size=chunk_size, temporal_context=temporal_context):
            seq_dev = [d.to(device) for d in seq]
            optimizer.zero_grad(set_to_none=True)
            logits = model(seq_dev)
            y_logit = core.concat_logits(logits)
            y_true = core.concat_targets(seq_dev).to(device)
            loss = criterion(y_logit, y_true)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(cfg["grad_clip_norm"]))
            optimizer.step()
            batch_losses.append(float(loss.item()))

        val_true, val_prob = predict_probs(
            model,
            val_snaps,
            chunk_size=chunk_size,
            temporal_context=temporal_context,
            device=device,
        )
        val_metrics = core.compute_metrics(val_true, val_prob, threshold=0.5)
        train_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_f1": float(val_metrics["f1"]),
                "val_recall": float(val_metrics["recall"]),
                "val_far": float(val_metrics["FAR"]),
            }
        )

        scheduler.step(float(val_metrics["f1"]))
        if float(val_metrics["f1"]) > best_val_f1 + 1e-6:
            best_val_f1 = float(val_metrics["f1"])
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= int(cfg["patience"]):
            log(f"{run_name}: early stopping at epoch {epoch} (best epoch={best_epoch}, best val_f1={best_val_f1:.4f})")
            break

    model.load_state_dict(best_state)
    val_true, val_prob = predict_probs(
        model,
        val_snaps,
        chunk_size=chunk_size,
        temporal_context=temporal_context,
        device=device,
    )
    test_true, test_prob = predict_probs(
        model,
        test_snaps,
        chunk_size=chunk_size,
        temporal_context=temporal_context,
        device=device,
    )
    threshold, _ = select_threshold_with_far(
        val_true,
        val_prob,
        far_target=float(cfg["threshold_far_target"]),
        cfg=cfg,
    )
    test_metrics = core.compute_metrics(test_true, test_prob, threshold=threshold)
    return {
        "run_name": run_name,
        "model": model,
        "state_dict": deepcopy(model.state_dict()),
        "threshold": float(threshold),
        "best_epoch": int(best_epoch),
        "history": history,
        "test_metrics": test_metrics,
    }


def train_baselines(
    train_snaps: List[core.Data], val_snaps: List[core.Data], test_snaps: List[core.Data], cfg: Dict[str, Any]
) -> List[Dict[str, Any]]:
    x_train, y_train = core.flatten_snapshots(train_snaps)
    x_val, y_val = core.flatten_snapshots(val_snaps)
    x_test, y_test = core.flatten_snapshots(test_snaps)

    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            random_state=int(cfg["seed"]),
            n_jobs=-1,
            class_weight="balanced_subsample",
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=2000,
            random_state=int(cfg["seed"]),
            class_weight="balanced",
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            random_state=int(cfg["seed"]),
            early_stopping=True,
            max_iter=200,
        ),
    }

    results: List[Dict[str, Any]] = []
    for model_name, clf in models.items():
        try:
            clf.fit(x_train, y_train)
            val_prob = clf.predict_proba(x_val)[:, 1]
            test_prob = clf.predict_proba(x_test)[:, 1]
            threshold, _ = select_threshold_with_far(
                y_val,
                val_prob,
                far_target=float(cfg["threshold_far_target"]),
                cfg=cfg,
            )
            metrics = core.compute_metrics(y_test, test_prob, threshold=threshold)
            results.append(
                {
                    "model": model_name,
                    "threshold": float(threshold),
                    "test_metrics": metrics,
                }
            )
        except ValueError as exc:
            log(f"Skipping baseline '{model_name}' due to training error: {exc}")
    return results


def prepare_dataframe(cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, List[str]]:
    core.set_seed(int(cfg["seed"]))
    raw_df = core.load_and_merge_cicids(Path(cfg["data_dir"]), cfg["max_rows_per_file"])
    raw_df = core.add_binary_label(raw_df)
    raw_df = core.clean_dataset(raw_df)
    raw_df = core.prepare_time_and_endpoints(raw_df)
    raw_df = add_snapshot_ids(raw_df, cfg)
    edge_cols = resolve_edge_features(raw_df)
    return (raw_df, edge_cols)


def run_single_seed(
    base_cfg: Dict[str, Any],
    raw_df: pd.DataFrame,
    edge_cols: List[str],
    seed: int,
    epoch_override: int | None = None,
) -> Dict[str, Any]:
    cfg = deepcopy(base_cfg)
    cfg["seed"] = int(seed)
    if epoch_override is not None:
        cfg["num_epochs"] = int(epoch_override)
    sync_core_config(cfg)
    core.set_seed(int(seed))

    split_info = split_snapshot_ids(raw_df, cfg)
    train_ids = set(split_info["train_ids"])
    val_ids = set(split_info["val_ids"])
    test_ids = set(split_info["test_ids"])

    split_frames = {
        "train": raw_df[raw_df["snapshot_id"].isin(train_ids)].copy(),
        "val": raw_df[raw_df["snapshot_id"].isin(val_ids)].copy(),
        "test": raw_df[raw_df["snapshot_id"].isin(test_ids)].copy(),
    }

    transform_params, scaler = fit_train_only_edge_transform(split_frames["train"], edge_cols)
    for split_name in ("train", "val", "test"):
        split_frames[split_name] = apply_train_only_edge_transform(
            split_frames[split_name], edge_cols=edge_cols, params=transform_params, scaler=scaler
        )

    split_snaps = build_split_snapshots(split_frames, edge_cols=edge_cols, cfg=cfg)
    train_snaps = split_snaps["train"]
    val_snaps = split_snaps["val"]
    test_snaps = split_snaps["test"]

    temporal = train_temporal_gnn(
        train_snaps=train_snaps,
        val_snaps=val_snaps,
        test_snaps=test_snaps,
        cfg=cfg,
        temporal_context=True,
        run_name="TemporalGNN_LSTM",
    )
    non_temporal = train_temporal_gnn(
        train_snaps=train_snaps,
        val_snaps=val_snaps,
        test_snaps=test_snaps,
        cfg=cfg,
        temporal_context=False,
        run_name="TemporalGNN_NoTemporal",
    )
    baseline_runs = train_baselines(train_snaps, val_snaps, test_snaps, cfg)

    model_rows: List[Dict[str, Any]] = []
    for item in (
        {
            "model": temporal["run_name"],
            "threshold": temporal["threshold"],
            "seed": seed,
            **temporal["test_metrics"],
        },
        {
            "model": non_temporal["run_name"],
            "threshold": non_temporal["threshold"],
            "seed": seed,
            **non_temporal["test_metrics"],
        },
    ):
        model_rows.append(item)
    for b in baseline_runs:
        model_rows.append(
            {
                "model": b["model"],
                "threshold": b["threshold"],
                "seed": seed,
                **b["test_metrics"],
            }
        )

    comparison_df = pd.DataFrame(model_rows).sort_values("f1", ascending=False).reset_index(drop=True)
    ablation_df = comparison_df[comparison_df["model"].isin(["TemporalGNN_LSTM", "TemporalGNN_NoTemporal"])].copy()
    split_edge_report = {
        "train": core.split_edge_stats(train_snaps),
        "val": core.split_edge_stats(val_snaps),
        "test": core.split_edge_stats(test_snaps),
    }

    model_path = Path(cfg["models_dir"]) / f"temporal_gnn_lstm_seed{seed}.pt"
    torch.save(temporal["state_dict"], model_path)
    temporal_model_kwargs = {
        "node_in": int(train_snaps[0].x.shape[1]),
        "edge_in": int(train_snaps[0].edge_attr.shape[1]),
        "hidden_dim": int(cfg["hidden_dim"]),
        "lstm_hidden": int(cfg["lstm_hidden"]),
        "dropout": float(cfg["dropout"]),
    }

    return {
        "seed": seed,
        "cfg": cfg,
        "comparison_df": comparison_df,
        "ablation_df": ablation_df,
        "split_info": split_info,
        "split_edge_report": split_edge_report,
        "temporal_model_path": str(model_path),
        "temporal_history": temporal["history"],
        "temporal_threshold": float(temporal["threshold"]),
        "edge_cols": edge_cols,
        "transform_params": transform_params,
        "scaler_mean": scaler.mean_.astype(float).tolist(),
        "scaler_scale": scaler.scale_.astype(float).tolist(),
        "temporal_model_kwargs": temporal_model_kwargs,
    }


def build_significance_report(seed_metrics: pd.DataFrame, anchor_model: str = "TemporalGNN_LSTM") -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if seed_metrics.empty:
        return pd.DataFrame(rows)

    for model_name in sorted(seed_metrics["model"].unique()):
        if model_name == anchor_model:
            continue
        a = seed_metrics[seed_metrics["model"] == anchor_model][["seed", "f1"]].rename(columns={"f1": "f1_anchor"})
        b = seed_metrics[seed_metrics["model"] == model_name][["seed", "f1"]].rename(columns={"f1": "f1_other"})
        paired = a.merge(b, on="seed", how="inner").sort_values("seed")
        n = len(paired)
        if n < 2:
            rows.append(
                {
                    "anchor_model": anchor_model,
                    "compared_model": model_name,
                    "n_seeds": n,
                    "mean_diff_f1": float("nan"),
                    "ttest_pvalue": float("nan"),
                    "wilcoxon_pvalue": float("nan"),
                }
            )
            continue
        diff = paired["f1_anchor"] - paired["f1_other"]
        t_p = float(ttest_rel(paired["f1_anchor"], paired["f1_other"]).pvalue)
        try:
            w_p = float(wilcoxon(paired["f1_anchor"], paired["f1_other"]).pvalue)
        except ValueError:
            w_p = float("nan")
        rows.append(
            {
                "anchor_model": anchor_model,
                "compared_model": model_name,
                "n_seeds": int(n),
                "mean_diff_f1": float(diff.mean()),
                "ttest_pvalue": t_p,
                "wilcoxon_pvalue": w_p,
            }
        )
    return pd.DataFrame(rows)


def write_manifest(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run reproducible temporal GNN training and IEEE-ready reporting.")
    parser.add_argument("--config", type=str, default="config.yml", help="Path to YAML config.")
    parser.add_argument("--seed", type=int, default=None, help="Override primary seed.")
    parser.add_argument("--epochs", type=int, default=None, help="Override num_epochs for primary run.")
    parser.add_argument(
        "--split-mode",
        type=str,
        default=None,
        choices=["per_day_temporal", "chronological", "stratified_snapshot"],
        help="Override split mode.",
    )
    parser.add_argument("--skip-significance", action="store_true", help="Skip multi-seed significance runs.")
    parser.add_argument(
        "--significance-seeds",
        type=str,
        default=None,
        help="Comma-separated seeds for significance (e.g., 42,52,62).",
    )
    parser.add_argument("--fast-smoke", action="store_true", help="Run a quick smoke configuration for verification.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))

    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    if args.epochs is not None:
        cfg["num_epochs"] = int(args.epochs)
    if args.split_mode is not None:
        cfg["split_mode"] = str(args.split_mode)
    if args.significance_seeds is not None:
        cfg["seeds_for_significance"] = parse_seed_list(args.significance_seeds)

    if args.fast_smoke:
        cfg["max_rows_per_file"] = 5000
        cfg["max_train_snapshots"] = 40
        cfg["max_val_snapshots"] = 20
        cfg["max_test_snapshots"] = 20
        cfg["num_epochs"] = 2
        cfg["significance_epochs"] = 1
        cfg["seeds_for_significance"] = [int(cfg["seed"])]
        cfg["verbose"] = False

    ensure_dirs(cfg)
    log("Loading and preprocessing raw dataframe...")
    raw_df, edge_cols = prepare_dataframe(cfg)
    log(
        f"Prepared rows={len(raw_df):,}, snapshots={raw_df['snapshot_id'].nunique():,}, "
        f"attack_ratio={raw_df['binary_label'].mean():.4f}"
    )

    log(f"Running primary experiment (seed={cfg['seed']}, split_mode={cfg['split_mode']})...")
    main_result = run_single_seed(cfg, raw_df, edge_cols, seed=int(cfg["seed"]), epoch_override=int(cfg["num_epochs"]))

    eval_dir = Path(cfg["evaluation_dir"])
    comparison_path = eval_dir / "model_comparison.csv"
    ablation_path = eval_dir / "ablation_results.csv"
    manifest_path = eval_dir / "run_manifest.json"
    deployment_path = eval_dir / "deployment_artifacts.json"
    seed_metrics_path = eval_dir / "seed_metrics.csv"
    significance_path = eval_dir / "significance_report.csv"

    main_result["comparison_df"].to_csv(comparison_path, index=False)
    main_result["ablation_df"].to_csv(ablation_path, index=False)
    log(f"Wrote {comparison_path}")
    log(f"Wrote {ablation_path}")

    seed_metrics_frames = [main_result["comparison_df"].copy()]

    if not args.skip_significance:
        seeds = parse_seed_list(cfg.get("seeds_for_significance", []))
        seeds = sorted(set(seeds))
        extra_seeds = [s for s in seeds if s != int(cfg["seed"])]
        for seed in extra_seeds:
            log(f"Running significance seed={seed} (epochs={cfg['significance_epochs']})...")
            seed_result = run_single_seed(
                cfg,
                raw_df,
                edge_cols,
                seed=seed,
                epoch_override=int(cfg["significance_epochs"]),
            )
            seed_metrics_frames.append(seed_result["comparison_df"].copy())

    seed_metrics_df = pd.concat(seed_metrics_frames, ignore_index=True)
    seed_metrics_df.to_csv(seed_metrics_path, index=False)
    log(f"Wrote {seed_metrics_path}")

    significance_df = build_significance_report(seed_metrics_df)
    significance_df.to_csv(significance_path, index=False)
    log(f"Wrote {significance_path}")

    manifest = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "primary_seed": int(cfg["seed"]),
        "config": cfg,
        "leakage_guard": {
            "fit_on_train_only": True,
            "description": "Edge feature clip/log/standardization fitted on train split only and applied to val/test.",
        },
        "split_info": main_result["split_info"],
        "split_edge_report": main_result["split_edge_report"],
        "temporal_model_path": main_result["temporal_model_path"],
        "deployment_artifacts_path": str(deployment_path),
    }
    write_manifest(manifest_path, manifest)
    log(f"Wrote {manifest_path}")

    deployment = {
        "timestamp_utc": manifest["timestamp_utc"],
        "seed": int(cfg["seed"]),
        "window_size": str(cfg["window_size"]),
        "snapshot_chunk_size": int(cfg["snapshot_chunk_size"]),
        "max_edges_per_snapshot": int(cfg["max_edges_per_snapshot"]),
        "threshold": float(main_result["temporal_threshold"]),
        "edge_feature_cols": list(main_result["edge_cols"]),
        "transform_params": main_result["transform_params"],
        "scaler_mean": main_result["scaler_mean"],
        "scaler_scale": main_result["scaler_scale"],
        "model_state_dict_path": main_result["temporal_model_path"],
        "model_kwargs": main_result["temporal_model_kwargs"],
        "split_mode": str(cfg["split_mode"]),
        "leakage_guard": {
            "fit_on_train_only": True,
            "description": "Edge feature transform and scaler fitted using train split only.",
        },
    }
    write_manifest(deployment_path, deployment)
    log(f"Wrote {deployment_path}")
    log("Done.")


if __name__ == "__main__":
    main()
