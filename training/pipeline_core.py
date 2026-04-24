"""Core pipeline utilities extracted from main_pipeline.ipynb for reproducible runs.
This file is auto-generated and then manually maintained.
"""

import os
import json
import warnings
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import permutation_importance
NOTEBOOK_START = time.perf_counter()

def format_seconds(sec: float) -> str:
    sec = float(max(0.0, sec))
    if sec < 60:
        return f'{sec:.1f}s'
    if sec < 3600:
        return f'{sec / 60:.1f}m'
    return f'{sec / 3600:.2f}h'

def log_step(msg: str):
    elapsed = time.perf_counter() - NOTEBOOK_START
    print(f'[{format_seconds(elapsed):>8}] {msg}', flush=True)

def progress_table(stage: str, current: int, total: int, start_time: float, extra: str='') -> None:
    total = max(int(total), 1)
    current = min(max(int(current), 0), total)
    pct = 100.0 * current / total
    bar_w = 24
    filled = int(bar_w * current / total)
    bar = '#' * filled + '-' * (bar_w - filled)
    elapsed = time.perf_counter() - start_time
    eta = elapsed / current * (total - current) if current > 0 else float('nan')
    eta_txt = format_seconds(eta) if np.isfinite(eta) else '--'
    line = f'{stage:<14} [{bar}] {current:>4}/{total:<4} {pct:5.1f}% | elapsed={format_seconds(elapsed)} | ETA={eta_txt}'
    if extra:
        line += f' | {extra}'
    print('\r' + line, end='', flush=True)
    if current >= total:
        print()
CONFIG = {'seed': 42, 'data_dir': Path('../data'), 'window_size': '300s', 'max_rows_per_file': 200000, 'num_epochs': 40, 'lr': 0.0008, 'weight_decay': 0.0001, 'dropout': 0.2, 'hidden_dim': 64, 'lstm_hidden': 64, 'patience': 8, 'batch_stream_size': 5, 'snapshot_chunk_size': 4, 'split_mode': 'per_day_temporal', 'split_train': 0.7, 'split_val': 0.15, 'split_test': 0.15, 'grad_clip_norm': 1.0, 'focal_gamma': 1.5, 'max_edges_per_snapshot': 3000, 'max_train_snapshots': 500, 'max_val_snapshots': 150, 'max_test_snapshots': 150, 'threshold_far_target': 0.3, 'threshold_min': 0.05, 'threshold_max': 0.95, 'threshold_step': 0.01, 'chunk_log_interval': 1, 'verbose': True, 'compact_progress': True}

def set_seed(seed: int=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip() for c in df.columns]
    seen = {}
    new_cols = []
    for c in df.columns:
        if c not in seen:
            seen[c] = 0
            new_cols.append(c)
        else:
            seen[c] += 1
            new_cols.append(f'{c}__dup{seen[c]}')
    df.columns = new_cols
    return df

def load_and_merge_cicids(data_dir: Path, max_rows_per_file=None) -> pd.DataFrame:
    required_cols = {'Label', 'Flow Duration', 'Destination Port', 'Total Fwd Packets', 'Total Backward Packets', 'Total Length of Fwd Packets', 'Total Length of Bwd Packets', 'Flow Bytes/s', 'Flow Packets/s', 'Source IP', 'Src IP', 'source_ip', 'Destination IP', 'Dst IP', 'destination_ip', 'Flow ID', 'Flow_ID', 'Timestamp', 'Flow Start', 'Date'}
    csv_paths = sorted(data_dir.glob('*.csv'))
    log_step(f'Loading {len(csv_paths)} CSV files from {data_dir} (max_rows_per_file={max_rows_per_file})')
    frames = []
    total_rows = 0
    t_all = time.perf_counter()
    for i, csv_path in enumerate(csv_paths, start=1):
        t_file = time.perf_counter()

        def _use_col(c):
            return str(c).strip() in required_cols
        tmp = pd.read_csv(csv_path, low_memory=True, nrows=max_rows_per_file, usecols=_use_col)
        tmp = standardize_columns(tmp)
        tmp['day_file'] = csv_path.name
        for col in tmp.columns:
            if col in {'day_file', 'Label', 'Source IP', 'Src IP', 'source_ip', 'Destination IP', 'Dst IP', 'destination_ip', 'Flow ID', 'Flow_ID', 'Timestamp', 'Flow Start', 'Date'}:
                continue
            tmp[col] = pd.to_numeric(tmp[col], errors='coerce', downcast='float')
        frames.append(tmp)
        total_rows += len(tmp)
        log_step(f'[{i}/{len(csv_paths)}] {csv_path.name}: rows={len(tmp):,}, cols={len(tmp.columns)}, time={format_seconds(time.perf_counter() - t_file)}')
    merged = pd.concat(frames, axis=0, ignore_index=True)
    log_step(f'Merged dataframe shape={merged.shape} (total_rows_read={total_rows:,}) in {format_seconds(time.perf_counter() - t_all)}')
    return merged

def locate_column(df: pd.DataFrame, candidates: List[str]) -> str:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return ''

def add_binary_label(df: pd.DataFrame) -> pd.DataFrame:
    t0 = time.perf_counter()
    label_col = locate_column(df, ['Label'])
    if not label_col:
        raise ValueError('Label column was not found.')
    df[label_col] = df[label_col].astype(str).str.strip()
    df['binary_label'] = np.where(df[label_col].str.upper() == 'BENIGN', 0, 1).astype(np.int8)
    log_step(f"Binary labels created from '{label_col}' in {format_seconds(time.perf_counter() - t0)}")
    return df

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    t0 = time.perf_counter()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    before = len(df)
    df = df.dropna(axis=0).reset_index(drop=True)
    after = len(df)
    log_step(f'Clean step: removed {before - after:,} rows containing NaN/Inf; remaining={after:,}; time={format_seconds(time.perf_counter() - t0)}')
    return df

def prepare_time_and_endpoints(df: pd.DataFrame) -> pd.DataFrame:
    t0 = time.perf_counter()
    src_col = locate_column(df, ['Source IP', 'Src IP', 'source_ip'])
    dst_col = locate_column(df, ['Destination IP', 'Dst IP', 'destination_ip'])
    if not src_col or not dst_col:
        flow_id_col = locate_column(df, ['Flow ID', 'Flow_ID'])
        if flow_id_col:
            parts = df[flow_id_col].astype(str).str.split('-', expand=True)
            if parts.shape[1] >= 2:
                df['Source IP'] = parts[0]
                df['Destination IP'] = parts[1]
                src_col, dst_col = ('Source IP', 'Destination IP')
                log_step('Endpoint fallback: extracted Source/Destination IP from Flow ID.')
    if not src_col or not dst_col:
        dport_col = locate_column(df, ['Destination Port'])
        if not dport_col:
            raise ValueError('Cannot construct endpoints: Destination Port and IP columns are missing.')
        synthetic_src = pd.util.hash_pandas_object(df[[dport_col]].astype(str), index=False) % 5000
        df['Source IP'] = 'SRC_' + synthetic_src.astype(str)
        df['Destination IP'] = 'DSTPORT_' + df[dport_col].astype(str)
        src_col, dst_col = ('Source IP', 'Destination IP')
        log_step('Endpoint fallback: generated synthetic Source/Destination endpoints.')
    ts_col = locate_column(df, ['Timestamp', 'Flow Start', 'Date'])
    if ts_col:
        ts = pd.to_datetime(df[ts_col], errors='coerce')
        log_step(f"Using time column '{ts_col}'")
    else:
        dur_col = locate_column(df, ['Flow Duration'])
        if not dur_col:
            raise ValueError('No Timestamp or Flow Duration column found to build temporal snapshots.')
        duration_us = pd.to_numeric(df[dur_col], errors='coerce').fillna(0).clip(lower=0)
        ts = pd.Timestamp('2017-01-01') + pd.to_timedelta(duration_us.cumsum(), unit='us')
        log_step('Time fallback: built synthetic timeline from cumulative Flow Duration.')
    df['event_time'] = ts
    df = df.dropna(subset=['event_time']).reset_index(drop=True)
    df['Source IP'] = df[src_col].astype(str)
    df['Destination IP'] = df[dst_col].astype(str)
    log_step(f'Prepared endpoints and event_time in {format_seconds(time.perf_counter() - t0)}')
    return df
EDGE_FEATURE_CANDIDATES = ['Flow Duration', 'Total Fwd Packets', 'Total Backward Packets', 'Total Length of Fwd Packets', 'Total Length of Bwd Packets', 'Flow Bytes/s', 'Flow Packets/s']

def encode_and_normalize(df: pd.DataFrame, edge_cols: List[str]) -> Tuple[pd.DataFrame, StandardScaler]:
    t0 = time.perf_counter()
    df = df.copy()
    protected = {'binary_label', 'Source IP', 'Destination IP', 'event_time', 'day_file'}
    object_cols = [c for c in df.columns if df[c].dtype == 'object' and c not in protected]
    if object_cols:
        df = pd.get_dummies(df, columns=object_cols, drop_first=True)
    for c in edge_cols:
        s = pd.to_numeric(df[c], errors='coerce').fillna(0.0).astype(np.float64)
        low = float(s.quantile(0.001))
        high = float(s.quantile(0.999))
        s = s.clip(lower=low, upper=high)
        if float(s.min()) >= 0.0:
            s = np.log1p(s)
        else:
            s = np.sign(s) * np.log1p(np.abs(s))
        df[c] = s.astype(np.float32)
    scaler = StandardScaler()
    df[edge_cols] = scaler.fit_transform(df[edge_cols])
    log_step(f'Encoded+normalized dataframe in {format_seconds(time.perf_counter() - t0)}; shape={df.shape}')
    return (df, scaler)

def build_snapshot_graphs(df: pd.DataFrame, edge_cols: List[str]) -> List[Data]:
    t0 = time.perf_counter()
    snapshots = []
    grouped = list(df.sort_values('event_time').groupby('snapshot_id', sort=True))
    total_groups = len(grouped)
    max_edges = int(CONFIG.get('max_edges_per_snapshot', 0) or 0)
    log_step(f"Graph construction started: {total_groups} snapshot groups (max_edges_per_snapshot={max_edges or 'none'})")
    for k, (sid, g) in enumerate(grouped, start=1):
        if len(g) < 2:
            continue
        if max_edges > 0 and len(g) > max_edges:
            g = g.sample(n=max_edges, random_state=CONFIG['seed']).sort_values('event_time')
        src_nodes = g['Source IP'].astype(str)
        dst_nodes = g['Destination IP'].astype(str)
        node_names = pd.Index(pd.concat([src_nodes, dst_nodes]).unique())
        node_to_idx = {n: i for i, n in enumerate(node_names)}
        src_idx = src_nodes.map(node_to_idx).to_numpy()
        dst_idx = dst_nodes.map(node_to_idx).to_numpy()
        edge_index = torch.tensor(np.vstack([src_idx, dst_idx]), dtype=torch.long)
        edge_attr = torch.tensor(g[edge_cols].to_numpy(dtype=np.float32), dtype=torch.float32)
        edge_y = torch.tensor(g['binary_label'].to_numpy(dtype=np.float32), dtype=torch.float32)
        out_mean = g.groupby('Source IP')[edge_cols].mean()
        in_mean = g.groupby('Destination IP')[edge_cols].mean()
        conn_counts = (g['Source IP'].value_counts() + g['Destination IP'].value_counts()).fillna(0)
        node_feats = []
        for n in node_names:
            out_vec = out_mean.loc[n].to_numpy() if n in out_mean.index else np.zeros(len(edge_cols), dtype=np.float64)
            in_vec = in_mean.loc[n].to_numpy() if n in in_mean.index else np.zeros(len(edge_cols), dtype=np.float64)
            mean_vec = (out_vec + in_vec) / 2.0
            count = float(np.log1p(conn_counts.get(n, 0.0)))
            node_feats.append(np.concatenate([mean_vec, [count]], axis=0))
        x_np = np.asarray(node_feats, dtype=np.float32)
        if x_np.shape[0] > 1:
            count_col = x_np[:, -1]
            x_np[:, -1] = (count_col - float(count_col.mean())) / (float(count_col.std()) + 1e-06)
        x = torch.tensor(x_np, dtype=torch.float32)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=edge_y)
        data.snapshot_id = int(sid)
        data.node_names = node_names.to_list()
        if 'day_file' in g.columns:
            data.day_file = str(g['day_file'].mode().iloc[0])
        else:
            data.day_file = 'unknown'
        snapshots.append(data)
        if CONFIG.get('verbose', True) and (k % 25 == 0 or k == total_groups):
            elapsed = time.perf_counter() - t0
            rate = k / max(elapsed, 1e-09)
            remain = (total_groups - k) / max(rate, 1e-09)
            log_step(f'Graph progress: {k}/{total_groups} groups processed, built={len(snapshots)}, elapsed={format_seconds(elapsed)}, ETA={format_seconds(remain)}')
    log_step(f'Graph construction finished: snapshots={len(snapshots)}, total_time={format_seconds(time.perf_counter() - t0)}')
    return snapshots
import networkx as nx

def _split_sequence(seq: List[Data], train: float, val: float, test: float):
    n = len(seq)
    if n < 3:
        return (seq[:1], seq[1:2], seq[2:])
    n_train = max(1, int(n * train))
    n_val = max(1, int(n * val))
    n_test = n - n_train - n_val
    if n_test < 1:
        n_test = 1
        if n_train > n_val:
            n_train = max(1, n_train - 1)
        else:
            n_val = max(1, n_val - 1)
    train_snaps = seq[:n_train]
    val_snaps = seq[n_train:n_train + n_val]
    test_snaps = seq[n_train + n_val:]
    return (train_snaps, val_snaps, test_snaps)

def temporal_split_snapshots(snapshots: List[Data], train=0.7, val=0.15, test=0.15, mode: str='chronological'):
    assert abs(train + val + test - 1.0) < 1e-08
    snapshots = sorted(snapshots, key=lambda d: int(d.snapshot_id))
    n = len(snapshots)
    if n < 3:
        raise ValueError('Need at least 3 snapshots for train/val/test split.')
    if mode == 'stratified_snapshot':
        from sklearn.model_selection import train_test_split
        idx_all = np.arange(n)
        attack_ratio = np.array([float(d.y.float().mean().item()) for d in snapshots], dtype=np.float64)
        median_ratio = float(np.median(attack_ratio))
        strata = (attack_ratio >= median_ratio).astype(int)
        if np.unique(strata).size < 2:
            strata = None
        idx_train, idx_temp = train_test_split(idx_all, test_size=1.0 - train, random_state=int(CONFIG.get('seed', 42)), stratify=strata, shuffle=True)
        temp_val_fraction = val / max(val + test, 1e-12)
        strata_temp = strata[idx_temp] if strata is not None else None
        if strata_temp is not None and np.unique(strata_temp).size < 2:
            strata_temp = None
        idx_val, idx_test = train_test_split(idx_temp, test_size=1.0 - temp_val_fraction, random_state=int(CONFIG.get('seed', 42)), stratify=strata_temp, shuffle=True)
        train_snaps = [snapshots[i] for i in sorted(idx_train)]
        val_snaps = [snapshots[i] for i in sorted(idx_val)]
        test_snaps = [snapshots[i] for i in sorted(idx_test)]
        return (train_snaps, val_snaps, test_snaps)
    if mode == 'per_day_temporal':
        day_groups: Dict[str, List[Data]] = {}
        for d in snapshots:
            day_key = str(getattr(d, 'day_file', 'unknown'))
            day_groups.setdefault(day_key, []).append(d)
        train_snaps, val_snaps, test_snaps = ([], [], [])
        for day_key in sorted(day_groups.keys()):
            day_seq = sorted(day_groups[day_key], key=lambda d: int(d.snapshot_id))
            tr, va, te = _split_sequence(day_seq, train, val, test)
            train_snaps.extend(tr)
            val_snaps.extend(va)
            test_snaps.extend(te)
        if len(train_snaps) == 0 or len(val_snaps) == 0 or len(test_snaps) == 0:
            log_step('Per-day split fallback triggered; switching to chronological split.')
            train_snaps, val_snaps, test_snaps = _split_sequence(snapshots, train, val, test)
        train_snaps = sorted(train_snaps, key=lambda d: int(d.snapshot_id))
        val_snaps = sorted(val_snaps, key=lambda d: int(d.snapshot_id))
        test_snaps = sorted(test_snaps, key=lambda d: int(d.snapshot_id))
        return (train_snaps, val_snaps, test_snaps)
    return _split_sequence(snapshots, train, val, test)

def cap_snapshots(snaps: List[Data], max_count: int) -> List[Data]:
    max_count = int(max_count or 0)
    if max_count <= 0 or len(snaps) <= max_count:
        return snaps
    idx = np.linspace(0, len(snaps) - 1, num=max_count, dtype=int)
    idx = np.unique(idx)
    return [snaps[i] for i in idx]

def split_edge_stats(snaps: List[Data]) -> Dict[str, float]:
    total = int(sum((int(d.y.numel()) for d in snaps)))
    pos = int(sum((float(d.y.sum().item()) for d in snaps)))
    ratio = float(pos / max(total, 1))
    return {'edges': total, 'attack_edges': pos, 'attack_ratio': ratio}

class TemporalEdgeGNN(nn.Module):

    def __init__(self, node_in: int, edge_in: int, hidden_dim: int=64, lstm_hidden: int=64, dropout: float=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm_hidden = lstm_hidden
        self.gat1 = GATConv(node_in, hidden_dim, heads=2, concat=False, dropout=dropout)
        self.gat2 = GATConv(hidden_dim, hidden_dim, heads=1, concat=False, dropout=dropout)
        self.edge_mlp = nn.Sequential(nn.Linear(hidden_dim * 2 + edge_in, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.temporal_lstm = nn.LSTM(input_size=hidden_dim, hidden_size=lstm_hidden, num_layers=1, batch_first=True)
        self.classifier = nn.Sequential(nn.Linear(hidden_dim + lstm_hidden, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def _run_lstm_safe(self, seq: torch.Tensor) -> torch.Tensor:
        try:
            out, _ = self.temporal_lstm(seq)
            return out
        except RuntimeError as exc:
            msg = str(exc).lower()
            if 'cudnn_status_not_supported' in msg or 'cudnn' in msg:
                with torch.backends.cudnn.flags(enabled=False):
                    out, _ = self.temporal_lstm(seq)
                    return out
            raise

    def forward(self, snapshots: List[Data]) -> List[torch.Tensor]:
        edge_hidden_list: List[torch.Tensor] = []
        context_list: List[torch.Tensor] = []
        for d in snapshots:
            x = F.elu(self.gat1(d.x, d.edge_index))
            x = F.dropout(x, p=0.2, training=self.training)
            x = F.elu(self.gat2(x, d.edge_index))
            src = d.edge_index[0]
            dst = d.edge_index[1]
            edge_input = torch.cat([x[src], x[dst], d.edge_attr], dim=1)
            edge_hidden = self.edge_mlp(edge_input)
            edge_hidden_list.append(edge_hidden)
            graph_context = edge_hidden.mean(dim=0, keepdim=True)
            context_list.append(graph_context)
        context_seq = torch.stack(context_list, dim=1)
        temporal_out = self._run_lstm_safe(context_seq).squeeze(0)
        logits = []
        for i, edge_hidden in enumerate(edge_hidden_list):
            t_context = temporal_out[i].unsqueeze(0).expand(edge_hidden.size(0), -1)
            logit = self.classifier(torch.cat([edge_hidden, t_context], dim=1)).squeeze(-1)
            logits.append(logit)
        return logits

def concat_logits(logits: List[torch.Tensor]) -> torch.Tensor:
    return torch.cat([z.reshape(-1) for z in logits], dim=0)

def concat_targets(snapshots: List[Data]) -> torch.Tensor:
    return torch.cat([d.y.float().reshape(-1) for d in snapshots], dim=0)

class FocalBCEWithLogits(nn.Module):

    def __init__(self, pos_weight: torch.Tensor, gamma: float=1.5):
        super().__init__()
        self.register_buffer('pos_weight', pos_weight.reshape(1))
        self.gamma = float(max(gamma, 0.0))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none', pos_weight=self.pos_weight)
        probs = torch.sigmoid(logits)
        pt = torch.where(targets > 0.5, probs, 1.0 - probs)
        focal_factor = (1.0 - pt).pow(self.gamma)
        return (focal_factor * bce).mean()

def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray, min_t: float=0.05, max_t: float=0.95, step: float=0.01) -> Tuple[float, float]:
    thresholds = np.arange(min_t, max_t + 1e-12, step)
    best_t = 0.5
    best_f1 = -1.0
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        cur_f1 = f1_score(y_true, y_pred, zero_division=0)
        if cur_f1 > best_f1:
            best_f1 = float(cur_f1)
            best_t = float(t)
    return (best_t, best_f1)

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float=0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        roc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc = float('nan')
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    far = fp / (fp + tn + 1e-12)
    dr = tp / (tp + fn + 1e-12)
    return {'accuracy': float(acc), 'precision': float(prec), 'recall': float(rec), 'f1': float(f1), 'roc_auc': float(roc), 'FAR': float(far), 'DR': float(dr), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}

def snapshot_chunks(snapshots: List[Data], chunk_size: int):
    for i in range(0, len(snapshots), chunk_size):
        yield snapshots[i:i + chunk_size]

def move_chunk_to_device(snap_chunk: List[Data], device: torch.device) -> List[Data]:
    return [d.to(device) for d in snap_chunk]

def evaluate_temporal(model: nn.Module, snapshots: List[Data], chunk_size: int=4) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    model.eval()
    y_prob_parts, y_true_parts = ([], [])
    t_eval = time.perf_counter()
    with torch.no_grad():
        chunks = list(snapshot_chunks(snapshots, chunk_size))
        total_chunks = len(chunks)
        for i, chunk in enumerate(chunks, start=1):
            t_chunk = time.perf_counter()
            chunk_dev = move_chunk_to_device(chunk, device)
            logits = model(chunk_dev)
            y_logit = concat_logits(logits).detach().cpu().numpy()
            y_prob_parts.append(1.0 / (1.0 + np.exp(-y_logit)))
            y_true_parts.append(concat_targets(chunk_dev).detach().cpu().numpy().astype(int))
            del chunk_dev, logits
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if CONFIG.get('compact_progress', True):
                progress_table(stage='Eval', current=i, total=total_chunks, start_time=t_eval, extra=f'chunk={format_seconds(time.perf_counter() - t_chunk)}')
            elif CONFIG.get('verbose', True) and (i % max(1, int(CONFIG.get('chunk_log_interval', 1))) == 0 or i == total_chunks):
                elapsed = time.perf_counter() - t_eval
                rate = i / max(elapsed, 1e-09)
                remain = (total_chunks - i) / max(rate, 1e-09)
                log_step(f'Eval progress: chunk {i}/{total_chunks} | chunk_time={format_seconds(time.perf_counter() - t_chunk)} | ETA={format_seconds(remain)}')
    y_prob = np.concatenate(y_prob_parts, axis=0)
    y_true = np.concatenate(y_true_parts, axis=0)
    metrics = compute_metrics(y_true, y_prob)
    log_step(f'Evaluation complete in {format_seconds(time.perf_counter() - t_eval)}')
    return (y_true, y_prob, metrics)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class TemporalEdgeGNN_TemporalAttention(nn.Module):

    def __init__(self, node_in, edge_in, hidden_dim=128, gat_heads=2, lstm_hidden=64, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gat1 = GATConv(node_in, hidden_dim, heads=gat_heads, concat=False, dropout=dropout)
        self.gat2 = GATConv(hidden_dim, hidden_dim, heads=1, concat=False, dropout=dropout)
        self.edge_encoder = nn.Sequential(nn.Linear(edge_in, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.temporal_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_score = nn.Linear(hidden_dim, 1)
        self.lstm = nn.LSTM(hidden_dim, lstm_hidden, batch_first=True)
        self.classifier = nn.Sequential(nn.Linear(lstm_hidden, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, snap_seq):
        device = next(self.parameters()).device
        edge_seq = []
        for d in snap_seq:
            x = d.x.to(device)
            edge_index = d.edge_index.to(device)
            edge_attr = d.edge_attr.to(device)
            x = self.gat1(x, edge_index)
            x = F.elu(x)
            x = self.gat2(x, edge_index)
            x = F.elu(x)
            src, dst = (edge_index[0], edge_index[1])
            node_pair = x[src] * x[dst] + (x[src] + x[dst]) * 0.5
            e_enc = self.edge_encoder(edge_attr)
            edge_repr = node_pair + e_enc
            edge_repr = self.temporal_proj(edge_repr)
            edge_seq.append(edge_repr)
        seq_stack = torch.stack(edge_seq, dim=0)
        seq_stack = seq_stack.transpose(0, 1)
        proj = torch.tanh(seq_stack)
        scores = self.attn_score(proj).squeeze(-1)
        attn_w = torch.softmax(scores, dim=1).unsqueeze(-1)
        seq_reweighted = seq_stack * attn_w
        lstm_out, (h_n, c_n) = self.lstm(seq_reweighted)
        edge_final = h_n.squeeze(0)
        logits = self.classifier(edge_final).squeeze(-1)
        probs = torch.sigmoid(logits)
        return (probs, logits)

def flatten_snapshots(snaps: List[Data]) -> Tuple[np.ndarray, np.ndarray]:
    x_parts, y_parts = ([], [])
    for d in snaps:
        x_parts.append(d.edge_attr.detach().cpu().numpy().astype(np.float32))
        y_parts.append(d.y.detach().cpu().numpy().astype(np.int64))
    return (np.concatenate(x_parts, axis=0), np.concatenate(y_parts, axis=0))

def select_threshold_with_far(y_true: np.ndarray, y_prob: np.ndarray, far_target: float) -> Tuple[float, Dict[str, float]]:
    rows = []
    for t in np.arange(float(CONFIG.get('threshold_min', 0.05)), float(CONFIG.get('threshold_max', 0.95)) + 1e-12, float(CONFIG.get('threshold_step', 0.01))):
        m = compute_metrics(y_true, y_prob, threshold=float(t))
        rows.append((float(t), m))
    feasible = [(t, m) for t, m in rows if m['FAR'] <= far_target]
    if feasible:
        t_best, m_best = max(feasible, key=lambda x: (x[1]['f1'], x[1]['recall']))
        return (t_best, m_best)
    t_best, m_best = max(rows, key=lambda x: x[1]['recall'] * max(0.0, 1.0 - x[1]['FAR']))
    return (t_best, m_best)
import itertools
import time
import pandas as pd

def train_model_quick(model, train_snaps, val_snaps, epochs=5, lr=0.001, weight_decay=1e-05, device='cpu'):
    device = torch.device(device)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    for ep in range(epochs):
        model.train()
        epoch_loss = 0.0
        for snaps in train_snaps:
            opt.zero_grad()
            snap_seq = snaps if isinstance(snaps, (list, tuple)) else [snaps]
            probs, logits = model(snap_seq)
            y_ref = snap_seq[0]
            y = y_ref.y.to(device).float()
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
    model.eval()
    with torch.no_grad():
        all_probs = []
        all_y = []
        for snaps in val_snaps:
            snap_seq = snaps if isinstance(snaps, (list, tuple)) else [snaps]
            probs, logits = model(snap_seq)
            all_probs.append(probs.detach().cpu().numpy())
            all_y.append(snap_seq[0].y.detach().cpu().numpy())
    probs = np.concatenate(all_probs, axis=0)
    y = np.concatenate(all_y, axis=0)
    m = compute_metrics(y, probs)
    return m

def run_hpo_grid(grid, epochs=5, device='cpu'):
    if not ('train_snaps' in globals() and 'val_snaps' in globals()):
        print('train_snaps/val_snaps not found in globals; please run this notebook in the main pipeline kernel where data is prepared.')
        return None
    combos = list(itertools.product(*grid.values()))
    cols = list(grid.keys())
    rows = []
    for vals in combos:
        params = dict(zip(cols, vals))
        print(f'Running HPO trial: {params}')
        node_in_local = train_snaps[0].x.shape[1] if isinstance(train_snaps, list) and hasattr(train_snaps[0], 'x') else CONFIG.get('node_feat_dim', 16)
        edge_in_local = train_snaps[0].edge_attr.shape[1] if isinstance(train_snaps, list) and hasattr(train_snaps[0], 'edge_attr') else CONFIG.get('edge_attr_dim', 8)
        model = TemporalEdgeGNN_TemporalAttention(node_in=node_in_local, edge_in=edge_in_local, hidden_dim=params.get('hidden_dim', 128), gat_heads=params.get('gat_heads', 2), lstm_hidden=params.get('lstm_hidden', 64), dropout=params.get('dropout', 0.3))
        start = time.time()
        m = train_model_quick(model, train_snaps, val_snaps, epochs=epochs, lr=params.get('lr', 0.001), weight_decay=params.get('weight_decay', 1e-05), device=device)
        elapsed = time.time() - start
        row = params.copy()
        row.update({'val_accuracy': m.get('accuracy'), 'val_f1': m.get('f1'), 'val_recall': m.get('recall'), 'val_FAR': m.get('FAR'), 'time_s': elapsed})
        rows.append(row)
    df = pd.DataFrame(rows)
    outpath = '../evaluation/hpo_results_quick.csv'
    df.to_csv(outpath, index=False)
    print(f'HPO grid finished. Results written to {outpath}')
    return df
import time
import os
import time
import json
import numpy as np
import torch
import matplotlib.pyplot as plt

def select_threshold_with_far_local(y_true, y_prob, far_target=CONFIG.get('threshold_far_target', 0.3)):
    best = None
    for t in np.arange(0.01, 0.99, 0.01):
        m = compute_metrics(y_true, y_prob, threshold=float(t))
        if m['FAR'] <= far_target:
            if best is None or m['f1'] > best[1]['f1']:
                best = (t, m)
    if best is not None:
        return best
    rows = [(t, compute_metrics(y_true, y_prob, threshold=float(t))) for t in np.arange(0.01, 0.99, 0.01)]
    t_best, m_best = max(rows, key=lambda x: x[1]['f1'])
    return (t_best, m_best)
import os, json, numpy as np, pandas as pd, torch
from sklearn.linear_model import LogisticRegression

def get_probs_for_snaps(model, snaps):
    probs_list = []
    with torch.no_grad():
        for snaps_elem in snaps:
            snap_seq = snaps_elem if isinstance(snaps_elem, (list, tuple)) else [snaps_elem]
            p, _ = model(snap_seq)
            probs_list.append(p.detach().cpu().numpy())
    if len(probs_list) == 0:
        return np.array([])
    return np.concatenate(probs_list, axis=0)

def select_threshold_with_far_local(y_true, y_prob, far_target):
    best = None
    for t in np.arange(0.01, 0.99, 0.01):
        m = compute_metrics(y_true, y_prob, threshold=float(t))
        if m['FAR'] <= far_target:
            if best is None or m['f1'] > best[1]['f1']:
                best = (t, m)
    if best is not None:
        return best
    rows = [(t, compute_metrics(y_true, y_prob, threshold=float(t))) for t in np.arange(0.01, 0.99, 0.01)]
    return max(rows, key=lambda x: x[1]['f1'])
import matplotlib.pyplot as plt, seaborn as sns
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns, torch
from torch_geometric.data import Data

def forward_model(model, snap_seq, requires_grad=False):
    if requires_grad:
        seq = []
        for s in snap_seq:
            s_new = Data(x=s.x.clone().detach().to(device), edge_index=s.edge_index.clone().detach().to(device), edge_attr=s.edge_attr.clone().detach().to(device) if getattr(s, 'edge_attr', None) is not None else None, y=s.y.clone().detach().to(device) if getattr(s, 'y', None) is not None else None)
            if s_new.edge_attr is not None:
                s_new.edge_attr.requires_grad_(True)
            seq.append(s_new)
        out = model(seq)
    else:
        seq = [s.to(device) if hasattr(s, 'to') else s for s in snap_seq]
        with torch.no_grad():
            out = model(seq)
    if isinstance(out, tuple) and len(out) >= 2:
        probs = out[0]
        aux = out[1]
    else:
        probs = out
        aux = None
    return (probs, aux, seq)
import pprint

def stream_snapshots(snapshots: List[Data], chunk_size: int=5):
    for i in range(0, len(snapshots), chunk_size):
        yield snapshots[i:i + chunk_size]

def realtime_predict(model: nn.Module, stream_chunks: List[List[Data]], threshold: float=0.5):
    model.eval()
    stream_records = []
    with torch.no_grad():
        for chunk in stream_chunks:
            chunk_dev = [d.to(device) for d in chunk]
            try:
                logits = model(chunk_dev)
                probs = [torch.sigmoid(z).detach().cpu().numpy() for z in logits]
            except RuntimeError as exc:
                msg = str(exc).lower()
                if 'stack expects' in msg or 'equal size' in msg or 'stack expects each tensor' in msg:
                    probs = []
                    for d in chunk:
                        out = model([d])
                        if isinstance(out, tuple) or isinstance(out, list):
                            p = out[0]
                        else:
                            p = out
                        p_np = p.detach().cpu().numpy() if isinstance(p, torch.Tensor) else np.asarray(p)
                        probs.append(p_np)
                else:
                    raise
            for d, p in zip(chunk, probs):
                stream_records.append({'snapshot_id': int(d.snapshot_id), 'mean_attack_probability': float(np.mean(p)), 'predicted_attack_snapshot': int(np.mean(p) >= threshold)})
    return pd.DataFrame(stream_records).sort_values('snapshot_id')
