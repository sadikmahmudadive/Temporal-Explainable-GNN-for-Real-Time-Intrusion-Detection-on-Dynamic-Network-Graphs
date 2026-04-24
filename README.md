# Temporal Explainable GNN for Real-Time Intrusion Detection

This repository now includes a reproducible training/evaluation CLI for paper-ready experiments.

## What Was Added for IEEE Readiness

- Reproducible runner: `training/run_training.py`
- Config file: `config.yml`
- Leakage guard: edge feature transforms are fitted on **train only** and applied to val/test
- Explicit split policy (default): `per_day_temporal` (strict temporal-style split)
- Baseline comparison output (RF, LR, MLP + GNN variants)
- Ablation output (`TemporalGNN_LSTM` vs `TemporalGNN_NoTemporal`)
- Multi-seed significance report support

## Project Structure

```text
project_root/
|-- config.yml
|-- data/
|-- evaluation/
|-- models/
|-- notebooks/
|-- training/
|   |-- __init__.py
|   |-- pipeline_core.py
|   |-- run_training.py
|-- requirements.txt
|-- README.md
```

## Environment Setup

```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run Reproducible Training

From repository root:

```powershell
python -m training.run_training --config config.yml
```

Quick smoke test:

```powershell
python -m training.run_training --config config.yml --fast-smoke
```

This training run also generates `evaluation/deployment_artifacts.json`, which is required for realtime inference.

## Run Realtime Detection

Tail an appending flow CSV stream and emit snapshot-level alerts:

```powershell
python -m training.realtime_detect --deployment evaluation/deployment_artifacts.json --input-csv data/live_flows.csv
```

Optional useful flags:

```powershell
python -m training.realtime_detect --deployment evaluation/deployment_artifacts.json --input-csv data/live_flows.csv --poll-seconds 1 --max-rows-per-read 2000 --output-csv evaluation/realtime_alerts.csv
```

## Single-File GUI App

Run one GUI that controls both training and realtime detection:

```powershell
python temporal_gnn_gui_app.py
```

The GUI includes:
- Embedded editable config (replaces manual `config.yml` editing)
- Training start/stop controls
- Realtime detection start/stop controls
- Live process logs

## Key Output Artifacts

After a run, the following are generated in `evaluation/`:

- `model_comparison.csv`
- `ablation_results.csv`
- `seed_metrics.csv`
- `significance_report.csv`
- `run_manifest.json`
- `deployment_artifacts.json`

Model checkpoints are written to `models/` (for example `temporal_gnn_lstm_seed42.pt`).

## Publication Notes

- For strict real-time claims, use `split_mode: per_day_temporal` or `chronological`.
- `stratified_snapshot` is available for analysis but is not strict temporal deployment.
- Significance testing is based on paired tests across seeds in `significance_report.csv`.
