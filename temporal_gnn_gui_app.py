"""Single-file GUI app to run training and realtime detection.

This app wraps:
- training.run_training
- training.realtime_detect

It also embeds editable default config content, so `config.yml` is optional.
"""

from __future__ import annotations

import json
import queue
import subprocess
import sys
import threading
from pathlib import Path
from tkinter import BooleanVar, END, StringVar, Tk, filedialog, messagebox
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText


DEFAULT_CONFIG = {
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
    "lr": 0.0008,
    "weight_decay": 0.0001,
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


class TemporalGNNGui(Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Temporal GNN IDS - Unified GUI")
        self.geometry("1200x820")
        self.minsize(1050, 740)

        self._repo_root = Path(__file__).resolve().parent
        self._log_queue: queue.Queue[tuple[str, str]] = queue.Queue()
        self._procs: dict[str, subprocess.Popen | None] = {"training": None, "realtime": None}

        self._init_vars()
        self._build_ui()
        self.after(120, self._drain_log_queue)

    def _init_vars(self) -> None:
        self.train_seed = StringVar(value="")
        self.train_epochs = StringVar(value="")
        self.train_split_mode = StringVar(value="")
        self.train_significance_seeds = StringVar(value="")
        self.train_fast_smoke = BooleanVar(value=False)
        self.train_skip_significance = BooleanVar(value=False)

        self.rt_deployment = StringVar(value="evaluation/deployment_artifacts.json")
        self.rt_input_csv = StringVar(value="data/live_flows.csv")
        self.rt_poll_seconds = StringVar(value="2")
        self.rt_max_rows = StringVar(value="5000")
        self.rt_context_size = StringVar(value="")
        self.rt_output_csv = StringVar(value="evaluation/realtime_alerts.csv")
        self.rt_device = StringVar(value="auto")
        self.rt_emit_open_window = BooleanVar(value=False)
        self.rt_once = BooleanVar(value=False)

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=10)
        root.pack(fill="both", expand=True)

        notebook = ttk.Notebook(root)
        notebook.pack(fill="both", expand=True)

        train_tab = ttk.Frame(notebook, padding=10)
        realtime_tab = ttk.Frame(notebook, padding=10)
        logs_tab = ttk.Frame(notebook, padding=10)
        notebook.add(train_tab, text="Training")
        notebook.add(realtime_tab, text="Realtime Detection")
        notebook.add(logs_tab, text="Logs")

        self._build_training_tab(train_tab)
        self._build_realtime_tab(realtime_tab)
        self._build_logs_tab(logs_tab)

    def _build_training_tab(self, tab: ttk.Frame) -> None:
        top = ttk.LabelFrame(tab, text="Training Controls", padding=10)
        top.pack(fill="x")

        row0 = ttk.Frame(top)
        row0.pack(fill="x", pady=2)
        ttk.Label(row0, text="Seed (optional)", width=22).pack(side="left")
        ttk.Entry(row0, textvariable=self.train_seed, width=18).pack(side="left")
        ttk.Label(row0, text="Epochs (optional)", width=18).pack(side="left", padx=(18, 0))
        ttk.Entry(row0, textvariable=self.train_epochs, width=18).pack(side="left")
        ttk.Label(row0, text="Split mode (optional)", width=20).pack(side="left", padx=(18, 0))
        ttk.Combobox(
            row0,
            textvariable=self.train_split_mode,
            values=["", "per_day_temporal", "chronological", "stratified_snapshot"],
            width=22,
            state="readonly",
        ).pack(side="left")

        row1 = ttk.Frame(top)
        row1.pack(fill="x", pady=2)
        ttk.Label(row1, text="Significance seeds (optional, comma-separated)", width=40).pack(side="left")
        ttk.Entry(row1, textvariable=self.train_significance_seeds, width=40).pack(side="left")

        row2 = ttk.Frame(top)
        row2.pack(fill="x", pady=6)
        ttk.Checkbutton(row2, text="Fast smoke", variable=self.train_fast_smoke).pack(side="left")
        ttk.Checkbutton(row2, text="Skip significance", variable=self.train_skip_significance).pack(side="left", padx=(20, 0))

        row3 = ttk.Frame(top)
        row3.pack(fill="x", pady=(8, 0))
        ttk.Button(row3, text="Run Training", command=self._run_training).pack(side="left")
        ttk.Button(row3, text="Stop Training", command=lambda: self._stop_process("training")).pack(side="left", padx=8)

        cfg_frame = ttk.LabelFrame(tab, text="Embedded Config Editor (JSON)", padding=10)
        cfg_frame.pack(fill="both", expand=True, pady=(10, 0))

        cfg_btn_row = ttk.Frame(cfg_frame)
        cfg_btn_row.pack(fill="x", pady=(0, 6))
        ttk.Button(cfg_btn_row, text="Load From File", command=self._load_config_into_editor).pack(side="left")
        ttk.Button(cfg_btn_row, text="Save Editor To File", command=self._save_editor_to_file).pack(side="left", padx=8)
        ttk.Button(cfg_btn_row, text="Reset To Defaults", command=self._reset_editor_defaults).pack(side="left")

        self.config_editor = ScrolledText(cfg_frame, wrap="none", height=20)
        self.config_editor.pack(fill="both", expand=True)
        self._reset_editor_defaults()

    def _build_realtime_tab(self, tab: ttk.Frame) -> None:
        frm = ttk.LabelFrame(tab, text="Realtime Detection Controls", padding=10)
        frm.pack(fill="x")

        def add_row(label: str, var: StringVar, browse: bool = False, save_browse: bool = False) -> None:
            row = ttk.Frame(frm)
            row.pack(fill="x", pady=2)
            ttk.Label(row, text=label, width=26).pack(side="left")
            ent = ttk.Entry(row, textvariable=var)
            ent.pack(side="left", fill="x", expand=True)
            if browse:
                ttk.Button(row, text="Browse", command=lambda v=var: self._browse_file(v)).pack(side="left", padx=6)
            if save_browse:
                ttk.Button(row, text="Save As", command=lambda v=var: self._browse_save_file(v)).pack(side="left", padx=6)

        add_row("Deployment JSON", self.rt_deployment, browse=True)
        add_row("Input CSV Stream", self.rt_input_csv, browse=True)
        add_row("Output Alerts CSV", self.rt_output_csv, save_browse=True)

        row4 = ttk.Frame(frm)
        row4.pack(fill="x", pady=2)
        ttk.Label(row4, text="Poll seconds", width=26).pack(side="left")
        ttk.Entry(row4, textvariable=self.rt_poll_seconds, width=12).pack(side="left")
        ttk.Label(row4, text="Max rows/read", width=16).pack(side="left", padx=(16, 0))
        ttk.Entry(row4, textvariable=self.rt_max_rows, width=12).pack(side="left")
        ttk.Label(row4, text="Context size", width=13).pack(side="left", padx=(16, 0))
        ttk.Entry(row4, textvariable=self.rt_context_size, width=12).pack(side="left")
        ttk.Label(row4, text="Device", width=10).pack(side="left", padx=(16, 0))
        ttk.Combobox(row4, textvariable=self.rt_device, values=["auto", "cpu", "cuda"], state="readonly", width=10).pack(side="left")

        row5 = ttk.Frame(frm)
        row5.pack(fill="x", pady=(8, 0))
        ttk.Checkbutton(row5, text="Emit open window", variable=self.rt_emit_open_window).pack(side="left")
        ttk.Checkbutton(row5, text="Run once", variable=self.rt_once).pack(side="left", padx=(20, 0))

        row6 = ttk.Frame(frm)
        row6.pack(fill="x", pady=(8, 0))
        ttk.Button(row6, text="Start Realtime Detection", command=self._run_realtime).pack(side="left")
        ttk.Button(row6, text="Stop Realtime Detection", command=lambda: self._stop_process("realtime")).pack(side="left", padx=8)

        hint = ttk.Label(
            tab,
            text=(
                "Tip: run training first to generate evaluation/deployment_artifacts.json, "
                "then start realtime detection."
            ),
            foreground="#555",
        )
        hint.pack(anchor="w", pady=(10, 0))

    def _build_logs_tab(self, tab: ttk.Frame) -> None:
        ctrl = ttk.Frame(tab)
        ctrl.pack(fill="x")
        ttk.Button(ctrl, text="Clear Logs", command=self._clear_logs).pack(side="left")

        self.log_box = ScrolledText(tab, wrap="word", state="normal")
        self.log_box.pack(fill="both", expand=True, pady=(8, 0))
        self._append_log("system", f"GUI ready in {self._repo_root}")

    def _append_log(self, source: str, text: str) -> None:
        self.log_box.insert(END, f"[{source}] {text.rstrip()}\n")
        self.log_box.see(END)

    def _clear_logs(self) -> None:
        self.log_box.delete("1.0", END)

    def _load_config_into_editor(self) -> None:
        path = filedialog.askopenfilename(
            title="Load config file",
            filetypes=[("YAML/JSON", "*.yml *.yaml *.json"), ("All files", "*.*")],
            initialdir=str(self._repo_root),
        )
        if not path:
            return
        text = Path(path).read_text(encoding="utf-8")
        self.config_editor.delete("1.0", END)
        self.config_editor.insert("1.0", text)
        self._append_log("gui", f"Loaded config into editor: {path}")

    def _save_editor_to_file(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Save config editor content",
            defaultextension=".yml",
            filetypes=[("YAML", "*.yml"), ("JSON", "*.json"), ("All files", "*.*")],
            initialdir=str(self._repo_root),
            initialfile="config_from_gui.yml",
        )
        if not path:
            return
        Path(path).write_text(self.config_editor.get("1.0", END), encoding="utf-8")
        self._append_log("gui", f"Saved editor config: {path}")

    def _reset_editor_defaults(self) -> None:
        txt = json.dumps(DEFAULT_CONFIG, indent=2)
        self.config_editor.delete("1.0", END)
        self.config_editor.insert("1.0", txt + "\n")

    def _browse_file(self, var: StringVar) -> None:
        path = filedialog.askopenfilename(initialdir=str(self._repo_root))
        if path:
            var.set(path)

    def _browse_save_file(self, var: StringVar) -> None:
        path = filedialog.asksaveasfilename(initialdir=str(self._repo_root))
        if path:
            var.set(path)

    def _write_runtime_config(self) -> Path:
        raw = self.config_editor.get("1.0", END).strip()
        if not raw:
            raise ValueError("Config editor is empty.")
        runtime_config = self._repo_root / "_gui_runtime_config.yml"
        runtime_config.write_text(raw + "\n", encoding="utf-8")
        return runtime_config

    def _run_training(self) -> None:
        if self._procs["training"] is not None:
            messagebox.showwarning("Training already running", "A training process is already running.")
            return

        try:
            config_path = self._write_runtime_config()
        except Exception as exc:
            messagebox.showerror("Invalid config", f"Could not prepare config content:\n{exc}")
            return

        cmd = [sys.executable, "-m", "training.run_training", "--config", str(config_path)]
        if self.train_seed.get().strip():
            cmd += ["--seed", self.train_seed.get().strip()]
        if self.train_epochs.get().strip():
            cmd += ["--epochs", self.train_epochs.get().strip()]
        if self.train_split_mode.get().strip():
            cmd += ["--split-mode", self.train_split_mode.get().strip()]
        if self.train_significance_seeds.get().strip():
            cmd += ["--significance-seeds", self.train_significance_seeds.get().strip()]
        if self.train_fast_smoke.get():
            cmd.append("--fast-smoke")
        if self.train_skip_significance.get():
            cmd.append("--skip-significance")

        self._start_process("training", cmd)

    def _run_realtime(self) -> None:
        if self._procs["realtime"] is not None:
            messagebox.showwarning("Realtime already running", "A realtime process is already running.")
            return
        if not self.rt_input_csv.get().strip():
            messagebox.showerror("Missing input", "Please provide an input CSV stream path.")
            return

        cmd = [
            sys.executable,
            "-m",
            "training.realtime_detect",
            "--deployment",
            self.rt_deployment.get().strip(),
            "--input-csv",
            self.rt_input_csv.get().strip(),
            "--poll-seconds",
            self.rt_poll_seconds.get().strip() or "2",
            "--max-rows-per-read",
            self.rt_max_rows.get().strip() or "5000",
            "--output-csv",
            self.rt_output_csv.get().strip() or "evaluation/realtime_alerts.csv",
            "--device",
            self.rt_device.get().strip() or "auto",
        ]
        if self.rt_context_size.get().strip():
            cmd += ["--context-size", self.rt_context_size.get().strip()]
        if self.rt_emit_open_window.get():
            cmd.append("--emit-open-window")
        if self.rt_once.get():
            cmd.append("--once")

        self._start_process("realtime", cmd)

    def _start_process(self, kind: str, cmd: list[str]) -> None:
        self._append_log("gui", f"Starting {kind}: {' '.join(cmd)}")
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(self._repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )
        except Exception as exc:
            messagebox.showerror("Process start failed", f"Failed to start {kind}:\n{exc}")
            return

        self._procs[kind] = proc
        t = threading.Thread(target=self._stream_proc_output, args=(kind, proc), daemon=True)
        t.start()

    def _stream_proc_output(self, kind: str, proc: subprocess.Popen) -> None:
        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                self._log_queue.put((kind, line.rstrip("\n")))
        finally:
            code = proc.wait()
            self._log_queue.put((kind, f"Process exited with code {code}"))
            self._procs[kind] = None

    def _stop_process(self, kind: str) -> None:
        proc = self._procs.get(kind)
        if proc is None:
            self._append_log("gui", f"No active {kind} process.")
            return
        self._append_log("gui", f"Stopping {kind} process...")
        try:
            proc.terminate()
        except Exception as exc:
            self._append_log("gui", f"Terminate error ({kind}): {exc}")

    def _drain_log_queue(self) -> None:
        try:
            while True:
                src, msg = self._log_queue.get_nowait()
                self._append_log(src, msg)
        except queue.Empty:
            pass
        self.after(120, self._drain_log_queue)


def main() -> None:
    app = TemporalGNNGui()
    app.mainloop()


if __name__ == "__main__":
    main()

