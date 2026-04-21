Simulator GUI (Streamlit)

This folder provides a standalone GUI to run model-driven simulations outside the notebook.

Quick start

1. Install requirements (recommended in a venv):

```bash
pip install streamlit pyvis plotly imageio torch torchvision torchaudio
# Also install your project's existing dependencies (torch_geometric, etc.)
```

2. Prepare snapshots: save each torch_geometric.Data snapshot into files:

snapshots/test/snap_000.pt
snapshots/test/snap_001.pt
snapshots/val/...
snapshots/train/...

Use `torch.save(data, path)` in your notebook to export.

3. Start app:

```bash
streamlit run src/simulator_streamlit.py
```

Model loader options

- If you saved the full model using `torch.save(model)` the app will try to load it directly.
- If you saved only `model.state_dict()` then provide a small Python module (upload via the sidebar) that defines:

```python
# example user_loader.py
from my_models import TemporalEdgeGNN_TemporalAttention
import torch

def build_model(checkpoint_path, device='cpu'):
    model = TemporalEdgeGNN_TemporalAttention(...)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    return model.to(device)
```

Outputs

- CSV and plot files are written to `evaluation/simulator/` by default when running a simulation.

Notes

- The app attempts to use `pyvis` for interactive network visualizations. If `pyvis` is not installed it falls back to static `matplotlib` network drawings.
- This app expects the environment to have the same Python package versions as the training environment (notably `torch` and `torch_geometric`).

If you want, I can:
- Add a packaged executable (PyInstaller) to distribute the GUI,
- Add an example `user_loader.py` for the models in this repo,
- Wire the app to read snapshots directly from notebook variables via a tiny RPC bridge (advanced).

Packaging with PyInstaller

1. Ensure you have a virtual environment and installed simulator requirements:

```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements-simulator.txt
pip install pyinstaller
```

2. Build (Windows):

```powershell
.\build_executable.bat
```

Or (POSIX):

```bash
./build_executable.sh
```

3. After the build, the single-file executable will appear in `dist/simulator_gui` (or `dist/simulator_gui.exe` on Windows).

Notes about packaging:
- PyInstaller bundles Python and packages; the resulting binary can be large.
- `torch_geometric` may have native extensions that are challenging to bundle; if you rely on it, consider shipping a small wrapper that loads snapshots via CPU-only `torch.load` and pass them into the bundled app, or distribute via a platform-specific installer.
- Test the built binary on target platforms (Windows, Linux). If you want, I can attempt a local build and smoke-test here.
