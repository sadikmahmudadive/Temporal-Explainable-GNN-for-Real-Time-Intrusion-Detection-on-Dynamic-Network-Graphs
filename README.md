# Temporal Explainable GNN Environment Setup

This project is configured for:
- PyTorch + CUDA acceleration
- PyTorch Geometric (PyG)
- Large graph processing workflows

## Project Structure

```text
project_root/
|-- venv/
|-- data/
|-- notebooks/
|-- src/
|-- requirements.txt
|-- README.md
```

## Environment Setup

### 1) Create virtual environment

Recommended interpreter: Python 3.10.

If Python 3.10 is installed:

```powershell
py -3.10 -m venv venv
```

Current workspace setup used Python 3.11 (3.10 not present on this machine):

```powershell
py -3.11 -m venv venv
```

### 2) Activate environment

Windows (PowerShell):

```powershell
.\venv\Scripts\Activate.ps1
```

Windows (CMD):

```cmd
venv\Scripts\activate
```

Linux/Mac:

```bash
source venv/bin/activate
```

### 3) Upgrade pip

```powershell
python -m pip install --upgrade pip
```

## Core Dependencies

### PyTorch (CUDA 11.8)

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### PyTorch Geometric

Install binary extensions matching your installed torch/cuda wheel.

For torch 2.7 + cu118:

```powershell
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.7.0+cu118.html
pip install torch-geometric
```

### Additional libraries

```powershell
pip install pandas scikit-learn matplotlib seaborn networkx tqdm jupyter ipykernel
```

## Jupyter Kernel

```powershell
python -m ipykernel install --user --name gnn-env
```

## Verify GPU Support

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO_CUDA_DEVICE")
```

Expected `torch.cuda.is_available()` output: `True`.

## Reproducibility

Installed dependencies were exported to:
- `requirements.txt`
