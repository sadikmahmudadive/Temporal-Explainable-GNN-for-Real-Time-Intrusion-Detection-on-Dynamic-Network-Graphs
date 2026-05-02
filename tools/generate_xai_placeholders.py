from pathlib import Path
from PIL import Image

# This script creates placeholder SHAP and LIME images by copying existing explainability artifacts.
# Run from repository root: python tools/generate_xai_placeholders.py

repo_root = Path(__file__).resolve().parents[1]
src_dir = repo_root / 'evaluation' / 'plots' / 'explain'
if not src_dir.exists():
    raise SystemExit(f"Explain plots directory not found: {src_dir}")

att0 = src_dir / 'attention_snapshot_0.png'
if not att0.exists():
    # fallback to any available png in explain dir
    candidates = list(src_dir.glob('*.png'))
    if candidates:
        att0 = candidates[0]
    else:
        raise SystemExit('No PNGs found to use as placeholders in evaluation/plots/explain/')

target_shap = src_dir / 'shap_summary.png'
target_lime = src_dir / 'lime_instance_0.png'

print(f'Creating placeholders:\n  {att0} -> {target_shap}\n  {att0} -> {target_lime}')
img = Image.open(att0)
img.save(target_shap)
img.save(target_lime)
print('Placeholders created.')
