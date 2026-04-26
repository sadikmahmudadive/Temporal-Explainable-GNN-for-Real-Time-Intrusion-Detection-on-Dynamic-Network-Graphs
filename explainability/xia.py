"""Explainability helpers (XIA) integrating SHAP and LIME.

This module provides lightweight wrappers to run SHAP (KernelExplainer)
and LIME (LimeTabularExplainer) against a PyTorch model by exposing a
NumPy-compatible prediction function.

Usage (high-level):
  from explainability.xia import numpy_predict_fn_factory, explain_with_shap_kernel, explain_with_lime

  predict_fn = numpy_predict_fn_factory(model, device='cpu')
  shap_vals = explain_with_shap_kernel(predict_fn, background_data, samples)
  lime_exp = explain_with_lime(predict_fn, train_data, instance)

The functions intentionally use SHAP's KernelExplainer for broad
compatibility and LIME's tabular explainer. They work with tabular
feature arrays (2D NumPy arrays).
"""
from typing import Callable, List, Optional, Sequence

import numpy as np
import torch


def numpy_predict_fn_factory(model: torch.nn.Module, device: str = "cpu", batch_size: int = 256) -> Callable[[np.ndarray], np.ndarray]:
    """Return a predict_fn(X) -> probs numpy function for SHAP/LIME.

    - `model` should be a PyTorch nn.Module that returns logits or
      probabilities when given a tensor of shape (N, features).
    - The returned function accepts a 2D NumPy array and returns a
      2D NumPy array of class probabilities (shape (N, C)).
    """

    model = model.to(device)
    model.eval()

    def predict_fn(x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            x = x.reshape(1, -1)
        n = x.shape[0]
        outputs: List[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, n, batch_size):
                batch = x[i : i + batch_size]
                t = torch.from_numpy(batch.astype(np.float32)).to(device)
                out = model(t)
                if isinstance(out, (tuple, list)):
                    out = out[0]
                probs = torch.softmax(out, dim=-1).cpu().numpy()
                outputs.append(probs)
        return np.vstack(outputs)

    return predict_fn


def explain_with_shap_kernel(predict_fn: Callable[[np.ndarray], np.ndarray],
                             background_data: np.ndarray,
                             samples: np.ndarray,
                             nsamples: Optional[int] = 100) -> List[np.ndarray]:
    """Run SHAP KernelExplainer on `samples` with `background_data`.

    Returns the SHAP values as produced by `explainer.shap_values`.
    Note: KernelExplainer can be slow; reduce `nsamples` for speed.
    """
    import shap

    explainer = shap.KernelExplainer(lambda x: predict_fn(x), background_data)
    shap_values = explainer.shap_values(samples, nsamples=nsamples)
    return shap_values


def explain_with_lime(predict_fn: Callable[[np.ndarray], np.ndarray],
                      train_data: np.ndarray,
                      instance: np.ndarray,
                      feature_names: Optional[Sequence[str]] = None,
                      class_names: Optional[Sequence[str]] = None,
                      num_features: int = 10):
    """Run LIME Tabular explainer for one `instance` and return explanation.

    Returns the LIME explanation object (has `as_list()` and `as_map()`).
    """
    from lime.lime_tabular import LimeTabularExplainer

    explainer = LimeTabularExplainer(
        training_data=train_data,
        feature_names=list(feature_names) if feature_names is not None else None,
        class_names=list(class_names) if class_names is not None else None,
        discretize_continuous=True,
    )

    exp = explainer.explain_instance(instance.astype(float), predict_fn, num_features=num_features)
    return exp
