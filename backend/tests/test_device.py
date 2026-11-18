from unittest.mock import patch

import torch

from backend.app.ml import (EyeFeatureClassifier, FeatureClassifier,
                            select_device)


def test_device_selection_uses_cuda_and_supports_cpu_fallback():
    selected=select_device()
    assert selected.type==("cuda" if torch.cuda.is_available() else "cpu")
    with patch("torch.cuda.is_available",return_value=False):
        assert select_device().type=="cpu"


def test_custom_models_run_on_selected_device_with_synthetic_tensors_only():
    device=select_device()
    gesture=FeatureClassifier(6).to(device).eval()
    eye=EyeFeatureClassifier(2).to(device).eval()
    with torch.inference_mode():
        assert gesture(torch.zeros((2,63),device=device)).shape==(2,6)
        assert eye(torch.zeros((2,24),device=device)).shape==(2,2)
    assert all(model is None for model in __import__("backend.app.ml",fromlist=["registry"]).registry.models.values())
