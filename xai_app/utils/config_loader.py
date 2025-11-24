# xai_app/utils/config_loader.py
import os
from omegaconf import DictConfig
from hydra import compose, initialize_config_dir

from xai_app.utils.paths import get_latec_config_dir, ensure_latec_on_syspath


def load_latec_config(modality: str = "image", config_name: str = "explain") -> DictConfig:
    """
    Load hierarchical LATEC-style Hydra configs (e.g., explain.yaml, eval.yaml).
    This points to third_party/latec/configs.
    """
    ensure_latec_on_syspath()
    cfg_dir = get_latec_config_dir()

    overrides = [
        f"data.modality={modality}",
        f"explain_method={modality}",
    ]

    with initialize_config_dir(config_dir=str(cfg_dir), version_base="1.3"):
        cfg = compose(config_name=config_name, overrides=overrides)

    print(f"[INFO] Loaded LATEC config ({modality}, {config_name})")
    return cfg
