# xai_app/utils/paths.py
from pathlib import Path
import sys


def get_project_root() -> Path:
    """
    Detect the project root:
    xai_app/utils/paths.py
       -> xai_app/utils
       -> xai_app
       -> project_root
    """
    return Path(__file__).resolve().parents[2]


def get_latec_root() -> Path:
    """
    Path: <project_root>/third_party/latec
    """
    return get_project_root() / "third_party" / "latec"


def get_latec_config_dir() -> Path:
    """
    Path: <project_root>/third_party/latec/configs
    """
    return get_latec_root() / "configs"


def get_latec_src_dir() -> Path:
    """
    Path: <project_root>/third_party/latec/src
    """
    return get_latec_root() / "src"


def ensure_latec_on_syspath():
    """
    Ensure that LATEC's package root is available as `src.*`.
    That means adding <project_root>/third_party/latec (not /src!).
    """
    latec_root = get_latec_root()

    if latec_root.exists():
        if str(latec_root) not in sys.path:
            sys.path.insert(0, str(latec_root))
            print(f"[INFO] Added LATEC to PYTHONPATH: {latec_root}")
    else:
        print(f"[WARN] LATEC root not found: {latec_root}")

