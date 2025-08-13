import os
from typing import Any, Dict

try:
    import yaml
except Exception as exc:
    raise ImportError("PyYAML is required. Please install with 'pip install pyyaml'.") from exc


def load_config(config_path: str) -> Dict[str, Any]:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    return data


def get_param(config: Dict[str, Any], key: str, default: Any) -> Any:
    return config.get(key, default)


