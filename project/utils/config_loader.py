from pathlib import Path
import yaml
from typing import Dict, Any


def load_config(config_path: str = None) -> Dict[str, Any]:
    if config_path is None:
        base_dir = Path(__file__).parent.parent
        config_path = base_dir / "config" / "config.yaml"
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    
    return config