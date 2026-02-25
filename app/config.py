from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"

ENV_PREFIX = "STOCKER__"  # e.g. STOCKER__FETCH__DAYS=3


def _parse_env_value(raw: str) -> Any:
    s = raw.strip()

    # bool
    if s.lower() in {"true", "false"}:
        return s.lower() == "true"

    # int
    try:
        if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
            return int(s)
    except Exception:
        pass

    # float
    try:
        return float(s) if any(c in s for c in [".", "e", "E"]) else s
    except Exception:
        return s


def _set_nested(d: Dict[str, Any], keys: list[str], value: Any) -> None:
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def _apply_env_overrides(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Env override format:
      STOCKER__FETCH__DAYS=3
      STOCKER__APP__DEBUG=true
      STOCKER__SENTIMENT__MODEL=vader

    Rules:
      - Split by "__"
      - Lowercase keys to match YAML
      - Values auto-cast to bool/int/float when possible
    """
    for k, v in os.environ.items():
        if not k.startswith(ENV_PREFIX):
            continue
        path = k[len(ENV_PREFIX):]
        parts = [p.strip().lower() for p in path.split("__") if p.strip()]
        if not parts:
            continue
        _set_nested(cfg, parts, _parse_env_value(v))
    return cfg


@dataclass(frozen=True)
class Settings:
    raw: Dict[str, Any]

    @property
    def app(self) -> Dict[str, Any]:
        return self.raw.get("app", {})

    @property
    def fetch(self) -> Dict[str, Any]:
        return self.raw.get("fetch", {})

    @property
    def sentiment(self) -> Dict[str, Any]:
        return self.raw.get("sentiment", {})

    @property
    def orchestrator(self) -> Dict[str, Any]:
        return self.raw.get("orchestrator", {})


_SETTINGS: Settings | None = None


def get_settings(config_path: Path | None = None) -> Settings:
    global _SETTINGS
    if _SETTINGS is not None:
        return _SETTINGS

    path = config_path or Path(os.environ.get("STOCKER_CONFIG", str(DEFAULT_CONFIG_PATH)))

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    cfg = _apply_env_overrides(cfg)

    # convenience: allow PORT env to override app.port (common in deploys)
    if "PORT" in os.environ:
        try:
            cfg.setdefault("app", {})
            cfg["app"]["port"] = int(os.environ["PORT"])
        except Exception:
            pass

    _SETTINGS = Settings(raw=cfg)
    return _SETTINGS
