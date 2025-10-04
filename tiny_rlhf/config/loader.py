"""Configuration loader utilities."""
from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml

from .schema import ExperimentConfig, RegistryConfig

ENV_PREFIX = "TINY_RLHF"


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping at {path}, got {type(data).__name__}")
    return data


def _deep_merge(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    result = deepcopy(base)
    for key, value in extra.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _resolve_default(entry: Any, base_dir: Path) -> Dict[str, Any]:
    if isinstance(entry, str):
        return _load_yaml((base_dir / entry).resolve())
    if isinstance(entry, dict):
        resolved: Dict[str, Any] = {}
        for section, value in entry.items():
            if isinstance(value, str) and value.endswith(('.yml', '.yaml')):
                resolved_section = _load_yaml((base_dir / value).resolve())
                resolved = _deep_merge(resolved, resolved_section)
            elif isinstance(value, dict):
                resolved = _deep_merge(resolved, {section: value})
            else:
                resolved = _deep_merge(resolved, {section: value})
        return resolved
    raise TypeError(f"Unsupported default entry type: {type(entry)}")


def _apply_env_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    result = deepcopy(config)
    prefix = ENV_PREFIX + "__"
    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue
        path = key[len(prefix):].lower().split("__")
        _set_by_path(result, path, _interpret_env_value(value))
    return result


def _interpret_env_value(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _set_by_path(config: Dict[str, Any], path: Iterable[str], value: Any) -> None:
    cursor = config
    path_list = list(path)
    for key in path_list[:-1]:
        if key not in cursor or not isinstance(cursor[key], dict):
            cursor[key] = {}
        cursor = cursor[key]
    cursor[path_list[-1]] = value


def _apply_cli_overrides(config: Dict[str, Any], overrides: Iterable[str]) -> Dict[str, Any]:
    result = deepcopy(config)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override '{override}'. Expected format key=value")
        key, raw_value = override.split("=", 1)
        path = key.split(".")
        _set_by_path(result, path, _interpret_env_value(raw_value))
    return result


def load_experiment_config(path: str | Path, overrides: Iterable[str] | None = None) -> ExperimentConfig:
    config_path = Path(path).resolve()
    data = _load_yaml(config_path)
    base_dir = config_path.parent

    merged: Dict[str, Any] = {}
    for entry in data.pop("defaults", []):
        merged = _deep_merge(merged, _resolve_default(entry, base_dir))

    merged = _deep_merge(merged, data)
    merged = _apply_env_overrides(merged)
    if overrides:
        merged = _apply_cli_overrides(merged, overrides)

    return ExperimentConfig.parse_obj(merged)


def load_registry(path: str | Path) -> RegistryConfig:
    config_path = Path(path).resolve()
    data = _load_yaml(config_path)
    return RegistryConfig.parse_obj(data)
