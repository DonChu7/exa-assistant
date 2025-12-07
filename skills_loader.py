import os
import re
from pathlib import Path
import copy
from typing import Any, Dict, Iterable, List, Mapping, Optional

import yaml

_PLACEHOLDER_PATTERN = re.compile(r"^\$\{([^}:]+)(:-([^}]*))?\}$")


def _resolve_env_value(value: str, environ: Mapping[str, str]) -> str:
    match = _PLACEHOLDER_PATTERN.match(value)
    if match:
        var_name = match.group(1)
        default = match.group(3) or ""
        return environ.get(var_name, default)
    return os.path.expandvars(value)


def _resolve_env_map(raw: Mapping[str, Any], environ: Mapping[str, str]) -> Dict[str, str]:
    resolved = {}
    for key, val in raw.items():
        if val is None:
            continue
        if isinstance(val, str):
            resolved[key] = _resolve_env_value(val, environ)
        else:
            resolved[key] = str(val)
    return resolved


class SkillEntry(object):
    def __init__(self, name, server_id, command, env, tools, presets, logs, owners, path):
        self.name = name
        self.server_id = server_id
        self.command = command
        self.env = env
        self.tools = tools
        self.presets = presets or []
        self.logs = logs
        self.owners = owners
        self.path = path

    def presets_copy(self) -> List[Dict[str, Any]]:
        return copy.deepcopy(self.presets)


class SkillCatalog(object):
    def __init__(self, by_name, by_server_id):
        self.by_name = by_name
        self.by_server_id = by_server_id

    def get(self, name):
        return self.by_name.get(name)

    def get_by_server(self, server_id):
        return self.by_server_id.get(server_id)

    def __iter__(self):
        return iter(self.by_name.values())

    def collect_presets(self) -> Dict[str, List[Dict[str, Any]]]:
        presets = {}
        for entry in self:
            presets[entry.name] = entry.presets_copy()
        return presets


def _parse_front_matter(text: str) -> Optional[Dict[str, Any]]:
    if not text.startswith("---"):
        return None
    parts = text.split("---", 2)
    if len(parts) < 3:
        return None
    front_matter = parts[1]
    return yaml.safe_load(front_matter) or {}


def load_skill_catalog(directory: str = "skills") -> SkillCatalog:
    base_path = Path(directory)
    entries = []
    if not base_path.exists():
        return SkillCatalog({}, {})

    for skill_file in sorted(base_path.glob("**/skill.md")):
        text = skill_file.read_text(encoding="utf-8")
        meta = _parse_front_matter(text)
        if not meta:
            continue
        name = meta.get("name")
        server_id = meta.get("server_id")
        command = meta.get("launch", {}).get("command") or []
        raw_env = meta.get("launch", {}).get("env") or {}
        if not name or not server_id or not command:
            continue
        if isinstance(command, str):
            command_list = command.split()
        else:
            command_list = [str(item) for item in command]
        env_map = _resolve_env_map(raw_env, environ=os.environ)
        tools = meta.get("tools") or []
        presets = meta.get("presets") or []
        logs = meta.get("logs") or {}
        owners = meta.get("owners") or []
        entry = SkillEntry(
            name=name,
            server_id=server_id,
            command=command_list,
            env=env_map,
            tools=tools,
            presets=presets,
            logs=logs,
            owners=owners,
            path=skill_file,
        )
        entries.append(entry)

    by_name = {entry.name: entry for entry in entries}
    by_server = {entry.server_id: entry for entry in entries}
    return SkillCatalog(by_name, by_server)


__all__ = [
    "SkillEntry",
    "SkillCatalog",
    "load_skill_catalog",
]
