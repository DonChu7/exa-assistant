from typing import Dict, List, Optional

from skills_loader import SkillCatalog, SkillEntry, load_skill_catalog


class SkillsIndex(object):
    def __init__(self, catalog: Optional[SkillCatalog] = None) -> None:
        self._catalog = catalog or load_skill_catalog()

    @property
    def catalog(self) -> SkillCatalog:
        return self._catalog

    def commands_by_name(self) -> Dict[str, List[str]]:
        return {entry.name: entry.command for entry in self._catalog}

    def env_by_name(self) -> Dict[str, Dict[str, str]]:
        return {entry.name: entry.env for entry in self._catalog}

    def presets_by_name(self) -> Dict[str, List[Dict[str, object]]]:
        return {entry.name: entry.presets for entry in self._catalog}


__all__ = ["SkillsIndex", "load_skill_catalog", "SkillCatalog", "SkillEntry"]
