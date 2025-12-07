import asyncio
import copy
import re
from typing import Any, Callable, Dict, Iterable, List, Mapping

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, create_model

from skills_loader import load_skill_catalog

_PLACEHOLDER_PATTERN = re.compile(r"\{([^{}]+)\}")


def _extract_placeholders(value: Any) -> List[str]:
    placeholders: List[str] = []
    if isinstance(value, str):
        placeholders.extend(_PLACEHOLDER_PATTERN.findall(value))
    elif isinstance(value, dict):
        for item in value.values():
            placeholders.extend(_extract_placeholders(item))
    elif isinstance(value, list):
        for item in value:
            placeholders.extend(_extract_placeholders(item))
    return placeholders


def _resolve_placeholders(payload: Any, mapping: Mapping[str, Any]) -> Any:
    if isinstance(payload, str):
        placeholders = _PLACEHOLDER_PATTERN.findall(payload)
        if placeholders:
            missing = [name for name in placeholders if name not in mapping]
            if missing:
                raise KeyError(f"Missing placeholder values: {', '.join(missing)}")
            return payload.format(**mapping)
        return payload
    if isinstance(payload, dict):
        return {k: _resolve_placeholders(v, mapping) for k, v in payload.items()}
    if isinstance(payload, list):
        return [_resolve_placeholders(item, mapping) for item in payload]
    return payload


def _make_args_model(skill: str, preset_name: str, payload: Dict[str, Any]) -> type[BaseModel]:
    placeholders = sorted(set(_extract_placeholders(payload)))
    if not placeholders:
        return create_model(f"PresetArgs_{skill}_{preset_name}")
    fields = {
        name: (str, Field(..., description=f"Value for placeholder {{{name}}}"))
        for name in placeholders
    }
    model_name = f"PresetArgs_{skill.replace('-', '_')}_{preset_name.replace('-', '_')}"
    return create_model(model_name, **fields)  # type: ignore[arg-type]


def build_preset_tools(
    *,
    client_callers: Mapping[str, Callable[[str, Dict[str, Any]], Any]],
    skip_names: Iterable[str] | None = None,
) -> List[StructuredTool]:
    catalog = load_skill_catalog()
    skip = set(skip_names or [])
    generated: List[StructuredTool] = []

    for entry in catalog:
        caller = client_callers.get(entry.name)
        if caller is None:
            continue
        for preset in entry.presets or []:
            preset_name = preset.get("name")
            payload = preset.get("payload") or {}
            base_tool = payload.get("tool")
            base_args = payload.get("args", {})
            if not preset_name or not base_tool:
                continue
            tool_name = f"{entry.name}_{preset_name}_preset"
            if tool_name in skip:
                continue

            args_model = _make_args_model(entry.name, preset_name, base_args)
            description = preset.get("description") or f"Preset for {base_tool}"
            template_snapshot = copy.deepcopy(base_args)

            def _run(
                tool_input: BaseModel,
                _caller: Callable[[str, Dict[str, Any]], Any] = caller,
                _base_tool: str = base_tool,
                _template: Dict[str, Any] = template_snapshot,
            ) -> Any:
                mapping = tool_input.model_dump() if isinstance(tool_input, BaseModel) else dict(tool_input)
                resolved_args = _resolve_placeholders(copy.deepcopy(_template), mapping)
                return _caller(_base_tool, resolved_args)

            async def _arun(
                tool_input: BaseModel,
                _caller: Callable[[str, Dict[str, Any]], Any] = caller,
                _base_tool: str = base_tool,
                _template: Dict[str, Any] = template_snapshot,
            ) -> Any:
                mapping = tool_input.model_dump() if isinstance(tool_input, BaseModel) else dict(tool_input)
                resolved_args = _resolve_placeholders(copy.deepcopy(_template), mapping)
                return await asyncio.to_thread(_caller, _base_tool, resolved_args)

            structured = StructuredTool.from_function(
                func=_run,
                coroutine=_arun,
                name=tool_name,
                description=description,
                args_schema=args_model,
            )
            generated.append(structured)

    return generated


__all__ = ["build_preset_tools"]
