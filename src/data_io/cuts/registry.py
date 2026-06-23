from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import Callable


@dataclass(frozen=True)
class CutSpec:
    name: str
    cls: type
    description: str
    module: str
    parameters: tuple[tuple[str, str], ...]
    example: str


_REGISTRY: dict[str, CutSpec] = {}


def register_cut(
    name: str | None = None,
    description: str | None = None,
    parameters: dict[str, str] | None = None,
    example: str | None = None,
) -> Callable[[type], type]:
    def decorator(cls: type) -> type:
        cut_name = name or cls.__name__
        if cut_name in _REGISTRY:
            raise ValueError(f"Cut '{cut_name}' is already registered.")

        spec = CutSpec(
            name=cut_name,
            cls=cls,
            description=description or (cls.__doc__ or "").strip(),
            module=cls.__module__,
            parameters=_describe_parameters(cls, parameters or {}),
            example=example or _example_instantiation(cls),
        )
        _REGISTRY[cut_name] = spec
        cls.cut_name = cut_name
        cls.cut_description = spec.description
        cls.cut_parameters = dict(spec.parameters)
        cls.cut_example = spec.example
        return cls

    return decorator


def get_registered_cuts(query: str | None = None) -> dict[str, CutSpec]:
    if query is None:
        return dict(_REGISTRY)
    query_lower = query.lower()
    return {
        cut_name: spec
        for cut_name, spec in _REGISTRY.items()
        if _matches_query(spec, query_lower)
    }


def format_registered_cuts(query: str | None = None) -> str:
    specs = get_registered_cuts(query)
    if not specs:
        label = query if query is not None else "query"
        return f"No registered cuts matched {label!r}."

    lines: list[str] = ["cuts:"]
    for cut_name in sorted(specs):
        spec = specs[cut_name]
        description = f" - {spec.description}" if spec.description else ""
        lines.append(f"  {cut_name} ({spec.cls.__name__}){description}")
        if spec.parameters:
            lines.append("    parameters:")
            for parameter_name, parameter_description in spec.parameters:
                lines.append(f"      {parameter_name}: {parameter_description}")
        else:
            lines.append("    parameters: none")
        lines.append(f"    example: {spec.example}")
    return "\n".join(lines)


def print_registered_cuts(query: str | None = None) -> None:
    print(format_registered_cuts(query))


def _describe_parameters(cls: type, descriptions: dict[str, str]) -> tuple[tuple[str, str], ...]:
    signature = inspect.signature(cls.__init__)
    described: list[tuple[str, str]] = []
    for parameter_name, parameter in signature.parameters.items():
        if parameter_name == "self":
            continue
        if parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        rendered = descriptions.get(parameter_name)
        if rendered is None:
            rendered = _render_parameter(parameter)
        described.append((parameter_name, rendered))
    return tuple(described)


def _render_parameter(parameter: inspect.Parameter) -> str:
    annotation = ""
    if parameter.annotation is not inspect._empty:
        annotation = f"{_annotation_text(parameter.annotation)}"
    if parameter.default is inspect._empty:
        return annotation or "required"
    default_text = repr(parameter.default)
    if annotation:
        return f"{annotation}, default={default_text}"
    return f"default={default_text}"


def _example_instantiation(cls: type) -> str:
    signature = inspect.signature(cls.__init__)
    arguments = []
    for parameter_name, parameter in signature.parameters.items():
        if parameter_name == "self":
            continue
        if parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if parameter.default is inspect._empty:
            arguments.append(f"{parameter_name}=...")
        else:
            arguments.append(f"{parameter_name}={repr(parameter.default)}")
    return f"{cls.__name__}({', '.join(arguments)})"


def _annotation_text(annotation) -> str:
    if isinstance(annotation, type):
        return annotation.__name__
    return str(annotation).replace("typing.", "")


def _matches_query(spec: CutSpec, query_lower: str) -> bool:
    haystacks = (
        spec.name.lower(),
        spec.cls.__name__.lower(),
        spec.module.lower(),
        spec.description.lower(),
    )
    return any(query_lower in haystack for haystack in haystacks)
