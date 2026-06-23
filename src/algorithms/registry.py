from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import Callable


@dataclass(frozen=True)
class AlgorithmSpec:
    task: str
    name: str
    cls: type
    description: str
    module: str
    parameters: tuple[tuple[str, str], ...]
    example: str


_REGISTRY: dict[str, dict[str, AlgorithmSpec]] = {
    "tracklet": {},
    "vertex": {},
    "pattern": {},
    "validation": {},
}


def register_algorithm(
    task: str,
    name: str | None = None,
    description: str | None = None,
    parameters: dict[str, str] | None = None,
    example: str | None = None,
) -> Callable[[type], type]:
    if task not in _REGISTRY:
        candidates = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"Unknown algorithm task '{task}'. Candidates are: {candidates}")

    def decorator(cls: type) -> type:
        algorithm_name = name or cls.__name__
        task_registry = _REGISTRY[task]
        if algorithm_name in task_registry:
            raise ValueError(
                f"Algorithm '{algorithm_name}' is already registered under task '{task}'."
            )

        spec = AlgorithmSpec(
            task=task,
            name=algorithm_name,
            cls=cls,
            description=description or (cls.__doc__ or "").strip(),
            module=cls.__module__,
            parameters=_describe_parameters(cls, parameters or {}),
            example=example or _example_instantiation(cls),
        )
        task_registry[algorithm_name] = spec
        cls.algorithm_task = task
        cls.algorithm_name = algorithm_name
        cls.algorithm_description = spec.description
        cls.algorithm_parameters = dict(spec.parameters)
        cls.algorithm_example = spec.example
        return cls

    return decorator


def get_registered_algorithms(query: str | None = None) -> dict[str, dict[str, AlgorithmSpec]]:
    if query is None:
        return {name: dict(specs) for name, specs in _REGISTRY.items()}

    if query in _REGISTRY:
        return {query: dict(_REGISTRY[query])}

    query_lower = query.lower()
    grouped: dict[str, dict[str, AlgorithmSpec]] = {}
    for task_name, specs in _REGISTRY.items():
        matched = {
            algorithm_name: spec
            for algorithm_name, spec in specs.items()
            if _matches_query(spec, query_lower)
        }
        if matched:
            grouped[task_name] = matched
    return grouped


def format_registered_algorithms(query: str | None = None) -> str:
    grouped = get_registered_algorithms(query)
    if not grouped:
        label = query if query is not None else "query"
        return f"No registered algorithms matched {label!r}."

    lines: list[str] = []
    for task_name in sorted(grouped):
        specs = grouped[task_name]
        lines.append(f"{task_name}:")
        if not specs:
            lines.append("  (none)")
            continue
        for algorithm_name in sorted(specs):
            spec = specs[algorithm_name]
            description = f" - {spec.description}" if spec.description else ""
            lines.append(f"  {algorithm_name} ({spec.cls.__name__}){description}")
            if spec.parameters:
                lines.append("    parameters:")
                for parameter_name, parameter_description in spec.parameters:
                    lines.append(f"      {parameter_name}: {parameter_description}")
            else:
                lines.append("    parameters: none")
            lines.append(f"    example: {spec.example}")
    return "\n".join(lines)


def print_registered_algorithms(query: str | None = None) -> None:
    print(format_registered_algorithms(query))


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


def _matches_query(spec: AlgorithmSpec, query_lower: str) -> bool:
    haystacks = (
        spec.task.lower(),
        spec.name.lower(),
        spec.cls.__name__.lower(),
        spec.module.lower(),
        spec.description.lower(),
    )
    return any(query_lower in haystack for haystack in haystacks)
