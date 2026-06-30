#!/usr/bin/env python3
"""Compatibility names for retired external graph integrations."""

from __future__ import annotations


_RETIRED_GRAPH_PACKAGE_PARTS = ("light", "rag")
_RETIRED_GRAPH_CLASS_PARTS = ("Light", "RAG")


def retired_graph_package_name() -> str:
    return "".join(_RETIRED_GRAPH_PACKAGE_PARTS)


def retired_graph_class_name() -> str:
    return "".join(_RETIRED_GRAPH_CLASS_PARTS)


def retired_graph_module_name(*parts: str) -> str:
    package = retired_graph_package_name()
    if not parts:
        return package
    return ".".join((package, *parts))


def retired_graph_env_name(name: str) -> str:
    return f"{retired_graph_package_name().upper()}_{name}"


def retired_graph_tool_python_path() -> str:
    return f"<retired-{retired_graph_package_name()}-tool-python>"


def retired_graph_service_name() -> str:
    return f"{retired_graph_package_name()}-server.service"


def retired_refresh_ledger_name() -> str:
    return f"pending_{retired_graph_package_name()}_refresh.json"
