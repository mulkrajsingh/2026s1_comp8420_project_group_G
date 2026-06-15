"""Load LLM module helpers without colliding with the integration ``app`` package.

The repository contains two ``app`` trees (integration and modules/llm). These
loaders import the LLM runtime and ReAct loop by file location so integration
code never shadows the member module package.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from functools import lru_cache
from pathlib import Path

_LLM_PACKAGE = "_comp8420_llm_app"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def _llm_package():
    package_dir = _repo_root() / "modules" / "llm" / "app"
    init_path = package_dir / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        _LLM_PACKAGE,
        init_path,
        submodule_search_locations=[str(package_dir)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load LLM package: {package_dir}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def _react_loop_module():
    _llm_package()
    return importlib.import_module(f"{_LLM_PACKAGE}.react_loop")


@lru_cache(maxsize=1)
def _runtime_module():
    _llm_package()
    return importlib.import_module(f"{_LLM_PACKAGE}.runtime")


def build_llm_runtime(ollama_host: str):
    """Return a configured Ollama runtime from the LLM module."""
    return _runtime_module().build_runtime(ollama_host)


def run_react_topic_rag(*args, **kwargs):
    """Run the LLM module ReAct topic-RAG loop with the supplied callbacks."""
    return _react_loop_module().run_react_topic_rag(*args, **kwargs)
