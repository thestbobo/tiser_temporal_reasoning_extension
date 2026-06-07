from __future__ import annotations

import builtins
import importlib
import sys

import pytest


ML_DEPENDENCIES = {"torch", "transformers", "peft", "trl"}


def test_src_and_utils_import_without_ml_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    for module_name in list(sys.modules):
        if module_name == "src" or module_name.startswith("src."):
            monkeypatch.delitem(sys.modules, module_name, raising=False)

    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        top_level = name.split(".", 1)[0]
        if top_level in ML_DEPENDENCIES:
            raise AssertionError(f"ML dependency imported during lightweight smoke: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    importlib.import_module("src")
    importlib.import_module("src.utils")
