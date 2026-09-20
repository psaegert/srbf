"""The documentation is checked against the package it describes.

RECALL: what a user can reach (every CLI flag, adapter type, suite catalog, derived metric) is named in the docs.
PRECISION: what the docs name exists (flags, importable names, repository paths), their examples parse, and the prose
keeps to the current behaviour of the current release: no version history, no infrastructure names, no stale model.
"""
from __future__ import annotations

import argparse
import ast
import importlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
PAGES = sorted(DOCS.glob("*.md"))
PUBLIC = PAGES + [ROOT / "README.md", ROOT / "CONTRIBUTING.md"]


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _blocks(text: str, *languages: str) -> list[str]:
    return [m.group(2) for m in re.finditer(r"```(\w+)\n(.*?)```", text, flags=re.S) if m.group(1) in languages]


def _all_docs() -> str:
    return "\n".join(_text(p) for p in PAGES)


# ---- recall ------------------------------------------------------------------------------------------------------
def _subparsers() -> dict[str, argparse.ArgumentParser]:
    from srbf.__main__ import build_parser
    action = next(a for a in build_parser()._actions if isinstance(a, argparse._SubParsersAction))
    return dict(action.choices)


def test_every_command_and_flag_is_documented() -> None:
    text = _text(DOCS / "cli.md")
    for name, parser in _subparsers().items():
        assert f"srbf {name}" in text, f"`srbf {name}` is not in docs/cli.md"
        section = text.split(f"## `srbf {name}`", 1)
        assert len(section) == 2, f"docs/cli.md has no section for `srbf {name}`"
        body = section[1].split("\n## ", 1)[0]
        for action in parser._actions:
            for flag in action.option_strings:
                if flag.startswith("--") and flag != "--help":
                    assert f"`{flag}" in body, f"`srbf {name} {flag}` is not documented"


def test_every_documented_flag_exists() -> None:
    parsers = _subparsers()
    for page in PUBLIC:
        for block in _blocks(_text(page), "bash", "console", "sh"):
            for line in re.split(r"\n(?!\s)", block.replace("\\\n", " ")):
                m = re.match(r"\s*(?:[A-Z_]+=\S+\s+)*(?:python -m )?srbf (\w+)(.*)", line)
                if not m or m.group(1) not in parsers:
                    continue
                known = {f for a in parsers[m.group(1)]._actions for f in a.option_strings}
                for flag in re.findall(r"(?<=\s)(--?[a-zA-Z][\w-]*)", m.group(2).split("#")[0]):
                    assert flag in known, f"{page.name}: `srbf {m.group(1)} {flag}` does not exist"


def test_every_adapter_type_is_documented() -> None:
    from srbf.config import _ADAPTER_REGISTRY
    text = _text(DOCS / "models.md") + _text(DOCS / "adapters.md")
    for name in _ADAPTER_REGISTRY:
        assert f"`{name}`" in text, f"adapter type `{name}` is not documented"
    assert "package.module:function" in text or "module:function" in text, "the builder form of `type` is not documented"


def test_every_suite_catalog_is_documented() -> None:
    from srbf.suites import SRBF_CATALOGS
    text = _text(DOCS / "benchmarks.md")
    for name in SRBF_CATALOGS:
        assert f"`{name}`" in text, f"catalog `{name}` is not in docs/benchmarks.md"
    assert str(len(SRBF_CATALOGS)) in text


def test_every_derived_metric_is_documented() -> None:
    """Every column that metric derivation adds has a definition on the metrics page."""
    import numpy as np
    from srbf import derive_metrics
    from srbf.sample_metadata import build_base_metadata   # noqa: F401  (the raw columns come from here)
    raw = {"skeleton": [["+", "x1", "<constant>"]], "expression": [["+", "x1", "1.5"]], "variables": [["x1"]], "variable_names": [["x1"]],
           "x": [np.linspace(0, 1, 8).reshape(-1, 1)], "y": [np.linspace(0, 1, 8).reshape(-1, 1) + 1.5],
           "x_val": [np.linspace(1, 2, 8).reshape(-1, 1)], "y_val": [np.linspace(1, 2, 8).reshape(-1, 1) + 1.5],
           "y_pred": [np.linspace(0, 1, 8).reshape(-1, 1) + 1.5], "y_pred_val": [np.linspace(1, 2, 8).reshape(-1, 1) + 1.5],
           "predicted_expression": ["x1 + 1.5"], "predicted_expression_prefix": [["+", "x1", "1.5"]],
           "predicted_skeleton_prefix": [["+", "x1", "<constant>"]], "prediction_success": [True], "placeholder": [False],
           "fit_time": [0.1], "benchmark_eq_id": ["a"]}
    scored = derive_metrics(raw, operator_arity={"+": 2})
    text = _text(DOCS / "metrics.md")
    for key in sorted(set(scored) - set(raw)):
        assert f"`{key}`" in text, f"derived column `{key}` is not defined in docs/metrics.md"


# ---- precision ---------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("page", PUBLIC, ids=lambda p: p.name)
def test_python_examples_parse_and_their_imports_resolve(page: Path) -> None:
    for block in _blocks(_text(page), "python"):
        tree = ast.parse(block)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] == "srbf":
                module = importlib.import_module(node.module)
                for alias in node.names:
                    assert hasattr(module, alias.name), f"{page.name}: `from {node.module} import {alias.name}` does not resolve"


@pytest.mark.parametrize("page", PUBLIC, ids=lambda p: p.name)
def test_yaml_examples_parse(page: Path) -> None:
    from srbf import register_sweep_yaml
    register_sweep_yaml()
    for block in _blocks(_text(page), "yaml"):
        yaml.safe_load(block)


@pytest.mark.parametrize("page", PUBLIC, ids=lambda p: p.name)
def test_repository_paths_in_the_docs_exist(page: Path) -> None:
    for path in set(re.findall(r"(?<![\w/.{}-])((?:configs|scripts|src|tests|envs|assets)/[\w./-]+\.(?:py|yaml|yml|md|txt|json|svg))", _text(page))):
        if "mymethod" in path or "<" in path:
            continue
        assert (ROOT / path).exists(), f"{page.name} names {path}, which is not in the repository"


# What never belongs on a public page. Every entry is a way the docs have gone wrong before.
FACADE = {
    r"\b(?:since|from|as of|until|before|in) (?:srbf |flash-ansr |simplipy )?v?\d+\.\d+": "version history: say what holds now; the CHANGELOG keeps history",
    r"\bpre-\d+\.\d+|\(srbf \d+\.\d+|\bpost-\d+\.\d+": "version history",
    r"\bcarved out\b|\bused to\b|\bno longer\b|\bformerly\b|\bpreviously\b|\bwe (?:now|decided|chose|moved)\b": "history or an internal decision",
    r"/home/\w+|/Users/\w+": "a path on somebody's machine",
    r"\bTODO\b|\bFIXME\b|\bWP\d+\b|\bowner\b|\binternal(?:ly)?\b": "internal vocabulary",
    r"v25\.0-T7|v24\b|v23\.\d-(?!val)": "a model that is not the current reference",
    r"—|&mdash;": "em-dash: use a colon, a semicolon, parentheses or a new sentence",
    r"(?i)\bhonest(?:ly)?\b|\bnever silently\b|\bnot aspirational\b|\bsimply\b|\bjust\b(?! as)|\bof course\b|\bobviously\b": "protesting or filler tone",
}

# Names that must not reach a public page are not spelled out in a public test either: a maintainer's machine
# keeps them in a git-ignored file, and the lint merges them in where that file exists.
_LOCAL_PATTERNS = ROOT / "results-site" / "private" / "banned_patterns.json"
if _LOCAL_PATTERNS.is_file():
    FACADE.update(json.loads(_LOCAL_PATTERNS.read_text()))


@pytest.mark.parametrize("page", PUBLIC, ids=lambda p: p.name)
def test_the_facade_is_clean(page: Path) -> None:
    text = _text(page)
    prose = re.sub(r"```.*?```", "", text, flags=re.S)
    found = []
    for pattern, why in FACADE.items():
        for m in re.finditer(pattern, prose if "em-dash" not in why else text):
            found.append(f"{page.name}: {m.group(0)!r} ({why}): ...{prose[max(0, m.start() - 50):m.end() + 30]!r}")
    assert not found, "\n".join(found)


def test_the_site_builds_strictly(tmp_path: Path) -> None:
    """Every internal link and anchor resolves: mkdocs reports a broken one as a warning, --strict makes it fatal."""
    if importlib.util.find_spec("mkdocs") is None or importlib.util.find_spec("material") is None:
        pytest.skip("mkdocs-material is not installed")
    # no -q: strict mode counts the warnings that are LOGGED, so a quiet build passes whatever is broken
    out = subprocess.run([sys.executable, "-m", "mkdocs", "build", "--strict", "-d", str(tmp_path / "site")], cwd=ROOT, capture_output=True, text=True)
    shutil.rmtree(tmp_path / "site", ignore_errors=True)
    warnings = [line for line in out.stderr.splitlines() if line.startswith("WARNING")]
    assert out.returncode == 0 and not warnings, "\n".join(warnings) or out.stderr[-3000:]
