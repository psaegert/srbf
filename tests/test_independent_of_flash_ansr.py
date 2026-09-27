"""srbf does not need flash-ansr (owner 2026-09-27). Only the adapters built on it and the two baselines that fit their
constants with its refiner import it, when they are built. Checked in a subprocess in which flash-ansr cannot be
imported, so the check holds whether or not flash-ansr is installed here."""
import json
import os
import subprocess
import sys

SCRIPT = r"""
import importlib, json, pkgutil, sys
sys.modules["flash_ansr"] = None          # as if flash-ansr were not installed
import srbf
out = {"needs_flash_ansr": [], "other_errors": {}}
for info in pkgutil.walk_packages(srbf.__path__, "srbf."):
    try:
        importlib.import_module(info.name)
    except Exception as exc:  # another optional package (a method's own) may be missing too; only flash-ansr counts
        if "flash_ansr" in repr(exc):
            out["needs_flash_ansr"].append(info.name)
        else:
            out["other_errors"][info.name] = repr(exc)[:200]
from srbf.config import build_model_adapter
try:
    build_model_adapter({"type": "flash_ansr", "model_path": "x", "evaluation_config": {}})
except ImportError as exc:
    out["flash_ansr_adapter"] = str(exc)
import srbf.config
srbf.config.resolve_simplipy_engine = lambda cfg, adapter_name: object()   # no engine download for this check
adapter = build_model_adapter({"type": "subprocess", "worker": "example", "python": sys.executable})
out["subprocess_adapter"] = type(adapter).__name__
print(json.dumps(out))
"""


def test_srbf_imports_and_builds_a_worker_adapter_without_flash_ansr():
    proc = subprocess.run([sys.executable, "-c", SCRIPT], capture_output=True, text=True, timeout=600,
                          env={**os.environ, "SRBF_ROOT": os.getcwd()})
    assert proc.returncode == 0, proc.stderr[-2000:]
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    # the two refining baselines fit their constants with flash-ansr's refiner; nothing else may need it
    assert all(m.startswith("srbf.baselines") for m in out["needs_flash_ansr"]), out["needs_flash_ansr"]
    assert "pip install 'srbf[flash-ansr]'" in out["flash_ansr_adapter"]
    assert out["subprocess_adapter"] == "SubprocessAdapter"
