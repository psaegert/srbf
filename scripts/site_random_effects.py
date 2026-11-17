"""The averaging the results site uses, for the scripts and the page's parity cases: srbf.aggregate, loaded from this
checkout's source (src/srbf/aggregate.py) rather than from the installed srbf, so the site is built with the averaging
of the code it is built from, whatever srbf the environment has. Every name of srbf.aggregate is available here.
"""
from __future__ import annotations

import importlib.util
import os
import sys

_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "srbf", "aggregate.py")
_SPEC = importlib.util.spec_from_file_location("_srbf_aggregate_checkout", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE   # its dataclasses look the module up by name
_SPEC.loader.exec_module(_MODULE)
globals().update({k: v for k, v in vars(_MODULE).items() if not k.startswith("__")})
