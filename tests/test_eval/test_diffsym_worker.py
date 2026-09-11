"""The diffsym worker's pure parts: the two translations between diffsym and srbf (the pre-0.12
vocabulary, the variable names) and the out-of-range-variable filter. The model itself is not
loaded here -- `srbf check -c configs/evaluation/baselines/diffsym_fastsrb.yaml` is that smoke."""
import sys

import pytest

from srbf.subprocess_adapter import BUILTIN_WORKERS
from srbf.worker import MODELS_DIR


@pytest.fixture(scope="module")
def worker():
    """The worker module, imported the way the runner imports it (helpers on the path)."""
    pytest.importorskip("torch")
    sys.path.insert(0, str(MODELS_DIR.parent))
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("diffsym_worker", MODELS_DIR / "diffsym_worker.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(MODELS_DIR.parent))


def test_it_is_a_builtin_worker():
    assert "diffsym" in BUILTIN_WORKERS and BUILTIN_WORKERS["diffsym"].is_file()


class TestOutOfRangeVariables:
    """The decoder's variable vocabulary is the checkpoint's (x1..x8), not the problem's: on a
    4-variable problem it still proposes `x8`, which is not an answer to THAT problem."""

    def test_a_variable_the_problem_lacks_is_out_of_range(self, worker):
        assert worker._out_of_range(["+", "x1", "x8"], 4) is True
        assert worker._out_of_range(["+", "x1", "x4"], 4) is False
        assert worker._out_of_range(["*", "<constant>", "x1"], 1) is False
        assert worker._out_of_range(["*", "<constant>", "x2"], 1) is True

    def test_non_variable_tokens_pass(self, worker):
        assert worker._out_of_range(["pow2", "x1", "1.5", "<constant>", "exp"], 1) is False


class TestTranslations:
    """What leaves the worker is diffsym's own expression in srbf's vocabulary and the catalog's names."""

    def test_legacy_vocabulary_and_names(self, worker):
        from srbf_worker_helpers import prefix_to_infix, respell_legacy_prefix, substitute_placeholders
        legacy = ["+", "pow2", "x1", "*", "<constant>", "pow1_3", "x2"]      # x1^2 + c * cbrt(x2)
        realized = substitute_placeholders(respell_legacy_prefix(legacy), [2.5])
        assert realized == ["+", "pow", "x1", "2", "*", "2.5", "rootn", "x2", "3"]
        names = {"x1": "v1", "x2": "v2"}
        infix = prefix_to_infix([names.get(t, t) for t in realized])
        assert "v1" in infix and "v2" in infix and "x1" not in infix
        assert "pow1_3" not in infix and "rootn" in infix

    def test_the_helpers_are_importable_from_the_models_directory(self):
        sys.path.insert(0, str(MODELS_DIR.parent))
        try:
            import srbf_worker_helpers
            assert hasattr(srbf_worker_helpers, "respell_legacy_prefix")
        finally:
            sys.path.remove(str(MODELS_DIR.parent))


class TestNormalization:
    """Sampling must mirror training: the tnet path normalises y, the set-transformer path does not."""

    def test_auto_follows_the_encoder(self, worker):
        assert worker._resolve_normalization({"coord_encoder_type": "tnet"}, "auto") == "arcsinh"
        assert worker._resolve_normalization({"coord_encoder_type": "set_transformer"}, "auto") is None
        assert worker._resolve_normalization({}, "auto") == "arcsinh"                  # tnet is the default
        assert worker._resolve_normalization({"coord_encoder_type": "tnet"}, "none") is None
        assert worker._resolve_normalization({"coord_encoder_type": "tnet"}, "symlog") == "symlog"
