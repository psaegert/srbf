"""srbf's own baselines. The polynomial family needs NumPy only; the sampling and enumeration baselines fit their
constants with flash-ansr's refiner, so they are imported when first named and srbf imports without flash-ansr."""
from typing import TYPE_CHECKING, Any

from .polynomial_model import PolynomialModel

if TYPE_CHECKING:
    from .brute_force_model import BruteForceModel
    from .lample_charton_model import LampleChartonModel

__all__ = [
    "LampleChartonModel",
    "BruteForceModel",
    "PolynomialModel",
]


def __getattr__(name: str) -> Any:
    if name == "LampleChartonModel":
        from .lample_charton_model import LampleChartonModel
        return LampleChartonModel
    if name == "BruteForceModel":
        from .brute_force_model import BruteForceModel
        return BruteForceModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
