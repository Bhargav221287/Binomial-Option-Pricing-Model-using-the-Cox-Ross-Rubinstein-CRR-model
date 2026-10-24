"""CRR Binomial Option Pricing & Greeks Engine.

A production-grade, zero-dependency quantitative finance library implementing
the Cox-Ross-Rubinstein (1979) binomial model for European and American options,
lattice-based and numerical Greeks extraction, and terminal ASCII lattice visualization.
"""

from crr_engine.models import (
    ExerciseStyle,
    OptionType,
    OptionContract,
    CRRBinomialTree,
    BinomialNode,
    BlackScholesBenchmark,
)
from crr_engine.greeks import (
    GreeksResult,
    GreeksEngine,
)
from crr_engine.visualizer import (
    TreeVisualizer,
)

__version__ = "1.0.0"
__author__ = "P Sai Bhargav"

__all__ = [
    "ExerciseStyle",
    "OptionType",
    "OptionContract",
    "CRRBinomialTree",
    "BinomialNode",
    "BlackScholesBenchmark",
    "GreeksResult",
    "GreeksEngine",
    "TreeVisualizer",
]
