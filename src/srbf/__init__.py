"""srbf: the symbolic-regression evaluation framework, carved from flash-ansr.

Engine + model adapters + benchmarks + metrics for evaluating symbolic-regression models.
Depends one-way on flash-ansr (srbf imports flash-ansr; flash-ansr never imports srbf).
"""
from srbf.eval.evaluation import Evaluation
from srbf.eval.run_config import EvaluationRunPlan, build_evaluation_run

__all__ = ["Evaluation", "EvaluationRunPlan", "build_evaluation_run"]
__version__ = "0.1.0"
