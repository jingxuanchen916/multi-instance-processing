"""
Evaluation module for running LLM tests.

Structure:
- evaluators/: Dataset-specific evaluator implementations
- runner.py: Main orchestration logic
"""

from .evaluators import BaseEvaluator, TweetEvaluator, NEREvaluator
from .runner import EvaluationRunner

__all__ = [
    'BaseEvaluator',
    'TweetEvaluator',
    'NEREvaluator',
    'EvaluationRunner'
]
