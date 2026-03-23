"""
Dataset-specific evaluators.

Each evaluator wraps a dataset and provides:
- Ground truth retrieval
- Dataset parameter naming
"""

from .base_evaluator import BaseEvaluator
from .tweet_evaluator import TweetEvaluator
from .ner_evaluator import NEREvaluator
from .wsd_evaluator import WSDEvaluator
from .sentiment_evaluator import SentimentEvaluator
from .news_evaluator import NewsEvaluator
from .language_evaluator import LanguageEvaluator
from .arithmetic_evaluator import ArithmeticEvaluator
from .parity_evaluator import ParityEvaluator

__all__ = ['BaseEvaluator', 'TweetEvaluator', 'NEREvaluator', 'WSDEvaluator', 'SentimentEvaluator', 'NewsEvaluator', 'LanguageEvaluator', 'ArithmeticEvaluator', 'ParityEvaluator']
