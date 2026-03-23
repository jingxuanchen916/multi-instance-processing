"""
Ground Truth Generator
Generates ground truth answers for evaluation questions.
Supports multiple datasets: 'tweets', 'ner', 'wsd', 'sentiment', 'news', 'language', 'arithmetic', 'parity'.
"""

import logging
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)


# ========================================
# SHARED UTILITIES
# ========================================

def _read_lines(file_path: str, cast, *, skip_blank: bool = False) -> list:
    """
    Read file lines with optional casting and blank line filtering.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = (line.strip() for line in f)
        if skip_blank:
            lines = (line for line in lines if line)
        return [cast(line) for line in lines]


def _validate_indices(indices: List[int], max_idx: int) -> None:
    """
    Ensure indices stay inside available range.
    """
    if any(idx < 0 or idx > max_idx for idx in indices):
        raise ValueError(f"Invalid indices. Valid range: 0 to {max_idx}")


def _slice_by_indices(values: List[Any], indices: List[int]) -> List[Any]:
    """
    Fetch items by index order.
    """
    return [values[i] for i in indices]


def _load_paired_data(
    text_file: str,
    labels_file: str,
    indices: List[int],
    *,
    text_cast=str,
    label_cast=int,
    skip_blank: bool = False,
    text_name: str = "items",
    label_name: str = "labels"
) -> Tuple[List[Any], List[Any]]:
    """
    Load paired text/label files with shared validation and slicing.
    """
    texts = _read_lines(text_file, text_cast, skip_blank=skip_blank)
    labels = _read_lines(labels_file, label_cast, skip_blank=skip_blank)

    if len(texts) != len(labels):
        raise ValueError(f"Mismatch: {len(texts)} {text_name} but {len(labels)} {label_name}")

    _validate_indices(indices, len(texts) - 1)
    return _slice_by_indices(texts, indices), _slice_by_indices(labels, indices)


# ========================================
# GROUND TRUTH GENERATOR CLASS
# ========================================

class GroundTruthGenerator:
    """
    Generates and manages ground truth for evaluation runs.
    
    Ground truth is generated dynamically based on:
    - Question IDs
    - Instance selections (name + indices)
    - Dataset type
    """
    
    def __init__(self, run_dir: Path):
        """
        Initialize ground truth generator.
        
        Args:
            run_dir: Output directory for this evaluation run
        """
        self.run_dir = run_dir
        self.ground_truth_file = run_dir / "ground_truth.csv"

    def _generate_with_dataset_files(
        self,
        dataset,
        questions: List[str],
        selections: List[Tuple[str, List[int]]],
        *,
        dataset_label: str,
        generate_func
    ) -> None:
        """
        Shared routine for datasets with per-question text/label files.
        """
        results = []
        total = len(questions) * len(selections)
        
        logger.info(f"Generating {total} ground truth values for {dataset_label} with question-specific data files...")
        
        for question_id in questions:
            # Get question config to find data file
            question_config = dataset.questions.get(question_id)
            if not question_config:
                raise ValueError(f"Question {question_id} not found in dataset")
            
            # Get data file for this question
            data_file_name = question_config.get('data_file', dataset.default_data_file)
            
            # Build file paths
            text_file = str(dataset.data_dir / f"{data_file_name}_text.txt")
            labels_file = str(dataset.data_dir / f"{data_file_name}_label.txt")
            
            for selection_name, indices in selections:
                try:
                    # Generate ground truth for this combination
                    ground_truth = generate_func(text_file, labels_file, question_id, indices)
                    
                    results.append({
                        'question_id': question_id,
                        'selection_name': selection_name,
                        'indices': str(indices),
                        'ground_truth': ground_truth
                    })
                except Exception as e:
                    logger.error(
                        f"Failed to generate ground truth for {question_id}/{selection_name} "
                        f"with data file {data_file_name}: {e}"
                    )
                    raise
        
        # Save to CSV
        df = pd.DataFrame(results)
        df.to_csv(self.ground_truth_file, index=False)
        
        logger.info(f"Ground truth saved to {self.ground_truth_file}")

    def generate_wsd_with_dataset(
        self,
        dataset,
        questions: List[str],
        selections: List[Tuple[str, List[int]]]
    ):
        """
        Generate ground truth for WSD dataset with question-specific data files.
        """
        self._generate_with_dataset_files(
            dataset,
            questions,
            selections,
            dataset_label="WSD",
            generate_func=generate_wsd_answer
        )

    def generate(
        self,
        dataset_type: str,
        data_file: str,
        questions: List[str],
        selections: List[Tuple[str, List[int]]],
        labels_file: str = None
    ):
        """
        Generate ground truth for all question-selection combinations.
        Saves to run_dir/ground_truth.csv with columns:
        - question_id: Question identifier (e.g., 'Q1')
        - selection_name: Selection identifier (e.g., 'first_10', 'window_5_start0')
        - indices: List of indices as string representation
        - ground_truth: The ground truth answer
        
        Args:
            dataset_type: Type of dataset ('TweetDataset', 'NERDataset', 'WSDDataset', 'SentimentDataset', 'NewsDataset', 'LanguageDataset', 'ArithmeticDataset', 'ParityDataset')
            data_file: Path to data file
            questions: List of question IDs
            selections: List of (selection_name, indices) tuples
            labels_file: Path to labels file
        """
        dataset_generators = {
            'TweetDataset': generate_tweet_answer,
            'NERDataset': generate_ner_answer,
            'WSDDataset': generate_wsd_answer,
            'SentimentDataset': generate_sentiment_answer,
            'NewsDataset': generate_news_answer,
            'LanguageDataset': generate_language_answer,
            'ArithmeticDataset': generate_arithmetic_answer,
            'ParityDataset': generate_parity_answer,
        }

        if dataset_type not in dataset_generators:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

        if labels_file is None:
            dataset_label_name = dataset_type.replace('Dataset', '').strip() or "this"
            raise ValueError(f"labels_file is required for {dataset_label_name} dataset")

        generate_func = dataset_generators[dataset_type]
        
        # Generate all ground truths
        results = []
        total = len(questions) * len(selections)
        
        logger.info(f"Generating {total} ground truth values...")
        
        for question_id in questions:
            for selection_name, indices in selections:
                try:
                    # Generate ground truth for this combination
                    # All datasets now require labels_file
                    ground_truth = generate_func(data_file, labels_file, question_id, indices)
                    
                    results.append({
                        'question_id': question_id,
                        'selection_name': selection_name,
                        'indices': str(indices),  # Store as string representation
                        'ground_truth': ground_truth
                    })
                except Exception as e:
                    logger.error(
                        f"Failed to generate ground truth for {question_id}/{selection_name}: {e}"
                    )
                    raise
        
        # Save to CSV
        df = pd.DataFrame(results)
        df.to_csv(self.ground_truth_file, index=False)
        
        logger.info(f"Ground truth saved to {self.ground_truth_file}")
    
    def exists(self) -> bool:
        """
        Check if ground truth file exists.
        
        Returns:
            True if ground truth CSV exists
        """
        return self.ground_truth_file.exists()
    
    def get_file_path(self) -> Path:
        """
        Get path to ground truth CSV file.
        
        Returns:
            Path to ground_truth.csv
        """
        return self.ground_truth_file


# ========================================
# TWEETS DATASET FUNCTIONS
# ========================================

def load_tweets_labels_by_indices(text_file: str, labels_file: str, indices: List[int]) -> Tuple[List[str], List[int]]:
    """
    Load tweets and labels by specific indices.
    
    Args:
        text_file: Path to the tweets text file
        labels_file: Path to the labels file (count of 'women' per tweet)
        indices: List of 0-based indices to load
        
    Returns:
        Tuple of (tweets, labels) at specified indices
    """
    return _load_paired_data(
        text_file,
        labels_file,
        indices,
        text_name="tweets",
        label_name="labels"
    )


def count_word_women_by_indices(text_file: str, labels_file: str, indices: List[int]) -> int:
    """
    Q1: Count total occurrences of the word 'women' across all tweets.
    
    Args:
        text_file: Path to the tweets text file (not used, kept for consistency)
        labels_file: Path to the labels file
        indices: List of tweet indices to process
        
    Returns:
        Total count of 'women' occurrences
    """
    _, labels = load_tweets_labels_by_indices(text_file, labels_file, indices)
    
    # Sum all the counts
    return sum(labels)


def count_word_women_per_tweet_by_indices(text_file: str, labels_file: str, indices: List[int]) -> Dict[str, int]:
    """
    Q2: Count occurrences of 'women' in each tweet and return dict with per-tweet counts and total.
    
    Args:
        text_file: Path to the tweets text file (not used, kept for consistency)
        labels_file: Path to the labels file
        indices: List of tweet indices to process
        
    Returns:
        Dict with per-tweet counts and total:
        {"1": count, "2": count, ..., "total": sum}
    """
    _, labels = load_tweets_labels_by_indices(text_file, labels_file, indices)
    
    result = {}
    
    # Add per-tweet counts (1-indexed keys)
    for i, count in enumerate(labels, 1):
        result[str(i)] = count
    
    # Add total
    result["total"] = sum(labels)
    
    return result


# ========================================
# NER DATASET FUNCTIONS
# ========================================

def load_ner_by_indices(text_file: str, labels_file: str, indices: List[int]) -> tuple[List[str], List[Dict]]:
    """
    Load NER sentences and labels for specific indices.
    
    Args:
        text_file: Path to the sentences text file
        labels_file: Path to the labels file (JSON dicts per line)
        indices: List of 0-based sentence indices
        
    Returns:
        Tuple of (sentences list, labels list) for the specified indices
    """
    return _load_paired_data(
        text_file,
        labels_file,
        indices,
        text_cast=str,
        label_cast=json.loads,
        text_name="sentences",
        label_name="labels"
    )


def count_entity_person_by_indices(text_file: str, labels_file: str, indices: List[int]) -> int:
    """
    Q1: Count total occurrences of PERSON entities across all sentences.
    
    Args:
        text_file: Path to the sentences text file
        labels_file: Path to the labels file
        indices: List of sentence indices to process
        
    Returns:
        Total count of PERSON entities
    """
    _, labels = load_ner_by_indices(text_file, labels_file, indices)
    
    # Count total PERSON entities (length of PER list for each sentence)
    total_count = sum(len(label.get('PER', [])) for label in labels)
    
    return total_count


def count_entity_person_with_ids_by_indices(text_file: str, labels_file: str, indices: List[int]) -> Dict[str, int]:
    """
    Q2: Count PERSON entities per sentence and return dict with per-sentence counts and total.
    
    Args:
        text_file: Path to the sentences text file
        labels_file: Path to the labels file
        indices: List of sentence indices to process
        
    Returns:
        Dict with per-sentence counts and total:
        {"1": count, "2": count, ..., "total": total}
    """
    _, labels = load_ner_by_indices(text_file, labels_file, indices)
    
    result = {}
    total = 0
    
    # Build result with per-sentence counts (1-indexed keys)
    for i, label in enumerate(labels, 1):
        count = len(label.get('PER', []))
        result[str(i)] = count
        total += count
    
    # Add total
    result['total'] = total
    
    return result


# ========================================
# WSD DATASET FUNCTIONS
# ========================================

def load_wsd_data_by_indices(text_file: str, labels_file: str, indices: List[int]) -> tuple[List[str], List[int]]:
    """
    Load WSD paragraphs and labels for specific indices.
    
    Args:
        text_file: Path to the paragraphs text file
        labels_file: Path to the labels file
        indices: List of 0-based paragraph indices
        
    Returns:
        Tuple of (paragraphs list, labels list) for the specified indices
    """
    return _load_paired_data(
        text_file,
        labels_file,
        indices,
        text_name="paragraphs",
        label_name="labels"
    )


def count_apple_company_by_indices(text_file: str, labels_file: str, indices: List[int]) -> int:
    """
    Q1: Count how many paragraphs where 'apple' means the company (label = 0).
    
    Args:
        text_file: Path to the paragraphs text file
        labels_file: Path to the labels file
        indices: List of paragraph indices to process
        
    Returns:
        Count of paragraphs where apple means the company
    """
    _, labels = load_wsd_data_by_indices(text_file, labels_file, indices)
    
    # Count how many labels are 0 (company)
    company_count = labels.count(0)
    
    return company_count


def classify_apple_meaning_by_indices(text_file: str, labels_file: str, indices: List[int]) -> Dict[str, int]:
    """
    Q2: Classify apple meaning for each paragraph and return dict with per-paragraph labels and counts.
    
    Args:
        text_file: Path to paragraphs text file
        labels_file: Path to labels file
        indices: List of paragraph indices to classify
        
    Returns:
        Dict with per-paragraph classifications and summary counts:
        {"1": 0, "2": 1, ..., "company": count, "fruit": count}
    """
    _, labels = load_wsd_data_by_indices(text_file, labels_file, indices)
    
    result = {}
    
    # Add per-paragraph classifications (1-indexed keys)
    for i, label in enumerate(labels, start=1):
        result[str(i)] = label
    
    # Add summary counts
    result["company"] = labels.count(0)
    result["fruit"] = labels.count(1)
    
    return result

# ========================================
# NEWS DATASET FUNCTIONS
# ========================================

def load_news_by_indices(text_file: str, labels_file: str, indices: List[int]) -> Tuple[List[str], List[int]]:
    """
    Load news articles and labels by specific indices.
    
    Args:
        text_file: Path to the news text file
        labels_file: Path to the labels file
        indices: List of 0-based indices to load
        
    Returns:
        Tuple of (articles, labels) at specified indices
    """
    return _load_paired_data(
        text_file,
        labels_file,
        indices,
        text_name="articles",
        label_name="labels"
    )


def count_tech_news_by_indices(text_file: str, labels_file: str, indices: List[int]) -> int:
    """
    Q1: Count how many news articles belong to the 'tech' category (label=4).
    
    Args:
        text_file: Path to the news text file
        labels_file: Path to the labels file
        indices: List of 0-based indices to process
        
    Returns:
        Count of tech articles
    """
    _, labels = load_news_by_indices(text_file, labels_file, indices)
    
    # Count tech articles (label=4)
    tech_count = sum(1 for label in labels if label == 4)
    
    return tech_count


def classify_news_categories_by_indices(text_file: str, labels_file: str, indices: List[int]) -> Dict[str, Any]:
    """
    Q2: Classify each article into 5 categories and provide summary counts.
    
    Args:
        text_file: Path to the news text file
        labels_file: Path to the labels file
        indices: List of 0-based indices to process
        
    Returns:
        Dict with per-article classifications and summary counts
        Format: {"1": 0, "2": 1, ..., "business": count, "entertainment": count, 
                 "politics": count, "sport": count, "tech": count}
    """
    _, labels = load_news_by_indices(text_file, labels_file, indices)
    
    # Build result dict
    result = {}
    
    # Add per-article classifications (1-indexed keys)
    for i, label in enumerate(labels, 1):
        result[str(i)] = label
    
    # Add summary counts for each category
    result['business'] = sum(1 for label in labels if label == 0)
    result['entertainment'] = sum(1 for label in labels if label == 1)
    result['politics'] = sum(1 for label in labels if label == 2)
    result['sport'] = sum(1 for label in labels if label == 3)
    result['tech'] = sum(1 for label in labels if label == 4)
    
    return result


def classify_tech_binary_by_indices(text_file: str, labels_file: str, indices: List[int]) -> Dict[str, Any]:
    """
    Q3: Binary classification - tech (1) vs non-tech (0) with summary counts.
    
    Args:
        text_file: Path to the news text file
        labels_file: Path to the labels file
        indices: List of 0-based indices to process
        
    Returns:
        Dict with per-article binary classifications and summary counts
        Format: {"1": 0, "2": 1, ..., "tech": count, "non_tech": count}
    """
    _, labels = load_news_by_indices(text_file, labels_file, indices)
    
    # Build result dict
    result = {}
    
    # Add per-article binary classifications (1-indexed keys)
    # 1 if tech (label=4), 0 if non-tech (labels 0,1,2,3)
    for i, label in enumerate(labels, 1):
        result[str(i)] = 1 if label == 4 else 0
    
    # Add summary counts
    tech_count = sum(1 for label in labels if label == 4)
    non_tech_count = len(labels) - tech_count
    
    result['tech'] = tech_count
    result['non_tech'] = non_tech_count
    
    return result


def generate_news_answer(text_file: str, labels_file: str, question_id: str, indices: List[int]) -> Any:
    """
    Generate ground truth answer for a single news question and specific indices.
    
    Args:
        text_file: Path to the news text file
        labels_file: Path to the labels file
        question_id: Question identifier (e.g., 'Q1', 'Q2', 'Q3')
        indices: List of article indices to process
        
    Returns:
        Ground truth answer (type depends on question)
    """
    question_funcs = {
        'Q1': count_tech_news_by_indices,
        'Q2': classify_news_categories_by_indices,
        'Q3': classify_tech_binary_by_indices
    }
    
    if question_id not in question_funcs:
        raise ValueError(f"Unknown news question ID: {question_id}")
    
    return question_funcs[question_id](text_file, labels_file, indices)


# ========================================
# GROUND TRUTH GENERATION (INDICES-BASED)
# ========================================

def generate_tweet_answer(tweet_file: str, labels_file: str, question_id: str, indices: List[int]) -> Any:
    """
    Generate ground truth answer for a single tweet question and specific indices.
    
    Args:
        tweet_file: Path to the tweets text file
        labels_file: Path to the labels file (count of 'women' per tweet)
        question_id: Question identifier (e.g., 'Q1', 'Q2')
        indices: List of tweet indices to process
        
    Returns:
        Ground truth answer (type depends on question)
    """
    question_funcs = {
        'Q1': count_word_women_by_indices,
        'Q2': count_word_women_per_tweet_by_indices
    }
    
    if question_id not in question_funcs:
        raise ValueError(f"Unknown question ID: {question_id}")
    
    return question_funcs[question_id](tweet_file, labels_file, indices)


def generate_ner_answer(text_file: str, labels_file: str, question_id: str, indices: List[int]) -> Any:
    """
    Generate ground truth answer for a single NER question and specific indices.
    
    Args:
        text_file: Path to the sentences text file
        labels_file: Path to the labels file
        question_id: Question identifier (e.g., 'Q1', 'Q2')
        indices: List of sentence indices to process
        
    Returns:
        Ground truth answer (type depends on question)
    """
    question_funcs = {
        'Q1': count_entity_person_by_indices,
        'Q2': count_entity_person_with_ids_by_indices
    }
    
    if question_id not in question_funcs:
        raise ValueError(f"Unknown NER question ID: {question_id}")
    
    return question_funcs[question_id](text_file, labels_file, indices)


def generate_wsd_answer(text_file: str, labels_file: str, question_id: str, indices: List[int]) -> Any:
    """
    Generate ground truth answer for a single WSD question and specific indices.
    
    Args:
        text_file: Path to the paragraphs text file
        labels_file: Path to the labels file (ground truth)
        question_id: Question identifier (e.g., 'Q1', 'Q2', 'Q3', 'Q4')
        indices: List of paragraph indices to process
        
    Returns:
        Ground truth answer (type depends on question)
    """
    question_funcs = {
        'Q1': count_apple_company_by_indices,
        'Q2': classify_apple_meaning_by_indices
    }
    
    if question_id not in question_funcs:
        raise ValueError(f"Unknown WSD question ID: {question_id}")
    
    return question_funcs[question_id](text_file, labels_file, indices)


# ========================================
# SENTIMENT DATASET FUNCTIONS
# ========================================

def load_sentiment_by_indices(text_file: str, labels_file: str, indices: List[int]) -> Tuple[List[str], List[int]]:
    """
    Load movie reviews and labels by specific indices.
    
    Args:
        text_file: Path to the reviews text file
        labels_file: Path to the labels file
        indices: List of 0-based indices to load
        
    Returns:
        Tuple of (reviews, labels) at specified indices
    """
    return _load_paired_data(
        text_file,
        labels_file,
        indices,
        text_name="reviews",
        label_name="labels"
    )


def count_positive_reviews_by_indices(text_file: str, labels_file: str, indices: List[int]) -> int:
    """
    Q1: Count how many reviews are positive (label=1).
    
    Args:
        text_file: Path to the reviews text file
        labels_file: Path to the labels file
        indices: List of 0-based indices to process
        
    Returns:
        Count of positive reviews
    """
    _, labels = load_sentiment_by_indices(text_file, labels_file, indices)
    
    # Count positive reviews (label=1)
    positive_count = sum(1 for label in labels if label == 1)
    
    return positive_count


def classify_all_reviews_by_indices(text_file: str, labels_file: str, indices: List[int]) -> Dict[str, Any]:
    """
    Q2: Classify each review and provide summary counts.
    
    Args:
        text_file: Path to the reviews text file
        labels_file: Path to the labels file
        indices: List of 0-based indices to process
        
    Returns:
        Dict with per-review classifications and summary counts
        Format: {"1": 0, "2": 1, ..., "negative": count, "positive": count}
    """
    _, labels = load_sentiment_by_indices(text_file, labels_file, indices)
    
    # Build result dict
    result = {}
    
    # Add per-review classifications (1-indexed keys)
    for i, label in enumerate(labels, 1):
        result[str(i)] = label
    
    # Add summary counts
    negative_count = sum(1 for label in labels if label == 0)
    positive_count = sum(1 for label in labels if label == 1)
    
    result['negative'] = negative_count
    result['positive'] = positive_count
    
    return result


def generate_sentiment_answer(text_file: str, labels_file: str, question_id: str, indices: List[int]) -> Any:
    """
    Generate ground truth answer for a single sentiment question and specific indices.
    
    Args:
        text_file: Path to the reviews text file
        labels_file: Path to the labels file
        question_id: Question identifier (e.g., 'Q1', 'Q2')
        indices: List of review indices to process
        
    Returns:
        Ground truth answer (type depends on question)
    """
    question_funcs = {
        'Q1': count_positive_reviews_by_indices,
        'Q2': classify_all_reviews_by_indices
    }
    
    if question_id not in question_funcs:
        raise ValueError(f"Unknown sentiment question ID: {question_id}")
    
    return question_funcs[question_id](text_file, labels_file, indices)


# ========================================
# LANGUAGE DATASET FUNCTIONS
# ========================================

def load_language_by_indices(text_file: str, labels_file: str, indices: List[int]) -> Tuple[List[str], List[int]]:
    """
    Load language paragraphs and labels by specific indices.
    
    Args:
        text_file: Path to the paragraphs text file
        labels_file: Path to the labels file
        indices: List of 0-based indices to load
        
    Returns:
        Tuple of (paragraphs, labels) at specified indices
    """
    return _load_paired_data(
        text_file,
        labels_file,
        indices,
        text_name="paragraphs",
        label_name="labels"
    )


def count_english_by_indices(text_file: str, labels_file: str, indices: List[int]) -> int:
    """
    Q1: Count how many paragraphs are in English (label=0).
    
    Args:
        text_file: Path to the paragraphs text file
        labels_file: Path to the labels file
        indices: List of 0-based indices to process
        
    Returns:
        Count of English paragraphs
    """
    _, labels = load_language_by_indices(text_file, labels_file, indices)
    
    # Count English paragraphs (label=0)
    english_count = sum(1 for label in labels if label == 0)
    
    return english_count


def classify_language_all_by_indices(text_file: str, labels_file: str, indices: List[int]) -> Dict[str, Any]:
    """
    Q2: Classify each paragraph into 4 languages and provide summary counts.
    
    Args:
        text_file: Path to the paragraphs text file
        labels_file: Path to the labels file
        indices: List of 0-based indices to process
        
    Returns:
        Dict with per-paragraph classifications and summary counts
        Format: {"1": 0, "2": 1, ..., "english": count, "chinese": count, 
                 "persian": count, "spanish": count}
    """
    _, labels = load_language_by_indices(text_file, labels_file, indices)
    
    # Build result dict
    result = {}
    
    # Add per-paragraph classifications (1-indexed keys)
    for i, label in enumerate(labels, 1):
        result[str(i)] = label
    
    # Add summary counts for each language
    result['english'] = sum(1 for label in labels if label == 0)
    result['chinese'] = sum(1 for label in labels if label == 1)
    result['persian'] = sum(1 for label in labels if label == 2)
    result['spanish'] = sum(1 for label in labels if label == 3)
    
    return result


def classify_english_binary_by_indices(text_file: str, labels_file: str, indices: List[int]) -> Dict[str, Any]:
    """
    Q3: Binary classification - English (1) vs non-English (0) with summary counts.
    
    Args:
        text_file: Path to the paragraphs text file
        labels_file: Path to the labels file
        indices: List of 0-based indices to process
        
    Returns:
        Dict with per-paragraph binary classifications and summary counts
        Format: {"1": 0, "2": 1, ..., "english": count, "non_english": count}
    """
    _, labels = load_language_by_indices(text_file, labels_file, indices)
    
    # Build result dict
    result = {}
    
    # Add per-paragraph binary classifications (1-indexed keys)
    # 1 if English (label=0), 0 if non-English (labels 1,2,3)
    for i, label in enumerate(labels, 1):
        result[str(i)] = 1 if label == 0 else 0
    
    # Add summary counts
    english_count = sum(1 for label in labels if label == 0)
    non_english_count = len(labels) - english_count
    
    result['english'] = english_count
    result['non_english'] = non_english_count
    
    return result


def generate_language_answer(text_file: str, labels_file: str, question_id: str, indices: List[int]) -> Any:
    """
    Generate ground truth answer for a single language question and specific indices.
    
    Args:
        text_file: Path to the paragraphs text file
        labels_file: Path to the labels file
        question_id: Question identifier (e.g., 'Q1', 'Q2', 'Q3')
        indices: List of paragraph indices to process
        
    Returns:
        Ground truth answer (type depends on question)
    """
    question_funcs = {
        'Q1': count_english_by_indices,
        'Q2': classify_language_all_by_indices,
        'Q3': classify_english_binary_by_indices
    }
    
    if question_id not in question_funcs:
        raise ValueError(f"Unknown language question ID: {question_id}")
    
    return question_funcs[question_id](text_file, labels_file, indices)


# ========================================
# PARITY DATASET FUNCTIONS
# ========================================

def load_parity_by_indices(text_file: str, labels_file: str, indices: List[int]) -> Tuple[List[str], List[int]]:
    """
    Load parity numbers and labels by specific indices.
    
    Args:
        text_file: Path to the numbers text file
        labels_file: Path to the labels file
        indices: List of 0-based indices to load
        
    Returns:
        Tuple of (numbers, labels) at specified indices
    """
    return _load_paired_data(
        text_file,
        labels_file,
        indices,
        text_name="numbers",
        label_name="labels"
    )


def count_odd_by_indices(text_file: str, labels_file: str, indices: List[int]) -> int:
    """
    Q1: Count how many numbers are odd (label=1).
    
    Args:
        text_file: Path to the numbers text file
        labels_file: Path to the labels file
        indices: List of 0-based indices to process
        
    Returns:
        Count of odd numbers
    """
    _, labels = load_parity_by_indices(text_file, labels_file, indices)
    
    # Count odd numbers (label=1)
    odd_count = sum(1 for label in labels if label == 1)
    
    return odd_count


def classify_parity_all_by_indices(text_file: str, labels_file: str, indices: List[int]) -> Dict[str, Any]:
    """
    Q2: Classify each number as odd (1) or even (0) and provide summary counts.
    
    Args:
        text_file: Path to the numbers text file
        labels_file: Path to the labels file
        indices: List of 0-based indices to process
        
    Returns:
        Dict with per-number classifications and summary counts
        Format: {"1": 0, "2": 1, ..., "odd": count, "even": count}
    """
    _, labels = load_parity_by_indices(text_file, labels_file, indices)
    
    # Build result dict
    result = {}
    
    # Add per-number classifications (1-indexed keys)
    for i, label in enumerate(labels, 1):
        result[str(i)] = label
    
    # Add summary counts for odd and even
    result['odd'] = sum(1 for label in labels if label == 1)
    result['even'] = sum(1 for label in labels if label == 0)
    
    return result


def generate_parity_answer(text_file: str, labels_file: str, question_id: str, indices: List[int]) -> Any:
    """
    Generate ground truth answer for a single parity question and specific indices.
    
    Args:
        text_file: Path to the numbers text file
        labels_file: Path to the labels file
        question_id: Question identifier (e.g., 'Q1', 'Q2')
        indices: List of number indices to process
        
    Returns:
        Ground truth answer (type depends on question)
    """
    question_funcs = {
        'Q1': count_odd_by_indices,
        'Q2': classify_parity_all_by_indices,
    }
    
    if question_id not in question_funcs:
        raise ValueError(f"Unknown parity question ID: {question_id}")
    
    return question_funcs[question_id](text_file, labels_file, indices)


# ========================================
# ARITHMETIC DATASET FUNCTIONS
# ========================================

def load_arithmetic_by_indices(text_file: str, labels_file: str, indices: List[int]) -> Tuple[List[str], List[str]]:
    """
    Load arithmetic questions and answers by specific indices.
    
    Args:
        text_file: Path to the questions text file
        labels_file: Path to the answers file
        indices: List of 0-based indices to load
        
    Returns:
        Tuple of (questions, answers) at specified indices
    """
    return _load_paired_data(
        text_file,
        labels_file,
        indices,
        label_cast=str,
        text_name="questions",
        label_name="answers"
    )


def solve_arithmetic_by_indices(text_file: str, labels_file: str, indices: List[int]) -> str:
    """
    Q1: Solve all arithmetic questions and return the sum of all answers.
    
    Args:
        text_file: Path to the questions text file
        labels_file: Path to the answers file
        indices: List of 0-based indices to process
        
    Returns:
        The sum of all answers as a string
    """
    from decimal import Decimal, getcontext
    
    _, answers = load_arithmetic_by_indices(text_file, labels_file, indices)
    
    if not answers:
        return ""
    
    # If there's only one answer, return it directly after formatting
    if len(answers) == 1:
        answer = answers[0]
        # Format answer without unnecessary trailing zeros
        if '.' in answer:
            answer = answer.rstrip('0').rstrip('.')
        return answer
    
    # Use Decimal for precise arithmetic
    getcontext().prec = 50  # Set high precision
    total = sum(Decimal(ans) for ans in answers)
    
    # Format the result without unnecessary trailing zeros
    result = str(total)
    
    # Remove trailing zeros after decimal point if present
    if '.' in result:
        result = result.rstrip('0').rstrip('.')
    
    return result


def solve_arithmetic_with_ids_by_indices(text_file: str, labels_file: str, indices: List[int]) -> Dict[str, str]:
    """
    Q2: Solve multiple arithmetic questions and return answers with IDs and sum.
    
    Args:
        text_file: Path to the questions text file
        labels_file: Path to the answers file
        indices: List of 0-based indices to process
        
    Returns:
        Dict with per-question answers (1-indexed keys) and sum
        Format: {"1": "answer1", "2": "answer2", ..., "sum": "total"}
    """
    from decimal import Decimal, getcontext
    
    _, answers = load_arithmetic_by_indices(text_file, labels_file, indices)
    
    # Build result dict with individual answers (1-indexed keys)
    # Remove trailing zeros from each answer
    result = {}
    for i, answer in enumerate(answers, 1):
        # Format answer without unnecessary trailing zeros
        formatted_answer = answer
        if '.' in formatted_answer:
            formatted_answer = formatted_answer.rstrip('0').rstrip('.')
        result[str(i)] = formatted_answer
    
    # Use Decimal for precise arithmetic
    getcontext().prec = 50  # Set high precision
    total = sum(Decimal(ans) for ans in answers)
    
    # Format the sum without unnecessary trailing zeros
    sum_str = str(total)
    
    # Remove trailing zeros after decimal point if present
    if '.' in sum_str:
        sum_str = sum_str.rstrip('0').rstrip('.')
    
    result['sum'] = sum_str
    
    return result


def generate_arithmetic_answer(text_file: str, labels_file: str, question_id: str, indices: List[int]) -> Any:
    """
    Generate ground truth answer for a single arithmetic question and specific indices.
    
    Args:
        text_file: Path to the questions text file
        labels_file: Path to the answers file
        question_id: Question identifier (e.g., 'Q1', 'Q2')
        indices: List of question indices to process
        
    Returns:
        Ground truth answer (type depends on question)
    """
    question_funcs = {
        'Q1': solve_arithmetic_by_indices,
        'Q2': solve_arithmetic_with_ids_by_indices
    }
    
    if question_id not in question_funcs:
        raise ValueError(f"Unknown arithmetic question ID: {question_id}")
    
    return question_funcs[question_id](text_file, labels_file, indices)
