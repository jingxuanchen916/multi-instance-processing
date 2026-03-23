"""
Model Utilities
Helper functions for model information and token cost calculations.
"""

import logging
import pandas as pd
from typing import Dict
from pathlib import Path

logger = logging.getLogger(__name__)


def load_model_info(config_dir: Path) -> pd.DataFrame:
    """
    Load model information (pricing, context length, etc.) from CSV.
    
    Args:
        config_dir: Directory containing model_info.csv
        
    Returns:
        DataFrame with model info indexed by model name
    """
    model_info_file = config_dir / "model_info.csv"
    
    if not model_info_file.exists():
        logger.warning(f"Model info file not found: {model_info_file}")
        logger.warning("Token costs will not be calculated. Add config/model_info.csv to enable cost calculation.")
        return pd.DataFrame()
    
    df = pd.read_csv(model_info_file)
    df.set_index('model_name', inplace=True)
    logger.info(f"Loaded model info for {len(df)} models")
    return df


def calculate_token_cost(
    model: str,
    usage: Dict[str, int],
    model_info: pd.DataFrame
) -> Dict[str, float]:
    """
    Calculate token costs based on usage and model pricing.
    
    Args:
        model: Model identifier
        usage: Usage dictionary with prompt_tokens and completion_tokens
        model_info: DataFrame with model pricing info
        
    Returns:
        Dictionary with prompt_cost, completion_cost, total_cost, and token counts
    """
    costs = {
        'prompt_tokens': usage.get('prompt_tokens', 0),
        'completion_tokens': usage.get('completion_tokens', 0),
        'total_tokens': usage.get('total_tokens', 0),
        'prompt_cost': None,
        'completion_cost': None,
        'total_cost': None
    }
    
    # Check if we have pricing info for this model
    if model_info.empty or model not in model_info.index:
        logger.debug(f"No pricing info for model: {model}")
        return costs
    
    model_pricing = model_info.loc[model]
    
    # Get per-1M token prices and convert to per-token prices
    prompt_price_per_1m = model_pricing.get('prompt_price_per_1m')
    completion_price_per_1m = model_pricing.get('completion_price_per_1m')
    
    # Calculate costs if pricing is available
    if pd.notna(prompt_price_per_1m) and pd.notna(completion_price_per_1m):
        # Convert per-1M prices to per-token prices
        prompt_price_per_token = float(prompt_price_per_1m) / 1_000_000
        completion_price_per_token = float(completion_price_per_1m) / 1_000_000
        
        prompt_cost = usage.get('prompt_tokens', 0) * prompt_price_per_token
        completion_cost = usage.get('completion_tokens', 0) * completion_price_per_token
        
        costs['prompt_cost'] = prompt_cost
        costs['completion_cost'] = completion_cost
        costs['total_cost'] = prompt_cost + completion_cost
    
    return costs
