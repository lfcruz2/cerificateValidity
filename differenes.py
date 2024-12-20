import pandas as pd
import numpy as np
from typing import Callable, Dict, Tuple, List, Any
import seaborn as sns
import matplotlib.pyplot as plt

# ========================================
# Previous code assumptions and setup
# ========================================

cols_to_compare = [c for c in df.columns if c not in ['PAIR_ID', 'TRANS_TYPE']]
numeric_cols = [c for c in cols_to_compare if pd.api.types.is_numeric_dtype(df[c])]
datetime_cols = [c for c in cols_to_compare if pd.api.types.is_datetime64_any_dtype(df[c])]
text_cols = [c for c in cols_to_compare if c not in numeric_cols + datetime_cols]

special_numeric_config = {
    'amount': {
        'type': 'function',
        'function': lambda diff: 0.0 if diff > 10_000_000 else (1.0 - diff / 10_000_000)
    },
    'Rate': {
        'type': 'categories',
        'categories': [
            {'max_diff': 0.0001, 'similarity': 1.0},
            {'max_diff': 0.001, 'similarity': 0.5},
            {'max_diff': None, 'similarity': 0.0}
        ]
    }
}

def similarity_numeric_general(val_cancel: float, val_replace: float) -> float:
    diff = abs(val_cancel - val_replace)
    denom = max(1, abs(val_cancel), abs(val_replace))
    return max(0, 1 - (diff / denom))

def apply_special_numeric_rule(column_name: str, diff: float, config: Dict[str, Any]) -> float:
    if column_name not in config:
        return None

    rule = config[column_name]
    rule_type = rule.get('type')

    if rule_type == 'function':
        func = rule.get('function')
        if callable(func):
            return func(diff)
        else:
            raise ValueError(f"No valid function found for column {column_name}")

    elif rule_type == 'categories':
        categories = rule.get('categories', [])
        for cat in categories:
            max_diff = cat['max_diff']
            if max_diff is None or diff <= max_diff:
                return cat['similarity']
        return 0.0
    else:
        return None

def parse_datetime_with_formats(value: Any, formats: List[str]) -> pd.Timestamp:
    if isinstance(value, pd.Timestamp):
        return value
    if not isinstance(value, str):
        return None
    
    for fmt in formats:
        try:
            return pd.to_datetime(value, format=fmt)
        except ValueError:
            continue
    return None

def parse_datetime(value: Any, known_formats: List[str] = None) -> pd.Timestamp:
    if pd.api.types.is_datetime64_any_dtype(pd.Series([value])):
        return value

    parsed = pd.to_datetime(value, errors='coerce')
    if not pd.isnull(parsed):
        return parsed

    if known_formats is None:
        known_formats = [
            '%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', 
            '%Y-%m-%d %H:%M:%S', '%d/%m/%Y %H:%M:%S', '%m/%d/%Y %H:%M:%S'
        ]
    return parse_datetime_with_formats(value, known_formats)

def similarity_datetime(val_cancel: Any, val_replace: Any) -> float:
    dt_cancel = parse_datetime(val_cancel)
    dt_replace = parse_datetime(val_replace)
    if dt_cancel is None or dt_replace is None:
        return 0.0
    diff_days = abs((dt_cancel - dt_replace).total_seconds()) / 86400
    return 1.0 / (1.0 + diff_days)

def similarity_text(val_cancel: Any, val_replace: Any) -> float:
    return 1.0 if val_cancel == val_replace else 0.0

def extract_pairs(df: pd.DataFrame) -> Dict[Any, Tuple[pd.Series, pd.Series]]:
    pairs = {}
    for pid, group in df.groupby('PAIR_ID'):
        row_cancel = group[group['TRANS_TYPE'] == 'CANCELLED'].iloc[0]
        row_replace = group[group['TRANS_TYPE'] == 'REPLACED'].iloc[0]
        pairs[pid] = (row_cancel, row_replace)
    return pairs

pairs = extract_pairs(df)

def calculate_similarities(pairs: Dict[Any, Tuple[pd.Series, pd.Series]], 
                           cols: List[str], 
                           similarity_func: Callable[[Any, Any, str], float]) -> pd.DataFrame:
    results = []
    for pid, (row_cancel, row_replace) in pairs.items():
        row_res = {'PAIR_ID': pid}
        for c in cols:
            row_res[c] = similarity_func(row_cancel[c], row_replace[c], c)
        results.append(row_res)
    return pd.DataFrame(results)

def calculate_numeric_similarities(pairs: Dict[Any, Tuple[pd.Series, pd.Series]], numeric_cols: List[str]) -> pd.DataFrame:
    def numeric_similarity_wrapper(val_cancel, val_replace, col_name):
        diff = abs(val_cancel - val_replace)
        special_result = apply_special_numeric_rule(col_name, diff, special_numeric_config)
        if special_result is not None:
            return special_result
        else:
            return similarity_numeric_general(val_cancel, val_replace)
    return calculate_similarities(pairs, numeric_cols, numeric_similarity_wrapper)

def calculate_datetime_similarities(pairs: Dict[Any, Tuple[pd.Series, pd.Series]], datetime_cols: List[str]) -> pd.DataFrame:
    def datetime_similarity_wrapper(val_cancel, val_replace, col_name):
        return similarity_datetime(val_cancel, val_replace)
    return calculate_similarities(pairs, datetime_cols, datetime_similarity_wrapper)

def calculate_text_similarities(pairs: Dict[Any, Tuple[pd.Series, pd.Series]], text_cols: List[str]) -> pd.DataFrame:
    def text_similarity_wrapper(val_cancel, val_replace, col_name):
        return similarity_text(val_cancel, val_replace)
    return calculate_similarities(pairs, text_cols, text_similarity_wrapper)

def save_to_csv(df: pd.DataFrame, filename: str) -> None:
    df.to_csv(filename, index=False)

# Calculate the similarity dataframes
numeric_similarity_df = calculate_numeric_similarities(pairs, numeric_cols)
datetime_similarity_df = calculate_datetime_similarities(pairs, datetime_cols)
text_similarity_df = calculate_text_similarities(pairs, text_cols)

# Example of saving results
save_to_csv(numeric_similarity_df, 'numeric_similarities.csv')
save_to_csv(datetime_similarity_df, 'datetime_similarities.csv')
save_to_csv(text_similarity_df, 'text_similarities.csv')

print("Numeric Similarities:\n", numeric_similarity_df)
print("Datetime Similarities:\n", datetime_similarity_df)
print("Text Similarities:\n", text_similarity_df)

# ---------------------------------------
# 2) Function to visualize similarity matrix with seaborn
# ---------------------------------------
def plot_similarity_matrix(sim_df: pd.DataFrame, title: str):
    """
    Plot a similarity matrix for the columns in a given similarity DataFrame.
    This function:
    - Assumes sim_df has a 'PAIR_ID' column and other columns representing similarity scores.
    - Computes the correlation among the similarity columns (excluding PAIR_ID).
    - Plots a heatmap using seaborn.

    Parameters:
    - sim_df: The similarity DataFrame (with one row per PAIR_ID, columns = similarity scores).
    - title: Title for the plot.
    """
    # Set PAIR_ID as index, if present
    if 'PAIR_ID' in sim_df.columns:
        sim_df = sim_df.set_index('PAIR_ID', drop=True)
    
    # Compute the correlation matrix of the similarity columns
    corr = sim_df.corr()
    
    # Create a seaborn heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='viridis', fmt='.2f')
    plt.title(title)
    plt.tight_layout()
    plt.show()

# Example usage of plot_similarity_matrix for each type:
plot_similarity_matrix(numeric_similarity_df, "Numeric Similarity Matrix")
plot_similarity_matrix(datetime_similarity_df, "Datetime Similarity Matrix")
plot_similarity_matrix(text_similarity_df, "Text Similarity Matrix")
