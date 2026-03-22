import pandas as pd
import re
import os

# Load the complete dataset
_script_dir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(_script_dir, '..', 'JFK_Pages_Merged.csv'))

print(f"Total pages before filtering: {len(df)}")
before = len(df)
df = df[df['content'].notna() & (df['content'].str.strip() != '')]
print(f"Removed {before - len(df)} rows with empty or null content")
print(f"Total pages after filtering: {len(df)}")

# Validate that all pages from the 55 missing CSV are present in the merged dataset
missing_csv = pd.read_csv(os.path.join(_script_dir, '..', 'jfk_categorization_55missing.csv'))
missing_keys = set(zip(missing_csv['file_id'].astype(str).str.strip(), missing_csv['page_number'].astype(str).str.strip()))
merged_keys = set(zip(df['file_id'].astype(str).str.strip(), df['page_number'].astype(str).str.strip()))
not_found = missing_keys - merged_keys
if not_found:
    print(f"WARNING: {len(not_found)} pages from jfk_categorization_55missing.csv are not in the merged dataset:")
    for key in sorted(not_found):
        print(f"  file_id={key[0]}, page_number={key[1]}")
else:
    print(f"All {len(missing_keys)} pages from jfk_categorization_55missing.csv are present in the merged dataset.")

# Compute basic metrics
df['char_count'] = df['content'].str.len()
df['word_count'] = df['content'].str.split().str.len()
df['line_count'] = df['content'].str.count('\n') + 1

# Text composition metrics
def uppercase_ratio(text):
    if pd.isna(text):
        return 0.0
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return 0.0
    upper = sum(1 for c in alpha if c.isupper())
    return upper / len(alpha)

def numeric_ratio(text):
    if pd.isna(text):
        return 0.0
    tokens = text.split()
    if not tokens:
        return 0.0
    numeric = sum(1 for t in tokens if t.isdigit())
    return numeric / len(tokens)

def short_token_ratio(text):
    if pd.isna(text):
        return 0.0
    tokens = text.split()
    if not tokens:
        return 0.0
    short = sum(1 for t in tokens if len(t) <= 2)
    return short / len(tokens)

def unique_word_ratio(text):
    if pd.isna(text):
        return 0.0
    tokens = text.split()
    if not tokens:
        return 0.0
    unique = len(set(tokens))
    return unique / len(tokens)

df['uppercase_ratio'] = df['content'].apply(uppercase_ratio)
df['numeric_ratio'] = df['content'].apply(numeric_ratio)
df['short_token_ratio'] = df['content'].apply(short_token_ratio)
df['unique_word_ratio'] = df['content'].apply(unique_word_ratio)

# Structure detection
def code_like_line_ratio(text):
    if pd.isna(text):
        return 0.0
    lines = text.split('\n')
    if not lines:
        return 0.0
    code_like = 0
    for line in lines:
        line = line.strip()
        if re.match(r'^[A-Z]+/\d+(/[A-Z]+)?$', line):
            code_like += 1
    return code_like / len(lines)

df['code_like_line_ratio'] = df['content'].apply(code_like_line_ratio)

# Keyword detection flags
df['contains_distribution_keyword'] = df['content'].str.lower().str.contains(r'\b(?:distribution|division|station|hqs)\b', na=False)
df['contains_classification_keyword'] = df['content'].str.lower().str.contains(r'\b(?:secret|confidential|classified)\b', na=False)

# Heuristic flags
df['is_low_content_page'] = df['word_count'] < 40
df['is_likely_distribution_page'] = (df['code_like_line_ratio'] > 0.3) | df['contains_distribution_keyword']
df['is_likely_cover_page'] = (df['word_count'] < 60) & (df['uppercase_ratio'] > 0.5)

# Save the dataset
os.makedirs(os.path.join(_script_dir, '..', 'data'), exist_ok=True)
df.to_csv(os.path.join(_script_dir, '..', 'data', 'pages_phase1_structural.csv'), index=False)