import pandas as pd
import os

# Load the dataset
_script_dir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(_script_dir, '..', 'data', 'pages_phase1_structural.csv'))

print("=== Basic Dataset Information ===")
print(f"Number of rows: {df.shape[0]}")
print(f"Column names: {list(df.columns)}")
print("\nMissing values per column:")
print(df.isnull().sum())

print("\n=== Descriptive Statistics for Numeric Columns ===")
numeric_cols = ['char_count', 'word_count', 'line_count', 'uppercase_ratio', 'numeric_ratio', 'short_token_ratio', 'unique_word_ratio', 'code_like_line_ratio']
print(df[numeric_cols].describe())

print("\n=== Counts for Boolean Flags ===")
flags = ['is_low_content_page', 'is_likely_distribution_page', 'is_likely_cover_page']
for flag in flags:
    print(f"{flag}: {df[flag].sum()} True, {len(df) - df[flag].sum()} False")

print("\n=== Example Pages ===")

def print_examples(title, examples_df):
    print(f"\n{title}:")
    for idx, row in examples_df.iterrows():
        print(f"File ID: {row['file_id']}, Page: {row['page_number']}")
        print(f"Word count: {row['word_count']}, Uppercase ratio: {row['uppercase_ratio']:.3f}, Code-like line ratio: {row['code_like_line_ratio']:.3f}")
        print(f"Content (first 500 chars): {str(row['content'])[:500]}")
        print("---")

# Low content pages
low_content = df[df['is_low_content_page']].head(5)
print_examples("Low Content Pages", low_content)

# Likely distribution pages
dist_pages = df[df['is_likely_distribution_page']].head(5)
print_examples("Likely Distribution Pages", dist_pages)

# Likely cover pages
cover_pages = df[df['is_likely_cover_page']].head(5)
print_examples("Likely Cover Pages", cover_pages)

# Highest code_like_line_ratio
high_code = df.sort_values('code_like_line_ratio', ascending=False).head(5)
print_examples("Pages with Highest Code-like Line Ratio", high_code)

# Highest uppercase_ratio
high_upper = df.sort_values('uppercase_ratio', ascending=False).head(5)
print_examples("Pages with Highest Uppercase Ratio", high_upper)

# Random pages not flagged
not_flagged = df[~(df['is_low_content_page'] | df['is_likely_distribution_page'] | df['is_likely_cover_page'])].sample(5, random_state=42)
print_examples("Random Non-Flagged Pages", not_flagged)
