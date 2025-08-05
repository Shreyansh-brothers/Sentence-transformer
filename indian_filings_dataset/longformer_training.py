import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm

# Enable tqdm for pandas
tqdm.pandas()

# Load the dataset
df = pd.read_csv(r'D:\Python_files\indian_filings_dataset\comprehensive_realistic_mda_dataset.csv')

# Load FinBERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")

# Calculate token lengths
def get_token_length(text):
    return len(tokenizer.tokenize(str(text)))

print("📊 Calculating token lengths for comprehensive MDA dataset...")

# Use progress_apply after enabling tqdm for pandas
df["token_length"] = df["text"].progress_apply(get_token_length)

# Statistics
print(f"\n🔢 Token Length Statistics:")
print(df["token_length"].describe())

# Distribution by sentiment
print(f"\n📊 Token Length by Sentiment:")
for sentiment in df['sentiment_label'].unique():
    subset = df[df['sentiment_label'] == sentiment]
    print(f"  {sentiment}: Mean={subset['token_length'].mean():.0f}, "
          f"Range={subset['token_length'].min()}-{subset['token_length'].max()}")

# Chunking requirements
print(f"\n📦 Chunking Analysis:")
print(f"  • Entries ≤ 512 tokens: {(df['token_length'] <= 512).sum()}")
print(f"  • Entries needing 2 chunks: {((df['token_length'] > 512) & (df['token_length'] <= 1024)).sum()}")
print(f"  • Entries needing 3+ chunks: {(df['token_length'] > 1024).sum()}")

# Show longest entries
print(f"\n📏 Top 5 longest entries by tokens:")
longest = df.nlargest(5, 'token_length')[['sentiment_label', 'token_length']]
print(longest)
