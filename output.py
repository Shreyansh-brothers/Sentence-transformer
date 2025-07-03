# # Example: Aggregate sentiment
# import pandas as pd
#
# df = pd.read_csv(r"D:\Python_files\fully_merged.csv")
# sentiment_score = df.groupby("company")["label"].value_counts(normalize=True).unstack().fillna(0)
#
# # Add a custom score
# sentiment_score["score"] = (
#     sentiment_score.get("positive", 0) - sentiment_score.get("negative", 0)
# )
# print(sentiment_score.sort_values("score", ascending=False))

# import pandas as pd
#
# # Load the dataset
# df = pd.read_csv(r"D:\Python_files\fully_merged.csv")
#
# # Clean columns
# df.columns = df.columns.str.strip().str.lower()
#
# # Group by company and calculate label proportions
# sentiment_df = (
#     df.groupby('company')['label']
#     .value_counts(normalize=True)
#     .unstack(fill_value=0)
# )
#
# # Add sentiment score: positive - negative
# sentiment_df['score'] = sentiment_df.get('positive', 0) - sentiment_df.get('negative', 0)
#
# # Define suggested action
# def suggest_action(score):
#     if score > 0.5:
#         return "Buy"
#     elif score < 0:
#         return "Sell"
#     else:
#         return "Hold"
#
# sentiment_df['action'] = sentiment_df['score'].apply(suggest_action)
#
# # Reset index for saving
# sentiment_df = sentiment_df.reset_index()
#
# # Save final CSV
# sentiment_df.to_csv("company_sentiment_scores.csv", index=False)
# print("✅ Saved: company_sentiment_scores.csv")


# import pandas as pd
#
# # Load sentiment scores (output from previous step)
# df = pd.read_csv(r"D:\Python_files\company_sentiment_scores.csv")
#
# # Assign risk level
# def assign_risk(score):
#     if score >= 0.6:
#         return 'low'
#     elif score >= 0.3:
#         return 'medium'
#     else:
#         return 'high'
#
# df['risk'] = df['score'].apply(assign_risk)
#
# # User inputs
# user_budget = 10000
# user_risk_appetite = 'medium'  # can be 'low', 'medium', or 'high'
# est_stock_price = 500
#
# # Filter based on risk
# def filter_by_risk(df, user_risk):
#     if user_risk == 'low':
#         return df[df['risk'] == 'low']
#     elif user_risk == 'medium':
#         return df[df['risk'].isin(['low', 'medium'])]
#     else:
#         return df
#
# filtered = filter_by_risk(df[df['action'] == 'Buy'], user_risk_appetite)
#
# # Budget-based allocation
# n = min(len(filtered), user_budget // est_stock_price)
# recommended = filtered.nlargest(n, 'score').copy()
# recommended['allocated_amount'] = user_budget // n if n else 0
#
# # Export
# recommended.to_csv("final_recommendations.csv", index=False)
# print("✅ Stock recommendations saved to: final_recommendations.csv")


import pandas as pd
import yfinance as yf
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier

# Load your labeled CSV
df = pd.read_csv(r"D:\Python_files\fully_merged.csv")
# Example columns: company, article, label

# 1. Load trained SBERT + RF model
model = SentenceTransformer('all-MiniLM-L6-v2')
clf = RandomForestClassifier()  # Already trained with your pipeline

# 2. Predict sentiment embedding
embeddings = model.encode(df['article'].tolist(), show_progress_bar=True)
df['predicted_label'] = clf.predict(embeddings)

# 3. Fetch live stock price
def fetch_price(ticker):
    try:
        info = yf.Ticker(ticker + ".NS")
        return info.info.get('regularMarketPrice')
    except Exception:
        return None

df['price'] = df['company'].apply(fetch_price)

# 4. Aggregate sentiment per company
agg = df.groupby('company')['predicted_label'].value_counts(normalize=True).unstack(fill_value=0)
agg['score'] = agg.get('positive',0) - agg.get('negative',0)

# 5. Merge price
prices = df[['company', 'price']].drop_duplicates().set_index('company')
result = agg.join(prices)

print(result.reset_index())
