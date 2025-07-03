# import pandas as pd
# from sentence_transformers import SentenceTransformer
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import classification_report, confusion_matrix
# from imblearn.over_sampling import SMOTE
#
# # Load dataset
# df = pd.read_csv(r"D:\Python_files\fully_merged.csv")  # Use your latest merged file
# df = df.dropna(subset=['article', 'label'])
# df = df[df['label'].isin(['positive', 'neutral', 'negative'])]
#
# # SBERT Embedding
# model = SentenceTransformer('all-MiniLM-L6-v2')  # Or 'paraphrase-MiniLM-L12-v2' (slower but better)
# embeddings = model.encode(df['article'].tolist(), show_progress_bar=True)
#
# # Prepare labels
# y = df['label'].values
#
# # Oversample to balance
# sm = SMOTE(random_state=42)
# X_resampled, y_resampled = sm.fit_resample(embeddings, y)
#
# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(
#     X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42
# )
#
# # Choose classifier:
# clf = RandomForestClassifier(n_estimators=100, random_state=42)
# # clf = LogisticRegression(max_iter=1000, class_weight='balanced')  # Uncomment to try Logistic Regression
#
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
#
# # Results
# print("\n✅ SBERT + Classifier Results")
# print(classification_report(y_test, y_pred, zero_division=0))
#
# print("\n🔍 Confusion Matrix:")
# print(confusion_matrix(y_test, y_pred))


# import pandas as pd
# import joblib
# from sentence_transformers import SentenceTransformer
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report, confusion_matrix
# from imblearn.over_sampling import SMOTE
# from sklearn.pipeline import Pipeline
# from sklearn.base import BaseEstimator, TransformerMixin
#
# # === 1. Custom Transformer to wrap SBERT ===
# class SBERTTransformer(BaseEstimator, TransformerMixin):
#     def __init__(self, model_name='all-MiniLM-L6-v2'):
#         self.model_name = model_name
#         self.model = SentenceTransformer(model_name)
#
#     def fit(self, X, y=None):
#         return self
#
#     def transform(self, X):
#         return self.model.encode(X.tolist(), show_progress_bar=True)
#
# # === 2. Load dataset ===
# df = pd.read_csv(r"D:\Python_files\fully_merged.csv")
# df = df.dropna(subset=['article', 'label'])
# df = df[df['label'].isin(['positive', 'neutral', 'negative'])]
#
# X = df['article']
# y = df['label'].values
#
# # === 3. Get SBERT embeddings ===
# sbert = SBERTTransformer()
# embeddings = sbert.transform(X)
#
# # === 4. Oversample ===
# sm = SMOTE(random_state=42)
# X_resampled, y_resampled = sm.fit_resample(embeddings, y)
#
# # === 5. Train-test split ===
# X_train, X_test, y_train, y_test = train_test_split(
#     X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42
# )
#
# # === 6. Train classifier ===
# clf = RandomForestClassifier(n_estimators=100, random_state=42)
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
#
# # === 7. Save full pipeline ===
# full_pipeline = Pipeline([
#     ('sbert', sbert),
#     ('clf', clf)
# ])
# joblib.dump(full_pipeline, "sbert_rf_pipeline.pkl")
# print("✅ Full SBERT + RandomForest pipeline saved as 'sbert_rf_pipeline.pkl'")
#
# # === 8. Evaluation ===
# print("\n✅ SBERT + Classifier Results")
# print(classification_report(y_test, y_pred, zero_division=0))
# print("\n🔍 Confusion Matrix:")
# print(confusion_matrix(y_test, y_pred))

# 1. Define SBERTTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sentence_transformers import SentenceTransformer

class SBERTTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.model.encode(X, show_progress_bar=True)

# 2. Imports
import pandas as pd
import joblib
import yfinance as yf

# 3. Load dataset
df = pd.read_csv(r"D:\Python_files\fully_merged.csv")
df = df.dropna(subset=["article", "company"])
df["article"] = df["article"].astype(str)

# 4. Load model pipeline
pipeline = joblib.load(r"D:\Python_files\models\sbert_rf_pipeline.pkl")
df["predicted_label"] = pipeline.predict(df["article"].tolist())

# 5. Define company-to-ticker map
ticker_map = {
    "Adani Green": "ADANIGREEN",
    "Apollo Hospitals": "APOLLOHOSP",
    "Ashok Leyland": "ASHOKLEY",
    "Ashoka Buildcon": "ASHOKA",
    "Avenue Supermarts (DMart)": "DMART",
    "BHEL": "BHEL",
    "Bajaj Auto": "BAJAJ-AUTO",
    "Bharat Forge": "BHARATFORG",
    "Bikaji Foods": "BIKAJI",
    "Cipla": "CIPLA",
    "Dabur India": "DABUR",
    "Delhivery": "DELHIVERY",
    "Dilip Buildcon": "DBL",
    "Divi’s Laboratories": "DIVISLAB",
    "Dr. Reddy’s Labs": "DRREDDY",
    "Eicher Motors": "EICHERMOT",
    "Godrej Consumer": "GODREJCP",
    "HDFC": "HDFC.BO",
    "Havells India": "HAVELLS",
    "ICICI": "ICICIBANK",
    "IDFC First Bank": "IDFCFIRSTB",
    "IRB Infrastructure": "IRB",
    "IRCTC": "IRCTC",
    "ITC Ltd.": "ITC",
    "Infosys": "INFY",
    "JSW Steel": "JSWSTEEL",
    "Kotak Mahindra Bank": "KOTAKBANK",
    "Mahindra & Mahindra": "M&M",
    "Marico": "MARICO",
    "Maruti Suzuki": "MARUTI",
    "NTPC": "NTPC",
    "Nazara Technologies": "NAZARA",
    "Nestle India": "NESTLEIND",
    "Nykaa": "NYKAA.NS",
    "ONGC": "ONGC",
    "Oberoi Realty": "OBEROIRLTY",
    "PNB Bank": "PNB",
    "PNC Infratech": "PNCINFRA",
    "Paytm (One97)": "PAYTM",
    "SBI": "SBIN",
    "Siemens India": "SIEMENS",
    "Sun Pharma": "SUNPHARMA",
    "TCS": "TCS",
    "Tata Power": "TATAPOWER",
    "Tata motors": "TATAMOTORS",
    "Tech Mahindra": "TECHM",
    "Zomato": "ZOMATO.NS"
}

# 6. Fetch stock prices
def fetch_price(ticker):
    try:
        return yf.Ticker(ticker + ".NS").info.get("regularMarketPrice", None)
    except:
        return None

df["ticker"] = df["company"].map(ticker_map)
df["price"] = df["ticker"].apply(lambda t: fetch_price(t) if pd.notna(t) else None)

# 7. Aggregate sentiment
agg = df.groupby("company")["predicted_label"].value_counts(normalize=True).unstack(fill_value=0)
agg["score"] = agg.get("positive", 0) - agg.get("negative", 0)

# 8. Merge with price
prices = df[["company", "price"]].drop_duplicates().set_index("company")
result = agg.join(prices)

# 9. Output
print(result.reset_index())
