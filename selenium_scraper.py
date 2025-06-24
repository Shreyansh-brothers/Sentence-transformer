# import feedparser
# import pandas as pd
# from urllib.parse import quote_plus
# from datetime import datetime
# import os
#
# def scrape_google_news(company_name, max_articles=50):
#     search_query = quote_plus(f"{company_name} stock")
#     rss_url = f"https://news.google.com/rss/search?q={search_query}&hl=en-IN&gl=IN&ceid=IN:en"
#
#     print(f"🔍 Scraping: {company_name}")
#     feed = feedparser.parse(rss_url)
#     headlines = []
#
#     for entry in feed.entries[:max_articles]:
#         headlines.append({
#             "headline": entry.title,
#             "link": entry.link,
#             "timestamp": entry.published if 'published' in entry else datetime.now().isoformat(),
#             "company": company_name,
#             "label": ""  # For manual labeling
#         })
#
#     return headlines
#
# if __name__ == "__main__":
#     companies = [
#         "Infosys",
#         "Tata Motors",
#         "Reliance Industries",
#         "HDFC Bank",
#         "Adani Enterprises"
#     ]
#
#     all_news = []
#     for company in companies:
#         try:
#             all_news.extend(scrape_google_news(company, max_articles=30))
#         except Exception as e:
#             print(f"⚠️ Failed to scrape {company}: {e}")
#
#     df = pd.DataFrame(all_news)
#
#     os.makedirs("data", exist_ok=True)
#     output_path = "data/google_news_by_company.csv"
#     df.to_csv(output_path, index=False, encoding="utf-8")
#     print(f"\n✅ Saved {len(df)} records to {output_path}")


import feedparser
import pandas as pd
from urllib.parse import quote_plus
from datetime import datetime

def scrape_google_news(company_name, max_articles=30):
    search_query = quote_plus(f"{company_name} stock")
    rss_url = f"https://news.google.com/rss/search?q={search_query}&hl=en-IN&gl=IN&ceid=IN:en"
    feed = feedparser.parse(rss_url)
    headlines = []

    for entry in feed.entries[:max_articles]:
        headlines.append({
            "headline": entry.title,
            "link": entry.link,
            "timestamp": entry.published if 'published' in entry else datetime.now().isoformat(),
            "company": company_name,
            "label": ""  # for manual labeling
        })

    return headlines

# Company list
companies = ["Infosys", "Tata Motors", "Reliance Industries", "HDFC Bank", "Adani Enterprises"]

# Gather headlines
all_data = []
for company in companies:
    all_data.extend(scrape_google_news(company, 30))

# Save to Excel
df = pd.DataFrame(all_data)
df.to_excel("google_news_by_company.xlsx", index=False)
print("✅ Saved to google_news_by_company.xlsx")


