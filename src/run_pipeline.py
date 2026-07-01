import re
import time
import joblib
import requests
import pandas as pd

try:
    from recommendation import load_processed_company_profiles
except ImportError:
    from src.recommendation import load_processed_company_profiles


HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}


def build_queries(brand: str):
    brand = brand.strip()

    queries = [
        brand,
        f"{brand} egypt",
        f"{brand} egypt network",
        f"{brand} egypt internet",
        f"{brand} مصر",
    ]

    arabic_map = {
        "vodafone": ["فودافون", "فودافون مصر", "فودافون نت", "فودافون شبكة"],
        "orange": ["اورنج", "اورنج مصر", "اورنج نت", "اورنج شبكة"],
        "etisalat": ["اتصالات", "اتصالات مصر", "اتصالات نت", "اتصالات شبكة"],
        "we": ["وي", "we egypt", "te data", "tedata", "وي مصر"],
    }

    brand_lower = brand.lower()

    for key, values in arabic_map.items():
        if key in brand_lower:
            queries.extend(values)

    seen = set()
    final_queries = []
    for q in queries:
        q_clean = q.strip().lower()
        if q_clean not in seen:
            seen.add(q_clean)
            final_queries.append(q)

    return final_queries


def is_egypt_relevant(text: str, brand: str):
    text = str(text).lower()
    brand = brand.lower()

    egypt_keywords = [
        "egypt", "مصر", "cairo", "alexandria",
        "القاهرة", "الاسكندرية", "egp", "جنيه"
    ]

    brand_keywords = {
        "vodafone": ["vodafone egypt", "vodafone", "فودافون", "فودافون مصر"],
        "orange": ["orange egypt", "orange", "اورنج", "اورنج مصر"],
        "etisalat": ["etisalat egypt", "etisalat", "اتصالات", "اتصالات مصر"],
        "we": ["we egypt", "te data", "tedata", "وي", "وي مصر"]
    }

    selected_brand_keywords = []
    for key, vals in brand_keywords.items():
        if key in brand:
            selected_brand_keywords = vals
            break

    if selected_brand_keywords:
        has_brand = any(k.lower() in text for k in selected_brand_keywords)
    else:
        has_brand = brand in text

    has_egypt = any(k.lower() in text for k in egypt_keywords)

    return has_brand and has_egypt


def get_posts(query: str, limit: int = 20):
    encoded_query = requests.utils.quote(query)
    url = f"https://www.reddit.com/search.json?q={encoded_query}&limit={limit}&sort=relevance"
    response = requests.get(url, headers=HEADERS, timeout=30)

    if response.status_code != 200:
        return []

    data = response.json()
    posts = data["data"]["children"]

    results = []

    for post in posts:
        post_data = post["data"]

        results.append({
            "record_type": "post",
            "post_id": post_data.get("id", ""),
            "query": query,
            "title": post_data.get("title", ""),
            "text": post_data.get("selftext", ""),
            "subreddit": post_data.get("subreddit", ""),
            "score": post_data.get("score", 0),
            "num_comments": post_data.get("num_comments", 0),
            "permalink": post_data.get("permalink", ""),
            "created_utc": post_data.get("created_utc", "")
        })

    return results


def get_comments(permalink: str, query: str):
    if not permalink:
        return []

    url = f"https://www.reddit.com{permalink}.json"
    response = requests.get(url, headers=HEADERS, timeout=30)

    if response.status_code != 200:
        return []

    comments = []

    try:
        data = response.json()
        comment_list = data[1]["data"]["children"]

        for c in comment_list:
            if c.get("kind") != "t1":
                continue

            comment_data = c["data"]

            comments.append({
                "record_type": "comment",
                "post_id": comment_data.get("link_id", ""),
                "query": query,
                "title": "",
                "text": comment_data.get("body", ""),
                "subreddit": comment_data.get("subreddit", ""),
                "score": comment_data.get("score", 0),
                "num_comments": 0,
                "permalink": permalink,
                "created_utc": comment_data.get("created_utc", "")
            })

    except Exception:
        pass

    return comments


def collect_brand_data(brand: str, limit_per_query: int = 10):
    def _run_search(egypt_only: bool):
        queries = build_queries(brand)
        all_rows = []

        print("\nQueries:")
        for q in queries:
            print("-", q)

        for query in queries:
            print(f"\nCollecting: {query}")
            posts = get_posts(query=query, limit=limit_per_query)

            for post in posts:
                post["brand"] = brand
                all_rows.append(post)

                comments = get_comments(post["permalink"], query=query)
                for comment in comments:
                    comment["brand"] = brand
                    all_rows.append(comment)

                time.sleep(1)

        df = pd.DataFrame(all_rows)

        if not df.empty:
            df = df.drop_duplicates(subset=["record_type", "post_id", "text"])
            df["combined_text"] = df["title"].fillna("") + " " + df["text"].fillna("")
            if egypt_only:
                df["egypt_relevant"] = df["combined_text"].apply(lambda x: is_egypt_relevant(x, brand))
                df = df[df["egypt_relevant"] == True].copy()

        return df

    egypt_df = _run_search(egypt_only=True)
    if not egypt_df.empty:
        return egypt_df

    print("No Egypt-specific results found. Expanding to global Reddit search.")
    fallback_df = _run_search(egypt_only=False)
    if not fallback_df.empty and "egypt_relevant" in fallback_df.columns:
        fallback_df = fallback_df.drop(columns=["egypt_relevant"])
    return fallback_df


def clean_text(text):
    text = str(text).lower()
    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)
    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)
    # Remove Arabic diacritics (Tashkeel)
    text = re.sub(r"[\u064B-\u0652]", "", text)
    # Remove punctuation, symbols, emojis, and numbers (keep only English letters, Arabic letters, and spaces)
    text = re.sub(r"[^a-z\u0621-\u064A\s]", "", text)
    # Normalize whitespaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_data(df: pd.DataFrame, vectorizer=None) -> pd.DataFrame:
    """Prepare raw text data for the model by cleaning, tokenizing, and lemmatizing it."""
    vocab_set = set(vectorizer.vocabulary_.keys()) if vectorizer is not None else set()

    def preprocess_text(text):
        cleaned = clean_text(text)
        # Tokenization step
        tokens = cleaned.split()
        
        # Lemmatization & Stemming step
        lemmatized_tokens = []
        for t in tokens:
            stemmed = t
            # Simple English suffix stemming
            if len(t) > 4:
                if t.endswith("sses"):
                    stemmed = t[:-2]
                elif t.endswith("ies"):
                    stemmed = t[:-3] + "y"
                elif t.endswith("s") and not t.endswith("us") and not t.endswith("is") and not t.endswith("as"):
                    stemmed = t[:-1]
            # Simple Arabic prefix stemming
            elif len(t) > 4 and t.startswith("ال"):
                stemmed = t[2:]
            
            # Fallback check
            if stemmed in vocab_set:
                lemmatized_tokens.append(stemmed)
            else:
                lemmatized_tokens.append(t)
                
        return " ".join(lemmatized_tokens)

    df["clean_text"] = df["combined_text"].fillna("").apply(preprocess_text)
    df = df[df["clean_text"].str.strip() != ""].copy()
    return df


def predict_sentiment(df: pd.DataFrame):
    model = joblib.load("models/svm_model.pkl")
    vectorizer = joblib.load("models/vectorizer.pkl")

    # Run the preprocessing step
    df = preprocess_data(df, vectorizer)

    X = vectorizer.transform(df["clean_text"])
    df["predicted_sentiment"] = model.predict(X)

    return df


def summarize_results(df: pd.DataFrame):
    counts = df["predicted_sentiment"].value_counts()
    percentages = (df["predicted_sentiment"].value_counts(normalize=True) * 100).round(2)

    summary_df = pd.DataFrame({
        "Count": counts,
        "Percentage": percentages
    })

    return summary_df


def main():
    brand = input("Enter brand name: ").strip()

    if not brand:
        print("Brand name cannot be empty.")
        return

    df = collect_brand_data(brand=brand, limit_per_query=10)

    if df.empty:
        print("No Egypt-relevant Reddit data found for this brand.")
        return

    raw_path = f"data/raw/{brand.lower().replace(' ', '_')}_raw.csv"
    df.to_csv(raw_path, index=False, encoding="utf-8-sig")

    df = predict_sentiment(df)

    processed_path = f"data/processed/{brand.lower().replace(' ', '_')}_predicted.csv"
    df.to_csv(processed_path, index=False, encoding="utf-8-sig")

    summary_df = summarize_results(df)

    print("\nFinal shape:", df.shape)
    print("\nSentiment Summary:")
    print(summary_df)

    print("\nSample Results:")
    print(df[["record_type", "query", "combined_text", "predicted_sentiment"]].head(10))

    print(f"\nSaved raw data to: {raw_path}")
    print(f"Saved predicted data to: {processed_path}")

    ranking_df = load_processed_company_profiles()
    if not ranking_df.empty:
        best_company = ranking_df.iloc[0]
        print("\nRecommendation System:")
        print(
            f"Best company: {best_company['brand']} "
            f"({best_company['recommendation_score']:.1f}/100)"
        )
        print("\nCompany Ranking:")
        print(ranking_df[[
            "rank", "brand", "recommendation_score",
            "positive_pct", "negative_pct", "total_records", "reason"
        ]].to_string(index=False))


if __name__ == "__main__":
    main()
