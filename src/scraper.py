import os
import requests
import pandas as pd
import time

# Reddit OAuth2 application-only auth.
# Set env vars: REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET
# (or pass them directly to collect_brand_data)
REDDIT_USER_AGENT = "EgyptBrandSentiment/1.0"

_token: str | None = None
_token_expiry: float = 0.0


def _get_oauth_token(client_id: str, client_secret: str) -> str | None:
    global _token, _token_expiry
    now = time.time()
    if _token and now < _token_expiry - 60:
        return _token
    try:
        resp = requests.post(
            "https://www.reddit.com/api/v1/access_token",
            auth=(client_id, client_secret),
            data={"grant_type": "client_credentials"},
            headers={"User-Agent": REDDIT_USER_AGENT},
            timeout=15,
        )
        if resp.status_code == 200:
            data = resp.json()
            _token = data.get("access_token")
            _token_expiry = now + data.get("expires_in", 3600)
            return _token
    except Exception:
        pass
    return None


def _auth_headers(token: str | None) -> dict:
    if token:
        return {"Authorization": f"bearer {token}", "User-Agent": REDDIT_USER_AGENT}
    return {"User-Agent": REDDIT_USER_AGENT}


def _reddit_url(url: str, token: str | None) -> str:
    if token and url.startswith("https://www.reddit.com"):
        return url.replace("https://www.reddit.com", "https://oauth.reddit.com", 1)
    return url


def _parse_post(post_data: dict, query: str) -> dict:
    return {
        "record_type": "post",
        "post_id": post_data.get("id", ""),
        "query": query,
        "title": post_data.get("title", ""),
        "text": post_data.get("selftext", ""),
        "subreddit": post_data.get("subreddit", ""),
        "score": post_data.get("score", 0),
        "num_comments": post_data.get("num_comments", 0),
        "permalink": post_data.get("permalink", ""),
        "created_utc": post_data.get("created_utc", ""),
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


def get_posts(query: str, limit: int = 20, token: str | None = None):
    encoded_query = requests.utils.quote(query)
    url = _reddit_url(
        f"https://www.reddit.com/search.json?q={encoded_query}&limit={limit}&sort=relevance",
        token
    )
    response = requests.get(url, headers=_auth_headers(token), timeout=30)

    if response.status_code != 200:
        print(f"[scraper] get_posts returned {response.status_code} for query: {query}")
        return []

    data = response.json()
    posts = data["data"]["children"]

    results = []

    for post in posts:
        results.append(_parse_post(post["data"], query))

    return results


def get_comments(permalink: str, query: str, token: str | None = None):
    if not permalink:
        return []

    url = _reddit_url(f"https://www.reddit.com{permalink}.json", token)
    response = requests.get(url, headers=_auth_headers(token), timeout=30)

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


def collect_brand_data(brand: str, limit_per_query: int = 20,
                       client_id: str | None = None,
                       client_secret: str | None = None):
    client_id = client_id or os.environ.get("REDDIT_CLIENT_ID", "")
    client_secret = client_secret or os.environ.get("REDDIT_CLIENT_SECRET", "")
    token = _get_oauth_token(client_id, client_secret) if (client_id and client_secret) else None
    if not token:
        print("[scraper] No OAuth token — requests may return 403. "
              "Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET env vars.")

    def _run_search(egypt_only: bool):
        queries = build_queries(brand)
        all_rows = []

        print("Queries that will be used:")
        for q in queries:
            print("-", q)

        for query in queries:
            print(f"\nCollecting posts for query: {query}")
            posts = get_posts(query=query, limit=limit_per_query, token=token)

            for post in posts:
                post["brand"] = brand
                all_rows.append(post)

                comments = get_comments(post["permalink"], query=query, token=token)
                for comment in comments:
                    comment["brand"] = brand
                    all_rows.append(comment)

                time.sleep(1)

        df = pd.DataFrame(all_rows)

        if not df.empty:
            df = df.drop_duplicates(subset=["record_type", "post_id", "text"])
            df["combined_text"] = df["title"].fillna("") + " " + df["text"].fillna("")
            if egypt_only:
                df["egypt_relevant"] = df["combined_text"].apply(
                    lambda x: is_egypt_relevant(x, brand)
                )
                df = df[df["egypt_relevant"] == True].copy()

        return df

    egypt_df = _run_search(egypt_only=True)
    if not egypt_df.empty:
        return egypt_df

    print("[scraper] No Egypt-specific results found. Expanding to global Reddit search.")
    fallback_df = _run_search(egypt_only=False)
    if not fallback_df.empty and "egypt_relevant" in fallback_df.columns:
        fallback_df = fallback_df.drop(columns=["egypt_relevant"])
    return fallback_df


if __name__ == "__main__":
    brand = "Vodafone"
    df = collect_brand_data(brand=brand, limit_per_query=10)

    print("\nFinal shape:", df.shape)
    print(df.head())

    df.to_csv("data/raw/reddit_raw.csv", index=False, encoding="utf-8-sig")
    print("\nSaved to data/raw/reddit_raw.csv")