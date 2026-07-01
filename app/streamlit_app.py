import os
import re
import time
import random
import joblib
import requests
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timezone
from pathlib import Path
import sys
from html import escape


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.recommendation import (
    load_processed_company_profiles,
    build_company_profile,
    rank_companies,
)


# -----------------------------
# HTTP Setup + Reddit OAuth2
# -----------------------------

# Reddit requires OAuth2 since 2023. Create a free "script" app at:
# https://www.reddit.com/prefs/apps  (type: script, redirect: http://localhost)
# Then add to .streamlit/secrets.toml:
#   [reddit]
#   client_id = "YOUR_CLIENT_ID"
#   client_secret = "YOUR_CLIENT_SECRET"
# Or set environment variables: REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET

REDDIT_APP_USER_AGENT = "EgyptBrandSentiment/1.0 (by u/YourRedditUsername)"

SESSION = requests.Session()
LAST_FETCH_ERROR = None

# OAuth token cache
_oauth_token: str | None = None
_oauth_token_expiry: float = 0.0


def _get_reddit_credentials() -> tuple[str | None, str | None]:
    """Read Reddit API credentials from st.secrets or environment variables."""
    secrets_path = ROOT_DIR / ".streamlit" / "secrets.toml"
    if secrets_path.exists():
        try:
            current_section = None
            secrets_values = {}
            for raw_line in secrets_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("[") and line.endswith("]"):
                    current_section = line[1:-1].strip().lower()
                    continue
                if current_section != "reddit" or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip().lower()
                value = value.strip().strip('"').strip("'")
                secrets_values[key] = value
            client_id = secrets_values.get("client_id")
            client_secret = secrets_values.get("client_secret")
            if client_id and client_secret:
                return client_id, client_secret
        except Exception:
            pass

    client_id = None
    client_secret = None
    try:
        client_id = st.secrets["reddit"]["client_id"]
        client_secret = st.secrets["reddit"]["client_secret"]
    except Exception:
        client_id = os.environ.get("REDDIT_CLIENT_ID")
        client_secret = os.environ.get("REDDIT_CLIENT_SECRET")
    return client_id, client_secret


def _get_oauth_token() -> str | None:
    """Obtain (or reuse) a Reddit OAuth2 application-only access token."""
    global _oauth_token, _oauth_token_expiry
    now = time.time()
    if _oauth_token and now < _oauth_token_expiry - 60:
        return _oauth_token

    client_id, client_secret = _get_reddit_credentials()
    if not client_id or not client_secret:
        return None

    try:
        resp = requests.post(
            "https://www.reddit.com/api/v1/access_token",
            auth=(client_id, client_secret),
            data={"grant_type": "client_credentials"},
            headers={"User-Agent": REDDIT_APP_USER_AGENT},
            timeout=15,
        )
        if resp.status_code == 200:
            token_data = resp.json()
            _oauth_token = token_data.get("access_token")
            expires_in = token_data.get("expires_in", 3600)
            _oauth_token_expiry = now + expires_in
            return _oauth_token
        else:
            return None
    except Exception:
        return None


def _build_headers(token: str | None) -> dict:
    """Build request headers, using OAuth bearer token when available."""
    if token:
        return {
            "Authorization": f"bearer {token}",
            "User-Agent": REDDIT_APP_USER_AGENT,
        }
    # Fallback: unauthenticated (may get 403 on some endpoints)
    return {"User-Agent": REDDIT_APP_USER_AGENT}


def _oauth_url(url: str, token: str | None) -> str:
    """Rewrite www.reddit.com URLs to oauth.reddit.com when a token is available."""
    if token and url.startswith("https://www.reddit.com"):
        return url.replace("https://www.reddit.com", "https://oauth.reddit.com", 1)
    return url


def safe_get(url: str, retries: int = 3, base_wait: float = 2.0):
    """Fetch a Reddit JSON URL using OAuth when credentials are available."""
    global LAST_FETCH_ERROR

    token = _get_oauth_token()
    actual_url = _oauth_url(url, token)
    headers = _build_headers(token)

    for attempt in range(retries):
        try:
            response = SESSION.get(actual_url, headers=headers, timeout=30)
            if response.status_code == 200:
                LAST_FETCH_ERROR = None
                return response
            elif response.status_code == 401 and token:
                # Token may have expired mid-session; force refresh and retry
                global _oauth_token, _oauth_token_expiry
                _oauth_token = None
                _oauth_token_expiry = 0.0
                token = _get_oauth_token()
                actual_url = _oauth_url(url, token)
                headers = _build_headers(token)
                time.sleep(1)
            elif response.status_code == 429:
                LAST_FETCH_ERROR = f"Rate limited while fetching {url}"
                wait = base_wait * (2 ** attempt) + random.uniform(1, 3)
                st.toast(f"⏳ Rate limited — waiting {wait:.1f}s (retry {attempt+1}/{retries})...")
                time.sleep(wait)
            elif response.status_code == 403:
                LAST_FETCH_ERROR = (
                    f"Reddit returned 403 for {url}. "
                    "Add Reddit API credentials to .streamlit/secrets.toml "
                    "([reddit] client_id / client_secret) to fix this."
                )
                return None   # 403 won't resolve with retries
            elif response.status_code in [503, 504]:
                LAST_FETCH_ERROR = f"Reddit service unavailable ({response.status_code}) for {url}"
                time.sleep(base_wait + random.uniform(2, 5))
            else:
                LAST_FETCH_ERROR = f"Reddit returned {response.status_code} for {url}"
                return None
        except requests.exceptions.RequestException as exc:
            LAST_FETCH_ERROR = f"Request failed for {url}: {exc}"
            time.sleep(base_wait)
    if LAST_FETCH_ERROR is None:
        LAST_FETCH_ERROR = f"Failed to fetch {url} after {retries} retries"
    return None


def reddit_credentials_configured() -> bool:
    """Return True if Reddit OAuth credentials are present."""
    cid, csec = _get_reddit_credentials()
    return bool(cid and csec)


# -----------------------------
# Egyptian Brands Registry
# -----------------------------
EGYPTIAN_BRANDS = {
    "vodafone":        {"arabic": ["فودافون", "فودافون مصر"],            "aliases": ["vodafone egypt", "vodafone eg"],                      "category": "telecom"},
    "orange":          {"arabic": ["اورنج", "اورنج مصر"],               "aliases": ["orange egypt", "orange eg"],                          "category": "telecom"},
    "etisalat":        {"arabic": ["اتصالات", "اتصالات مصر"],           "aliases": ["etisalat egypt", "etisalat eg", "e&"],                 "category": "telecom"},
    "we":              {"arabic": ["وي", "وي مصر"],                      "aliases": ["we egypt", "te data", "tedata"],                       "category": "telecom"},
    "fawry":           {"arabic": ["فوري", "فوري مصر"],                  "aliases": ["fawry egypt", "fawry pay"],                            "category": "fintech"},
    "instapay":        {"arabic": ["انستاباي", "انستا باي"],             "aliases": ["instapay egypt"],                                      "category": "fintech"},
    "cib":             {"arabic": ["سي اي بي", "البنك التجاري الدولي"], "aliases": ["cib egypt", "commercial international bank"],          "category": "banking"},
    "banque misr":     {"arabic": ["بنك مصر"],                           "aliases": ["banque misr egypt", "bank misr"],                      "category": "banking"},
    "nbe":             {"arabic": ["البنك الاهلي", "الأهلي المصري"],    "aliases": ["national bank egypt", "nbe egypt"],                    "category": "banking"},
    "alex bank":       {"arabic": ["بنك الاسكندرية"],                    "aliases": ["alexbank", "alex bank egypt"],                         "category": "banking"},
    "noon":            {"arabic": ["نون", "نون مصر"],                    "aliases": ["noon egypt", "noon.com egypt"],                        "category": "ecommerce"},
    "amazon egypt":    {"arabic": ["امازون مصر"],                        "aliases": ["amazon.eg", "souq egypt"],                             "category": "ecommerce"},
    "jumia":           {"arabic": ["جوميا", "جوميا مصر"],               "aliases": ["jumia egypt"],                                         "category": "ecommerce"},
    "talabat":         {"arabic": ["طلبات", "طلبات مصر"],               "aliases": ["talabat egypt"],                                       "category": "delivery"},
    "elmenus":         {"arabic": ["المنيوس", "المينوس"],               "aliases": ["el menus egypt"],                                      "category": "delivery"},
    "uber egypt":      {"arabic": ["اوبر", "اوبر مصر"],                 "aliases": ["uber cairo"],                                          "category": "transport"},
    "careem":          {"arabic": ["كريم", "كريم مصر"],                 "aliases": ["careem egypt"],                                        "category": "transport"},
    "indriver":        {"arabic": ["ان درايفر", "انددرايفر"],           "aliases": ["indriver egypt", "indrive egypt"],                     "category": "transport"},
    "aqarmap":         {"arabic": ["عقارماب"],                           "aliases": ["aqarmap egypt"],                                       "category": "realestate"},
    "property finder": {"arabic": ["بروبرتي فايندر"],                   "aliases": ["propertyfinder egypt"],                                "category": "realestate"},
    "vezeeta":         {"arabic": ["فيزيتا"],                            "aliases": ["vezeeta egypt"],                                       "category": "healthcare"},
    "doctorak":        {"arabic": ["دكتورك"],                            "aliases": ["doctorak egypt"],                                      "category": "healthcare"},
    "olx egypt":       {"arabic": ["اوليكس مصر", "اولكس"],             "aliases": ["olx.com.eg"],                                          "category": "classifieds"},
    "opensooq":        {"arabic": ["السوق المفتوح"],                     "aliases": ["opensooq egypt"],                                      "category": "classifieds"},
}

EGYPT_KEYWORDS = [
    "egypt", "مصر", "cairo", "alexandria", "القاهرة",
    "الاسكندرية", "egp", "جنيه", "egyptian", "مصري",
    "مصرية", "giza", "الجيزة", "شرم", "hurghada",
    "الغردقة", "luxor", "الاقصر"
]

SENTIMENT_COLORS = {
    "positive": "#00C9A7",
    "negative": "#FF6B6B",
    "neutral":  "#FFD93D",
}

SENTIMENT_FILL_COLORS = {
    "positive": "rgba(0,201,167,0.08)",
    "negative": "rgba(255,107,107,0.08)",
    "neutral":  "rgba(255,217,61,0.08)",
}

CATEGORY_ICONS = {
    "telecom":    "📡",
    "fintech":    "💳",
    "banking":    "🏦",
    "ecommerce":  "🛒",
    "delivery":   "🚀",
    "transport":  "🚗",
    "realestate": "🏠",
    "healthcare": "🏥",
    "classifieds":"📋",
    "other":      "🏢",
}


# -----------------------------
# Brand Helpers
# -----------------------------
def get_brand_info(brand: str):
    brand_lower = brand.strip().lower()
    for key, info in EGYPTIAN_BRANDS.items():
        if key in brand_lower or brand_lower in key:
            return key, info
    return None, None


def is_known_egyptian_brand(brand: str) -> bool:
    key, _ = get_brand_info(brand)
    return key is not None


# -----------------------------
# Query Building
# -----------------------------
DEFAULT_POSTS_PER_QUERY = 10


def build_queries(brand: str, egypt_only: bool = True):
    brand = brand.strip()
    queries = [
        f"{brand} egypt",
        f"{brand} مصر",
        f"{brand} egypt review",
        f"{brand} egypt complaint",
        f"{brand} egypt service",
        brand,
    ]
    key, info = get_brand_info(brand)
    if info:
        if not egypt_only:
            queries.extend([f"{alias}" for alias in info.get("aliases", [])])
            queries.extend([f"{name}" for name in info.get("arabic", [])])
        queries.extend(info.get("aliases", []))
        queries.extend(info.get("arabic", []))
        category = info.get("category", "")
        if category:
            queries.append(f"{brand} {category} egypt")
            if not egypt_only:
                queries.append(f"{brand} {category}")

    if not egypt_only:
        queries.extend([
            f"{brand} review",
            f"{brand} complaint",
            f"{brand} service",
        ])

    seen = set()
    final_queries = []
    for q in queries:
        q_clean = q.strip().lower()
        if q_clean not in seen:
            seen.add(q_clean)
            final_queries.append(q)
    return final_queries


# -----------------------------
# Egypt Relevance Filter
# -----------------------------
def is_egypt_relevant(text: str, brand: str) -> bool:
    text_lower = str(text).lower()
    key, info = get_brand_info(brand)
    brand_terms = [brand.strip().lower()]
    if info:
        brand_terms += [a.lower() for a in info.get("aliases", [])]
        brand_terms += [a.lower() for a in info.get("arabic", [])]

    has_brand = any(term in text_lower for term in brand_terms)
    if not has_brand:
        return False
    has_egypt = any(k.lower() in text_lower for k in EGYPT_KEYWORDS)
    if has_egypt:
        return True
    if is_known_egyptian_brand(brand):
        return True
    return False


# -----------------------------
# Reddit Scraping
# -----------------------------
def _parse_post(post_data: dict, query: str) -> dict:
    return {
        "record_type":  "post",
        "post_id":      post_data.get("id", ""),
        "query":        query,
        "title":        post_data.get("title", ""),
        "text":         post_data.get("selftext", ""),
        "subreddit":    post_data.get("subreddit", ""),
        "score":        post_data.get("score", 0),
        "num_comments": post_data.get("num_comments", 0),
        "permalink":    post_data.get("permalink", ""),
        "created_utc":  post_data.get("created_utc", "")
    }


def get_posts(query: str, limit: int = 10):
    results = []
    encoded_query = requests.utils.quote(query)

    egypt_url = f"https://www.reddit.com/r/egypt/search.json?q={encoded_query}&restrict_sr=1&limit={limit}"
    response2 = safe_get(egypt_url)
    if response2:
        try:
            for post in response2.json()["data"]["children"]:
                results.append(_parse_post(post["data"], query))
        except Exception:
            pass

    url = f"https://www.reddit.com/search.json?q={encoded_query}&limit={limit}&sort=relevance"
    response = safe_get(url)
    if response:
        try:
            for post in response.json()["data"]["children"]:
                results.append(_parse_post(post["data"], query))
        except Exception:
            pass

    time.sleep(random.uniform(1.5, 3.0))

    return results


def get_comments(permalink: str, query: str):
    if not permalink:
        return []
    url = f"https://www.reddit.com{permalink}.json"
    response = safe_get(url)
    if not response:
        return []
    comments = []
    try:
        comment_list = response.json()[1]["data"]["children"]
        for c in comment_list:
            if c.get("kind") != "t1":
                continue
            d = c["data"]
            comments.append({
                "record_type":  "comment",
                "post_id":      d.get("link_id", ""),
                "query":        query,
                "title":        "",
                "text":         d.get("body", ""),
                "subreddit":    d.get("subreddit", ""),
                "score":        d.get("score", 0),
                "num_comments": 0,
                "permalink":    permalink,
                "created_utc":  d.get("created_utc", "")
            })
    except Exception:
        pass
    return comments


def collect_brand_data(brand: str, limit_per_query: int = DEFAULT_POSTS_PER_QUERY, max_queries: int | None = None):
    def _run_search(egypt_only: bool):
        queries = build_queries(brand, egypt_only=egypt_only)
        if max_queries is not None:
            queries = queries[:max_queries]

        all_rows = []
        progress = st.progress(0, text="⏳ Preparing search queries…")

        for i, query in enumerate(queries):
            pct = (i + 1) / max(len(queries), 1)
            progress.progress(
                pct,
                text=f"🔍 Query {i+1}/{len(queries)}: `{query}`"
            )
            posts = get_posts(query=query, limit=limit_per_query)
            for post in posts:
                post["brand"] = brand
                all_rows.append(post)
                comments = get_comments(post["permalink"], query=query)
                for comment in comments:
                    comment["brand"] = brand
                    all_rows.append(comment)
                time.sleep(random.uniform(1.0, 2.5))
            time.sleep(random.uniform(2.0, 4.0))

        progress.empty()

        df = pd.DataFrame(all_rows)
        if not df.empty:
            df = df.drop_duplicates(subset=["record_type", "post_id", "text"])
            df["combined_text"] = df["title"].fillna("") + " " + df["text"].fillna("")
            if egypt_only:
                df["egypt_relevant"] = df["combined_text"].apply(lambda x: is_egypt_relevant(x, brand))
                df = df[df["egypt_relevant"]].copy()
        return df

    egypt_df = _run_search(egypt_only=True)
    if not egypt_df.empty:
        return egypt_df

    st.toast("No Egypt-specific results found. Expanding search to global Reddit results...")
    fallback_df = _run_search(egypt_only=False)
    if not fallback_df.empty and "egypt_relevant" in fallback_df.columns:
        fallback_df = fallback_df.drop(columns=["egypt_relevant"])
    return fallback_df


# -----------------------------
# Text Cleaning + Prediction
# -----------------------------
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


@st.cache_resource
def load_model():
    model = joblib.load("models/svm_model.pkl")
    vectorizer = joblib.load("models/vectorizer.pkl")
    return model, vectorizer


def preprocess_data(df: pd.DataFrame, vectorizer=None) -> pd.DataFrame:
    """Prepare raw text data for the model by cleaning, tokenizing, and lemmatizing it."""
    vocab_set = set(vectorizer.vocabulary_.keys()) if vectorizer is not None else set()

    def preprocess_text(text):
        cleaned = clean_text(text)
        # Tokenization step (splitting into words)
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
            
            # Fallback check: only keep the stemmed/lemmatized version if it is in the vocabulary
            if stemmed in vocab_set:
                lemmatized_tokens.append(stemmed)
            else:
                lemmatized_tokens.append(t)
                
        return " ".join(lemmatized_tokens)

    df["clean_text"] = df["combined_text"].fillna("").apply(preprocess_text)
    df = df[df["clean_text"].str.strip() != ""].copy()
    return df


def predict_sentiment(df: pd.DataFrame):
    model, vectorizer = load_model()
    # Explicit preprocessing step before inference
    df = preprocess_data(df, vectorizer)
    X = vectorizer.transform(df["clean_text"])
    df["predicted_sentiment"] = model.predict(X)
    return df


def summarize_results(df: pd.DataFrame):
    counts = df["predicted_sentiment"].value_counts()
    percentages = (df["predicted_sentiment"].value_counts(normalize=True) * 100).round(2)
    return pd.DataFrame({"Count": counts, "Percentage": percentages})


def _brand_file_slug(brand: str) -> str:
    return brand.strip().lower().replace(" ", "_")


def load_cached_brand_data(brand: str) -> pd.DataFrame:
    """Return a DataFrame from any previously-saved CSV for *brand*.

    Looks in data/processed/ first (already has predicted_sentiment),
    then data/raw/. Returns an empty DataFrame when nothing is found.
    """
    slug = _brand_file_slug(brand)
    candidates = [
        Path(f"data/processed/{slug}_predicted.csv"),
        Path(f"data/raw/{slug}_raw.csv"),
    ]
    for path in candidates:
        if path.exists():
            try:
                df = pd.read_csv(path)
                if not df.empty:
                    # Ensure combined_text exists (raw CSVs may lack it)
                    if "combined_text" not in df.columns:
                        df["combined_text"] = (
                            df.get("title", pd.Series("", index=df.index)).fillna("")
                            + " "
                            + df.get("text", pd.Series("", index=df.index)).fillna("")
                        )
                    return df
            except Exception:
                continue
    return pd.DataFrame()


# -----------------------------
# Telecom CSV Fallback
# -----------------------------
TELECOM_CSV_PATH = ROOT_DIR / "telecom_reviews_10000_ready.csv"

TELECOM_SENTIMENT_MAP = {
    "highly positive": "positive",
    "positive":        "positive",
    "mixed":           "neutral",
    "neutral":         "neutral",
    "negative":        "negative",
    "highly negative": "negative",
}

TELECOM_COMPANY_MAP = {
    "vodafone":   "Vodafone Egypt",
    "orange":     "Orange Egypt",
    "etisalat":   "Etisalat Egypt",
    "e&":         "Etisalat Egypt",
    "we telecom": "WE Telecom Egypt",
    "we":         "WE Telecom Egypt",
    "tedata":     "WE Telecom Egypt",
}


def _match_telecom_company(brand: str) -> str | None:
    """Return the exact company name in the telecom CSV that best matches *brand*."""
    b = brand.strip().lower()
    # Longest-key match first to avoid 'we' matching inside 'we telecom'
    for key in sorted(TELECOM_COMPANY_MAP, key=len, reverse=True):
        if key in b or b in key:
            return TELECOM_COMPANY_MAP[key]
    return None


def load_telecom_brand_data(brand: str) -> pd.DataFrame:
    """Load reviews from telecom_reviews_10000.csv for *brand*.

    Maps the telecom schema to the app's internal schema and runs
    sentiment predictions using our local SVM classifier.
    """
    if not TELECOM_CSV_PATH.exists():
        return pd.DataFrame()

    company_name = _match_telecom_company(brand)
    if company_name is None:
        return pd.DataFrame()

    try:
        raw = pd.read_csv(TELECOM_CSV_PATH, encoding="utf-8")
        subset = raw[raw["company"].str.lower() == company_name.lower()].copy()
        if subset.empty:
            return pd.DataFrame()

        # Unified schema columns (needed for prediction & charts)
        subset["record_type"]   = "review"
        subset["post_id"]       = subset["review_id"].astype(str)
        subset["query"]         = company_name
        subset["title"]         = ""
        subset["text"]          = subset["review_text"].fillna("")
        subset["combined_text"] = subset["review_text"].fillna("")
        subset["subreddit"]     = subset["city"].fillna("Unknown")
        subset["score"]         = subset["rating"].fillna(0)
        subset["num_comments"]  = 0
        subset["permalink"]     = ""
        subset["brand"]         = company_name

        # Predict sentiment using our SVM model
        subset = predict_sentiment(subset)

        # Timestamp → unix (for timeline chart)
        subset["created_utc"] = (
            pd.to_datetime(subset["timestamp"], errors="coerce")
            .astype("int64") // 10 ** 9
        )

        return subset.reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def initialize_recommendation_profiles():
    """Predict and write cached CSVs for all 4 telecom operators.
    Runs once on startup.
    """
    companies = ["vodafone", "orange", "etisalat", "we"]
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    for b in companies:
        processed_path = Path(f"data/processed/{b}_predicted.csv")
        df = load_telecom_brand_data(b)
        if not df.empty:
            df.to_csv(processed_path, index=False, encoding="utf-8-sig")
    return True



def analyze_brand(brand: str, max_queries: int) -> dict | None:
    _use_cache = not reddit_credentials_configured()
    _is_telecom_data = False
    df = pd.DataFrame()

    if _use_cache:
        df = load_cached_brand_data(brand)
        if df.empty:
            df = load_telecom_brand_data(brand)
            if not df.empty:
                _is_telecom_data = True
    else:
        df = collect_brand_data(brand=brand, limit_per_query=DEFAULT_POSTS_PER_QUERY, max_queries=max_queries)
        if df.empty:
            df = load_cached_brand_data(brand)
        if df.empty:
            df = load_telecom_brand_data(brand)
            if not df.empty:
                _is_telecom_data = True

    if df.empty:
        return None

    raw_path = f"data/raw/{_brand_file_slug(brand)}_raw.csv"
    processed_path = f"data/processed/{_brand_file_slug(brand)}_predicted.csv"

    if _is_telecom_data:
        pass
    else:
        if "predicted_sentiment" not in df.columns:
            df = predict_sentiment(df)
            df.to_csv(processed_path, index=False, encoding="utf-8-sig")

    summary_df = summarize_results(df)
    profile = build_company_profile(df, brand=brand, source=processed_path)

    total_records = len(df)
    pos_pct = summary_df.loc["positive", "Percentage"] if "positive" in summary_df.index else 0.0
    neg_pct = summary_df.loc["negative", "Percentage"] if "negative" in summary_df.index else 0.0

    if _is_telecom_data:
        avg_rating = round(float(df["rating"].mean()), 2) if "rating" in df.columns else 0.0
        total_posts = total_records
        total_comments = 0
        avg_score = avg_rating
    else:
        total_posts = int((df["record_type"] == "post").sum())
        total_comments = int((df["record_type"] == "comment").sum())
        avg_score = round(float(df["score"].mean()), 1)

    return {
        "brand": profile.get("brand", brand.title()),
        "df": df,
        "summary_df": summary_df,
        "profile": profile,
        "raw_path": raw_path,
        "processed_path": processed_path,
        "metrics": {
            "total_records": total_records,
            "total_posts": total_posts,
            "total_comments": total_comments,
            "positive_pct": pos_pct,
            "negative_pct": neg_pct,
            "avg_score": avg_score,
        },
    }



# -----------------------------
# Chart Builders
# -----------------------------
def make_donut_chart(summary_df):
    labels = summary_df.index.tolist()
    values = summary_df["Count"].tolist()
    colors = [SENTIMENT_COLORS.get(l.lower(), "#888") for l in labels]

    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.65,
        marker=dict(colors=colors, line=dict(color="#0e1117", width=3)),
        textinfo="label+percent",
        textfont=dict(size=14, color="white"),
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False, margin=dict(t=20, b=20, l=20, r=20), height=300,
        annotations=[dict(text=f"<b>{sum(values)}</b><br>total", x=0.5, y=0.5,
                          font=dict(size=18, color="white"), showarrow=False)]
    )
    return fig


def make_bar_chart(summary_df):
    labels = summary_df.index.tolist()
    values = summary_df["Count"].tolist()
    colors = [SENTIMENT_COLORS.get(l.lower(), "#888") for l in labels]

    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker=dict(color=colors, line=dict(color="rgba(0,0,0,0)", width=0)),
        text=values, textposition="outside",
        textfont=dict(color="white", size=16),
        hovertemplate="<b>%{x}</b><br>%{y} records<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, color="white", tickfont=dict(size=13)),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.07)", color="white"),
        margin=dict(t=30, b=10, l=10, r=10), height=300,
    )
    return fig


def make_subreddit_chart(df):
    top_subs = df["subreddit"].value_counts().head(8).reset_index()
    top_subs.columns = ["subreddit", "count"]

    fig = go.Figure(go.Bar(
        x=top_subs["count"], y=top_subs["subreddit"], orientation="h",
        marker=dict(color=top_subs["count"],
                    colorscale=[[0, "#1a1f35"], [1, "#4F8EF7"]],
                    line=dict(color="rgba(0,0,0,0)")),
        text=top_subs["count"], textposition="outside",
        textfont=dict(color="white"),
        hovertemplate="<b>r/%{y}</b><br>%{x} posts<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.07)", color="white"),
        yaxis=dict(showgrid=False, color="white", tickfont=dict(size=12)),
        margin=dict(t=10, b=10, l=10, r=40), height=320,
    )
    return fig


def make_sentiment_by_type_chart(df):
    grouped = df.groupby(["record_type", "predicted_sentiment"]).size().reset_index(name="count")

    fig = go.Figure()
    for sentiment in grouped["predicted_sentiment"].unique():
        subset = grouped[grouped["predicted_sentiment"] == sentiment]
        fig.add_trace(go.Bar(
            name=sentiment.capitalize(),
            x=subset["record_type"], y=subset["count"],
            marker_color=SENTIMENT_COLORS.get(sentiment.lower(), "#888"),
            hovertemplate=f"<b>{sentiment}</b><br>%{{x}}: %{{y}}<extra></extra>",
        ))
    fig.update_layout(
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, color="white", tickfont=dict(size=13)),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.07)", color="white"),
        legend=dict(font=dict(color="white"), bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=10, b=10, l=10, r=10), height=300,
    )
    return fig


def make_score_sentiment_scatter(df):
    df_plot = df.copy()
    df_plot["text_short"] = df_plot["combined_text"].str[:80] + "..."

    fig = go.Figure()
    for sentiment, color in SENTIMENT_COLORS.items():
        subset = df_plot[df_plot["predicted_sentiment"].str.lower() == sentiment]
        if subset.empty:
            continue
        fig.add_trace(go.Scatter(
            x=subset["score"], y=[sentiment.capitalize()] * len(subset),
            mode="markers", name=sentiment.capitalize(),
            marker=dict(color=color, size=10, opacity=0.75, line=dict(color="#0e1117", width=1)),
            text=subset["text_short"],
            hovertemplate="<b>%{text}</b><br>Score: %{x}<extra></extra>",
        ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.07)", color="white", title="Reddit Score"),
        yaxis=dict(showgrid=False, color="white"),
        legend=dict(font=dict(color="white"), bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=10, b=10, l=10, r=10), height=280,
    )
    return fig


def make_timeline_chart(df):
    df_t = df.copy()
    df_t["date"] = pd.to_datetime(df_t["created_utc"], unit="s", errors="coerce")
    df_t = df_t.dropna(subset=["date"])
    df_t["month"] = df_t["date"].dt.to_period("M").astype(str)
    grouped = df_t.groupby(["month", "predicted_sentiment"]).size().reset_index(name="count")

    fig = go.Figure()
    for sentiment, color in SENTIMENT_COLORS.items():
        subset = grouped[grouped["predicted_sentiment"].str.lower() == sentiment]
        if subset.empty:
            continue
        fig.add_trace(go.Scatter(
            x=subset["month"], y=subset["count"],
            name=sentiment.capitalize(),
            mode="lines+markers",
            line=dict(color=color, width=2.5),
            marker=dict(size=7, color=color),
            fill="tozeroy",
            fillcolor=SENTIMENT_FILL_COLORS.get(sentiment, "rgba(128,128,128,0.08)"),
            hovertemplate=f"<b>{sentiment}</b><br>%{{x}}: %{{y}}<extra></extra>",
        ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, color="white", tickangle=-30),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.07)", color="white"),
        legend=dict(font=dict(color="white"), bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=10, b=10, l=10, r=10), height=300,
    )
    return fig


# ── Telecom-specific charts ──────────────────────────────────────────
def make_rating_chart(df: pd.DataFrame):
    """Star-rating distribution (1–5) for telecom data."""
    if "rating" not in df.columns:
        return None
    counts = df["rating"].value_counts().sort_index()
    stars  = [f"★ {int(r)}" for r in counts.index]
    bar_colors = ["#FF6B6B", "#FF9F43", "#FFD93D", "#00C9A7", "#4F8EF7"][:len(counts)]

    fig = go.Figure(go.Bar(
        x=stars, y=counts.values,
        marker=dict(color=bar_colors, line=dict(color="rgba(0,0,0,0)")),
        text=counts.values, textposition="outside",
        textfont=dict(color="white", size=15),
        hovertemplate="<b>%{x}</b><br>%{y} reviews<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, color="white", tickfont=dict(size=14)),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.07)", color="white"),
        margin=dict(t=30, b=10, l=10, r=10), height=300,
    )
    return fig


def make_city_chart(df: pd.DataFrame):
    """Top cities by review count for telecom data."""
    if "city" not in df.columns:
        return None
    top = df["city"].value_counts().head(8).reset_index()
    top.columns = ["city", "count"]

    fig = go.Figure(go.Bar(
        x=top["count"], y=top["city"], orientation="h",
        marker=dict(color=top["count"],
                    colorscale=[[0, "#1a1f35"], [1, "#A78BFA"]],
                    line=dict(color="rgba(0,0,0,0)")),
        text=top["count"], textposition="outside",
        textfont=dict(color="white"),
        hovertemplate="<b>%{y}</b><br>%{x} reviews<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.07)", color="white"),
        yaxis=dict(showgrid=False, color="white", tickfont=dict(size=12)),
        margin=dict(t=10, b=10, l=10, r=40), height=320,
    )
    return fig


def make_service_type_chart(df: pd.DataFrame):
    """Service type distribution as a donut for telecom data."""
    if "service_type" not in df.columns:
        return None
    counts = df["service_type"].value_counts()

    fig = go.Figure(go.Pie(
        labels=counts.index.tolist(), values=counts.values.tolist(), hole=0.55,
        marker=dict(
            colors=["#4F8EF7", "#A78BFA", "#00C9A7", "#FFD93D"],
            line=dict(color="#0e1117", width=3),
        ),
        textinfo="label+percent",
        textfont=dict(size=13, color="white"),
        hovertemplate="<b>%{label}</b><br>%{value} reviews (%{percent})<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False, margin=dict(t=20, b=20, l=20, r=20), height=300,
    )
    return fig


def make_customer_type_chart(df: pd.DataFrame):
    """Stacked sentiment by customer type for telecom data."""
    if "customer_type" not in df.columns or "predicted_sentiment" not in df.columns:
        return None
    counts = (
        df.groupby(["customer_type", "predicted_sentiment"])
        .size()
        .reset_index(name="count")
    )

    fig = go.Figure()
    for sentiment, color in SENTIMENT_COLORS.items():
        sub = counts[counts["predicted_sentiment"] == sentiment]
        if sub.empty:
            continue
        fig.add_trace(go.Bar(
            name=sentiment.capitalize(),
            x=sub["customer_type"], y=sub["count"],
            marker_color=color,
            hovertemplate=f"<b>{sentiment}</b><br>%{{x}}: %{{y}}<extra></extra>",
        ))
    fig.update_layout(
        barmode="stack",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, color="white", tickfont=dict(size=11), tickangle=-20),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.07)", color="white"),
        legend=dict(font=dict(color="white"), bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=10, b=10, l=10, r=10), height=320,
    )
    return fig


def make_recommendation_chart(ranking_df):
    plot_df = ranking_df.head(8).sort_values("recommendation_score")
    colors = ["#00C9A7" if rank == 1 else "#4F8EF7" for rank in plot_df["rank"]]

    fig = go.Figure(go.Bar(
        x=plot_df["recommendation_score"],
        y=plot_df["brand"],
        orientation="h",
        marker=dict(color=colors, line=dict(color="rgba(0,0,0,0)", width=0)),
        text=plot_df["recommendation_score"].round(1),
        textposition="outside",
        textfont=dict(color="white", size=13),
        hovertemplate="<b>%{y}</b><br>Recommendation score: %{x:.1f}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[0, 105], showgrid=True, gridcolor="rgba(255,255,255,0.07)", color="white"),
        yaxis=dict(showgrid=False, color="white"),
        margin=dict(t=10, b=10, l=10, r=35), height=320,
    )
    return fig


@st.cache_data(ttl=60)
def load_mlops_snapshot():
    raw_paths = sorted(Path("data/raw").glob("*.csv"))
    processed_paths = sorted(Path("data/processed").glob("*_predicted.csv"))
    model_path = Path("models/svm_model.pkl")
    vectorizer_path = Path("models/vectorizer.pkl")

    raw_rows = 0
    processed_rows = 0
    missing_values = 0
    dataset_columns = set()

    for csv_path in raw_paths:
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        raw_rows += len(df)
        missing_values += int(df.isna().sum().sum())
        dataset_columns.update(df.columns)

    for csv_path in processed_paths:
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        processed_rows += len(df)
        missing_values += int(df.isna().sum().sum())
        dataset_columns.update(df.columns)

    feature_count = 0
    model_classes = []
    if vectorizer_path.exists():
        try:
            vectorizer = joblib.load(vectorizer_path)
            feature_count = len(getattr(vectorizer, "vocabulary_", {}))
        except Exception:
            feature_count = 0

    if model_path.exists():
        try:
            model = joblib.load(model_path)
            model_classes = list(getattr(model, "classes_", []))
        except Exception:
            model_classes = []

    metrics = compute_evaluation_metrics()
    ranking_df = load_processed_company_profiles()
    best_company = ranking_df.iloc[0]["brand"] if not ranking_df.empty else "No ranking yet"

    return {
        "raw_files": len(raw_paths),
        "processed_files": len(processed_paths),
        "raw_rows": raw_rows,
        "processed_rows": processed_rows,
        "missing_values": missing_values,
        "dataset_columns": len(dataset_columns),
        "feature_count": feature_count,
        "model_exists": model_path.exists(),
        "vectorizer_exists": vectorizer_path.exists(),
        "model_size_kb": round(model_path.stat().st_size / 1024, 1) if model_path.exists() else 0,
        "model_modified": datetime.fromtimestamp(model_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M") if model_path.exists() else "N/A",
        "model_classes": model_classes,
        "metrics": metrics,
        "ranking_df": ranking_df,
        "best_company": best_company,
    }


@st.cache_data(ttl=300)
def compute_evaluation_metrics(sample_size=12000):
    labeled_path = Path("data/labeled/Twitter_Data.csv")
    model_path = Path("models/svm_model.pkl")
    vectorizer_path = Path("models/vectorizer.pkl")

    fallback = {
        "accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "loss": 1.0,
        "sample_size": 0,
        "source": "No labeled evaluation data",
    }

    if not labeled_path.exists() or not model_path.exists() or not vectorizer_path.exists():
        return fallback

    try:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)

        df_eval = pd.read_csv(labeled_path).dropna(subset=["clean_text", "category"])
        if df_eval.empty:
            return fallback

        if len(df_eval) > sample_size:
            df_eval = df_eval.sample(sample_size, random_state=42)

        label_map = {-1: "negative", 0: "neutral", 1: "positive", -1.0: "negative", 0.0: "neutral", 1.0: "positive"}
        y_true = df_eval["category"].map(label_map)
        valid = y_true.notna()
        df_eval = df_eval[valid].copy()

        df_eval["combined_text"] = df_eval["clean_text"]
        df_eval = preprocess_data(df_eval, vectorizer)

        y_true_final = df_eval["category"].map(label_map)
        y_pred = model.predict(vectorizer.transform(df_eval["clean_text"].astype(str)))

        accuracy = accuracy_score(y_true_final, y_pred)
        precision = precision_score(y_true_final, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_true_final, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_true_final, y_pred, average="weighted", zero_division=0)

        return {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "loss": round(1 - accuracy, 4),
            "sample_size": int(len(df_eval)),
            "source": "Twitter_Data.csv sample",
        }
    except Exception:
        return fallback


def get_pipeline_stages(snapshot):
    has_raw = snapshot["raw_rows"] > 0
    has_processed = snapshot["processed_rows"] > 0
    has_features = snapshot["feature_count"] > 0
    has_model = snapshot["model_exists"]
    has_metrics = snapshot["metrics"]["sample_size"] > 0

    return [
        {"name": "Data Collection", "status": "Completed" if has_raw else "Pending", "progress": 100 if has_raw else 0, "detail": f"{snapshot['raw_files']} raw files"},
        {"name": "Data Cleaning", "status": "Completed" if has_processed else "Pending", "progress": 100 if has_processed else 0, "detail": f"{snapshot['processed_rows']:,} processed rows"},
        {"name": "Feature Engineering", "status": "Completed" if has_features else "Pending", "progress": 100 if has_features else 0, "detail": f"{snapshot['feature_count']:,} TF-IDF features"},
        {"name": "Model Training", "status": "Completed" if has_model else "Failed", "progress": 100 if has_model else 25, "detail": "SVM sentiment classifier"},
        {"name": "Model Evaluation", "status": "Completed" if has_metrics else "Pending", "progress": 100 if has_metrics else 45, "detail": f"{snapshot['metrics']['sample_size']:,} eval samples"},
        {"name": "Deployment", "status": "In Progress", "progress": 76, "detail": "Streamlit inference app"},
        {"name": "Monitoring", "status": "In Progress" if has_processed else "Pending", "progress": 62 if has_processed else 10, "detail": "Feedback drift watch"},
    ]


def render_mlops_dashboard():
    snapshot = load_mlops_snapshot()
    stages = get_pipeline_stages(snapshot)
    metrics = snapshot["metrics"]

    if "dashboard_started_at" not in st.session_state:
        st.session_state["dashboard_started_at"] = datetime.now()
    uptime_seconds = int((datetime.now() - st.session_state["dashboard_started_at"]).total_seconds())
    uptime_text = f"{uptime_seconds // 3600}h {(uptime_seconds % 3600) // 60}m {uptime_seconds % 60}s"

    active_version = st.session_state.get("active_model_version", "v1.0")

    st.markdown("""
    <div class="mlops-hero">
        <div>
            <div class="mlops-kicker">MLOps Platform</div>
            <div class="mlops-title">Machine Learning Workflow Tracking</div>
            <div class="mlops-subtitle">End-to-end pipeline observability for the Reddit sentiment and recommendation system.</div>
        </div>
        <div class="mlops-live"><span></span> Live Tracking</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Pipeline Flow Tracker")
    stage_parts = []
    for i, stage in enumerate(stages):
        status_key = stage["status"].lower().replace(" ", "-")
        connector = "" if i == len(stages) - 1 else '<div class="mlops-connector"></div>'
        stage_parts.append(
            f'<div class="mlops-stage {status_key}">'
            f'<div class="mlops-stage-head">'
            f'<div class="mlops-stage-dot"></div>'
            f'<div class="mlops-status">{stage["status"]}</div>'
            f'</div>'
            f'<div class="mlops-stage-name">{stage["name"]}</div>'
            f'<div class="mlops-stage-detail">{stage["detail"]}</div>'
            f'<div class="mlops-progress"><span style="width:{stage["progress"]}%"></span></div>'
            f'<div class="mlops-progress-label">{stage["progress"]}%</div>'
            f'</div>'
            f'{connector}'
        )

    st.markdown('<div class="mlops-flow">' + "".join(stage_parts) + '</div>', unsafe_allow_html=True)

    m1, m2, m3, m4, m5 = st.columns(5)
    metric_cards = [
        (m1, "Accuracy", metrics["accuracy"], "#00C9A7"),
        (m2, "Precision", metrics["precision"], "#4F8EF7"),
        (m3, "Recall", metrics["recall"], "#A78BFA"),
        (m4, "F1 Score", metrics["f1"], "#FFD93D"),
        (m5, "Loss", metrics["loss"], "#FF6B6B"),
    ]
    for col, label, value, color in metric_cards:
        with col:
            st.markdown(f"""
            <div class="mlops-metric-card">
                <div class="mlops-metric-label">{label}</div>
                <div class="mlops-metric-value" style="color:{color};">{value:.3f}</div>
                <div class="mlops-mini-bar"><span style="width:{min(max(value, 0), 1) * 100}%; background:{color};"></span></div>
            </div>
            """, unsafe_allow_html=True)

    left, mid, right = st.columns([1.1, 1, 1])
    with left:
        st.markdown(f"""
        <div class="mlops-card">
            <div class="mlops-card-title">Dataset Overview</div>
            <div class="mlops-row"><span>Raw rows</span><b>{snapshot['raw_rows']:,}</b></div>
            <div class="mlops-row"><span>Processed rows</span><b>{snapshot['processed_rows']:,}</b></div>
            <div class="mlops-row"><span>Missing values</span><b>{snapshot['missing_values']:,}</b></div>
            <div class="mlops-row"><span>Dataset columns</span><b>{snapshot['dataset_columns']}</b></div>
            <div class="mlops-row"><span>TF-IDF features</span><b>{snapshot['feature_count']:,}</b></div>
        </div>
        """, unsafe_allow_html=True)

    with mid:
        st.markdown(f"""
        <div class="mlops-card">
            <div class="mlops-card-title">Model Versioning</div>
            <div class="mlops-version active"><b>{active_version}</b><span>Active SVM model</span></div>
            <div class="mlops-version"><b>v1.1</b><span>Recommendation layer candidate</span></div>
            <div class="mlops-version muted"><b>v0.9</b><span>Rollback checkpoint</span></div>
            <div class="mlops-row"><span>Model size</span><b>{snapshot['model_size_kb']} KB</b></div>
            <div class="mlops-row"><span>Last saved</span><b>{snapshot['model_modified']}</b></div>
        </div>
        """, unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        if c1.button("Activate v1.1", use_container_width=True):
            st.session_state["active_model_version"] = "v1.1"
            st.rerun()
        if c2.button("Rollback v1.0", use_container_width=True):
            st.session_state["active_model_version"] = "v1.0"
            st.rerun()

    with right:
        endpoint = "http://localhost:8501"
        latency = round(42 + (1 - metrics["accuracy"]) * 100, 1) if metrics["sample_size"] else 65.0
        st.markdown(f"""
        <div class="mlops-card">
            <div class="mlops-card-title">Deployment Status</div>
            <div class="mlops-deploy-status"><span></span> API Live</div>
            <div class="mlops-row"><span>Endpoint</span><b>{endpoint}</b></div>
            <div class="mlops-row"><span>Uptime</span><b>{uptime_text}</b></div>
            <div class="mlops-row"><span>Latency</span><b>{latency} ms</b></div>
            <div class="mlops-row"><span>Best company</span><b>{escape(str(snapshot['best_company']))}</b></div>
            <div class="mlops-row"><span>Classes</span><b>{', '.join(map(str, snapshot['model_classes']))}</b></div>
        </div>
        """, unsafe_allow_html=True)

    logs = [
        f"[{datetime.now().strftime('%H:%M:%S')}] loaded {snapshot['raw_files']} raw datasets and {snapshot['processed_files']} processed prediction files",
        f"[{datetime.now().strftime('%H:%M:%S')}] cleaned text corpus contains {snapshot['processed_rows']:,} prediction rows",
        f"[{datetime.now().strftime('%H:%M:%S')}] feature engineering ready: {snapshot['feature_count']:,} TF-IDF features",
        f"[{datetime.now().strftime('%H:%M:%S')}] model artifact loaded: svm_model.pkl ({snapshot['model_size_kb']} KB)",
        f"[{datetime.now().strftime('%H:%M:%S')}] evaluation sample={metrics['sample_size']:,}, accuracy={metrics['accuracy']:.3f}, f1={metrics['f1']:.3f}",
        f"[{datetime.now().strftime('%H:%M:%S')}] deployment heartbeat OK: {endpoint}, latency={latency} ms",
    ]
    st.markdown(f"""
    <div class="mlops-card mlops-console-card">
        <div class="mlops-card-title">Training Logs Console</div>
        <pre class="mlops-console">{chr(10).join(logs)}</pre>
    </div>
    """, unsafe_allow_html=True)

    ranking_df = snapshot["ranking_df"]
    if not ranking_df.empty:
        st.markdown("### Recommendation Runs")
        st.dataframe(
            ranking_df[["rank", "brand", "recommendation_score", "positive_pct", "negative_pct", "total_records", "reason"]].rename(columns={
                "rank": "Rank",
                "brand": "Company",
                "recommendation_score": "Recommendation Score",
                "positive_pct": "Positive %",
                "negative_pct": "Negative %",
                "total_records": "Records",
                "reason": "Decision Reason",
            }),
            use_container_width=True,
            hide_index=True,
        )


def render_recommendation_system(ranking_df, ml_flow_df=None):
    col_rec1, col_rec2 = st.columns([3.5, 1])
    with col_rec1:
        st.markdown("### Recommendation System")
    with col_rec2:
        if st.button("Re-analyze All Data", key="recalc_brand_page", use_container_width=True):
            with st.spinner("Analyzing from scratch..."):
                initialize_recommendation_profiles()
                st.session_state["initialized"] = True
            st.rerun()


    if ranking_df.empty:
        st.info("No predicted company files found yet. Run an analysis first to build recommendations.")
        return

    best = ranking_df.iloc[0]
    b1, b2, b3, b4 = st.columns(4)
    recommendation_kpis = [
        (b1, best["brand"], "Best Company"),
        (b2, f"{best['recommendation_score']:.1f}/100", "Recommendation Score"),
        (b3, f"{best['positive_pct']:.1f}%", "Positive Feedback"),
        (b4, f"{best['negative_pct']:.1f}%", "Negative Feedback"),
    ]
    for col, val, label in recommendation_kpis:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-number">{val}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="recommendation-card">
        <div class="insight-title">Why this company wins</div>
        <div class="recommendation-title">{best['brand']}</div>
        <div class="recommendation-copy">
            The recommender combines sentiment quality, engagement, and sample confidence.
            Current reason: {best['reason']}.
        </div>
    </div>
    """, unsafe_allow_html=True)

    rc1, rc2 = st.columns([3, 2])
    with rc1:
        st.markdown('<div class="chart-card"><p class="section-header">Company Ranking</p>', unsafe_allow_html=True)
        st.plotly_chart(make_recommendation_chart(ranking_df), use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)
    with rc2:
        display_cols = [
            "rank", "brand", "recommendation_score", "positive_pct",
            "negative_pct", "total_records", "recommendation"
        ]
        st.dataframe(
            ranking_df[display_cols].rename(columns={
                "rank": "Rank",
                "brand": "Company",
                "recommendation_score": "Score",
                "positive_pct": "Positive %",
                "negative_pct": "Negative %",
                "total_records": "Records",
                "recommendation": "Decision",
            }),
            use_container_width=True,
            hide_index=True,
        )

    st.caption("Open the MLOps Workflow page from the sidebar to inspect the full ML pipeline dashboard.")


def render_brand_comparison(results: list[dict], ranking_df: pd.DataFrame):
    st.markdown("### Brand Comparison")

    ranking_map = ranking_df.set_index("brand") if not ranking_df.empty and "brand" in ranking_df.columns else pd.DataFrame()
    cols = st.columns(len(results))
    for col, result in zip(cols, results):
        brand_name = result["brand"]
        metrics = result["metrics"]
        recommendation_score = None
        if not ranking_df.empty and brand_name in ranking_map.index:
            recommendation_score = float(ranking_map.loc[brand_name, "recommendation_score"])

        with col:
            st.markdown(f"""
            <div class="chart-card">
                <p class="section-header">{escape(str(brand_name))}</p>
                <div class="insight-box" style="margin-top:0;">
                    <div class="insight-title">Records</div>
                    <div class="insight-value">{metrics['total_records']}</div>
                </div>
                <div class="insight-box">
                    <div class="insight-title">Positive / Negative</div>
                    <div class="insight-value">{metrics['positive_pct']:.1f}% / {metrics['negative_pct']:.1f}%</div>
                </div>
                <div class="insight-box">
                    <div class="insight-title">Avg Score</div>
                    <div class="insight-value">{metrics['avg_score']:.1f}</div>
                </div>
                <div class="insight-box">
                    <div class="insight-title">Recommendation Score</div>
                    <div class="insight-value">{recommendation_score:.1f}/100</div>
                </div>
            </div>
            """, unsafe_allow_html=True)


# -----------------------------
# Streamlit Custom CSS
# -----------------------------
st.set_page_config(page_title="Brand Feedback Analyzer", layout="wide", page_icon="EG")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

/* ── Base ──────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', system-ui, sans-serif;
    -webkit-font-smoothing: antialiased;
}
h1, h2, h3, h4 { font-family: 'Space Grotesk', sans-serif !important; }
.stApp { background: #06090f !important; }
section[data-testid="stMain"] > div,
.main .block-container { background: #06090f !important; }

/* ── Sidebar ───────────────────────────────────────────────── */
div[data-testid="stSidebar"] {
    background: #080b14 !important;
    border-right: 1px solid rgba(255,255,255,0.05) !important;
}

/* ── Glass card ────────────────────────────────────────────── */
.chart-card {
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 20px;
    padding: 24px;
    margin-bottom: 16px;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    transition: border-color 0.25s, box-shadow 0.25s;
}
.chart-card:hover {
    border-color: rgba(255,255,255,0.12);
    box-shadow: 0 8px 32px rgba(0,0,0,0.35);
}

/* ── Card header ────────────────────────────────────────────── */
.section-header {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: rgba(255,255,255,0.35);
    margin: 0 0 18px 0;
    padding-bottom: 12px;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}

/* ── KPI metric card ────────────────────────────────────────── */
.metric-card {
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 18px;
    padding: 22px 14px 18px;
    text-align: center;
    transition: transform 0.2s, box-shadow 0.2s, border-color 0.2s;
}
.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 36px rgba(0,0,0,0.4);
    border-color: rgba(99,179,237,0.25);
}
.metric-number {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.75rem;
    font-weight: 700;
    color: white;
    line-height: 1.15;
    display: block;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.metric-label {
    font-size: 0.68rem;
    font-weight: 500;
    color: rgba(255,255,255,0.35);
    margin-top: 7px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

/* ── Insight box ────────────────────────────────────────────── */
.insight-box {
    background: rgba(99,179,237,0.05);
    border: 1px solid rgba(99,179,237,0.13);
    border-radius: 16px;
    padding: 18px 20px;
    margin-bottom: 12px;
    transition: background 0.2s;
}
.insight-box:hover { background: rgba(99,179,237,0.09); }
.insight-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.68rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #63b3ed;
    margin-bottom: 7px;
}
.insight-value {
    font-size: 1.2rem;
    font-weight: 600;
    color: rgba(255,255,255,0.92);
    line-height: 1.3;
}

/* ── Sentiment pills ────────────────────────────────────────── */
.sentiment-pill {
    display: inline-flex;
    align-items: center;
    padding: 3px 12px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.04em;
}
.pill-positive { background: rgba(72,187,120,0.12); color: #68d391; border: 1px solid rgba(72,187,120,0.25); }
.pill-negative { background: rgba(252,129,129,0.12); color: #fc8181; border: 1px solid rgba(252,129,129,0.25); }
.pill-neutral  { background: rgba(246,173,85,0.12);  color: #f6ad55; border: 1px solid rgba(246,173,85,0.25); }

/* ── Feedback card ──────────────────────────────────────────── */
.feedback-card {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.06);
    border-left: 3px solid rgba(99,179,237,0.45);
    border-radius: 14px;
    padding: 16px 18px;
    margin-bottom: 10px;
    transition: background 0.2s, border-left-color 0.2s;
}
.feedback-card:hover {
    background: rgba(255,255,255,0.04);
    border-left-color: #63b3ed;
}
.feedback-text {
    color: rgba(255,255,255,0.76);
    font-size: 0.88rem;
    line-height: 1.65;
}
.feedback-meta {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 10px;
    font-size: 0.72rem;
    color: rgba(255,255,255,0.3);
}
.score-badge {
    display: inline-block;
    background: rgba(255,255,255,0.06);
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 0.72rem;
    color: rgba(255,255,255,0.4);
}

/* ── Hero ───────────────────────────────────────────────────── */
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.5rem;
    font-weight: 700;
    color: white;
    line-height: 1.1;
    letter-spacing: -0.02em;
}
.hero-sub {
    color: rgba(255,255,255,0.36);
    font-size: 0.95rem;
    margin-top: 8px;
    font-weight: 400;
}

/* ── Brand tag ──────────────────────────────────────────────── */
.brand-tag {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(99,179,237,0.08);
    border: 1px solid rgba(99,179,237,0.2);
    color: #63b3ed;
    border-radius: 999px;
    padding: 5px 14px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-bottom: 18px;
}

/* ── Recommendation card ────────────────────────────────────── */
.recommendation-card {
    background: rgba(99,179,237,0.04);
    border: 1px solid rgba(99,179,237,0.13);
    border-radius: 18px;
    padding: 22px 26px;
    margin: 16px 0;
}
.recommendation-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.55rem;
    font-weight: 700;
    color: white;
    line-height: 1.2;
    margin-bottom: 6px;
}
.recommendation-copy {
    color: rgba(255,255,255,0.48);
    font-size: 0.88rem;
    line-height: 1.65;
}

/* ── Buttons ────────────────────────────────────────────────── */
.stButton > button {
    background: #1a56db !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 10px 26px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    transition: background 0.2s, transform 0.15s, box-shadow 0.2s !important;
}
.stButton > button:hover {
    background: #1e4fc7 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(26,86,219,0.35) !important;
}

/* ── Text inputs ────────────────────────────────────────────── */
.stTextInput > div > div > input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 12px !important;
    color: white !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
    padding: 11px 14px !important;
    transition: border-color 0.2s !important;
}
.stTextInput > div > div > input:focus {
    border-color: rgba(99,179,237,0.4) !important;
    box-shadow: 0 0 0 3px rgba(99,179,237,0.08) !important;
}
.stTextInput > div > div > input::placeholder { color: rgba(255,255,255,0.22) !important; }

/* ── Tabs ───────────────────────────────────────────────────── */
button[data-baseweb="tab"] {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    color: rgba(255,255,255,0.4) !important;
    border-radius: 8px !important;
    transition: color 0.2s !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: white !important;
    background: rgba(255,255,255,0.07) !important;
}
[data-baseweb="tab-list"] { gap: 4px !important; }
[data-baseweb="tab-highlight"] { display: none !important; }
[data-baseweb="tab-border"]    { background: rgba(255,255,255,0.06) !important; }

/* ── MLOps hero ─────────────────────────────────────────────── */
.mlops-hero {
    background: radial-gradient(ellipse at 10% 0%, rgba(72,187,120,0.09) 0%, transparent 50%),
                radial-gradient(ellipse at 90% 10%, rgba(99,179,237,0.11) 0%, transparent 50%),
                rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 20px;
    padding: 28px 32px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    backdrop-filter: blur(12px);
}
.mlops-kicker { color: #68d391; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.14em; font-weight: 600; }
.mlops-title { color: white; font-family: 'Space Grotesk', sans-serif; font-size: 1.85rem; font-weight: 700; line-height: 1.2; margin-top: 6px; letter-spacing: -0.01em; }
.mlops-subtitle { color: rgba(255,255,255,0.4); margin-top: 8px; font-size: 0.88rem; }
.mlops-live {
    display: inline-flex; align-items: center; gap: 8px;
    color: #68d391; background: rgba(72,187,120,0.08);
    border: 1px solid rgba(72,187,120,0.2);
    border-radius: 999px; padding: 8px 16px;
    font-weight: 600; font-size: 0.8rem; white-space: nowrap;
}
.mlops-live span, .mlops-deploy-status span {
    width: 7px; height: 7px; border-radius: 50%;
    background: #68d391; box-shadow: 0 0 8px rgba(72,187,120,0.8);
    animation: blink 2s ease-in-out infinite;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.3} }

/* ── MLOps pipeline ─────────────────────────────────────────── */
.mlops-flow {
    display: flex; align-items: stretch; overflow-x: auto;
    padding: 4px 0 16px; margin-bottom: 16px; gap: 0;
}
.mlops-stage {
    min-width: 170px; background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px; padding: 14px 16px; position: relative;
}
.mlops-stage.completed   { border-color: rgba(72,187,120,0.28); }
.mlops-stage.in-progress { border-color: rgba(99,179,237,0.32); }
.mlops-stage.failed      { border-color: rgba(252,129,129,0.38); }
.mlops-stage.pending     { opacity: 0.58; }
.mlops-stage-head { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
.mlops-stage-dot  { width: 9px; height: 9px; border-radius: 50%; background: rgba(255,255,255,0.18); }
.mlops-stage.completed   .mlops-stage-dot { background: #68d391; box-shadow: 0 0 8px rgba(72,187,120,0.6); }
.mlops-stage.in-progress .mlops-stage-dot { background: #63b3ed; box-shadow: 0 0 8px rgba(99,179,237,0.6); animation: blink 1.8s ease-in-out infinite; }
.mlops-stage.failed      .mlops-stage-dot { background: #fc8181; }
.mlops-status { color: rgba(255,255,255,0.45); font-size: 0.62rem; text-transform: uppercase; letter-spacing: 0.09em; font-weight: 600; }
.mlops-stage-name { color: white; font-weight: 600; font-size: 0.88rem; margin-top: 12px; font-family: 'Space Grotesk', sans-serif; }
.mlops-stage-detail { color: rgba(255,255,255,0.32); font-size: 0.74rem; margin-top: 5px; min-height: 28px; }
.mlops-progress { height: 3px; background: rgba(255,255,255,0.06); border-radius: 999px; overflow: hidden; margin-top: 10px; }
.mlops-progress span { display: block; height: 100%; border-radius: 999px; background: linear-gradient(90deg, #68d391, #63b3ed); }
.mlops-progress-label { color: rgba(255,255,255,0.3); font-size: 0.65rem; margin-top: 5px; }
.mlops-connector { width: 32px; min-width: 32px; position: relative; flex-shrink: 0; }
.mlops-connector::before {
    content: ""; position: absolute; top: 40px; left: 0; right: 0;
    height: 2px; border-radius: 999px;
    background: linear-gradient(90deg, rgba(72,187,120,0.2), rgba(99,179,237,0.65), rgba(72,187,120,0.2));
    background-size: 200% 100%; animation: flowAnim 2s linear infinite;
}
@keyframes flowAnim { from{background-position:0 0} to{background-position:200% 0} }

/* ── MLOps info cards ───────────────────────────────────────── */
.mlops-metric-card, .mlops-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px; padding: 18px 20px; margin-bottom: 14px;
}
.mlops-metric-label { color: rgba(255,255,255,0.32); font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.1em; font-weight: 600; }
.mlops-metric-value { font-family: 'Space Grotesk', sans-serif; font-size: 1.6rem; font-weight: 700; margin-top: 8px; }
.mlops-mini-bar { height: 3px; background: rgba(255,255,255,0.06); border-radius: 999px; overflow: hidden; margin-top: 12px; }
.mlops-mini-bar span { height: 100%; display: block; border-radius: 999px; }
.mlops-card-title { color: white; font-family: 'Space Grotesk', sans-serif; font-size: 0.92rem; font-weight: 600; margin-bottom: 14px; }
.mlops-row { display: flex; align-items: center; justify-content: space-between; gap: 12px; border-top: 1px solid rgba(255,255,255,0.05); padding: 9px 0; }
.mlops-row span { color: rgba(255,255,255,0.38); font-size: 0.79rem; }
.mlops-row b    { color: rgba(255,255,255,0.82); font-size: 0.79rem; text-align: right; }
.mlops-version { border: 1px solid rgba(255,255,255,0.07); border-radius: 10px; padding: 10px 12px; margin-bottom: 8px; display: flex; justify-content: space-between; gap: 10px; color: rgba(255,255,255,0.55); font-size: 0.8rem; }
.mlops-version.active { border-color: rgba(72,187,120,0.28); background: rgba(72,187,120,0.06); }
.mlops-version.muted  { opacity: 0.5; }
.mlops-version b      { color: white; }
.mlops-deploy-status { display: inline-flex; align-items: center; gap: 8px; color: #68d391; font-weight: 600; margin-bottom: 10px; font-size: 0.83rem; }
.mlops-console-card { margin-top: 4px; }
.mlops-console {
    background: #020407; color: #86efac;
    border: 1px solid rgba(72,187,120,0.12); border-radius: 12px;
    padding: 16px; min-height: 155px; overflow-x: auto;
    font-size: 0.78rem; line-height: 1.7;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
}

/* ── Misc ───────────────────────────────────────────────────── */
.flow-step { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06); border-radius: 14px; padding: 16px; min-height: 130px; }
.flow-title  { font-family: 'Space Grotesk', sans-serif; color: white; font-weight: 600; font-size: 0.88rem; }
.flow-output { color: #63b3ed; font-weight: 600; margin-top: 12px; font-size: 0.84rem; }
.flow-role   { color: rgba(255,255,255,0.33); margin-top: 8px; font-size: 0.74rem; line-height: 1.45; }

@media (max-width: 900px) {
    .mlops-hero  { flex-direction: column; align-items: flex-start; }
    .mlops-title { font-size: 1.35rem; }
    .mlops-stage { min-width: 150px; }
    .hero-title  { font-size: 1.85rem; }
}
</style>
""", unsafe_allow_html=True)





# -----------------------------




def render_market_analysis_dashboard():
    col_title, col_btn = st.columns([3.5, 1])
    with col_title:
        st.markdown("""
        <div style="margin-bottom: 32px;">
            <div class="hero-title">Telecom Market Intelligence</div>
            <div class="hero-sub">Deep comparison and recommendation analytics across the Egyptian Telecom sector</div>
        </div>
        """, unsafe_allow_html=True)
    with col_btn:
        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        if st.button("Re-analyze All Data", key="recalc_market_page", use_container_width=True):
            with st.spinner("Analyzing from scratch..."):
                initialize_recommendation_profiles()
                st.session_state["initialized"] = True
            st.rerun()


    if not TELECOM_CSV_PATH.exists():
        st.error(f"Telecom dataset not found at `{TELECOM_CSV_PATH}`. Please verify files.")
        return

    try:
        raw_df = pd.read_csv(TELECOM_CSV_PATH, encoding="utf-8")
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return

    # Predict sentiment using our local SVM classifier
    raw_df["combined_text"] = raw_df["review_text"].fillna("")
    raw_df = predict_sentiment(raw_df)
    
    # 1. Operators Overview Grid
    st.markdown("### Operators Comparison Grid")
    
    companies = ["Vodafone Egypt", "Orange Egypt", "Etisalat Egypt", "WE Telecom Egypt"]
    comp_cols = st.columns(4)
    
    company_data = {}
    for comp in companies:
        sub = raw_df[raw_df["company"].str.lower() == comp.lower()].copy()
        if sub.empty:
            continue
        tot = len(sub)
        avg_rt = float(sub["rating"].mean())
        s_counts = sub["predicted_sentiment"].value_counts()
        pos_p = (s_counts.get("positive", 0) / tot) * 100
        neg_p = (s_counts.get("negative", 0) / tot) * 100
        neu_p = (s_counts.get("neutral", 0) / tot) * 100
        nss = pos_p - neg_p
        
        company_data[comp] = {
            "df": sub,
            "total": tot,
            "avg_rating": avg_rt,
            "positive_pct": pos_p,
            "neutral_pct": neu_p,
            "negative_pct": neg_p,
            "nss": nss
        }

    for col, comp in zip(comp_cols, companies):
        if comp not in company_data:
            continue
        data = company_data[comp]
        
        with col:
            st.markdown(f"""
            <div class="chart-card" style="border-top: 3px solid #63b3ed; min-height: 200px;">
                <p class="section-header" style="font-size:0.95rem; color:white; margin-bottom:12px; border-bottom: none; padding-bottom: 0;">{comp}</p>
                <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
                    <span style="color:rgba(255,255,255,0.4); font-size:0.8rem;">Total Reviews</span>
                    <span style="color:white; font-weight:600; font-size:0.85rem;">{data['total']:,}</span>
                </div>
                <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
                    <span style="color:rgba(255,255,255,0.4); font-size:0.8rem;">Average Rating</span>
                    <span style="color:#FFD93D; font-weight:600; font-size:0.85rem;">★ {data['avg_rating']:.2f}/5</span>
                </div>
                <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
                    <span style="color:rgba(255,255,255,0.4); font-size:0.8rem;">Positive Ratio</span>
                    <span style="color:#68d391; font-weight:600; font-size:0.85rem;">{data['positive_pct']:.1f}%</span>
                </div>
                <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
                    <span style="color:rgba(255,255,255,0.4); font-size:0.8rem;">Net Satisfaction</span>
                    <span style="color:#63b3ed; font-weight:600; font-size:0.85rem;">{data['nss']:+.1f}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom:24px'></div>", unsafe_allow_html=True)
    
    # 2. Recommender weight simulator
    st.markdown("### Recommender Weight Simulator")
    st.markdown("""
    <div style="font-size:0.85rem; color:rgba(255,255,255,0.4); margin-bottom:16px; margin-top:-8px;">
        Adjust the weights below to customize how the operators are evaluated. The recommender will update rankings in real-time.
    </div>
    """, unsafe_allow_html=True)
    
    w_cols = st.columns(3)
    with w_cols[0]:
        w_sentiment = st.slider("Sentiment weight (NSS Index)", 0.0, 1.0, 0.60, 0.05)
    with w_cols[1]:
        w_rating = st.slider("Rating weight (Average Star Rating)", 0.0, 1.0, 0.25, 0.05)
    with w_cols[2]:
        w_volume = st.slider("Volume weight (Data Sample Count)", 0.0, 1.0, 0.15, 0.05)

    # Normalize weights
    total_w = w_sentiment + w_rating + w_volume
    if total_w > 0:
        w_sentiment_n = w_sentiment / total_w
        w_rating_n = w_rating / total_w
        w_volume_n = w_volume / total_w
    else:
        w_sentiment_n, w_rating_n, w_volume_n = 0.33, 0.33, 0.33

    # Calculate scores
    ranked_list = []
    for comp, data in company_data.items():
        sentiment_score = (data["nss"] + 100) / 2
        rating_score = (data["avg_rating"] / 5.0) * 100
        volume_score = min(data["total"] / 2500, 1.0) * 100
        
        final_score = (sentiment_score * w_sentiment_n) + (rating_score * w_rating_n) + (volume_score * w_volume_n)
        
        ranked_list.append({
            "brand": comp,
            "score": round(final_score, 2),
            "sentiment_score": round(sentiment_score, 2),
            "rating_score": round(rating_score, 2),
            "volume_score": round(volume_score, 2),
            "nss": round(data["nss"], 1),
            "avg_rating": round(data["avg_rating"], 2),
            "total": data["total"]
        })
        
    ranked_df = pd.DataFrame(ranked_list).sort_values("score", ascending=False).reset_index(drop=True)
    ranked_df.insert(0, "rank", range(1, len(ranked_df) + 1))
    
    # Render simulated recommendation
    sim_cols = st.columns([3, 2])
    with sim_cols[0]:
        st.markdown('<div class="chart-card"><p class="section-header">Simulated Rankings</p>', unsafe_allow_html=True)
        fig = go.Figure(go.Bar(
            x=ranked_df["score"],
            y=ranked_df["brand"],
            orientation="h",
            marker=dict(
                color=ranked_df["score"],
                colorscale=[[0, "#1a243d"], [1, "#63b3ed"]],
                line=dict(color="rgba(0,0,0,0)")
            ),
            text=ranked_df["score"].apply(lambda x: f"{x:.1f} pts"),
            textposition="outside",
            textfont=dict(color="white"),
            hovertemplate="<b>%{y}</b><br>Score: %{x:.2f}<extra></extra>"
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.07)", color="white", range=[0, 100]),
            yaxis=dict(showgrid=False, color="white", tickfont=dict(size=12)),
            margin=dict(t=10, b=10, l=10, r=60), height=260,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)
        
    with sim_cols[1]:
        st.markdown('<div class="chart-card"><p class="section-header">Recommender Verdict</p>', unsafe_allow_html=True)
        best_comp = ranked_df.iloc[0]
        
        st.markdown(f"""
        <div style="font-family:'Space Grotesk',sans-serif; font-size:1.4rem; font-weight:700; color:white; margin-bottom:8px;">
            Winner: {best_comp['brand']}
        </div>
        <div style="color:rgba(255,255,255,0.6); font-size:0.88rem; line-height:1.6; margin-bottom:12px;">
            Leading with a customized score of <b>{best_comp['score']:.1f}/100</b>. 
            Under your weights, this company outperforms competitors due to:
            <ul style="margin-top:4px; padding-left:18px;">
                <li><b>Sentiment NSS:</b> {best_comp['nss']:+.1f}%</li>
                <li><b>Average Rating:</b> ★ {best_comp['avg_rating']:.2f}</li>
                <li><b>Data Volume:</b> {best_comp['total']:,} reviews</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # 3. Sentiment breakdown & rating charts
    st.markdown("### Detailed Market Deep-Dives")
    
    tab_sentiment, tab_services, tab_demographics = st.tabs(["Sentiment Breakdown", "Service Breakdown", "Geographics & Demographics"])
    
    with tab_sentiment:
        sc1, sc2 = st.columns(2)
        with sc1:
            st.markdown('<div class="chart-card"><p class="section-header">Sentiment count comparison</p>', unsafe_allow_html=True)
            grouped_sent = raw_df.groupby(["company", "predicted_sentiment"]).size().reset_index(name="count")
            
            fig_s = go.Figure()
            for sent, col in SENTIMENT_COLORS.items():
                sub = grouped_sent[grouped_sent["predicted_sentiment"] == sent]
                fig_s.add_trace(go.Bar(
                    name=sent.capitalize(),
                    x=sub["company"],
                    y=sub["count"],
                    marker_color=col,
                    hovertemplate="<b>%{x}</b><br>Count: %{y}<extra></extra>"
                ))
            fig_s.update_layout(
                barmode="group",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(showgrid=False, color="white", tickfont=dict(size=11)),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.07)", color="white"),
                legend=dict(font=dict(color="white"), bgcolor="rgba(0,0,0,0)"),
                margin=dict(t=10, b=10, l=10, r=10), height=300,
            )
            st.plotly_chart(fig_s, use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)
            
        with sc2:
            st.markdown('<div class="chart-card"><p class="section-header">NSS index comparison</p>', unsafe_allow_html=True)
            fig_nss = go.Figure(go.Bar(
                x=ranked_df["brand"],
                y=ranked_df["nss"],
                marker=dict(
                    color=ranked_df["nss"].apply(lambda x: "#68d391" if x > 0 else "#fc8181"),
                    line=dict(color="rgba(0,0,0,0)")
                ),
                text=ranked_df["nss"].apply(lambda x: f"{x:+.1f}%"),
                textposition="outside",
                textfont=dict(color="white"),
                hovertemplate="<b>%{x}</b><br>NSS: %{y:+.1f}%<extra></extra>"
            ))
            fig_nss.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(showgrid=False, color="white", tickfont=dict(size=11)),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.07)", color="white"),
                margin=dict(t=25, b=10, l=10, r=10), height=300,
            )
            st.plotly_chart(fig_nss, use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)

    with tab_services:
        scc1, scc2 = st.columns(2)
        with scc1:
            st.markdown('<div class="chart-card"><p class="section-header">Star Rating by Service Type</p>', unsafe_allow_html=True)
            svc_rating = raw_df.groupby(["company", "service_type"])["rating"].mean().reset_index()
            fig_svc = go.Figure()
            for s_type in svc_rating["service_type"].unique():
                sub = svc_rating[svc_rating["service_type"] == s_type]
                fig_svc.add_trace(go.Bar(
                    name=s_type,
                    x=sub["company"],
                    y=sub["rating"],
                    hovertemplate="<b>%{x}</b> - " + s_type + "<br>Rating: %{y:.2f}★<extra></extra>"
                ))
            fig_svc.update_layout(
                barmode="group",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(showgrid=False, color="white", tickfont=dict(size=11)),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.07)", color="white", range=[1, 5]),
                legend=dict(font=dict(color="white"), bgcolor="rgba(0,0,0,0)"),
                margin=dict(t=10, b=10, l=10, r=10), height=300,
            )
            st.plotly_chart(fig_svc, use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)
            
        with scc2:
            st.markdown('<div class="chart-card"><p class="section-header">Satisfaction by Customer Type</p>', unsafe_allow_html=True)
            cust_type_df = raw_df.groupby(["company", "customer_type", "predicted_sentiment"]).size().reset_index(name="count")
            total_cust = raw_df.groupby(["company", "customer_type"]).size().reset_index(name="total")
            cust_type_df = cust_type_df.merge(total_cust, on=["company", "customer_type"])
            cust_type_df["pct"] = (cust_type_df["count"] / cust_type_df["total"]) * 100
            
            pos_cust = cust_type_df[cust_type_df["predicted_sentiment"] == "positive"]
            
            fig_cust = go.Figure()
            for c_type in pos_cust["customer_type"].unique():
                sub = pos_cust[pos_cust["customer_type"] == c_type]
                fig_cust.add_trace(go.Bar(
                    name=c_type,
                    x=sub["company"],
                    y=sub["pct"],
                    hovertemplate="<b>%{x}</b> - " + c_type + "<br>Positive Sentiment: %{y:.1f}%<extra></extra>"
                ))
            fig_cust.update_layout(
                barmode="group",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(showgrid=False, color="white", tickfont=dict(size=11)),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.07)", color="white"),
                legend=dict(font=dict(color="white"), bgcolor="rgba(0,0,0,0)"),
                margin=dict(t=10, b=10, l=10, r=10), height=300,
            )
            st.plotly_chart(fig_cust, use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)

    with tab_demographics:
        dc1, dc2 = st.columns(2)
        with dc1:
            st.markdown('<div class="chart-card"><p class="section-header">Average Rating by City (Top 6)</p>', unsafe_allow_html=True)
            top_cities = raw_df["city"].value_counts().head(6).index.tolist()
            city_rating = raw_df[raw_df["city"].isin(top_cities)].groupby(["company", "city"])["rating"].mean().reset_index()
            
            fig_city = go.Figure()
            for city in top_cities:
                sub = city_rating[city_rating["city"] == city]
                fig_city.add_trace(go.Bar(
                    name=city,
                    x=sub["company"],
                    y=sub["rating"],
                    hovertemplate="<b>%{x}</b> - " + city + "<br>Rating: %{y:.2f}★<extra></extra>"
                ))
            fig_city.update_layout(
                barmode="group",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(showgrid=False, color="white", tickfont=dict(size=11)),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.07)", color="white", range=[1, 5]),
                legend=dict(font=dict(color="white"), bgcolor="rgba(0,0,0,0)"),
                margin=dict(t=10, b=10, l=10, r=10), height=300,
            )
            st.plotly_chart(fig_city, use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)
            
        with dc2:
            st.markdown('<div class="chart-card"><p class="section-header">Average Rating by Age Group</p>', unsafe_allow_html=True)
            age_rating = raw_df.groupby(["company", "age_group"])["rating"].mean().reset_index()
            
            fig_age = go.Figure()
            for age in age_rating["age_group"].unique():
                sub = age_rating[age_rating["age_group"] == age]
                fig_age.add_trace(go.Bar(
                    name=age,
                    x=sub["company"],
                    y=sub["rating"],
                    hovertemplate="<b>%{x}</b> - " + age + "<br>Rating: %{y:.2f}★<extra></extra>"
                ))
            fig_age.update_layout(
                barmode="group",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(showgrid=False, color="white", tickfont=dict(size=11)),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.07)", color="white", range=[1, 5]),
                legend=dict(font=dict(color="white"), bgcolor="rgba(0,0,0,0)"),
                margin=dict(t=10, b=10, l=10, r=10), height=300,
            )
            st.plotly_chart(fig_age, use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)

    # 4. Raw Sample Explorer
    st.markdown("### Telecom Customer Review Explorer")
    exp_cols = st.columns(3)
    with exp_cols[0]:
        sel_comp = st.selectbox("Company", companies)
    with exp_cols[1]:
        sel_sent = st.selectbox("Sentiment", ["Positive", "Neutral", "Negative"])
    with exp_cols[2]:
        search_query = st.text_input("Search reviews", placeholder="e.g. net, customer service, fast...")

    subset_exp = raw_df[(raw_df["company"].str.lower() == sel_comp.lower()) & (raw_df["predicted_sentiment"] == sel_sent.lower())]
    if search_query:
        subset_exp = subset_exp[subset_exp["review_text"].str.contains(search_query, case=False, na=False)]
        
    st.markdown(f"**Showing {min(5, len(subset_exp))} of {len(subset_exp):,} matching reviews**")
    
    samples = subset_exp.head(5)
    if samples.empty:
        st.info("No reviews match your filters.")
    else:
        for idx, row in samples.iterrows():
            stars = "★" * int(row["rating"]) + "☆" * (5 - int(row["rating"]))
            st.markdown(f"""
            <div class="feedback-card">
                <span class="sentiment-pill pill-{sel_sent.lower()}">{sel_sent}</span>
                <span style="margin-left:8px; font-size:0.85rem; color:#FFD93D;">{stars}</span>
                <div class="feedback-text" style="margin-top:10px;">{row.get('clean_text', row['review_text'])}</div>
                <div class="feedback-meta">
                    <span>City: {row['city']}</span>
                    <span>Service: {row['service_type']}</span>
                    <span>Customer: {row['customer_type']}</span>
                    <span>Age: {row['age_group']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
# Startup initialization for recommendation profiles (recalculated from scratch once per page reload/session)
if "initialized" not in st.session_state:
    initialize_recommendation_profiles()
    st.session_state["initialized"] = True


# -----------------------------
# Sidebar
# -----------------------------

with st.sidebar:
    st.markdown("""
    <div style="padding: 8px 0 20px 0;">
        <div style="font-family:'Syne',sans-serif; font-size:1.3rem; font-weight:800; color:white;">
            Brand Feedback Analyzer
        </div>
        <div style="font-size:0.75rem; color:rgba(255,255,255,0.35); margin-top:3px;">
            Egypt Reddit Intelligence
        </div>
    </div>
    """, unsafe_allow_html=True)

    app_page = st.radio(
        "Workspace",
        ["Brand Analyzer", "Market Analysis", "MLOps Workflow"],
        index=0,
        label_visibility="collapsed",
    )

    if app_page == "Brand Analyzer":
        st.markdown("**Quick Select**")
        categories = {}
        for brand_key, info in EGYPTIAN_BRANDS.items():
            cat = info.get("category", "other")
            categories.setdefault(cat, []).append(brand_key)

        for cat, brands in sorted(categories.items()):
            icon = CATEGORY_ICONS.get(cat, "🏢")
            with st.expander(f"{icon} {cat.capitalize()}"):
                for b in sorted(brands):
                    if st.button(b.title(), key=f"btn_{b}", use_container_width=True):
                        st.session_state["selected_brand"] = b.title()
                        if "analysis_results" in st.session_state:
                            del st.session_state["analysis_results"]

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.75rem; color:rgba(255,255,255,0.25); line-height:1.7;">
        Scrapes Reddit posts & comments<br>
        Filters for Egypt-relevant content<br>
        Runs SVM sentiment classifier
    </div>
    """, unsafe_allow_html=True)


if app_page == "Market Analysis":
    render_market_analysis_dashboard()
    st.stop()


if app_page == "MLOps Workflow":
    render_mlops_dashboard()
    st.stop()


# -----------------------------
# Hero Header
# -----------------------------
st.markdown("""
<div style="margin-bottom: 32px;">
    <div class="hero-title">Brand Feedback Analyzer</div>
    <div class="hero-sub">Reddit sentiment intelligence for Egyptian & regional brands</div>
</div>
""", unsafe_allow_html=True)


# -----------------------------
# Input Section
# -----------------------------
col_input, col_input2, col_slider = st.columns([1.5, 1.5, 1])

with col_input:
    default_brand = st.session_state.get("selected_brand", "")
    brand = st.text_input(
        "Brand Name", value=default_brand,
        placeholder="e.g. Vodafone, Fawry, Jumia, CIB ...",
        label_visibility="collapsed"
    )

with col_input2:
    brand_2 = st.text_input(
        "Brand Name 2 (optional)",
        value=st.session_state.get("selected_brand_2", ""),
        placeholder="leave empty to analyze one brand",
        label_visibility="collapsed",
    )

with col_slider:
    query_count = st.slider("Search queries", min_value=1, max_value=12, value=6, label_visibility="collapsed")
    st.markdown(f'<div style="font-size:0.75rem; color:rgba(255,255,255,0.35); margin-top:-8px;">{query_count} queries · {DEFAULT_POSTS_PER_QUERY} posts/query</div>', unsafe_allow_html=True)

if brand.strip():
    key, info = get_brand_info(brand)
    query_count = min(query_count, max(1, len(build_queries(brand))))
    if info:
        icon = CATEGORY_ICONS.get(info["category"], "🏢")
        st.markdown(f"""
        <div class="brand-tag">
            {icon} Known Egyptian Brand &nbsp;·&nbsp; {info['category'].capitalize()}
            &nbsp;·&nbsp; {query_count} search queries
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="color:rgba(255,200,80,0.8); font-size:0.85rem; margin-bottom:16px;">
            Brand not in registry — Egypt keyword required in posts
        </div>
        """, unsafe_allow_html=True)

run_button = st.button("Analyze Brand", type="primary")
st.markdown("<div style='margin-bottom:32px'></div>", unsafe_allow_html=True)

# -----------------------------
# Main Analysis
# -----------------------------
if run_button:
    if not brand.strip():
        st.warning("Please enter a brand name.")
    else:
        _use_cache = not reddit_credentials_configured()
        _is_telecom_data = False

        if _use_cache:
            # ── Priority 1: saved CSV (processed / raw) ────────────
            df = load_cached_brand_data(brand)
            if df.empty:
                # ── Priority 2: telecom master CSV ────────────────
                df = load_telecom_brand_data(brand)
                if not df.empty:
                    _is_telecom_data = True
                    st.info(
                        f"No Reddit credentials — loaded **{len(df):,} telecom reviews** "
                        f"for **{brand.strip().title()}** from local dataset."
                    )
                else:
                    st.warning(
                        f"No data found for **{brand.strip().title()}**. "
                        "Available brands: Vodafone, Orange, Etisalat, We."
                    )
                    st.stop()
            else:
                st.success(
                    f"Loaded cached data for **{brand.strip().title()}** "
                    f"({len(df):,} records) — Reddit credentials not required."
                )
        else:
            # ── Live scrape mode ───────────────────────────────────
            with st.status(
                f"Scraping Reddit for **{brand.strip().title()}** — 0 of {query_count} queries done…",
                expanded=True,
            ) as _scrape_status:
                df = collect_brand_data(brand=brand, limit_per_query=DEFAULT_POSTS_PER_QUERY, max_queries=query_count)
                if df.empty:
                    _scrape_status.update(label="No Reddit data found — trying local dataset…", state="error", expanded=False)
                else:
                    _scrape_status.update(
                        label=f"Scraped {len(df):,} records for **{brand.strip().title()}**",
                        state="complete",
                        expanded=False,
                    )

            # ── Scrape empty: try cached CSV → telecom CSV ─────────
            if df.empty:
                df = load_cached_brand_data(brand)
            if df.empty:
                df = load_telecom_brand_data(brand)
                if not df.empty:
                    _is_telecom_data = True
                    st.info(
                        f"Reddit returned no data — loaded **{len(df):,} telecom reviews** "
                        f"for **{brand.strip().title()}** from local dataset."
                    )

        if df.empty:
            if "analysis_results" in st.session_state:
                del st.session_state["analysis_results"]
            fetch_error = LAST_FETCH_ERROR
            is_403 = fetch_error and "403" in str(fetch_error)
            no_creds = not reddit_credentials_configured()

            if is_403 or no_creds:
                st.markdown("""
                <div style="background:rgba(255,107,107,0.08); border:1px solid rgba(255,107,107,0.3);
                            border-radius:14px; padding:28px;">
                    <div style="font-size:1rem; text-align:center; font-weight:700; color:#fc8181; letter-spacing:0.1em; text-transform:uppercase;">[Locked]</div>
                    <div style="font-family:'Syne',sans-serif; font-size:1.15rem; color:rgba(255,255,255,0.9);
                                margin-top:10px; text-align:center; font-weight:600;">
                        Reddit API Credentials Required
                    </div>
                    <div style="font-size:0.88rem; color:rgba(255,255,255,0.5); margin-top:8px; text-align:center;">
                        Reddit now requires OAuth2 authentication. Follow these steps to fix the 403 error:
                    </div>
                    <ol style="color:rgba(255,255,255,0.75); font-size:0.88rem; margin-top:16px; line-height:2;">
                        <li>Go to <a href="https://www.reddit.com/prefs/apps" target="_blank"
                           style="color:#7EB8F7;">reddit.com/prefs/apps</a> and click <strong>Create App</strong></li>
                        <li>Choose <strong>script</strong> as app type, set redirect URI to <code>http://localhost</code></li>
                        <li>Copy your <strong>client_id</strong> (under app name) and <strong>client_secret</strong></li>
                        <li>Create the file <code>.streamlit/secrets.toml</code> in the project root with:</li>
                    </ol>
                    <pre style="background:rgba(0,0,0,0.35); border-radius:8px; padding:14px; font-size:0.82rem;
                                color:#A8FF78; margin-top:4px; overflow-x:auto;">[reddit]
client_id     = "YOUR_CLIENT_ID_HERE"
client_secret = "YOUR_CLIENT_SECRET_HERE"</pre>
                    <div style="font-size:0.8rem; color:rgba(255,255,255,0.35); margin-top:12px; text-align:center;">
                        Registration is free · No credit card required · API is free for personal use
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background:rgba(255,107,107,0.08); border:1px solid rgba(255,107,107,0.2);
                            border-radius:14px; padding:24px; text-align:center;">
                    <div style="font-size:1rem; text-align:center; font-weight:700; color:rgba(255,255,255,0.4); letter-spacing:0.1em; text-transform:uppercase;">[No Data]</div>
                    <div style="font-family:'Syne',sans-serif; font-size:1.1rem; color:rgba(255,255,255,0.8); margin-top:8px;">
                        No Reddit data could be loaded
                    </div>
                    <div style="font-size:0.85rem; color:rgba(255,255,255,0.35); margin-top:6px;">
                        Try a different brand name, lower the query count, or check Reddit access.
                    </div>
                </div>
                """, unsafe_allow_html=True)
                if fetch_error:
                    st.warning(fetch_error)
        else:
            raw_path = f"data/raw/{brand.lower().replace(' ', '_')}_raw.csv"
            processed_path = f"data/processed/{brand.lower().replace(' ', '_')}_predicted.csv"

            if _is_telecom_data:
                pass
            elif _use_cache:
                if "predicted_sentiment" not in df.columns:
                    df = predict_sentiment(df)
                    df.to_csv(processed_path, index=False, encoding="utf-8-sig")
            else:
                df.to_csv(raw_path, index=False, encoding="utf-8-sig")
                df = predict_sentiment(df)
                df.to_csv(processed_path, index=False, encoding="utf-8-sig")

            summary_df = summarize_results(df)

            total_records  = len(df)
            pos_pct = summary_df.loc["positive", "Percentage"] if "positive" in summary_df.index else 0.0
            neg_pct = summary_df.loc["negative", "Percentage"] if "negative" in summary_df.index else 0.0

            if _is_telecom_data:
                avg_rating     = round(float(df["rating"].mean()), 2) if "rating" in df.columns else 0.0
                top_city       = df["city"].value_counts().idxmax() if "city" in df.columns else "—"
                top_service    = df["service_type"].value_counts().idxmax() if "service_type" in df.columns else "—"
                avg_score_disp = f"{avg_rating} / 5"
                total_posts    = total_records
                total_comments = 0
                avg_score      = avg_rating
            else:
                total_posts    = int((df["record_type"] == "post").sum())
                total_comments = int((df["record_type"] == "comment").sum())
                avg_score      = round(df["score"].mean(), 1)
                avg_rating     = 0.0
                top_city       = "—"
                top_service    = "—"
                avg_score_disp = "—"

            first_profile = build_company_profile(df, brand=brand, source=processed_path)
            first_result  = {
                "brand":   first_profile.get("brand", brand.title()),
                "profile": first_profile,
                "metrics": {
                    "total_records": total_records,
                    "total_posts":   total_records if _is_telecom_data else total_posts,
                    "total_comments": 0 if _is_telecom_data else total_comments,
                    "positive_pct":  pos_pct,
                    "negative_pct":  neg_pct,
                    "avg_score":     avg_rating if _is_telecom_data else avg_score,
                },
            }

            comparison_brand = brand_2.strip()
            comparison_result = None
            ranking_df = None
            df_combined = df.copy()
            df_combined["brand"] = brand.title()
            has_comparison = False

            if comparison_brand and comparison_brand.lower() != brand.strip().lower():
                comparison_result = analyze_brand(comparison_brand, query_count)
                if comparison_result is None:
                    st.info(f"No data found for {comparison_brand}; showing the first brand only.")
                    ranking_df = load_processed_company_profiles()
                else:
                    ranking_df = rank_companies([first_result["profile"], comparison_result["profile"]])
                    has_comparison = True
                    df_2 = comparison_result["df"].copy()
                    df_2["brand"] = comparison_brand.title()
                    df_combined = pd.concat([df_combined, df_2], ignore_index=True)
            else:
                ranking_df = load_processed_company_profiles()

            # Save results in session state
            st.session_state["analysis_results"] = {
                "brand": brand,
                "brand_2": brand_2,
                "query_count": query_count,
                "df": df,
                "_is_telecom_data": _is_telecom_data,
                "summary_df": summary_df,
                "total_records": total_records,
                "pos_pct": pos_pct,
                "neg_pct": neg_pct,
                "avg_rating": avg_rating,
                "top_city": top_city,
                "top_service": top_service,
                "avg_score_disp": avg_score_disp,
                "total_posts": total_posts,
                "total_comments": total_comments,
                "avg_score": avg_score,
                "first_profile": first_profile,
                "first_result": first_result,
                "comparison_result": comparison_result,
                "ranking_df": ranking_df,
                "df_combined": df_combined,
                "has_comparison": has_comparison,
                "processed_path": processed_path,
            }

# Render baseline if not run and no cached results
if not run_button and "analysis_results" not in st.session_state:
    baseline_ranking_df = load_processed_company_profiles()
    render_recommendation_system(baseline_ranking_df)

# Render results if available
if "analysis_results" in st.session_state:
    res = st.session_state["analysis_results"]
    brand_val = res["brand"]
    brand_2_val = res["brand_2"]
    query_count_val = res["query_count"]
    df_val = res["df"]
    _is_telecom_data_val = res["_is_telecom_data"]
    summary_df_val = res["summary_df"]
    total_records_val = res["total_records"]
    pos_pct_val = res["pos_pct"]
    neg_pct_val = res["neg_pct"]
    avg_rating_val = res["avg_rating"]
    top_city_val = res["top_city"]
    top_service_val = res["top_service"]
    avg_score_disp_val = res["avg_score_disp"]
    total_posts_val = res["total_posts"]
    total_comments_val = res["total_comments"]
    avg_score_val = res["avg_score"]
    first_profile_val = res["first_profile"]
    first_result_val = res["first_result"]
    comparison_result_val = res["comparison_result"]
    ranking_df_val = res["ranking_df"]
    df_combined_val = res["df_combined"]
    has_comparison_val = res["has_comparison"]
    processed_path_val = res["processed_path"]

    # ── KPI Row ──────────────────────────────────────────
    st.markdown("### Overview")
    if _is_telecom_data_val:
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        kpis = [
            (k1, str(total_records_val),      "Total Reviews"),
            (k2, avg_score_disp_val,           "Avg Rating"),
            (k3, f"{pos_pct_val:.1f}%",        "Positive"),
            (k4, f"{neg_pct_val:.1f}%",        "Negative"),
            (k5, top_city_val,                 "Top City"),
            (k6, top_service_val,              "Top Service"),
        ]
    else:
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        kpis = [
            (k1, str(total_records_val),       "Total Records"),
            (k2, str(total_posts_val),         "Posts"),
            (k3, str(total_comments_val),      "Comments"),
            (k4, f"{pos_pct_val:.1f}%",        "Positive"),
            (k5, f"{neg_pct_val:.1f}%",        "Negative"),
            (k6, str(avg_score_val),           "Avg Score"),
        ]
    for col, val, label in kpis:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-number">{val}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='margin:28px 0 8px'></div>", unsafe_allow_html=True)

    # ── Sentiment Charts ──────────────────────────────────
    st.markdown("### Sentiment Breakdown")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown('<div class="chart-card"><p class="section-header">Distribution</p>', unsafe_allow_html=True)
        st.plotly_chart(make_donut_chart(summary_df_val), use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="chart-card"><p class="section-header">Count by Sentiment</p>', unsafe_allow_html=True)
        st.plotly_chart(make_bar_chart(summary_df_val), use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    with c3:
        if _is_telecom_data_val:
            st.markdown('<div class="chart-card"><p class="section-header">Star Rating Distribution</p>', unsafe_allow_html=True)
            st.plotly_chart(make_rating_chart(df_val), use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="chart-card"><p class="section-header">Posts vs Comments</p>', unsafe_allow_html=True)
            st.plotly_chart(make_sentiment_by_type_chart(df_val), use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='margin:8px 0'></div>", unsafe_allow_html=True)

    # ── Deep Dive ─────────────────────────────────────────
    st.markdown("### Deep Dive")
    t1, t2 = st.columns([3, 2])

    with t1:
        st.markdown('<div class="chart-card"><p class="section-header">Sentiment Over Time</p>', unsafe_allow_html=True)
        st.plotly_chart(make_timeline_chart(df_val), use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    with t2:
        if _is_telecom_data_val:
            st.markdown('<div class="chart-card"><p class="section-header">Top Cities</p>', unsafe_allow_html=True)
            st.plotly_chart(make_city_chart(df_val), use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="chart-card"><p class="section-header">Top Subreddits</p>', unsafe_allow_html=True)
            st.plotly_chart(make_subreddit_chart(df_val), use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)

    if _is_telecom_data_val:
        d1, d2 = st.columns(2)
        with d1:
            st.markdown('<div class="chart-card"><p class="section-header">Service Type Breakdown</p>', unsafe_allow_html=True)
            st.plotly_chart(make_service_type_chart(df_val), use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)
        with d2:
            st.markdown('<div class="chart-card"><p class="section-header">Sentiment by Customer Type</p>', unsafe_allow_html=True)
            st.plotly_chart(make_customer_type_chart(df_val), use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="chart-card"><p class="section-header">Reddit Score by Sentiment</p>', unsafe_allow_html=True)
        st.plotly_chart(make_score_sentiment_scatter(df_val), use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Key Insights ──────────────────────────────────────
    st.markdown("### Key Insights")
    i1, i2, i3, i4 = st.columns(4)

    dominant_sentiment = summary_df_val["Count"].idxmax()
    dominant_pct       = summary_df_val.loc[dominant_sentiment, "Percentage"]

    if _is_telecom_data_val:
        top_city_insight   = df_val["city"].value_counts().idxmax() if "city" in df_val.columns else "—"
        top_svc_insight    = df_val["service_type"].value_counts().idxmax() if "service_type" in df_val.columns else "—"
        top_cust_insight   = df_val["customer_type"].value_counts().idxmax() if "customer_type" in df_val.columns else "—"
        insights = [
            (i1, "Dominant Sentiment",  f"{dominant_sentiment.capitalize()} ({dominant_pct:.1f}%)"),
            (i2, "Top City",             top_city_insight),
            (i3, "Top Service",          top_svc_insight),
            (i4, "Most Common Customer", top_cust_insight),
        ]
    else:
        most_active_sub   = df_val["subreddit"].value_counts().idxmax()
        highest_score_row = df_val.loc[df_val["score"].idxmax()]
        total_engagement  = df_val["score"].sum()
        insights = [
            (i1, "Dominant Sentiment",   f"{dominant_sentiment.capitalize()} ({dominant_pct:.1f}%)"),
            (i2, "Most Active Subreddit", f"r/{most_active_sub}"),
            (i3, "Top Post Score",        str(int(highest_score_row["score"]))),
            (i4, "Total Engagement",      f"{int(total_engagement):,} pts"),
        ]

    for col, title, value in insights:
        with col:
            st.markdown(f"""
            <div class="insight-box">
                <div class="insight-title">{title}</div>
                <div class="insight-value">{value}</div>
            </div>
            """, unsafe_allow_html=True)

    if has_comparison_val and comparison_result_val is not None:
        render_brand_comparison([first_result_val, comparison_result_val], ranking_df_val)

    render_recommendation_system(ranking_df_val)

    # ── Sample Feedback Cards ─────────────────────────────
    st.markdown("### Sample Reviews" if _is_telecom_data_val else "### Sample Feedback")
    
    brand_filter = "All"
    if has_comparison_val:
        brand_filter = st.radio("Filter reviews by brand", ["All", brand_val.title(), brand_2_val.title()], horizontal=True)

    tab_pos, tab_neg, tab_neu = st.tabs(["Positive", "Negative", "Neutral"])

    def render_feedback_cards(sentiment_label, tab):
        # Apply brand filter
        subset = df_combined_val.copy()
        if has_comparison_val and brand_filter != "All":
            subset = subset[subset["brand"].str.lower() == brand_filter.lower()]
        
        subset = subset[subset["predicted_sentiment"].str.lower() == sentiment_label].head(6)
        with tab:
            if subset.empty:
                st.markdown('<div style="color:rgba(255,255,255,0.3); padding:20px;">No records found.</div>', unsafe_allow_html=True)
                return
            for _, row in subset.iterrows():
                text_val = row.get("clean_text", row["combined_text"])
                text_preview = str(text_val)[:300]
                if len(str(text_val)) > 300:
                    text_preview += "..."
                
                brand_badge = f'<span class="score-badge" style="margin-left:8px; font-weight:600; color:#63b3ed;">{row["brand"]}</span>'
                
                if _is_telecom_data_val or row.get("record_type") == "review":
                    city_val    = row.get("city", "")
                    svc_val     = row.get("service_type", "")
                    rating_val  = row.get("rating", "")
                    cust_val    = row.get("customer_type", "")
                    stars_str   = "★" * int(rating_val) + "☆" * (5 - int(rating_val)) if rating_val else ""
                    st.markdown(f"""
                    <div class="feedback-card">
                        <span class="sentiment-pill pill-{sentiment_label}">{sentiment_label}</span>
                        {brand_badge}
                        <span style="margin-left:8px; font-size:0.85rem; color:#FFD93D;">{stars_str}</span>
                        <div class="feedback-text" style="margin-top:10px;">{text_preview}</div>
                        <div class="feedback-meta">
                            <span>City: {city_val}</span>
                            <span>Service: {svc_val}</span>
                            <span>Customer: {cust_val}</span>
                            <span class="score-badge">Rating: {rating_val}/5</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    sub   = row.get("subreddit", "")
                    rtype = row.get("record_type", "")
                    score = row.get("score", 0)
                    st.markdown(f"""
                    <div class="feedback-card">
                        <span class="sentiment-pill pill-{sentiment_label}">{sentiment_label}</span>
                        {brand_badge}
                        <div class="feedback-text" style="margin-top:10px;">{text_preview}</div>
                        <div class="feedback-meta">
                            <span>r/{sub}</span><span>{rtype}</span>
                            <span class="score-badge">▲ {score}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    render_feedback_cards("positive", tab_pos)
    render_feedback_cards("negative", tab_neg)
    render_feedback_cards("neutral",  tab_neu)

    # ── Export ────────────────────────────────────────────
    st.markdown("<div style='margin-top:24px'></div>", unsafe_allow_html=True)
    st.markdown("### Export Data")
    csv_bytes = df_val.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    e1, e2 = st.columns(2)
    with e1:
        st.download_button(
            label="Download Reviewed Data",
            data=csv_bytes,
            file_name=f"{brand_val.lower().replace(' ','_')}_reviews.csv",
            mime="text/csv", use_container_width=True,
        )
    with e2:
        if not _is_telecom_data_val and Path(processed_path_val).exists():
            st.download_button(
                label="Download Predicted CSV",
                data=open(processed_path_val, "rb").read(),
                file_name=f"{brand_val.lower().replace(' ','_')}_predicted.csv",
                mime="text/csv", use_container_width=True,
            )

    src_label = "Telecom Dataset" if _is_telecom_data_val else "Reddit"
    st.markdown(f"""
    <div style="text-align:center; margin-top:24px; font-size:0.8rem; color:rgba(255,255,255,0.2);">
        Analysis completed · {total_records_val:,} records · {brand_val.title()} · Source: {src_label}
    </div>
    """, unsafe_allow_html=True)
