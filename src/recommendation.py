from __future__ import annotations

from pathlib import Path

import pandas as pd


SENTIMENT_WEIGHTS = {
    "positive": 1.0,
    "neutral": 0.35,
    "negative": -0.65,
}


def _brand_from_path(path: Path) -> str:
    name = path.stem
    if name.endswith("_predicted"):
        name = name[:-10]
    return name.replace("_", " ").title()


def build_company_profile(df: pd.DataFrame, brand: str | None = None, source: str = "") -> dict:
    if df.empty or "predicted_sentiment" not in df.columns:
        return {}

    data = df.copy()
    data["predicted_sentiment"] = data["predicted_sentiment"].fillna("neutral").astype(str).str.lower()
    data["score"] = pd.to_numeric(data.get("score", 0), errors="coerce").fillna(0)
    data["num_comments"] = pd.to_numeric(data.get("num_comments", 0), errors="coerce").fillna(0)

    if not brand:
        if "brand" in data.columns and data["brand"].notna().any():
            brand = str(data["brand"].dropna().iloc[0])
        else:
            brand = "Unknown"

    total_records = len(data)
    counts = data["predicted_sentiment"].value_counts()

    positive_count = int(counts.get("positive", 0))
    neutral_count = int(counts.get("neutral", 0))
    negative_count = int(counts.get("negative", 0))

    positive_pct = (positive_count / total_records) * 100 if total_records else 0
    neutral_pct = (neutral_count / total_records) * 100 if total_records else 0
    negative_pct = (negative_count / total_records) * 100 if total_records else 0

    sentiment_index = (
        positive_pct * SENTIMENT_WEIGHTS["positive"]
        + neutral_pct * SENTIMENT_WEIGHTS["neutral"]
        + negative_pct * SENTIMENT_WEIGHTS["negative"]
    )
    sentiment_index = max(0, min(100, sentiment_index))

    engagement = float(data["score"].clip(lower=0).sum() + (data["num_comments"].clip(lower=0).sum() * 0.25))
    confidence_index = min(total_records / 50, 1) * 100

    return {
        "brand": str(brand).title(),
        "source": source,
        "total_records": int(total_records),
        "positive_count": positive_count,
        "neutral_count": neutral_count,
        "negative_count": negative_count,
        "positive_pct": round(positive_pct, 2),
        "neutral_pct": round(neutral_pct, 2),
        "negative_pct": round(negative_pct, 2),
        "avg_score": round(float(data["score"].mean()), 2),
        "total_engagement": round(engagement, 2),
        "sentiment_index": round(sentiment_index, 2),
        "confidence_index": round(confidence_index, 2),
    }


def rank_companies(profiles: list[dict]) -> pd.DataFrame:
    profiles = [profile for profile in profiles if profile]
    if not profiles:
        return pd.DataFrame()

    ranking = pd.DataFrame(profiles)
    max_engagement = max(float(ranking["total_engagement"].max()), 1.0)
    ranking["engagement_index"] = (ranking["total_engagement"] / max_engagement * 100).clip(0, 100)

    ranking["recommendation_score"] = (
        ranking["sentiment_index"] * 0.65
        + ranking["engagement_index"] * 0.20
        + ranking["confidence_index"] * 0.15
    ).round(2)

    ranking = ranking.sort_values(
        ["recommendation_score", "positive_pct", "negative_pct", "total_records"],
        ascending=[False, False, True, False],
    ).reset_index(drop=True)

    ranking.insert(0, "rank", range(1, len(ranking) + 1))
    ranking["recommendation"] = ranking["rank"].apply(lambda rank: "Best company" if rank == 1 else "Compare")
    ranking["reason"] = ranking.apply(
        lambda row: (
            f"{row['positive_pct']:.1f}% positive, "
            f"{row['negative_pct']:.1f}% negative, "
            f"{int(row['total_records'])} records"
        ),
        axis=1,
    )
    return ranking


def load_processed_company_profiles(data_dir: str | Path = "data/processed") -> pd.DataFrame:
    data_path = Path(data_dir)
    profiles = []

    for csv_path in sorted(data_path.glob("*_predicted.csv")):
        if csv_path.stem == "reddit_predicted":
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue

        brand = _brand_from_path(csv_path)
        profiles.append(build_company_profile(df, brand=brand, source=str(csv_path)))

    return rank_companies(profiles)


def build_ml_flow_summary(raw_rows: int, predicted_rows: int, ranked_companies: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"step": "1. Data Collection", "output": f"{raw_rows} Reddit records", "ml_role": "Input data"},
            {"step": "2. Text Cleaning", "output": "clean_text", "ml_role": "Preprocessing"},
            {"step": "3. Vectorization", "output": "TF-IDF features", "ml_role": "Feature extraction"},
            {"step": "4. SVM Prediction", "output": f"{predicted_rows} sentiment labels", "ml_role": "Classification"},
            {"step": "5. Recommendation", "output": f"{ranked_companies} ranked companies", "ml_role": "Decision layer"},
        ]
    )
