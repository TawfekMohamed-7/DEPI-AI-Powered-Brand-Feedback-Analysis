import re
import joblib
import pandas as pd


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


def main():
    df = pd.read_csv("data/raw/reddit_raw.csv")

    print("Original shape:", df.shape)

    model = joblib.load("models/svm_model.pkl")
    vectorizer = joblib.load("models/vectorizer.pkl")

    # Preprocess raw data with vectorizer vocabulary awareness
    df = preprocess_data(df, vectorizer)

    X = vectorizer.transform(df["clean_text"])
    df["predicted_sentiment"] = model.predict(X)

    print("\nSentiment Counts:")
    print(df["predicted_sentiment"].value_counts())

    print("\nSentiment Percentages:")
    print((df["predicted_sentiment"].value_counts(normalize=True) * 100).round(2))

    print("\nSample Results:")
    print(df[["record_type", "query", "combined_text", "predicted_sentiment"]].head(10))

    df.to_csv("data/processed/reddit_predicted.csv", index=False, encoding="utf-8-sig")
    print("\nSaved to data/processed/reddit_predicted.csv")


if __name__ == "__main__":
    main()