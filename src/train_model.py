import re
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
from pathlib import Path

# Setup paths
ROOT_DIR = Path(__file__).resolve().parents[1]
LABELED_PATH = ROOT_DIR / "data" / "labeled" / "Twitter_Data.csv"
MODEL_SAVE_PATH = ROOT_DIR / "models" / "svm_model.pkl"
VEC_SAVE_PATH = ROOT_DIR / "models" / "vectorizer.pkl"

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

def tokenize(text: str) -> list[str]:
    return text.split()

def lemmatize_word(word: str) -> str:
    stemmed = word
    if len(word) > 4:
        if word.endswith("sses"):
            stemmed = word[:-2]
        elif word.endswith("ies"):
            stemmed = word[:-3] + "y"
        elif word.endswith("s") and not word.endswith("us") and not word.endswith("is") and not word.endswith("as"):
            stemmed = word[:-1]
    
    # Arabic simple stemming
    if len(word) > 4 and word.startswith("ال"):
        stemmed = word[2:]
        
    return stemmed

def preprocess_text(text):
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    lemmatized = [lemmatize_word(t) for t in tokens]
    return " ".join(lemmatized)

def main():
    print("Loading labeled dataset...")
    df = pd.read_csv(LABELED_PATH)
    print(f"Original shape: {df.shape}")
    
    # Drop rows with missing values
    df = df.dropna(subset=["clean_text", "category"]).copy()
    
    # Map category labels to positive, negative, neutral
    label_map = {-1: "negative", 0: "neutral", 1: "positive", -1.0: "negative", 0.0: "neutral", 1.0: "positive"}
    df["label"] = df["category"].map(label_map)
    df = df.dropna(subset=["label"]).copy()
    
    print("Preprocessing texts (cleaning, tokenizing, lemmatizing)...")
    df["processed_text"] = df["clean_text"].apply(preprocess_text)
    df = df[df["processed_text"].str.strip() != ""].copy()
    print(f"Preprocessed shape: {df.shape}")
    
    X = df["processed_text"]
    y = df["label"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Vectorizing text features using TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    print("Training SVM classifier (LinearSVC)...")
    svm_model = LinearSVC(random_state=42)
    svm_model.fit(X_train_vec, y_train)
    
    # Evaluate
    y_pred = svm_model.predict(X_test_vec)
    print("Linear SVM Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))
    
    # Save
    print(f"Saving SVM model to {MODEL_SAVE_PATH}...")
    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(svm_model, MODEL_SAVE_PATH)
    
    print(f"Saving Vectorizer to {VEC_SAVE_PATH}...")
    joblib.dump(vectorizer, VEC_SAVE_PATH)
    print("Training pipeline completed successfully!")

if __name__ == "__main__":
    main()
