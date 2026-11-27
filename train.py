from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "dataset" / "emails.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA)
required = {"label", "text"}
if not required.issubset(df.columns):
    raise ValueError("emails.csv must contain 'label' and 'text' columns.")
df = df.dropna(subset=["label", "text"])
df["label"] = df["label"].astype(str).str.lower().str.strip()
df["text"] = df["text"].astype(str).str.strip()
df = df[df["label"].isin(["spam", "ham"])]
if df["label"].nunique() != 2 or len(df) < 10:
    raise ValueError("Provide at least 10 labeled rows containing both spam and ham.")

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.25, random_state=42, stratify=df["label"]
)

candidates = {
    "Multinomial Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Linear SVM": LinearSVC(class_weight="balanced")
}
results = {}
best_name, best_model, best_f1 = None, None, -1

for name, estimator in candidates.items():
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2), max_features=20000)),
        ("classifier", estimator)
    ])
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    scores = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, pos_label="spam", zero_division=0),
        "recall": recall_score(y_test, predictions, pos_label="spam", zero_division=0),
        "f1": f1_score(y_test, predictions, pos_label="spam", zero_division=0),
        "confusion_matrix_labels": ["ham", "spam"],
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=["ham", "spam"]).tolist()
    }
    results[name] = scores
    print(f"\n{name}\n{classification_report(y_test, predictions, zero_division=0)}")
    if scores["f1"] > best_f1:
        best_name, best_model, best_f1 = name, pipeline, scores["f1"]

joblib.dump(best_model, MODEL_DIR / "spam_classifier.pkl")
with open(MODEL_DIR / "metrics.json", "w", encoding="utf-8") as f:
    json.dump({"best_model": best_name, "models": results}, f, indent=2)
print(f"\nBest model: {best_name} (spam F1: {best_f1:.3f})")
print(f"Saved model to: {MODEL_DIR / 'spam_classifier.pkl'}")
