import re
from typing import Tuple, Dict, List
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# os import is no longer needed as we are embedding the data directly

# Constants from model_training.py
FEATURE_COLUMNS = ["words", "links", "capital_words", "spam_word_count"]
LABEL_COLUMN = "is_spam"

# SPAM_KEYWORDS and URL_REGEX from spam_features.py
SPAM_KEYWORDS: List[str] = [
    "free", "winner", "win", "prize", "urgent", "limited", "offer",
    "bonus", "credit", "loan", "money", "cash", "discount", "click",
    "verify", "account", "password", "login", "confirm", "gift",
    "claim", "congratulations", "selected", "exclusive", "guaranteed",
    "iphone", "here", "won", "now", "collect"
]

URL_REGEX = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)

# Functions from model_training.py
# The load_dataset function was removed as its logic is now embedded into main.

def train_model(X: pd.DataFrame, y: pd.Series,
                test_size: float = 0.30,
                random_state: int = 42):
    """
    Split into 70/30 (train/test), train logistic regression,
    return model + split data.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    model = LogisticRegression(max_iter=2000, solver="lbfgs")
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test

def evaluate_model(model: LogisticRegression,
                   X_train: pd.DataFrame, y_train: pd.Series,
                   X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
    """
    Compute:
      - confusion matrix (test)
      - accuracy on train and test
      - classification report (precision/recall/f1)
      - test probabilities (for probability distribution plot)
    """
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    cm = confusion_matrix(y_test, y_pred_test)
    acc_test = accuracy_score(y_test, y_pred_test)
    acc_train = accuracy_score(y_train, y_pred_train)

    report = classification_report(y_test, y_pred_test, digits=4)

    probs_test = model.predict_proba(X_test)[:, 1]  # P(spam)

    return {
        "confusion_matrix": cm,
        "accuracy_test": acc_test,
        "accuracy_train": acc_train,
        "classification_report": report,
        "probs_test": probs_test,
        "y_pred_test": y_pred_test
    }

def plot_probability_distribution(probs_test: np.ndarray, y_test: pd.Series):
    """
    Plots the distribution of predicted probabilities for spam and non-spam emails.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(probs_test[y_test == 0], color='blue', label='Non-Spam', kde=True, stat='density', alpha=0.5, binwidth=0.05)
    sns.histplot(probs_test[y_test == 1], color='red', label='Spam', kde=True, stat='density', alpha=0.5, binwidth=0.05)
    plt.title('Distribution of Predicted Spam Probabilities')
    plt.xlabel('Predicted Probability of Spam')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.show()

# Function from spam_features.py
def extract_features(email_text: str) -> Dict[str, int]:
    """
    Convert a raw email string into the 4 numeric features used by the model.
    Returns a dict with keys:n      words, links, capital_words, spam_word_count
    """
    # word tokens (letters + optional apostrophes)
    tokens = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", email_text)
    words = len(tokens)

    # URL count
    links = len(URL_REGEX.findall(email_text))

    # count ALL-CAPS words (length >= 2)
    capital_words = sum(1 for t in tokens if len(t) >= 2 and t.isupper())

    # count occurrences of spam keywords (case-insensitive, whole-word)
    lower_text = email_text.lower()
    spam_word_count = 0
    for kw in SPAM_KEYWORDS:
        spam_word_count += len(re.findall(rf"\b{re.escape(kw)}\b", lower_text))

    return {
        "words": words,
        "links": links,
        "capital_words": capital_words,
        "spam_word_count": spam_word_count,
    }

# Functions from visualize.py
def plot_class_distribution(y: pd.Series, out_path: str, show: bool = False) -> None:
    counts = y.value_counts().sort_index()
    labels = ["Legitimate (0)", "Spam (1)"]

    plt.figure()
    plt.bar(labels, [counts.get(0, 0), counts.get(1, 0)], label="Email count")
    plt.title("Class Distribution in Dataset")
    plt.xlabel("Class")
    plt.ylabel("Number of emails")
    plt.legend()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()

def plot_confusion_matrix_heatmap(cm: np.ndarray, out_path: str, show: bool = False) -> None:
    plt.figure(figsize=(8, 6)) # Increased figure size for better readability
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=["Legitimate (0)", "Spam (1)"],
                yticklabels=["Legitimate (0)", "Spam (1)"])
    plt.title("Confusion Matrix Heatmap (Test Set)")
    plt.xlabel("Predicted class")
    plt.ylabel("Actual class")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()

def plot_coefficients_bar(feature_names, coefficients, out_path: str, show: bool = False) -> None:
    plt.figure(figsize=(10, 6))
    plt.bar(feature_names, coefficients, label="Coefficient")
    plt.title("Logistic Regression Coefficients (Feature Importance)")
    plt.xlabel("Feature")
    plt.ylabel("Coefficient value")
    plt.legend()
    plt.grid(axis='y')
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()

# Original main function, adapted to use consolidated functions and embedded data
def main():
    """
    Main function that orchestrates the execution of the program.
    """

    print("=" * 60)
    print("EMAIL SPAM CLASSIFICATION SYSTEM")
    print("=" * 60)

    # 1. Embedded dataset
    print("\n[1] Using embedded dataset...")
    data = """
words,links,capital_words,spam_word_count,is_spam
650,6,17,9,1
167,3,4,4,1
240,1,1,1,0
204,1,0,6,0
83,4,2,1,0
693,9,26,2,1
241,1,5,1,0
286,3,2,2,0
394,3,4,3,1
129,3,14,3,1
253,0,0,2,0
184,9,6,3,1
155,2,25,8,1
926,0,20,8,1
951,9,2,4,1
199,2,8,2,0
116,2,3,1,0
344,1,0,0,0
262,0,3,1,0
70,2,3,0,0
623,4,25,3,1
112,7,6,5,1
363,5,11,5,1
128,2,2,2,0
153,2,10,5,0
514,9,7,7,1
100,3,27,10,1
141,7,19,3,1
742,6,23,6,1
307,9,17,4,1
470,3,5,5,1
577,4,27,9,1
281,1,2,1,0
145,0,0,1,0
33,2,1,1,0
200,1,18,6,1
296,2,1,2,0
57,3,24,6,1
231,0,4,2,0
188,1,1,1,0
325,1,0,2,0
906,7,6,1,1
610,7,4,5,1
633,8,5,10,1
98,2,0,2,0
812,7,11,6,1
430,8,18,6,1
723,6,18,3,1
426,1,3,1,0
919,0,16,8,1
56,2,0,0,0
487,2,0,0,0
725,8,3,9,1
170,2,9,10,1
606,10,17,7,1
174,5,6,2,0
714,3,5,4,1
322,9,2,10,1
217,1,1,2,0
908,3,10,5,1
242,0,1,1,0
126,0,0,1,0
273,2,3,2,0
421,2,3,0,0
139,6,3,7,1
32,4,3,2,0
935,5,28,6,1
151,2,3,6,0
281,2,4,6,1
388,1,9,8,1
937,2,10,7,1
889,9,11,6,1
456,3,11,10,1
336,2,9,9,1
159,2,1,2,0
324,0,8,3,0
721,3,27,3,1
628,2,15,4,1
343,5,0,4,1
259,0,3,1,0
283,0,2,0,0
777,5,0,7,1
369,0,3,2,0
503,5,20,6,1
263,1,3,1,0
114,0,0,0,0
933,6,14,9,1
869,2,18,1,1
880,1,17,7,1
652,8,3,6,1
498,7,10,5,1
717,1,2,2,1
244,2,1,1,0
474,8,23,6,1
60,1,0,0,0
137,0,3,1,0
146,1,3,2,0
198,2,9,2,0
97,7,4,6,1
601,7,3,5,1
896,3,17,8,1
231,8,6,7,1
64,1,5,2,0
149,2,8,0,0
278,5,2,0,0
395,4,6,10,1
300,1,3,0,0
167,2,17,6,1
547,1,14,3,1
120,2,25,9,1
254,8,5,5,1
52,10,2,2,1
806,7,19,3,1
515,3,14,10,1
415,6,29,6,1
299,1,2,2,0
292,2,2,2,0
35,2,2,5,0
776,10,9,3,1
979,5,1,7,1
273,1,1,2,0
198,5,27,0,1
296,1,0,1,0
825,6,21,8,1
150,2,3,1,0
175,2,2,2,0
280,1,0,1,0
130,0,4,2,0
652,3,26,1,1
114,7,11,3,1
49,2,3,2,0
371,3,8,6,0
208,1,2,2,0
62,0,1,1,0
205,1,5,6,0
533,4,5,5,1
795,4,9,6,1
504,8,6,4,1
167,5,1,1,0
74,3,12,4,1
124,1,2,1,0
237,1,9,2,0
520,3,24,0,1
58,2,2,6,0
243,8,30,7,1
212,9,6,8,1
287,2,4,0,0
849,2,8,1,1
28,4,1,1,0
713,5,5,8,1
282,1,1,0,0
129,2,0,1,0
204,2,3,0,0
438,1,1,1,0
172,9,2,6,1
281,2,1,1,0
113,0,6,6,0
291,1,3,1,0
125,8,28,10,1
"""

    # Use io.StringIO to read the string data as if it were a file
    from io import StringIO
    df = pd.read_csv(StringIO(data))

    required = set(FEATURE_COLUMNS + [LABEL_COLUMN])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    X = df[FEATURE_COLUMNS]
    y = df[LABEL_COLUMN]

    # 2. Split data and train model
    print("[2] Splitting data and training Logistic Regression model...")
    model, X_train, X_test, y_train, y_test = train_model(X, y)

    # 3. Evaluate model
    print("[3] Evaluating model performance...")
    evaluation_results = evaluate_model(model, X_train, y_train, X_test, y_test)
    cm = evaluation_results["confusion_matrix"]
    probs_test = evaluation_results["probs_test"]

    print(f"\nTest Accuracy: {evaluation_results['accuracy_test']:.4f}")
    print(f"Classification Report:\n{evaluation_results['classification_report']}")

    # 4. Generate visualizations
    print("[4] Generating visualizations...")
    plot_class_distribution(y, out_path="class_distribution.png", show=True) # New: Class Distribution Plot
    plot_confusion_matrix_heatmap(cm, out_path="confusion_matrix.png", show=True)
    plot_probability_distribution(probs_test, y_test)

    # Add coefficients bar plot
    if hasattr(model, 'coef_') and len(model.coef_[0]) == len(FEATURE_COLUMNS):
        plot_coefficients_bar(FEATURE_COLUMNS, model.coef_[0], out_path="coefficients_bar.png", show=True)
    else:
        print("Could not plot coefficients bar: Model coefficients or feature columns mismatch.")

    # 5. Classify custom emails
    print("\n[5] Classifying custom email texts...")
    example_legitimate_email = "Hello, I hope this email finds you well. I'm writing to follow up on our last conversation."
    example_spam_email = "URGENT! You are a WINNER! Congratulations! You have won a FREE iPhone PRIZE! Click HERE NOW to claim your CASH! This is a limited time offer. Collect your bonus money! Verify your account at www.malicious-offer.xyz"

    custom_emails = {
        "Legitimate Email": example_legitimate_email,
        "Spam Email": example_spam_email
    }

    for email_type, email_text in custom_emails.items():
        features = extract_features(email_text)
        features_df = pd.DataFrame([features]) # Convert to DataFrame for prediction

        predicted_class = model.predict(features_df)[0]
        predicted_prob = model.predict_proba(features_df)[0][1] # Probability of being spam (class 1)

        print(f"\n--- {email_type} ---")
        print(f"Text: '{email_text[:70]}...' ")
        print(f"Extracted Features: {features}")
        print(f"Predicted Class: {'Spam' if predicted_class == 1 else 'Legitimate'} ({predicted_class})")
        print(f"Predicted Spam Probability: {predicted_prob:.4f}")

    print("\n✔ Program finished successfully!")


# Ensures the program runs only when executed directly
if __name__ == "__main__":
    main()
