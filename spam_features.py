import re
from typing import Tuple, Dict, List # Added List for SPAM_KEYWORDS
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Constants from model_training.py
FEATURE_COLUMNS = ["words", "links", "capital_words", "spam_word_count"]
LABEL_COLUMN = "is_spam"

# SPAM_KEYWORDS and URL_REGEX from spam_features.py
SPAM_KEYWORDS: List[str] = [
    "free", "winner", "win", "prize", "urgent", "limited", "offer",
    "bonus", "credit", "loan", "money", "cash", "discount", "click",
    "verify", "account", "password", "login", "confirm", "gift",
    "claim", "congratulations", "selected", "exclusive", "guaranteed"
]

URL_REGEX = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)

# Functions from model_training.py
def load_dataset(csv_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Load CSV and return X (features) and y (labels)."""
    df = pd.read_csv(csv_path)

    required = set(FEATURE_COLUMNS + [LABEL_COLUMN])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    X = df[FEATURE_COLUMNS]
    y = df[LABEL_COLUMN]
    return X, y

def train_model(X: pd.DataFrame, y: pd.Series,
                test_size: float = 0.30,
                random_state: int = 42):
    """
    Split into 70/30 (train/test), train logistic regression, return model + split data.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
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
    Returns a dict with keys:
      words, links, capital_words, spam_word_count
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

# Function from visualize.py
def plot_confusion_matrix_heatmap(cm: np.ndarray, out_path: str, show: bool = False) -> None:
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix Heatmap (Test Set)")
    plt.xlabel("Predicted class")
    plt.ylabel("Actual class")
    plt.xticks([0, 1], ["Legitimate (0)", "Spam (1)"])
    plt.yticks([0, 1], ["Legitimate (0)", "Spam (1)"])
    cbar = plt.colorbar()
    cbar.set_label("Count")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()

# Original main function, adapted to use consolidated functions and dummy data
def main():
    """
    Main function that orchestrates the execution of the program.
    """

    print("=" * 60)
    print("EMAIL SPAM CLASSIFICATION SYSTEM")
    print("=" * 60)

    # 1. Generate dummy dataset
    print("\n[1] Generating dummy dataset...")
    np.random.seed(42)
    n_samples = 200
    dummy_data = pd.DataFrame({
        "words": np.random.randint(5, 500, n_samples),
        "links": np.random.randint(0, 10, n_samples),
        "capital_words": np.random.randint(0, 50, n_samples),
        "spam_word_count": np.random.randint(0, 20, n_samples),
        "is_spam": np.random.randint(0, 2, n_samples)
    })

    # Introduce some correlation for spam to make the plot more meaningful
    dummy_data.loc[dummy_data["is_spam"] == 1, "spam_word_count"] = np.random.randint(10, 30, (dummy_data["is_spam"] == 1).sum())
    dummy_data.loc[dummy_data["is_spam"] == 1, "words"] = np.random.randint(100, 700, (dummy_data["is_spam"] == 1).sum())

    X = dummy_data[FEATURE_COLUMNS]
    y = dummy_data[LABEL_COLUMN]

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
    plot_confusion_matrix_heatmap(cm, out_path="confusion_matrix.png", show=True)
    plot_probability_distribution(probs_test, y_test)

    print("\n✔ Program finished successfully!")


# Ensures the program runs only when executed directly
if __name__ == "__main__":
    main()
