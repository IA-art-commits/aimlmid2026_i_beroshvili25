from typing import Tuple, Dict
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Global constants for feature and label columns (copied from 2h_p-wX9zi1R)
FEATURE_COLUMNS = ["words", "links", "capital_words", "spam_word_count"]
LABEL_COLUMN = "is_spam"

def load_dataset(csv_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Load CSV and return X (features) and y (labels).
    Includes dummy data generation if the file doesn't exist for demonstration.
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Warning: {csv_path} not found. Generating dummy data for demonstration.")
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
        df = dummy_data

    required = set(FEATURE_COLUMNS + [LABEL_COLUMN])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    X = df[FEATURE_COLUMNS]
    y = df[LABEL_COLUMN]
    return X, y

def preprocess_data(X: pd.DataFrame, y: pd.Series,
                test_size: float = 0.30,
                random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits data into training and testing sets.
    This replaces the original 'preprocessing.py' functionality.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test

def train_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Trains a Logistic Regression model.
    This replaces the original 'model.py' functionality.
    """
    model = LogisticRegression(max_iter=2000, solver="lbfgs")
    model.fit(X_train, y_train)
    return model

def evaluate_model_metrics(model: LogisticRegression,
                   X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes predictions and probabilities for evaluation.
    Aligns with parts of the original 'evaluation.py' functionality.
    """
    y_pred_test = model.predict(X_test)
    probs_test = model.predict_proba(X_test)[:, 1] # P(spam)
    return y_pred_test, probs_test

def plot_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray):
    """Plots the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Non-Spam', 'Spam'], yticklabels=['Non-Spam', 'Spam'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

def plot_roc_curve(y_true: pd.Series, y_prob: np.ndarray):
    """Plots the ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    """
    Main function that orchestrates the execution of the program.
    """

    print("=" * 60)
    print("EMAIL SPAM CLASSIFICATION SYSTEM")
    print("=" * 60)

    # 1. Load dataset
    print("\n[1] Loading dataset...")
    # The 'emails.csv' might not exist, so the load_dataset function now handles dummy data if not found.
    X, y = load_dataset("data/emails.csv")

    # 2. Preprocess data
    print("[2] Preprocessing data...")
    X_train, X_test, y_train, y_test = preprocess_data(X, y)

    # 3. Train model
    print("[3] Training Logistic Regression model...")
    model = train_logistic_regression(X_train, y_train)

    # 4. Evaluate model
    print("[4] Evaluating model performance...")
    y_pred, y_prob = evaluate_model_metrics(model, X_test, y_test)

    # 5. Visualizations
    print("[5] Generating visualizations...")
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_prob)

    print("\n✔ Program finished successfully!")


# Ensures the program runs only when executed directly
if __name__ == "__main__":
    main()
