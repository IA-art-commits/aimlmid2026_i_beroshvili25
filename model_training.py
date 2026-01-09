from typing import Tuple, Dict
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt # New import for plotting
import seaborn as sns # New import for enhanced plotting
import numpy as np # New import for dummy data generation

FEATURE_COLUMNS = ["words", "links", "capital_words", "spam_word_count"]
LABEL_COLUMN = "is_spam"

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

# Example usage when the script is run directly
if __name__ == "__main__":
    print("Running example for model_training.py...")
    # Create dummy data for demonstration
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


    X_dummy = dummy_data[FEATURE_COLUMNS]
    y_dummy = dummy_data[LABEL_COLUMN]

    # Train model
    model_dummy, X_train_dummy, X_test_dummy, y_train_dummy, y_test_dummy = train_model(X_dummy, y_dummy)

    # Evaluate model
    evaluation_results_dummy = evaluate_model(model_dummy, X_train_dummy, y_train_dummy, X_test_dummy, y_test_dummy)

    # Plot probability distribution
    plot_probability_distribution(evaluation_results_dummy["probs_test"], y_test_dummy)
    print("Example execution complete. A plot should have been generated.")
