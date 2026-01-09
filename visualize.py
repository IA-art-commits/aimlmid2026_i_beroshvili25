import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns # Added for potential future use or consistency with main.py

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

def plot_coefficients_bar(feature_names, coefficients, out_path: str, show: bool = False) -> None:
    plt.figure()
    plt.bar(feature_names, coefficients, label="Coefficient")
    plt.title("Logistic Regression Coefficients (Feature Importance)")
    plt.xlabel("Feature")
    plt.ylabel("Coefficient value")
    plt.legend()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()

# Added main execution block for demonstration and testing
if __name__ == "__main__":
    print("Running visualize.py demonstration...")
    np.random.seed(42)

    # Dummy data for plot_class_distribution
    dummy_y = pd.Series(np.random.randint(0, 2, 100))
    plot_class_distribution(dummy_y, out_path="class_distribution_dummy.png", show=True)

    # Dummy data for plot_confusion_matrix_heatmap
    dummy_cm = np.array([[70, 10], [5, 40]])
    plot_confusion_matrix_heatmap(dummy_cm, out_path="confusion_matrix_heatmap_dummy.png", show=True)

    # Dummy data for plot_coefficients_bar
    dummy_feature_names = ["words", "links", "capital_words", "spam_word_count"]
    dummy_coefficients = np.random.rand(len(dummy_feature_names)) * 2 - 1 # values between -1 and 1
    plot_coefficients_bar(dummy_feature_names, dummy_coefficients, out_path="coefficients_bar_dummy.png", show=True)

    print("visualize.py demonstration complete. Three plots should have been generated and displayed.")

