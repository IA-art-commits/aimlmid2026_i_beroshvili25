Github Link
https://github.com/IA-art-commits/aimlmid2026_i_beroshvili25.git


# Project Overview


This project encompasses two main tasks: calculating the Pearson correlation coefficient for a given dataset and building an email spam classification system using a Logistic Regression model.

## Task 1: Pearson Correlation Analysis

We calculated the Pearson correlation coefficient for a small set of data points to understand the linear relationship between two variables, `xi` and `yi`.

### Given Data Points
| i | xi    | yi    |
|---|-------|-------|
| 1 | -9.56 | 1.28  |
| 2 | -6.13 | 2.04  |
| 3 | -4.93 | -1.00 |
| 4 | -1.27 | 1.04  |
| 5 | 1.16  | -1.80 |
| 6 | 2.90  | 1.32  |
| 7 | 4.90  | -3.20 |
| 8 | 7.24  | -2.20 |

### Method
The Pearson correlation coefficient (r) was calculated using NumPy's `corrcoef` function, which measures the linear correlation between two sets of data. A scatter plot was also generated to visualize this relationship.

### Calculated Pearson Correlation Coefficient
**r = -0.6581**

This value indicates a moderately strong negative linear relationship between `xi` and `yi`.

## Task 2: Email Spam Classification

This task involved building and evaluating a Logistic Regression model for email spam classification, demonstrating a full machine learning workflow from data preparation to evaluation and custom classification.

### Data
The dataset used for training and testing the model was embedded directly within the `main.py` script as a multiline string. This dataset included features such as `words`, `links`, `capital_words`, `spam_word_count`, and the `is_spam` label.

### Model
A Logistic Regression model (`sklearn.linear_model.LogisticRegression`) was used for binary classification. The model was trained with `max_iter=2000` and `solver="lbfgs"`.

### Evaluation Metrics
The model's performance was evaluated using test accuracy and a detailed classification report on a held-out test set (30% of the data).

*   **Test Accuracy**: 0.9583
*   **Classification Report**:
    ```
                  precision    recall  f1-score   support

               0     0.9565    0.9565    0.9565        23
               1     0.9600    0.9600    0.9600        25

        accuracy                         0.9583        48
       macro avg     0.9583    0.9583    0.9583        48
    weighted avg     0.9583    0.9583    0.9583        48
    ```

### Generated Visualizations
The following plots were generated to provide insights into the model's performance and feature importance:
*   **Class Distribution Bar Chart**: Reveals that the dataset is relatively balanced, with 72 instances of legitimate emails and 78 instances of spam emails. This even distribution across classes is beneficial for model training, as it reduces the risk of bias towards a majority class.
*   **Confusion Matrix Heatmap**: Displays true positives, true negatives, false positives, and false negatives, saved as `confusion_matrix.png`.
*   **Distribution of Predicted Spam Probabilities**: Shows the probability scores assigned to emails for being spam, illustrating the model's confidence for each class.
*   **Logistic Regression Coefficients Bar Plot**: Illustrates the importance of each feature in the model's decision-making process, saved as `coefficients_bar.png`.

### Custom Email Classification with Feature Engineering Improvements

The `main.py` script included functionality to classify custom email texts using the trained model.

Initially, an example spam email (`"URGENT! You are a WINNER! Congratulations! You have won a FREE iPhone PRIZE! Click HERE NOW to claim your CASH! This is a limited time offer. Collect your bonus money! Verify your account at www.malicious-offer.xyz"`) was misclassified as 'Legitimate' due to an insufficient `SPAM_KEYWORDS` list. The `spam_word_count` feature for this email was low, resulting in a low predicted spam probability (around 0.05).

To address this, the `SPAM_KEYWORDS` list in the `extract_features` function was expanded in two iterations:
1.  **First Expansion**: Added `"iphone"` and `"here"`. This increased the `spam_word_count` and raised the predicted probability, but the email was still classified as 'Legitimate'.
2.  **Second Expansion**: Further added `"won"`, `"now"`, and `"collect"`. This final expansion significantly boosted the `spam_word_count` to 20 and resulted in the custom spam email being **correctly classified as 'Spam'** with a high predicted probability (0.9984).