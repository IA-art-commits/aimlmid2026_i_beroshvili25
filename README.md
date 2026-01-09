# aimlmid2026_i_beroshvili25
Midterm Exam Project AI &amp; Machine Learnin
gitithub Link 
# 📘 Midterm Exam Report
Github Upload Adrress: https://github.com/IA-art-commits/aimlmid2026_i_beroshvili25.git
## Statistical Analysis and Email Spam Classification

**Student:** I. Beroshvili
**Course:** Machine Learning / Data Analysis
**Exam Type:** Midterm Examination
**Programming Language:** Python

---

# **PART I – Pearson Correlation Coefficient**

## 1. Task Objective

The objective of this task is to calculate **Pearson’s correlation coefficient (r)** for a given set of observed data points extracted from a provided graph. The task requires:

* a clear mathematical explanation of the calculation,
* interpretation of the result,
* and a graphical visualization to support the analytical findings.

---

## 2. Observed Data

The following data points were extracted by hovering over the blue dots on the provided graph:

| i | xᵢ    | yᵢ    |
| - | ----- | ----- |
| 1 | -9.56 | 1.28  |
| 2 | -6.13 | 2.04  |
| 3 | -4.93 | -1.00 |
| 4 | -1.27 | 1.04  |
| 5 | 1.16  | -1.80 |
| 6 | 2.90  | 1.32  |
| 7 | 4.90  | -3.20 |
| 8 | 7.24  | -2.20 |

Number of observations:
[
n = 8
]

---

## 3. Pearson Correlation Formula

[
r =
\frac{\sum_{i=1}^{n}(x_i-\bar{x})(y_i-\bar{y})}
{\sqrt{\sum_{i=1}^{n}(x_i-\bar{x})^2 \cdot \sum_{i=1}^{n}(y_i-\bar{y})^2}}
]

Where:

* ( \bar{x} ) is the mean of x-values
* ( \bar{y} ) is the mean of y-values

---

## 4. Step-by-Step Calculation

### Step 1: Calculate Means

[
\bar{x} = -0.71125
]

[
\bar{y} = -0.315
]

---

### Step 2: Calculate Deviations

* Sum of cross-products:
  [
  \sum (x_i-\bar{x})(y_i-\bar{y}) \approx -52.80
  ]

* Sum of squared deviations:
  [
  \sum (x_i-\bar{x})^2 \approx 237.02
  ]
  [
  \sum (y_i-\bar{y})^2 \approx 27.14
  ]

---

### Step 3: Final Result

[
r = \frac{-52.80}{\sqrt{237.02 \cdot 27.14}} \approx \boxed{-0.66}
]

---

## 5. Interpretation

The value **r ≈ −0.66** indicates a **moderate negative linear correlation** between variables x and y.
This means that as x increases, y generally decreases, although the relationship is not perfectly linear.

---

## 6. Scatter Plot Visualization

To visually confirm the analytical result, a **scatter plot** was generated using **Python (matplotlib)**.

### Python Code

```python
import matplotlib.pyplot as plt

x = [-9.56, -6.13, -4.93, -1.27, 1.16, 2.90, 4.90, 7.24]
y = [1.28, 2.04, -1.00, 1.04, -1.80, 1.32, -3.20, -2.20]

plt.figure(figsize=(7, 5))
plt.scatter(x, y, label="Observed data points")
plt.xlabel("x values")
plt.ylabel("y values")
plt.title("Scatter Plot of Observed Data (Pearson Correlation)")
plt.legend()
plt.grid(True)
plt.show()
```

### Explanation

The scatter plot shows a **clear downward trend** of data points from left to right.
This visual behavior supports the **negative Pearson correlation coefficient**, confirming a moderate negative linear relationship between the two variables.

---

# **PART II – Email Spam Classification Using Logistic Regression**

## 1. Main Goal

The main goal of this task is to develop a **Python console application** that classifies emails into **Spam** and **Legitimate** categories using a **Logistic Regression model**, based on numerical features extracted from email content.

---

## 2. Dataset Upload

The provided dataset was uploaded to the repository:

```
data/i_beroshvili25_91478.csv
```

### Dataset Features

* `words` – total number of words in the email
* `links` – number of URLs
* `capital_words` – number of uppercase words
* `spam_word_count` – number of spam-related keywords
* `is_spam` – target label (1 = spam, 0 = legitimate)

---

## 3. Data Loading and Processing

The dataset is loaded using **pandas** and separated into features and labels:

```python
df = pd.read_csv(csv_path)
X = df[["words", "links", "capital_words", "spam_word_count"]]
y = df["is_spam"]
```

The data is split into:

* **70% training data**
* **30% testing data**

using stratified sampling to preserve class distribution.

---

## 4. Logistic Regression Model

### Model Selection

Logistic Regression is selected because:

* it is appropriate for binary classification,
* it provides probability-based predictions,
* and it allows clear interpretation through model coefficients.

### Model Training

```python
model = LogisticRegression(max_iter=2000, solver="lbfgs")
model.fit(X_train, y_train)
```

---

## 5. Model Coefficients

Each coefficient indicates the influence of a feature on spam probability:

* Positive coefficient → increases likelihood of spam
* Negative coefficient → decreases likelihood of spam

The coefficients are printed during training for transparency and interpretation.

---

## 6. Model Validation

### Confusion Matrix and Accuracy

The trained model is evaluated on the test dataset:

```python
confusion_matrix(y_test, y_pred)
accuracy_score(y_test, y_pred)
```

* **Accuracy** measures overall correctness.
* **Confusion Matrix** shows true positives, true negatives, false positives, and false negatives.

---

## 7. Email Text Classification

The application can classify **raw email text** entered by the user.

### Feature Extraction

Email text is parsed to extract:

* total word count,
* number of links,
* capitalized words,
* spam-related keywords.

These features are passed to the trained model to produce:

* spam probability,
* final classification result.

---

## 8. Manually Composed Emails

### Spam Email Example

```
URGENT!!! You have WON a FREE prize.
Click the link now to claim your reward!
```

**Explanation:**
Contains capital letters, spam keywords, urgency, and a call-to-action link.

---

### Legitimate Email Example

```
Hello,
Please find attached the meeting notes from yesterday.
Best regards.
```

**Explanation:**
Neutral tone, professional language, no spam indicators.

---

## 9. Visualizations

### Visualization A – Class Distribution

A bar chart shows the ratio of Spam vs Legitimate emails, helping detect dataset imbalance.

### Visualization B – Confusion Matrix Heatmap

A heatmap visualizes classification performance, clearly showing correct and incorrect predictions.

Each visualization includes:

* Python code,
* title,
* axis labels,
* explanatory text.

---

## 10. How to Run the Program

```bash
pip install -r requirements.txt
python main.py --mode train --data data/i_beroshvili25_91478.csv --show-plots
```

---

## 11. Conclusion

This midterm project demonstrates:

* statistical correlation analysis,
* supervised machine learning with Logistic Regression,
* feature extraction from raw text,
* model evaluation and visualization,
* practical email spam classification.

All tasks from the **Main Goal instructions** are fully satisfied with clear explanations and correct implementation.

---

✅ **End of README.md**

