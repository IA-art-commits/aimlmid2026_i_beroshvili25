# Email Spam Classification (Logistic Regression) — Console Application Report

**Student:** IA BEROSHVILI  
**Task:** Build a Python console application that classifies emails as **Spam** or **Legitimate** using Logistic Regression.

---

## 1) Dataset upload (Repository link) — 1 point

I uploaded the provided dataset to my GitHub repository.

- **Dataset file link:** *(replace with your GitHub link after upload)*  
  Example format: `https://github.com/<user>/<repo>/blob/main/data/i_beroshvili25_91478.csv`

**Local dataset path used during development (PyCharm):**  
`data/i_beroshvili25_91478.csv`

---

## 2) Training a Logistic Regression model on 70% of the data — 2 points

### 2.1 Data loading & processing code (with explanation) — 2 points

The dataset is loaded with **pandas**. I select the four numeric feature columns and the class label column:

- **Features:** `words`, `links`, `capital_words`, `spam_word_count`  
- **Label:** `is_spam` (1 = spam, 0 = legitimate)

To ensure the train and test sets preserve the original class ratio, I use **stratified splitting** and set a `random_state` for reproducibility.

**Code (model_training.py):**
```python
df = pd.read_csv(csv_path)

X = df[["words", "links", "capital_words", "spam_word_count"]]
y = df["is_spam"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
```

- **70%** of the data is used for training (`X_train`, `y_train`)
- **30%** is used for validation/testing (`X_test`, `y_test`)

**Split sizes (from program run):**
- Train size: **1750**
- Test size: **750**

---

### 2.2 Logistic Regression model (with code) — 1 point

I used **Logistic Regression** from `scikit-learn`. Logistic Regression is suitable for binary classification and outputs class probabilities through the sigmoid function.

**Code (model_training.py):**
```python
model = LogisticRegression(max_iter=2000, solver="lbfgs")
model.fit(X_train, y_train)
```

- `max_iter=2000` ensures convergence
- `lbfgs` is a reliable solver for logistic regression with numeric features

---

### 2.3 Coefficients found by the model — 1 point

After training, the model learned the following coefficients (feature order is the same as the dataset):

| Feature | Coefficient |
|---|---:|
| words | +0.007743 |
| links | +0.887587 |
| capital_words | +0.437373 |
| spam_word_count | +0.735358 |
| intercept | -9.446049 |

**Interpretation:**  
A **positive coefficient** means that as the feature increases, the probability of the email being classified as **spam** increases (holding other features constant). In this model, `links`, `capital_words`, and `spam_word_count` have the strongest positive influence, which matches typical spam patterns.

---

### 2.4 Source code link(s) — 1 point

I uploaded the application source code to my GitHub repository:

- **Main console application:** *(replace with your GitHub link)*  
  Example: `https://github.com/<user>/<repo>/blob/main/main.py`

- **Feature extraction module:** *(replace with your GitHub link)*  
  Example: `https://github.com/<user>/<repo>/blob/main/spam_features.py`

- **Training + evaluation module:** *(replace with your GitHub link)*  
  Example: `https://github.com/<user>/<repo>/blob/main/model_training.py`

---

## 3) Validation: Confusion Matrix and Accuracy on 30% test set — 1 + 2 points

### 3.1 Code used to compute Confusion Matrix and Accuracy — 2 points

After training, the model is validated on the 30% test set:

```python
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
```

### 3.2 Results (from program run) — 1 point

- **Accuracy:** **0.9693**  (≈ 96.93%)

- **Confusion Matrix** (rows = actual class, columns = predicted class):

\[
\begin{bmatrix}
TN & FP \\
FN & TP
\end{bmatrix}
=
\begin{bmatrix}
369 & 5 \\
18 & 358
\end{bmatrix}
\]

**Meaning of each value:**
- **TN** = 369 legitimate emails correctly predicted as legitimate  
- **FP** = 5 legitimate emails incorrectly predicted as spam  
- **FN** = 18 spam emails incorrectly predicted as legitimate  
- **TP** = 358 spam emails correctly predicted as spam

---

## 4) Email text classification: parsing → feature extraction → model prediction — 3 points

The application can classify a *new, raw email text* by extracting the same four features as in the dataset.

### 4.1 Feature extraction (spam_features.py)

```python
tokens = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", email_text)
words = len(tokens)

links = len(URL_REGEX.findall(email_text))

capital_words = sum(1 for t in tokens if len(t) >= 2 and t.isupper())

spam_word_count = 0
for kw in SPAM_KEYWORDS:
    spam_word_count += len(re.findall(rf"\b{re.escape(kw)}\b", email_text.lower()))
```

### 4.2 Prediction step (main.py)

```python
prob_spam = model.predict_proba(X_new)[0, 1]
prediction = 1 if prob_spam >= 0.5 else 0
```

The program prints both:
- extracted feature values
- spam probability `P(spam)`
- final class label (SPAM or LEGITIMATE)

---

## 5) Manually composed SPAM email (should be classified as spam) — 1 point

**Spam email text:**
> Subject: URGENT — YOU WON a FREE PRIZE!!!  
> Congratulations! You are a WINNER. Claim your FREE reward now.  
> Click the link to verify your account and receive CASH BONUS today:  
> https://example.com/claim-prize  
> LIMITED OFFER — ACT NOW!

**Why it is spam-like:**  
This email intentionally contains many typical spam indicators: multiple spam keywords (“free”, “winner”, “urgent”, “claim”, “bonus”), **a link**, and several **ALL-CAPS** words. These increase `spam_word_count`, `links`, and `capital_words`, which strongly increases the probability of the spam class.

---

## 6) Manually composed LEGITIMATE email (should be classified as legitimate) — 1 point

**Legitimate email text:**
> Subject: Project meeting agenda (Tuesday)  
> Hi team,  
> Please find the agenda for Tuesday’s project meeting.  
> We will review milestones and next steps.  
> Thanks,  
> [Your Name]

**Why it is legitimate-like:**  
It contains no promotional language, no suspicious keywords, no urgent calls to action, and no URLs. Therefore `spam_word_count`, `links`, and `capital_words` remain low, making the model more likely to predict the legitimate class.

---

## 7) Visualizations (2 required) — 4 points

All plots were generated using **matplotlib**, and each includes a title, axis labels, and a legend.

### Visualization A: Class Distribution Bar Chart
**Code (visualize.py):**
```python
counts = y.value_counts().sort_index()
labels = ["Legitimate (0)", "Spam (1)"]

plt.figure()
plt.bar(labels, [counts.get(0, 0), counts.get(1, 0)], label="Email count")
plt.title("Class Distribution in Dataset")
plt.xlabel("Class")
plt.ylabel("Number of emails")
plt.legend()
plt.savefig("outputs/class_distribution.png", dpi=200, bbox_inches="tight")
```

**Explanation (2–3 sentences):**  
This chart shows the number of emails in each class. The dataset is almost balanced between spam and legitimate, which helps prevent bias toward one class and makes accuracy a meaningful performance metric.

---

### Visualization B: Confusion Matrix Heatmap
**Code (visualize.py):**
```python
plt.figure()
plt.imshow(cm, interpolation="nearest")
plt.title("Confusion Matrix Heatmap (Test Set)")
plt.xlabel("Predicted class")
plt.ylabel("Actual class")
plt.xticks([0, 1], ["Legitimate (0)", "Spam (1)"])
plt.yticks([0, 1], ["Legitimate (0)", "Spam (1)"])
plt.colorbar(label="Count")
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center")
plt.savefig("outputs/confusion_matrix_heatmap.png", dpi=200, bbox_inches="tight")
```

**Explanation (2–3 sentences):**  
The diagonal cells (TN and TP) are large, meaning the model correctly classifies most emails. The off-diagonal cells (FP and FN) are small, showing relatively few mistakes. This aligns with the model’s accuracy of **96.93%**.

---

### (Optional) Visualization C: Feature importance via coefficients
A coefficient bar chart is also generated in `outputs/coefficients_bar.png`. It highlights which features contribute most to predicting spam.

---

## How to run the console application (PyCharm / terminal)

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Train + evaluate and save model (also generates plots):
> **Tip (PyCharm):** To open plots in a window *and* save them to `outputs/`, run training with `--show-plots`.

```bash
python main.py --mode train --data data/i_beroshvili25_91478.csv
# show plot windows too:
python main.py --mode train --data data/i_beroshvili25_91478.csv --show-plots
```

3. Classify a single email text:
```bash
python main.py --mode classify --model spam_lr_model.joblib --text "Your email text here..."
```

4. Interactive mode (paste email; blank line ends input):
```bash
python main.py --mode interactive --model spam_lr_model.joblib
```

---

## Output files created
- `spam_lr_model.joblib` (saved trained model)
- `outputs/class_distribution.png`
- `outputs/confusion_matrix_heatmap.png`
- `outputs/coefficients_bar.png` (optional extra visualization)
