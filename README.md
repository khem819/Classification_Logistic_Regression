Logistic Regression Binary Classification on Breast Cancer Dataset

Objective

Build a binary classification model using Logistic Regression to predict whether a breast tumor is malignant or benign.

Dataset

The project uses the Breast Cancer Wisconsin dataset available in Scikit-learn.

Target Classes
0 = Malignant (Cancerous)
1 = Benign (Non-cancerous)
Features

The dataset contains 30 numerical features such as:

Mean Radius
Mean Texture
Mean Perimeter
Mean Area
Mean Smoothness
and others
Libraries Used
pandas
matplotlib
scikit-learn

Install required libraries:

pip install pandas matplotlib scikit-learn
Workflow
1. Load Dataset
data = load_breast_cancer()

The dataset is loaded from Scikit-learn and converted into a Pandas DataFrame.

2. Split Dataset
train_test_split(X, y, test_size=0.2, random_state=42)
80% Training Data
20% Testing Data
3. Feature Scaling
scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

Standardization ensures all features have:

Mean = 0
Standard Deviation = 1

This improves Logistic Regression performance.

4. Train Logistic Regression Model
model = LogisticRegression()
model.fit(x_train, y_train)

The model learns the relationship between tumor features and class labels.

5. Make Predictions
Predicted Class
y_pred = model.predict(x_test)

Returns:

0 or 1
Predicted Probability
y_prob = model.predict_proba(x_test)[:,1]

Returns probability of belonging to class 1 (Benign).

Example:

[0.95, 0.12, 0.87]
Evaluation Metrics
Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

Example:

[[40  3]
 [ 1 70]]
Actual / Predicted	Malignant (0)	Benign (1)
Malignant (0)	40	3
Benign (1)	1	70
Precision
precision_score(y_test, y_pred)

Formula:

Precision=
TP+FP
TP
	​


Measures how many predicted positive cases are actually positive.

Higher precision means fewer false positives.

Recall
recall_score(y_test, y_pred)

Formula:

Recall=
TP+FN
TP
	​


Measures how many actual positive cases are correctly identified.

Higher recall means fewer false negatives.

ROC-AUC Score
auc = roc_auc_score(y_test, y_prob)

Measures how well the model separates the two classes.

Interpretation:

AUC Score	Performance
0.50	Random Guess
0.60–0.70	Poor
0.70–0.80	Fair
0.80–0.90	Good
0.90–1.00	Excellent

For ROC-AUC, the closer the score is to 1, the better.

ROC Curve

The ROC curve plots:

X-axis → False Positive Rate (FPR)
Y-axis → True Positive Rate (TPR)
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

Plot:

plt.plot(fpr, tpr)

A curve closer to the top-left corner indicates better performance.

Threshold Tuning

Default threshold:

0.5

Prediction rule:

Probability ≥ 0.5 → Class 1
Probability < 0.5 → Class 0

Custom threshold:

threshold = 0.3

y_pred_custom = (y_prob >= threshold).astype(int)
Why Change Threshold?

Lower Threshold (0.3):

More positive predictions
Higher Recall
Lower Precision

Higher Threshold (0.7):

Fewer positive predictions
Higher Precision
Lower Recall

Use threshold tuning when:

Missing a positive case is costly.
You want to balance Precision and Recall differently.
Sigmoid Function

Logistic Regression uses the Sigmoid Function:

p=
1+e
−z
1
	​


Where:

z = weighted sum of input features
Output is always between 0 and 1

This output is interpreted as probability.

Example:

z = 0  → p = 0.5
z = 2  → p = 0.88
z = -2 → p = 0.12
Output Produced

The program prints:

Confusion Matrix
Precision
Recall
ROC-AUC Score

and displays/saves:

roc_curve.png
Conclusion

This project demonstrates a complete Logistic Regression workflow:

Load Breast Cancer Dataset
Split into Train/Test sets
Standardize Features
Train Logistic Regression
Predict Classes and Probabilities
Evaluate using:
Confusion Matrix
Precision
Recall
ROC-AUC
Plot ROC Curve
Tune Classification Threshold

The Breast Cancer dataset is highly suitable for Logistic Regression and typically achieves excellent ROC-AUC scores close to 1.0.
