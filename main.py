import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import( 
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve
    )


data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

print(X.head())
print("Target Classes:", data.target_names)

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# train the model
model = LogisticRegression()
model.fit(x_train,y_train)


# make predictions
y_pred = model.predict(x_test)
y_prob = model.predict_proba(x_test)[:,1]

# create confusion matrix

cm = confusion_matrix(y_test,y_pred)
print("confusion matrix:")
print(cm)

precision = precision_score(y_test,y_pred)
recall = recall_score(y_test,y_pred)
print("Precision:",precision)
print("Recall:",recall)

auc = roc_auc_score(y_test, y_prob)
print("ROC-AUC Score:", auc)

# plot roc curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure(figsize=(8, 5))
plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig("roc_curve.png")
plt.show()

# Adjust Classification Threshold
threshold = 0.3

y_pred_custom = (y_prob >= threshold).astype(int)
print("Precision:", precision_score(y_test, y_pred_custom))
print("Recall:", recall_score(y_test, y_pred_custom))