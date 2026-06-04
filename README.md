**Logistic Regression Binary Classification on Breast Cancer Dataset
**
**Objective**

Build a binary classification model using Logistic Regression to predict whether a breast tumor is malignant or benign.

**Dataset**

The project uses the Breast Cancer Wisconsin dataset available in Scikit-learn.

Target Classes
0 = Malignant (Cancerous)
1 = Benign (Non-cancerous)

**Features**

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


1. Load Dataset

2. Split Dataset

3. Feature Scaling

4. Train Logistic Regression Model

5. Make Predictions

**Conclusion**

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
