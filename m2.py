# Import libraries
import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report

# Load Pima Indians Diabetes dataset (downloaded locally from UCI)
pima_data = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv',
                        header=None,
                        names=["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                               "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"])

# # Load UCI Diabetes dataset from sklearn
# uci_diabetes = load_diabetes()
# uci_data = pd.DataFrame(uci_diabetes.data, columns=uci_diabetes.feature_names)
# uci_data['target'] = uci_diabetes.target

print("Pima Dataset Shape:", pima_data.shape)
# print("UCI Diabetes Dataset Shape:", uci_data.shape)

# Display first few rows of both datasets
print("\nPima Indians Diabetes Dataset (first 5 rows):")
print(pima_data.head())

print("\nUCI Diabetes Dataset (first 5 rows):")
# print(uci_data.head())



# Function to compute univariate statistics
def univariate_analysis(data):
    stats_df = pd.DataFrame({
        'Feature': data.columns,
        'Mean': data.mean(),
        'Median': data.median(),
        'Mode': data.mode().iloc[0],
        'Variance': data.var(),
        'Std_Dev': data.std(),
        'Skewness': data.skew(),
        'Kurtosis': data.kurt()
    })
    return stats_df

# Univariate analysis on Pima dataset (excluding target column)
pima_uni_stats = univariate_analysis(pima_data.drop('Outcome', axis=1))
print("Pima Indians Diabetes Dataset - Univariate Analysis")
print(pima_uni_stats)

# Univariate analysis on UCI diabetes dataset (excluding target column)
# uci_uni_stats = univariate_analysis(uci_data.drop('target', axis=1))
# print("\nUCI Diabetes Dataset - Univariate Analysis")
# print(uci_uni_stats)


# Linear Regression on UCI Diabetes dataset
# X_uci = uci_data.drop('target', axis=1)
# y_uci = uci_data['target']

# X_train_uci, X_test_uci, y_train_uci, y_test_uci = train_test_split(X_uci, y_uci, test_size=0.3, random_state=42)

linear_model = LinearRegression()
# linear_model.fit(X_train_uci, y_train_uci)

# y_pred_uci = linear_model.predict(X_test_uci)

# print("\nLinear Regression - UCI Diabetes Dataset")
# print(f"Mean Squared Error: {mean_squared_error(y_test_uci, y_pred_uci):.2f}")
# print(f"R2 Score: {r2_score(y_test_uci, y_pred_uci):.2f}")

# Logistic Regression on Pima Indians Diabetes dataset
X_pima = pima_data.drop('Outcome', axis=1)
y_pima = pima_data['Outcome']

X_train_pima, X_test_pima, y_train_pima, y_test_pima = train_test_split(X_pima, y_pima, test_size=0.3, random_state=42)

logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train_pima, y_train_pima)

y_pred_pima = logistic_model.predict(X_test_pima)

print("\nLogistic Regression - Pima Indians Diabetes Dataset")
print(f"Accuracy: {accuracy_score(y_test_pima, y_pred_pima):.2f}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test_pima, y_pred_pima)}")
print(f"Classification Report:\n{classification_report(y_test_pima, y_pred_pima)}")

# Multiple Linear Regression on UCI Diabetes dataset
multiple_linear_model = LinearRegression()
# multiple_linear_model.fit(X_train_uci, y_train_uci)

# y_pred_multi = multiple_linear_model.predict(X_test_uci)

# print("\nMultiple Linear Regression - UCI Diabetes Dataset")
# print(f"Mean Squared Error: {mean_squared_error(y_test_uci, y_pred_multi):.2f}")
# print(f"R2 Score: {r2_score(y_test_uci, y_pred_multi):.2f}")
