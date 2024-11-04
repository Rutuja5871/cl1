# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Step 2: Load the dataset (Ensure the file is in the same directory or provide the correct path)
data = pd.read_csv('datasets/Social_Network_Ads.csv')

# Step 3: Explore the dataset
print("First 5 rows of the dataset:")
print(data.head())

# Check for missing values
print("\nMissing values in dataset:", data.isnull().sum())

# Step 4: Select Features and Target
X = data[['Age', 'EstimatedSalary']].values  # Features
y = data['Purchased'].values  # Target (0 or 1)

# Step 5: Split the dataset into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 6: Feature Scaling (Standardizing Age and Salary for better KNN performance)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 7: Initialize and Train the K-Nearest Neighbors Classifier
knn = KNeighborsClassifier(n_neighbors=5)  # You can change 'n_neighbors' as needed
knn.fit(X_train, y_train)

# Step 8: Make predictions on the test set
y_pred = knn.predict(X_test)

# Step 9: Evaluate the model performance

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy * 100:.2f}%")

# Error Rate (1 - Accuracy)
error_rate = 1 - accuracy
print(f"Error Rate: {error_rate * 100:.2f}%")

# Classification Report (Precision, Recall, F1-score)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
