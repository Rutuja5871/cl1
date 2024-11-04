import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the data
data = pd.read_csv('datasets/telecom_churn.csv')

# Step 2: Initial Inspection
print("Initial Data Overview:")
print(data.head())
print("\nData Types and Missing Values:")
print(data.info())

# Step 3: Remove Irrelevant Columns
# Assuming fields like 'customer_id' are identifiers and irrelevant to churn analysis
columns_to_drop = ['customer_id', "pincode", "num_dependents"]  # Replace with actual columns to drop
data.drop(columns=columns_to_drop, inplace=True, errors='ignore')

# Step 4: Data Cleaning
# Identify missing values and handle them
print("\nMissing Values Before Cleaning:")
print(data.isnull().sum())

# Fill or drop missing values in relevant columns
data['data_used'] = data['data_used'].apply(lambda x: np.nan if x < 0 else x)  # Removing outliers
data['data_used'].fillna(data['data_used'].median(), inplace=True)

# Step 5: Feature Engineering
# Adding tenure in days based on `date_of_registration`
data['date_of_registration'] = pd.to_datetime(data['date_of_registration'], format='%Y-%m-%d')
data['tenure_days'] = (datetime.now() - data['date_of_registration']).dt.days
data.drop(columns=['date_of_registration'], inplace=True)  # Drop original date column if no longer needed

# Step 6: Data Transformation
# Encoding categorical variables
data = pd.get_dummies(data, columns=['telecom_partner', 'gender'], drop_first=True)

# Standardize numerical columns
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
numerical_cols = ['age', 'estimated_salary', 'calls_made', 'sms_sent', 'data_used', 'tenure_days']
data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

# Display cleaned data
print("\nCleaned Data Overview:")
print(data.head())

# Step 7: Exploratory Analysis
# Analyze correlations with churn
plt.figure(figsize=(10, 6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix with Churn')
plt.show()

# Churn by Age
plt.figure(figsize=(10, 5))
sns.histplot(data=data, x='age', hue='churn', multiple='stack')
plt.title('Age Distribution by Churn')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

# Average values of key features by churn status
churn_analysis = data.groupby('churn')[numerical_cols].mean()
print("\nAverage Key Features by Churn Status:")
print(churn_analysis)

# Bar Plot for Average Key Features by Churn
churn_analysis.T.plot(kind='bar', figsize=(12, 6), color=['skyblue', 'salmon'])
plt.title('Average Key Features by Churn Status')
plt.xlabel('Features')
plt.ylabel('Average Value')
plt.legend(title='Churn')
plt.xticks(rotation=45)
plt.show()
