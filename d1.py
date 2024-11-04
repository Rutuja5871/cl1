import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load Data from Multiple Formats
csv_data = pd.read_csv('datasets/Sales Data/sales data.csv')
excel_data = pd.read_excel('datasets/Sales Data/sales data.xlsx')
json_data = pd.read_json('datasets/Sales Data/sales data.json')

# Combine all datasets into one DataFrame
combined_data = pd.concat([csv_data, excel_data, json_data], ignore_index=True)

# Display the first few rows of the combined dataset
print("Combined Sales Data:")
print(combined_data.head())

# Step 2: Data Cleaning
# Convert 'Date' to datetime format
combined_data['Date'] = pd.to_datetime(combined_data['Date'], format='%d-%m-%Y', errors='coerce')

# Check for missing values
print("\nMissing values in each column:")
print(combined_data.isnull().sum())

# Fill missing values (Example strategy: Fill with mean for numerical, mode for categorical)
combined_data['Rating'].fillna(combined_data['Rating'].mean(), inplace=True)
combined_data['Customer type'].fillna(combined_data['Customer type'].mode()[0], inplace=True)

# Drop rows where 'Total' or 'Unit price' is missing
combined_data.dropna(subset=['Total', 'Unit price'], inplace=True)

# Check the data types
print("\nData types:")
print(combined_data.dtypes)

# Step 3: Data Transformation
# Convert 'Gender' and 'Customer type' to categorical
combined_data['Gender'] = combined_data['Gender'].astype('category')
combined_data['Customer type'] = combined_data['Customer type'].astype('category')
# Create a new column for 'Total Profit' (Total - cogs)
combined_data['Total Profit'] = combined_data['Total'] - combined_data['cogs']

# Create a new column for 'Sales Month'
combined_data['Sales Month'] = combined_data['Date'].dt.month

print("-------------------------------------------")
print(combined_data.head())

# Step 4: Data Analysis
# Total sales by branch
total_sales_by_branch = combined_data.groupby('Branch')['Total'].sum().reset_index()

# Average rating by product line
average_rating_by_product_line = combined_data.groupby('Product line')['Rating'].mean().reset_index()

# Total profit by city
total_profit_by_city = combined_data.groupby('City')['Total Profit'].sum().reset_index()

# Count of sales by payment method
sales_count_by_payment = combined_data['Payment'].value_counts().reset_index()
sales_count_by_payment.columns = ['Payment Method', 'Count']

# Display results
print("\nTotal Sales by Branch:")
print(total_sales_by_branch)

print("\nAverage Rating by Product Line:")
print(average_rating_by_product_line)

print("\nTotal Profit by City:")
print(total_profit_by_city)

print("\nCount of Sales by Payment Method:")
print(sales_count_by_payment)

# Step 5: Visualization of Results
# Set up visualizations
plt.figure(figsize=(10, 6))
sns.barplot(data=total_sales_by_branch, x='Branch', y='Total', palette='viridis')
plt.title('Total Sales by Branch')
plt.xlabel('Branch')
plt.ylabel('Total Sales')
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(data=average_rating_by_product_line, x='Product line', y='Rating', palette='Blues')
plt.title('Average Rating by Product Line')
plt.xlabel('Product Line')
plt.ylabel('Average Rating')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(data=total_profit_by_city, x='City', y='Total Profit', palette='magma')
plt.title('Total Profit by City')
plt.xlabel('City')
plt.ylabel('Total Profit')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(data=sales_count_by_payment, x='Payment Method', y='Count', palette='coolwarm')
plt.title('Count of Sales by Payment Method')
plt.xlabel('Payment Method')
plt.ylabel('Count of Sales')
plt.xticks(rotation=45)
plt.show()
