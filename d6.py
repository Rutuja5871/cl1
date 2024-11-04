import pandas as pd

# Step 1: Load the Data from CSV
df = pd.read_csv('datasets/retail_sales_data.csv')

# Display the first few rows of the dataframe
print("Initial Data:")
print(df.head())

# Step 2: Data Cleaning
# Convert 'invoice_date' to datetime format
df['invoice_date'] = pd.to_datetime(df['invoice_date'], format='%d/%m/%Y')

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Optionally, handle missing values if any
# df.dropna(inplace=True)  # Uncomment if you want to drop rows with missing values

# Step 3: Data Aggregation
# Calculate total sales per region (shopping mall)
df['total_sales'] = df['quantity'] * df['price']
sales_per_region = df.groupby('shopping_mall').agg(
    total_quantity=('quantity', 'sum'),
    total_sales=('total_sales', 'sum'),
    total_transactions=('invoice_no', 'nunique')
).reset_index()

# Step 4: Identify Top-Performing Regions
top_regions = sales_per_region.sort_values(by='total_sales', ascending=False)

# Display the aggregated sales performance
print("\nSales Performance by Region:")
print(top_regions)

# Optionally, visualize the top regions
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 6))
sns.barplot(data=top_regions, x='total_sales', y='shopping_mall', palette='viridis')
plt.title('Top-Performing Regions by Total Sales')
plt.xlabel('Total Sales')
plt.ylabel('Shopping Mall')
plt.show()
