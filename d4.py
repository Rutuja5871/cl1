import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the Data from CSV
df = pd.read_csv("datasets/Real-Estate dataset.csv")

# Step 2: Inspect the Data
print("First few rows of the dataset:\n", df.head())
print("\nData Types and Missing Values:\n", df.info())

# Step 3: Data Cleaning - Handle Missing Values
# Option to drop rows with missing values
df = df.dropna()

# Step 4: Encode Categorical Variables
# Binary encoding for yes/no columns
binary_columns = ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea"]
df[binary_columns] = df[binary_columns].applymap(lambda x: 1 if x == "yes" else 0)

# One-hot encoding for furnishing status
df = pd.get_dummies(df, columns=["furnishingstatus"], drop_first=False)

# Step 5: Print the Columns to Check After Encoding
print("\nColumns after encoding:\n", df.columns)

# Step 6: Feature Engineering
# Calculate price per square foot area
df['price_per_sqft'] = df['price'] / df['area']

print(df.head())

# Step 7: Exploratory Data Analysis (EDA)

# Correlation heatmap to see relationships
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap="YlGnBu", fmt=".2f")
plt.title("Correlation Matrix of Real Estate Data")
plt.show()

# Boxplot to see price distribution across different furnishing statuses
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x="furnishingstatus_semi-furnished", y="price")
plt.title("Price Distribution by Furnishing Status (Semi-Furnished)")
plt.xlabel("Semi-Furnished Status")
plt.ylabel("Price")
plt.show()

# Boxplot for price distribution by number of bedrooms
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="bedrooms", y="price")
plt.title("Price Distribution by Number of Bedrooms")
plt.xlabel("Number of Bedrooms")
plt.ylabel("Price")
plt.show()

# Scatter plot for area vs price
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="area", y="price", hue="bedrooms", palette="viridis")
plt.title("Price vs Area Colored by Number of Bedrooms")
plt.xlabel("Area (sq ft)")
plt.ylabel("Price")
plt.legend(title="Number of Bedrooms")
plt.show()

# Step 8: Bar Graphs for Average Price Analysis

# Average price by number of bedrooms
avg_price_by_bedrooms = df.groupby('bedrooms')['price'].mean().reset_index()
plt.figure(figsize=(10, 6))
sns.barplot(data=avg_price_by_bedrooms, x='bedrooms', y='price', palette='Blues_d')
plt.title('Average Price by Number of Bedrooms')
plt.xlabel('Number of Bedrooms')
plt.ylabel('Average Price')
plt.xticks(rotation=0)
plt.show()

# Average price by number of bathrooms
avg_price_by_bathrooms = df.groupby('bathrooms')['price'].mean().reset_index()
plt.figure(figsize=(10, 6))
sns.barplot(data=avg_price_by_bathrooms, x='bathrooms', y='price', palette='Blues_d')
plt.title('Average Price by Number of Bathrooms')
plt.xlabel('Number of Bathrooms')
plt.ylabel('Average Price')
plt.xticks(rotation=0)
plt.show()

# Average price by presence of guestroom
avg_price_by_guests = df.groupby('guestroom')['price'].mean().reset_index()
plt.figure(figsize=(10, 6))
sns.barplot(data=avg_price_by_guests, x='guestroom', y='price', palette='Blues_d')
plt.title('Average Price by Guestroom Availability')
plt.xlabel('Guestroom Available (1 = Yes, 0 = No)')
plt.ylabel('Average Price')
plt.xticks(rotation=0)
plt.show()

# Average price by furnishing status
# Check the actual column names for furnishing status
print("Furnishing status columns:", df.filter(like='furnishingstatus').columns)

# Use the correct column name based on the previous output
avg_price_by_furnishing = df.groupby(df.filter(like='furnishingstatus').columns[0])['price'].mean().reset_index()
plt.figure(figsize=(10, 6))
sns.barplot(data=avg_price_by_furnishing, x=avg_price_by_furnishing.columns[0], y='price', palette='Blues_d')
plt.title('Average Price by Furnishing Status')
plt.xlabel('Furnishing Status')
plt.ylabel('Average Price')
plt.xticks(rotation=0)
plt.show()

# Step 9: Pie Chart for Furnishing Status Distribution

# Calculate the counts for each furnishing status
furnishing_status_counts = {
    'Furnished': df['furnishingstatus_furnished'].sum(),
    'Semi-Furnished': df['furnishingstatus_semi-furnished'].sum(),
    'Unfurnished': df['furnishingstatus_unfurnished'].sum()
}

# Plotting the pie chart
plt.figure(figsize=(8, 6))
plt.pie(furnishing_status_counts.values(), labels=furnishing_status_counts.keys(), autopct='%1.1f%%', startangle=90, colors=['lightblue', 'salmon', 'lime'])
plt.title('Distribution of Furnishing Status')
plt.show()

