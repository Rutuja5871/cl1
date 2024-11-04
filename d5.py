import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the Data from CSV
df = pd.read_csv('datasets/AQI Data Set.csv')

# Display the first few rows of the dataframe
print("Initial Data:")
print(df.head())

# Step 2: Data Cleaning
# Filling missing values using forward fill method
df.fillna(method='ffill', inplace=True)

# Convert 'Months' to datetime format
df['Mounths'] = pd.to_datetime(df['Mounths'], format='%b-%y')

# Step 3: Data Visualization
# Set up the figure and axes for multiple plots
fig, axs = plt.subplots(3, 2, figsize=(15, 12))

# Plotting AQI over the months
axs[0, 0].plot(df['Mounths'], df['AQI'], marker='o', color='blue')
axs[0, 0].set_title('AQI Over Time')
axs[0, 0].set_xlabel('Month')
axs[0, 0].set_ylabel('AQI')

# Plotting PM10 over the months
axs[0, 1].plot(df['Mounths'], df['PM10 in æg/m3'], marker='o', color='green')
axs[0, 1].set_title('PM10 Levels Over Time')
axs[0, 1].set_xlabel('Month')
axs[0, 1].set_ylabel('PM10 (µg/m³)')

# Plotting SO2 over the months
axs[1, 0].plot(df['Mounths'], df['SO2 in æg/m3'], marker='o', color='orange')
axs[1, 0].set_title('SO2 Levels Over Time')
axs[1, 0].set_xlabel('Month')
axs[1, 0].set_ylabel('SO2 (µg/m³)')

# Plotting NOx over the months
axs[1, 1].plot(df['Mounths'], df['NOx  in æg/m3'], marker='o', color='red')
axs[1, 1].set_title('NOx Levels Over Time')
axs[1, 1].set_xlabel('Month')
axs[1, 1].set_ylabel('NOx (µg/m³)')

# Plotting PM2.5 over the months
axs[2, 0].plot(df['Mounths'], df[' PM2.5  in æg/m3'], marker='o', color='purple')
axs[2, 0].set_title('PM2.5 Levels Over Time')
axs[2, 0].set_xlabel('Month')
axs[2, 0].set_ylabel('PM2.5 (µg/m³)')

# Plotting O3 over the months
axs[2, 1].plot(df['Mounths'], df['O3   in æg/m3'], marker='o', color='brown')
axs[2, 1].set_title('O3 Levels Over Time')
axs[2, 1].set_xlabel('Month')
axs[2, 1].set_ylabel('O3 (µg/m³)')

# Adjust layout
plt.tight_layout()
plt.show()

# Additional Visualizations

# Bar plot for average pollutant levels per month
average_pollutants = df.mean(numeric_only=True)
plt.figure(figsize=(12, 6))
average_pollutants.plot(kind='bar', color='lightblue')
plt.title('Average Pollutant Levels')
plt.xlabel('Pollutants')
plt.ylabel('Average Concentration')
plt.xticks(rotation=45)
plt.show()

# Box plot for pollutant levels
plt.figure(figsize=(15, 8))
sns.boxplot(data=df[['PM10 in æg/m3', 'SO2 in æg/m3', 'NOx  in æg/m3', ' PM2.5  in æg/m3', 'O3   in æg/m3']])
plt.title('Distribution of Pollutant Levels')
plt.ylabel('Concentration (µg/m³)')
plt.xticks(rotation=45)
plt.show()

# Heatmap for correlations
correlation_matrix = df[['PM10 in æg/m3', 'SO2 in æg/m3', 'NOx  in æg/m3', ' PM2.5  in æg/m3', 'O3   in æg/m3', 'AQI']].corr()
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Heatmap of Pollutants and AQI')
plt.show()

# Histogram for AQI distribution
plt.figure(figsize=(10, 6))
plt.hist(df['AQI'].dropna(), bins=10, color='skyblue', edgecolor='black')
plt.title('Distribution of AQI Values')
plt.xlabel('AQI')
plt.ylabel('Frequency')
plt.show()
