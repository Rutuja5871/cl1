# Step 1: Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Step 2: Load the Iris dataset
iris = load_iris()
X = iris.data  # Features: Sepal Length, Sepal Width, Petal Length, Petal Width
y = iris.target  # True labels (for reference only, not used in clustering)

# Step 3: Feature Scaling (K-Means is sensitive to feature magnitudes)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Determine the Optimal Number of Clusters using Elbow Method
inertia = []  # Store inertia for different K values
K = range(1, 11)  # Try K from 1 to 10

for k in K:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)  # Set n_init to suppress warning
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot the Elbow Curve
plt.figure(figsize=(8, 5))
plt.plot(K, inertia, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.show()

# Step 5: Train K-Means with the optimal number of clusters (let's assume K=3)
kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)  # Set n_init to suppress warning
y_kmeans = kmeans.fit_predict(X_scaled)

# Step 6: Visualize the Clusters (using the first two features for 2D visualization)
plt.figure(figsize=(8, 5))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_kmeans, cmap='viridis', marker='o')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', marker='x', label='Centroids')
plt.xlabel('Sepal Length (scaled)')
plt.ylabel('Sepal Width (scaled)')
plt.title('K-Means Clustering on Iris Dataset (K=3)')
plt.legend()
plt.show()

# Step 7: Display Cluster Assignments and Compare with True Labels (Optional)
df = pd.DataFrame(X, columns=iris.feature_names)
df['Cluster'] = y_kmeans
df['True Label'] = y  # For reference only

print("First 5 rows with Cluster Labels:")
print(df.head())

# !pip install --upgrade threadpoolctl scikit-learn --user
