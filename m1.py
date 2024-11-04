# Step 1: Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report

# Step 2: Load the Iris dataset
iris = load_iris()
X = iris.data  # Fe

# Step 8: Predict species for a new flower
new_flower = [[53.1, 0.5, 12.4, 0.2]]  # Example input (features)
new_flower_lda = lda.transform(new_flower)
pred_species = lda_classifier.predict(new_flower_lda)atures
y = iris.target  # Target labels (species)

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Apply LDA for dimensionality reduction
lda = LinearDiscriminantAnalysis(n_components=2)  # Reduce to 2 components
X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)

# Step 5: Train a classifier using LDA-transformed features
lda_classifier = LinearDiscriminantAnalysis()  # LDA can also be used as a classifier
lda_classifier.fit(X_train_lda, y_train)

# Step 6: Make predictions on the test set
y_pred = lda_classifier.predict(X_test_lda)

# Step 7: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
print('\nClassification Report:\n', classification_report(y_test, y_pred, target_names=iris.target_names))
print(f'Predicted Species: {iris.target_names[pred_species[0]]}')
