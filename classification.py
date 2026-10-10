
# AI Project 2: Data Classification Using AI

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load and understand the dataset
iris = load_iris()
X = iris.data
y = iris.target

print("AI DATA CLASSIFICATION PROJECT")
print("-" * 40)
print("Dataset: Iris Flower Dataset")
print("Number of samples:", X.shape[0])
print("Number of features:", X.shape[1])
print("Flower categories:", list(iris.target_names))
print("\nFeature names:", list(iris.feature_names))

# 2. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# 3. Create and train the classification model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# 4. Test the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\nModel: Decision Tree Classifier")
print(f"Accuracy: {accuracy:.2%}")

print("\nClassification Report:")
print(classification_report(
    y_test, y_pred, target_names=iris.target_names
))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 5. Predict the category of a new flower
new_flower = [[5.1, 3.5, 1.4, 0.2]]
prediction = model.predict(new_flower)[0]

print("\nExample Prediction")
print("Flower measurements:", new_flower[0])
print("Predicted flower category:", iris.target_names[prediction])
