
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score
)

# 1. Load dataset
file_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "diabetes.csv"
)

df = pd.read_csv(file_path)

print("First 5 Records:")
print(df.head())

# 2. Separate features and target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# 3. Handle invalid zero values
cols = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

X[cols] = X[cols].replace(0, float("nan"))

# 4. Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 5. Handle missing values
imputer = SimpleImputer(strategy="median")

X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# 6. Feature scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 7. Train KNN model
knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train)

# 8. Predict test data
y_pred = knn.predict(X_test)

# 9. Calculate performance metrics
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

accuracy = accuracy_score(y_test, y_pred)
error_rate = 1 - accuracy
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)

# 10. Display performance
print("\n--- Performance of KNN Algorithm ---")

print("Confusion Matrix:\n", cm)

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Error Rate: {error_rate * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")

# 11. Visualize confusion matrix
plt.figure(figsize=(6, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Non-Diabetic", "Diabetic"],
    yticklabels=["Non-Diabetic", "Diabetic"]
)

plt.title("KNN Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.tight_layout()
plt.show()

# 12. Accept user input
print("\n--- Enter Patient Details ---")

features = list(df.drop("Outcome", axis=1).columns)
values = []

for feature in features:
    value = float(input(f"Enter {feature}: "))
    values.append(value)

# 13. Prepare user data
user_data = pd.DataFrame(
    [values],
    columns=features
)

# Handle invalid zero values
user_data[cols] = user_data[cols].replace(0, float("nan"))

# Apply trained imputer
user_data = imputer.transform(user_data)

# Apply trained scaler
user_data_scaled = scaler.transform(user_data)

# 14. Predict user outcome
prediction = knn.predict(user_data_scaled)[0]

probabilities = knn.predict_proba(user_data_scaled)[0]

diabetic_index = list(knn.classes_).index(1)

probability = probabilities[ diabetic_index] * 100

# 15. Display prediction
print("\n--- Prediction Result ---")

if prediction == 1:
    print("Prediction: Diabetic")
else:
    print("Prediction: Non-Diabetic")

print(
    f"Estimated probability of diabetic class: "
    f"{probability:.2f}%"
)



# ==========================================
# SAMPLE PATIENT INPUT (REFERENCE)
# ==========================================

# Pregnancies: 2
# Glucose: 120
# BloodPressure: 70
# SkinThickness: 25
# Insulin: 80
# BMI: 28.5
# DiabetesPedigreeFunction: 0.45
# Age: 30

# ==========================================
# ENTER THE ABOVE VALUES IN THE CONSOLE
# ==========================================
