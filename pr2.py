
# Email Spam Detection using KNN and SVM

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# 1. Load Dataset
df = pd.read_csv("emails.csv")

print("Dataset:")
print(df.head())

print("\nDataset Shape:", df.shape)

# 2. Data Preprocessing
# Remove unnecessary email name column if present
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

if 'Email No.' in df.columns:
    df = df.drop('Email No.', axis=1)

# Remove duplicate records
df = df.drop_duplicates()

# Handle missing values
df = df.fillna(0)

print("\nMissing Values:", df.isnull().sum().sum())

# 3. Separate Features and Target
X = df.drop('Prediction', axis=1)
y = df['Prediction']

# 0 = Not Spam, 1 = Spam
print("\nClass Distribution:")
print(y.value_counts())

# 4. Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 5. Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 6. KNN Classification
knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train)

knn_pred = knn.predict(X_test)

# 7. SVM Classification
svm = SVC(kernel='linear')

svm.fit(X_train, y_train)

svm_pred = svm.predict(X_test)

# 8. Model Evaluation Function
def evaluate_model(name, y_test, y_pred):

    print("\n", name, "RESULT")

    print("Accuracy:",
          accuracy_score(y_test, y_pred))

    print("Precision:",
          precision_score(y_test, y_pred, zero_division=0))

    print("Recall:",
          recall_score(y_test, y_pred, zero_division=0))

    print("F1 Score:",
          f1_score(y_test, y_pred, zero_division=0))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)

    print("Confusion Matrix:")
    print(cm)

    return cm


# 9. Evaluate KNN
knn_cm = evaluate_model("KNN", y_test, knn_pred)

# 10. Evaluate SVM
svm_cm = evaluate_model("SVM", y_test, svm_pred)

# 11. Compare Model Performance
results = pd.DataFrame({
    'Model': ['KNN', 'SVM'],
    'Accuracy': [
        accuracy_score(y_test, knn_pred),
        accuracy_score(y_test, svm_pred)
    ],
    'Precision': [
        precision_score(y_test, knn_pred, zero_division=0),
        precision_score(y_test, svm_pred, zero_division=0)
    ],
    'Recall': [
        recall_score(y_test, knn_pred, zero_division=0),
        recall_score(y_test, svm_pred, zero_division=0)
    ],
    'F1 Score': [
        f1_score(y_test, knn_pred, zero_division=0),
        f1_score(y_test, svm_pred, zero_division=0)
    ]
})

print("\nMODEL COMPARISON:")
print(results)

# 12. Accuracy Comparison Graph
results.set_index('Model')[
    ['Accuracy', 'Precision', 'Recall', 'F1 Score']
].plot(kind='bar', figsize=(9, 5))

plt.title("KNN vs SVM Performance Comparison")
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.xticks(rotation=0)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

# 13. Confusion Matrix Visualization
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

sns.heatmap(
    knn_cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=['Not Spam', 'Spam'],
    yticklabels=['Not Spam', 'Spam'],
    ax=axes[0]
)
axes[0].set_title("KNN Confusion Matrix")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

sns.heatmap(
    svm_cm, annot=True, fmt='d', cmap='Greens',
    xticklabels=['Not Spam', 'Spam'],
    yticklabels=['Not Spam', 'Spam'],
    ax=axes[1]
)
axes[1].set_title("SVM Confusion Matrix")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

plt.tight_layout()
plt.show()

# 14. Best Model
best_model = results.loc[
    results['F1 Score'].idxmax(), 'Model'
]

print("\nBest Model based on F1 Score:", best_model)
