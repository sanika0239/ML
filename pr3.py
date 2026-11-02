
# ==========================================
# BANK CUSTOMER CHURN PREDICTION USING ANN
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# 1. READ DATASET
df = pd.read_csv("Churn_Modelling.csv")

print("First 5 Records:")
print(df.head())

print("\nDataset Shape:", df.shape)


# 2. DATA PREPROCESSING

df.drop(
    ['RowNumber', 'CustomerId', 'Surname'],
    axis=1,
    inplace=True
)

print("\nMissing Values:")
print(df.isnull().sum())

df.drop_duplicates(inplace=True)

df.dropna(inplace=True)


# 3. SEPARATE FEATURES AND TARGET

X = df.drop('Exited', axis=1)

y = df['Exited']

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)


# 4. IDENTIFY NUMERICAL AND CATEGORICAL FEATURES

numerical_features = [
    'CreditScore',
    'Age',
    'Tenure',
    'Balance',
    'NumOfProducts',
    'HasCrCard',
    'IsActiveMember',
    'EstimatedSalary'
]

categorical_features = [
    'Geography',
    'Gender'
]


# 5. SPLIT DATASET

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)


# 6. NORMALIZATION AND ENCODING

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),

        ('cat', OneHotEncoder(
            handle_unknown='ignore',
            sparse_output=False
        ), categorical_features)
    ]
)

# Fit on training data
X_train = preprocessor.fit_transform(X_train)

# Transform testing data
X_test = preprocessor.transform(X_test)

print("\nAfter Preprocessing:")
print("Training Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)


# 7. HANDLE CLASS IMBALANCE

classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=y_train
)

class_weights = dict(zip(classes, weights))

print("\nClass Weights:", class_weights)


# 8. BUILD ANN MODEL

model = Sequential([

    Dense(
        64,
        input_shape=(X_train.shape[1],),
        activation='relu'
    ),

    Dropout(0.3),

    Dense(
        32,
        activation='relu'
    ),

    Dropout(0.2),

    Dense(
        1,
        activation='sigmoid'
    )
])


# 9. COMPILE MODEL

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\nANN Model Summary:")
model.summary()


# 10. EARLY STOPPING

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)


# 11. TRAIN MODEL

history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    class_weight=class_weights,
    verbose=1
)


# 12. PREDICT ON TEST DATA

y_probability = model.predict(
    X_test,
    verbose=0
)

y_pred = (
    y_probability >= 0.5
).astype(int).ravel()


# 13. ACCURACY

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n===================================")
print("MODEL PERFORMANCE")
print("===================================")

print(
    "Model Accuracy:",
    round(accuracy * 100, 2),
    "%"
)


# 14. CONFUSION MATRIX

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Stay', 'Leave'],
    yticklabels=['Stay', 'Leave']
)

plt.title("Bank Customer Churn - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()


# 15. CLASSIFICATION REPORT

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=['Stay', 'Leave']
    )
)


# 16. ACCURACY AND LOSS GRAPH

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)

plt.plot(
    history.history['accuracy'],
    label='Training Accuracy'
)

plt.plot(
    history.history['val_accuracy'],
    label='Validation Accuracy'
)

plt.title("ANN Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()


plt.subplot(1, 2, 2)

plt.plot(
    history.history['loss'],
    label='Training Loss'
)

plt.plot(
    history.history['val_loss'],
    label='Validation Loss'
)

plt.title("ANN Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()


# 17. USER INPUT FOR CUSTOMER PREDICTION

print("\n===================================")
print("BANK CUSTOMER CHURN PREDICTION")
print("===================================")

print("\nEnter Customer Details:\n")

credit_score = float(
    input("Enter Credit Score (300-850): ")
)

geography = input(
    "Enter Geography (France/Germany/Spain): "
).strip().title()

gender = input(
    "Enter Gender (Male/Female): "
).strip().title()

age = float(
    input("Enter Age: ")
)

tenure = float(
    input("Enter Tenure (0-10 years): ")
)

balance = float(
    input("Enter Account Balance: ")
)

num_products = int(
    input("Enter Number of Products: ")
)

has_card = int(
    input("Has Credit Card? (1=Yes, 0=No): ")
)

active_member = int(
    input("Is Active Member? (1=Yes, 0=No): ")
)

estimated_salary = float(
    input("Enter Estimated Salary: ")
)


# 18. CREATE USER DATAFRAME

user_data = pd.DataFrame({

    'CreditScore': [credit_score],

    'Geography': [geography],

    'Gender': [gender],

    'Age': [age],

    'Tenure': [tenure],

    'Balance': [balance],

    'NumOfProducts': [num_products],

    'HasCrCard': [has_card],

    'IsActiveMember': [active_member],

    'EstimatedSalary': [estimated_salary]

})


# 19. APPLY SAME PREPROCESSING

# Use the already fitted preprocessor
# Do not fit it again on user data

user_data_processed = preprocessor.transform(
    user_data
)


# 20. PREDICT CUSTOMER CHURN

prediction_probability = model.predict(
    user_data_processed,
    verbose=0
)[0][0]


prediction = (
    1 if prediction_probability >= 0.5 else 0
)


# 21. DISPLAY FINAL RESULT

print("\n===================================")
print("CUSTOMER CHURN PREDICTION RESULT")
print("===================================")

print("\nCustomer Details:")

print("Credit Score:", credit_score)
print("Geography:", geography)
print("Gender:", gender)
print("Age:", age)
print("Tenure:", tenure)
print("Balance:", balance)
print("Number of Products:", num_products)
print("Has Credit Card:", has_card)
print("Active Member:", active_member)
print("Estimated Salary:", estimated_salary)


print(
    "\nChurn Probability:",
    round(prediction_probability * 100, 2),
    "%"
)


if prediction == 1:

    print("\nPrediction: CUSTOMER IS LIKELY TO EXIT.")

else:

    print("\nPrediction: CUSTOMER IS LIKELY TO STAY.")


print("\n===================================")




# Enter Credit Score (300-850): 650
# Enter Geography (France/Germany/Spain): Germany
# Enter Gender (Male/Female): Male
# Enter Age: 35
# Enter Tenure (0-10 years): 5
# Enter Account Balance: 120000
# Enter Number of Products: 2
# Has Credit Card? (1=Yes, 0=No): 1
# Is Active Member? (1=Yes, 0=No): 1
# Enter Estimated Salary: 80000
