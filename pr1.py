
# 1. IMPORT LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

# 2. LOAD DATASET

df = pd.read_csv("uber.csv")

print("First 5 Records:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistical Summary:")
print(df.describe())


# 3. DATA PRE-PROCESSING

# Remove unnecessary columns
if "Unnamed: 0" in df.columns:
    df.drop("Unnamed: 0", axis=1, inplace=True)

if "key" in df.columns:
    df.drop("key", axis=1, inplace=True)

# Remove missing values
df.dropna(inplace=True)

# Remove invalid fare values
df = df[df["fare_amount"] > 0]

# Remove invalid passenger counts
df = df[
    (df["passenger_count"] > 0) &
    (df["passenger_count"] <= 6)
]


# 4. FEATURE EXTRACTION FROM DATE-TIME

df["pickup_datetime"] = pd.to_datetime(
    df["pickup_datetime"],
    errors="coerce"
)

df.dropna(subset=["pickup_datetime"], inplace=True)

df["hour"] = df["pickup_datetime"].dt.hour
df["day"] = df["pickup_datetime"].dt.day
df["month"] = df["pickup_datetime"].dt.month
df["year"] = df["pickup_datetime"].dt.year
df["day_of_week"] = df["pickup_datetime"].dt.dayofweek

# Remove original datetime column
df.drop("pickup_datetime", axis=1, inplace=True)


# 5. CALCULATE DISTANCE USING HAVERSINE FORMULA

def haversine(lat1, lon1, lat2, lon2):

    R = 6371  # Radius of Earth in kilometers

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2 +
        np.cos(lat1) *
        np.cos(lat2) *
        np.sin(dlon / 2) ** 2
    )

    c = 2 * np.arcsin(np.sqrt(a))

    return R * c


df["distance_km"] = haversine(
    df["pickup_latitude"],
    df["pickup_longitude"],
    df["dropoff_latitude"],
    df["dropoff_longitude"]
)


# 6. REMOVE INVALID DISTANCE VALUES

df = df[
    (df["distance_km"] > 0) &
    (df["distance_km"] < 100)
]


# 7. OUTLIER DETECTION USING IQR

def remove_outliers(data, column):

    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    return data[
        (data[column] >= lower_limit) &
        (data[column] <= upper_limit)
    ]


# Remove outliers from numerical columns
for column in [
    "fare_amount",
    "distance_km",
    "passenger_count"
]:
    df = remove_outliers(df, column)


print("\nDataset Shape After Pre-processing:")
print(df.shape)


# 8. OUTLIER VISUALIZATION

plt.figure(figsize=(8, 5))

sns.boxplot(x=df["fare_amount"])

plt.title("Fare Amount - Outlier Detection")
plt.xlabel("Fare Amount")

plt.show()


# 9. CORRELATION ANALYSIS

plt.figure(figsize=(10, 8))

correlation = df.corr(numeric_only=True)

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.show()


# 10. SELECT FEATURES AND TARGET

features = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
    "distance_km",
    "hour",
    "day",
    "month",
    "year",
    "day_of_week"
]

X = df[features]

y = df["fare_amount"]


# 11. TRAIN-TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# 12. LINEAR REGRESSION

linear_model = LinearRegression()

linear_model.fit(X_train, y_train)

linear_predictions = linear_model.predict(X_test)

linear_r2 = r2_score(
    y_test,
    linear_predictions
)

linear_mae = mean_absolute_error(
    y_test,
    linear_predictions
)

linear_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        linear_predictions
    )
)


# 13. RANDOM FOREST REGRESSION

random_forest_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

random_forest_model.fit(
    X_train,
    y_train
)

rf_predictions = random_forest_model.predict(X_test)

rf_r2 = r2_score(
    y_test,
    rf_predictions
)

rf_mae = mean_absolute_error(
    y_test,
    rf_predictions
)

rf_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        rf_predictions
    )
)


# 14. DISPLAY RESULTS

print("\n---------------------------------------")
print("LINEAR REGRESSION PERFORMANCE")
print("---------------------------------------")

print("R2 Score :", linear_r2)
print("MAE      :", linear_mae)
print("RMSE     :", linear_rmse)


print("\n---------------------------------------")
print("RANDOM FOREST REGRESSION PERFORMANCE")
print("---------------------------------------")

print("R2 Score :", rf_r2)
print("MAE      :", rf_mae)
print("RMSE     :", rf_rmse)


# 15. MODEL COMPARISON

comparison = pd.DataFrame({

    "Model": [
        "Linear Regression",
        "Random Forest Regression"
    ],

    "R2 Score": [
        linear_r2,
        rf_r2
    ],

    "MAE": [
        linear_mae,
        rf_mae
    ],

    "RMSE": [
        linear_rmse,
        rf_rmse
    ]

})

print("\nModel Comparison:")
print(comparison)


# 16. ACTUAL VS PREDICTED VALUES

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    rf_predictions,
    alpha=0.5
)

plt.xlabel("Actual Fare")
plt.ylabel("Predicted Fare")

plt.title(
    "Actual vs Predicted Uber Fare - Random Forest"
)

plt.show()


# 17. BEST MODEL

if rf_r2 > linear_r2:
    print("\nBest Model: Random Forest Regression")
else:
    print("\nBest Model: Linear Regression")




# 1. IQR filtering: You remove outliers from passenger count as well. Since passenger count is discrete, removing values based on IQR might remove valid passenger counts. It is a choice in this implementation, not a strict requirement.

# 2. Distance: Haversine calculates straight-line distance. Actual road distance can be different.

# 3. Date features: Hour, day, month and day of week are treated as numerical values. These are useful, but time-related features can have cyclical patterns.

# 4. Model evaluation: Your best-model condition compares only R². Consider checking MAE and RMSE as well.

# 5. Feature leakage: Outlier thresholds are currently calculated using the complete dataset before the train-test split. In a more rigorous ML pipeline, preprocessing thresholds should be learned from training data only.
