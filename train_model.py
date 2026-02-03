import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from xgboost import XGBRegressor
from scipy.stats import randint, uniform

RANDOM_STATE = 42

df = pd.read_csv("car_prediction_data.csv")

CURRENT_YEAR = df["Year"].max()
df["Car_Age"] = CURRENT_YEAR - df["Year"]

df["Kms_Driven_Log"] = np.log1p(df["Kms_Driven"])
df["Age_x_Kms"] = df["Car_Age"] * df["Kms_Driven_Log"]
df["Age_x_Price"] = df["Car_Age"] * df["Present_Price"]

X = df.drop(columns=["Selling_Price", "Year", "Kms_Driven"])
y = np.log1p(df["Selling_Price"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

categorical_features = ["Fuel_Type", "Seller_Type", "Transmission"]
numerical_features = [
    "Present_Price",
    "Owner",
    "Car_Age",
    "Kms_Driven_Log",
    "Age_x_Kms",
    "Age_x_Price",
]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

xgb = XGBRegressor(
    objective="reg:squarederror",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", xgb),
    ]
)

param_dist = {
    "model__n_estimators": randint(400, 900),
    "model__max_depth": randint(4, 10),
    "model__learning_rate": uniform(0.03, 0.07),
    "model__subsample": uniform(0.6, 0.4),
    "model__colsample_bytree": uniform(0.6, 0.4),
    "model__min_child_weight": randint(1, 8),
    "model__gamma": uniform(0, 0.3),
    "model__reg_alpha": uniform(0, 0.5),  
    "model__reg_lambda": uniform(0.5, 1.5) 
}

search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_dist,
    n_iter=40,                    
    cv=5,
    scoring="neg_root_mean_squared_error",
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=1
)

print("Hyperparameter tuning started...")
search.fit(X_train, y_train)

best_model = search.best_estimator_

y_pred_log = best_model.predict(X_test)

y_pred = np.expm1(y_pred_log)
y_true = np.expm1(y_test)

print("\n--- Tuned XGBoost Performance ---")
print(f"R2   : {r2_score(y_true, y_pred):.4f}")
print(f"MAE  : {mean_absolute_error(y_true, y_pred):.4f}")
print(f"RMSE : {np.sqrt(mean_squared_error(y_true, y_pred)):.4f}")

print("\nBest Hyperparameters:")
print(search.best_params_)

joblib.dump(best_model, "car_price_model_xgboost_tuned.pkl")
print("\nModel saved as car_price_model_xgboost_tuned.pkl")
