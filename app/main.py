from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import os

app = FastAPI(title="Car Price Prediction API")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "car_price_model_xgboost_tuned.pkl")

try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded")
except Exception as e:
    print("Model loading failed:", e)
    model = None

class CarInput(BaseModel):
    Year: int = Field(..., ge=1990, le=datetime.now().year)
    Present_Price: float = Field(..., gt=0)
    Kms_Driven: int = Field(..., ge=0)
    Fuel_Type: Literal["Petrol", "Diesel", "CNG"]
    Seller_Type: Literal["Dealer", "Individual"]
    Transmission: Literal["Manual", "Automatic"]
    Owner: int = Field(..., ge=0, le=3)

@app.post("/predict")
def predict_price(car: CarInput):

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        data = car.model_dump()

        current_year = datetime.now().year
        car_age = current_year - data["Year"]

        kms_log = np.log1p(data["Kms_Driven"])
        age_x_kms = car_age * kms_log
        age_x_price = car_age * data["Present_Price"]

        df = pd.DataFrame([{
            "Present_Price": data["Present_Price"],
            "Owner": data["Owner"],
            "Car_Age": car_age,
            "Kms_Driven_Log": kms_log,
            "Age_x_Kms": age_x_kms,
            "Age_x_Price": age_x_price,
            "Fuel_Type": data["Fuel_Type"],
            "Seller_Type": data["Seller_Type"],
            "Transmission": data["Transmission"]
        }])

        log_price = model.predict(df)[0]
        price = np.expm1(log_price)

        return {
            "predicted_price_lakhs": round(float(price), 2)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}
