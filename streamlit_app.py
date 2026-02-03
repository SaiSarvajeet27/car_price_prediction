import streamlit as st
import requests
from datetime import datetime

# --------------------------------------------------
# App Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="centered"
)

API_URL = "http://127.0.0.1:8000/predict"

# --------------------------------------------------
# Title Section
# --------------------------------------------------
st.title("🚗 Used Car Price Prediction")
st.markdown(
    """
    Predict the **resale price of a car** using a machine learning model.
    Enter the car details below and get an instant estimate.
    """
)

st.divider()

# --------------------------------------------------
# Input Form
# --------------------------------------------------
with st.form("car_price_form"):
    col1, col2 = st.columns(2)

    with col1:
        year = st.number_input(
            "Manufacturing Year",
            min_value=1990,
            max_value=datetime.now().year,
            value=2018
        )

        present_price = st.number_input(
            "Showroom Price (in Lakhs)",
            min_value=0.1,
            value=7.5
        )

        kms_driven = st.number_input(
            "Kilometers Driven",
            min_value=0,
            step=1000,
            value=45000
        )

        owner = st.selectbox(
            "Number of Previous Owners",
            options=[0, 1, 2, 3]
        )

    with col2:
        fuel_type = st.selectbox(
            "Fuel Type",
            options=["Petrol", "Diesel", "CNG"]
        )

        seller_type = st.selectbox(
            "Seller Type",
            options=["Dealer", "Individual"]
        )

        transmission = st.selectbox(
            "Transmission Type",
            options=["Manual", "Automatic"]
        )

    submit = st.form_submit_button("🔍 Predict Price")

if submit:
    payload = {
        "Year": year,
        "Present_Price": present_price,
        "Kms_Driven": kms_driven,
        "Fuel_Type": fuel_type,
        "Seller_Type": seller_type,
        "Transmission": transmission,
        "Owner": owner
    }

    try:
        with st.spinner("Predicting price..."):
            response = requests.post(API_URL, json=payload, timeout=5)

        if response.status_code == 200:
            result = response.json()
            price = result["predicted_price_lakhs"]

            st.success("✅ Prediction Successful")
            st.metric(
                label="Estimated Selling Price",
                value=f"₹ {price} Lakhs"
            )

        else:
            st.error(f"❌ API Error ({response.status_code})")
            st.json(response.json())

    except requests.exceptions.ConnectionError:
        st.error("🚫 Cannot connect to API. Is FastAPI running?")
    except requests.exceptions.Timeout:
        st.error("⏱️ API request timed out. Try again.")
    except Exception as e:
        st.error(f"Unexpected error: {e}")

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.divider()
st.caption("Built with Streamlit + FastAPI + Machine Learning")
