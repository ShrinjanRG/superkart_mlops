"""Streamlit Web Application for SuperKart Sales Forecasting."""

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="SuperKart Sales Forecast", page_icon="🛒", layout="centered")

@st.cache_resource
def load_model(model_path: str = "models/best_model.joblib"):
    """Load the trained sales forecasting pipeline."""
    return joblib.load(model_path)


st.title("🛒 SuperKart Sales Revenue Forecaster")
st.markdown(
    "Enter product and store attributes below to predict **Product Store Sales Total**."
)

model = load_model()

STORE_LOOKUP = {
    "OUT001": {"year": 1987, "size": "High", "city": "Tier 2", "type": "Supermarket Type1"},
    "OUT002": {"year": 1998, "size": "Small", "city": "Tier 3", "type": "Food Mart"},
    "OUT003": {"year": 1999, "size": "Medium", "city": "Tier 1", "type": "Departmental Store"},
    "OUT004": {"year": 2009, "size": "Medium", "city": "Tier 2", "type": "Supermarket Type2"},
}

PRODUCT_TYPES = [
    "Baking Goods", "Breads", "Breakfast", "Canned", "Dairy",
    "Frozen Foods", "Fruits and Vegetables", "Hard Drinks",
    "Health and Hygiene", "Household", "Meat", "Others",
    "Seafood", "Snack Foods", "Soft Drinks", "Starchy Foods",
]

with st.form("prediction_form"):
    st.subheader("Product Details")
    col1, col2 = st.columns(2)
    with col1:
        product_prefix = st.selectbox("Product Category Prefix", ["FD", "NC", "DR"])
        product_type = st.selectbox("Product Type", PRODUCT_TYPES)
        product_sugar = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
    with col2:
        product_weight = st.number_input("Product Weight", min_value=1.0, max_value=30.0, value=12.65, step=0.1)
        product_area = st.number_input("Product Allocated Area", min_value=0.001, max_value=0.35, value=0.068, step=0.005, format="%.3f")
        product_mrp = st.number_input("Product MRP ($)", min_value=10.0, max_value=300.0, value=147.0, step=1.0)

    st.subheader("Store Details")
    store_id = st.selectbox("Store ID", list(STORE_LOOKUP.keys()))
    store_meta = STORE_LOOKUP[store_id]

    col3, col4 = st.columns(2)
    with col3:
        store_year = st.number_input("Store Establishment Year", value=store_meta["year"], disabled=True)
        store_size = st.text_input("Store Size", value=store_meta["size"], disabled=True)
    with col4:
        store_city = st.text_input("Store Location City Type", value=store_meta["city"], disabled=True)
        store_type = st.text_input("Store Type", value=store_meta["type"], disabled=True)

    submitted = st.form_submit_button("Predict Sales Revenue")

if submitted:
    input_df = pd.DataFrame(
        [
            {
                "Product_Weight": product_weight,
                "Product_Sugar_Content": product_sugar,
                "Product_Allocated_Area": product_area,
                "Product_Type": product_type,
                "Product_MRP": product_mrp,
                "Store_Id": store_id,
                "Store_Establishment_Year": int(store_year),
                "Store_Size": store_size,
                "Store_Location_City_Type": store_city,
                "Store_Type": store_type,
                "Product_Category_Prefix": product_prefix,
                "Store_Age": 2025 - int(store_year),
            }
        ]
    )

    prediction = float(model.predict(input_df)[0])
    st.success(f"Predicted Product Store Sales Total: **${prediction:,.2f}**")
    with st.expander("View Input DataFrame"):
        st.dataframe(input_df)
