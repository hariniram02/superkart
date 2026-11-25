import os
import requests
import streamlit as st

st.set_page_config(page_title="SuperKart Sales Prediction", page_icon="🛒", layout="centered")

st.title("SuperKart Sales Prediction App")
st.caption("Enter product and store attributes to predict `Product_Store_Sales_Total`.")

default_url = os.getenv("BACKEND_URL", "http://superkart-backend:5000/predict")
predict_url = st.text_input("Backend Predict Endpoint URL", value=default_url)

with st.form("superkart_form"):
    col1, col2 = st.columns(2)

    with col1:
        product_weight = st.number_input(
            "Product Weight", min_value=0.0, max_value=100.0, value=12.65, step=0.05
        )
        product_allocated_area = st.number_input(
            "Product Allocated Area",
            min_value=0.0,
            max_value=1.0,
            value=0.069,
            step=0.005,
            format="%.4f",
        )
        product_mrp = st.number_input(
            "Product MRP", min_value=0.0, max_value=1000.0, value=147.03, step=0.5
        )
        product_sugar_content = st.selectbox(
            "Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"]
        )
        product_type = st.selectbox(
            "Product Type",
            [
                "Fruits and Vegetables",
                "Snack Foods",
                "Frozen Foods",
                "Dairy",
                "Household",
                "Baking Goods",
                "Canned",
                "Health and Hygiene",
                "Soft Drinks",
                "Meat",
                "Breads",
                "Hard Drinks",
                "Others",
                "Starchy Foods",
                "Breakfast",
                "Seafood",
            ],
        )

    with col2:
        store_establishment_year = st.number_input(
            "Store Establishment Year",
            min_value=1980,
            max_value=2026,
            value=2000,
            step=1,
        )
        store_size = st.selectbox("Store Size", ["Small", "Medium", "High"])
        store_location_city_type = st.selectbox(
            "Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"]
        )
        store_type = st.selectbox(
            "Store Type",
            [
                "Supermarket Type1",
                "Supermarket Type2",
                "Departmental Store",
                "Food Mart",
            ],
        )
        product_category_prefix = st.selectbox(
            "Product Category Prefix", ["FD", "NC", "DR"]
        )

    submitted = st.form_submit_button("Predict Total Sales")

if submitted:
    store_age = 2026 - int(store_establishment_year)
    payload = {
        "Product_Weight": float(product_weight),
        "Product_Allocated_Area": float(product_allocated_area),
        "Product_MRP": float(product_mrp),
        "Product_Sugar_Content": product_sugar_content,
        "Product_Type": product_type,
        "Store_Establishment_Year": int(store_establishment_year),
        "Store_Size": store_size,
        "Store_Location_City_Type": store_location_city_type,
        "Store_Type": store_type,
        "Product_Category_Prefix": product_category_prefix,
        "Store_Age": store_age,
    }

    try:
        resp = requests.post(predict_url, json=payload, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        prediction = data["predictions"][0]
        st.success(f"Predicted `Product_Store_Sales_Total`: **${prediction:,.2f}**")
    except requests.exceptions.RequestException as err:
        st.error(f"Request to Flask backend failed: {err}")
