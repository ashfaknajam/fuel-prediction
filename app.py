import pandas as pd
import streamlit as st
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_PATH = Path(__file__).with_name("auto-mpg.csv.csv")


@st.cache_data
def load_and_train_model():
    df = pd.read_csv(DATA_PATH)
    df = df.rename(columns={"model year": "model_year", "car name": "car_name"})
    df["horsepower"] = pd.to_numeric(df["horsepower"], errors="coerce")
    df["horsepower"] = df["horsepower"].fillna(df["horsepower"].median())

    X = df.drop(columns=["mpg"])
    y = df["mpg"]

    numeric_features = ["cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year"]
    categorical_features = ["origin", "car_name"]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(n_estimators=250, random_state=42)),
        ]
    )

    model.fit(X, y)
    return model


model = load_and_train_model()


st.set_page_config(page_title="MPG Predictor", page_icon="🚗", layout="centered")
st.title("Fuel Efficiency Predictor")
st.write("Enter the vehicle details below to estimate MPG.")

with st.form("mpg_form"):
    cylinders = st.number_input("Cylinders", min_value=3, max_value=8, value=4, step=1)
    displacement = st.number_input("Displacement", min_value=70.0, max_value=500.0, value=120.0, step=1.0)
    horsepower = st.number_input("Horsepower", min_value=40.0, max_value=300.0, value=90.0, step=1.0)
    weight = st.number_input("Weight", min_value=1500, max_value=5000, value=3000, step=10)
    acceleration = st.number_input("Acceleration", min_value=5.0, max_value=25.0, value=15.0, step=0.1)
    model_year = st.slider("Model Year", min_value=70, max_value=82, value=76, step=1)
    origin = st.selectbox("Origin", options=[1, 2, 3], format_func=lambda x: {1: "USA", 2: "Europe", 3: "Japan"}[x])
    car_name = st.text_input("Car Name", value="Toyota Corolla")

    submitted = st.form_submit_button("Predict")

if submitted:
    input_df = pd.DataFrame(
        [{
            "cylinders": cylinders,
            "displacement": displacement,
            "horsepower": horsepower,
            "weight": weight,
            "acceleration": acceleration,
            "model_year": model_year,
            "origin": origin,
            "car_name": car_name,
        }]
    )

    prediction = model.predict(input_df)[0]
    st.success(f"Predicted MPG: {prediction:.2f}")
