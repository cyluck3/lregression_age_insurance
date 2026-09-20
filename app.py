import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 1. Generación de datos sintéticos con caché
@st.cache_data
def generate_data(size=1000, random_seed=42):
    np.random.seed(random_seed)
    dataset = {
        "age": np.random.randint(18, 60, size=size),
        "gender": np.random.choice(["M", "F"], size=size)
    }
    df = pd.DataFrame(dataset)
    df["insurance"] = np.where(df.age < 25, 0, 1)
    df["insurance_cost"] = np.where(df.insurance == 1, df.age ** 2, 0)
    return df


# 2. Entrenamiento de modelos con caché para alta velocidad
@st.cache_resource
def train_models(df):
    # Preprocesadores reutilizables
    cat_features = Pipeline([
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    
    num_features = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    # --- MODELO LOGÍSTICO (Clasificación) ---
    X_clf = df[["age", "gender"]]
    y_clf = df["insurance"]

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
    )

    transformer_clf = ColumnTransformer([
        ("num", num_features, ["age"]),
        ("cat", cat_features, ["gender"])
    ])

    model_clf = Pipeline([
        ("transformer", transformer_clf),
        ("model", LogisticRegression())
    ])
    model_clf.fit(X_train_c, y_train_c)

    # --- MODELO LINEAL (Regresión Polinomial) ---
    X_lm = df[["gender", "insurance", "age"]]
    y_lm = df["insurance_cost"]

    X_train_lm, X_test_lm, y_train_lm, y_test_lm = train_test_split(
        X_lm, y_lm, test_size=0.2, random_state=42
    )

    transformer_lm = ColumnTransformer([
        ("num", num_features, ["insurance", "age"]),
        ("cat", cat_features, ["gender"])
    ])

    model_lm = Pipeline([
        ("transformer", transformer_lm),
        ("Polynom", PolynomialFeatures(degree=3)),
        ("model", LinearRegression())
    ])
    model_lm.fit(X_train_lm, y_train_lm)

    y_pred_lm = model_lm.predict(X_test_lm)

    metrics_lm = {
        "r2": r2_score(y_test_lm, y_pred_lm),
        "mse": mean_squared_error(y_test_lm, y_pred_lm),
        "mae": mean_absolute_error(y_test_lm, y_pred_lm)
    }

    return model_clf, model_lm, metrics_lm


class InsuranceProjectApp:
    def __init__(self, df):
        self.df = df
        self.model, self.linear_model, self.metrics_lm = train_models(self.df)

    def predict_logistic(self, test_df):
        return self.model.predict(test_df)

    def predict_proba_logistic(self, test_df):
        return self.model.predict_proba(test_df)[0][1]

    def config(self):
        st.set_page_config(page_title="Insurance Cost Prediction", layout="wide")

    def header(self):
        st.title("📈 Insurance Cost Prediction")
        st.write("Predict the insurance cost based on age, gender, and insurance status.")

    def body(self):
        self.display_metrics()

        co1, co2 = st.columns(2)
        with co1:
            st.header("Graphic 1.1")
            st.write("Insurance cost is the y-axis, and age is the x-axis.")
            self.scatter_age_insurance()
        with co2:
            st.header("Graphic 1.2")
            st.write("Gender Distribution / Insurance Cost")
            self.bar_gender_insurance()

        co3, co4 = st.columns(2)
        with co3:
            st.header("Graphic 2.1")
            st.write("Logistic Regression Model learning the real probability curve.")
            self.scatter_sigmoid_insurance()
        with co4:
            st.header("Insurance Status Distribution")
            st.write("Insurance Status: 0: No Insurance, 1: Insurance")
            st.write(self.df["insurance"].value_counts())

            st.header("Insurance / Gender")
            st.write(self.df.groupby("gender")["insurance"].value_counts())

    def footer(self):
        co1, co2 = st.columns(2)

        with co1:
            st.header("Model (Logistic Regression)")
            st.markdown("### 🤖 How old are you?")

            age = st.slider("Age", 0, 100, 25)

            sc1, sc2 = st.columns(2)
            with sc1:
                malfem = st.radio("Gender", ["M", "F"])
            with sc2:
                st.header(f"{age} years old")

            input_data = pd.DataFrame({"age": [age], "gender": [malfem]})
            prediction = self.predict_logistic(input_data)[0]
            probability = self.predict_proba_logistic(input_data)

            self.scatter_model_dynamic(probability)

            if prediction == 1:
                st.success("Congratulations, you got insurance!")
            else:
                st.error("Sorry, no insurance for you yet!")

        with co2:
            st.header("Linear Regression Prediction")
            
            lm_input = pd.DataFrame({
                "gender": [malfem],
                "insurance": [prediction],
                "age": [age]
            })

            cost_prediction = self.linear_model.predict(lm_input)[0]

            sc1, sc2 = st.columns(2)
            with sc1:
                st.header(f"Your estimated insurance cost:")

            with sc2:
                st.title(f"${cost_prediction:,.2f}")

            st.write("---")

            st.header("LR Metrics (Model Performance)")
            st.write("---")
            sc2_5, sc3, sc4 = st.columns(3)
            with sc2_5:
                st.info(f"R² Score: {self.metrics_lm['r2']:.4f}")
                
            with sc3:
                st.info(f"MSE: {self.metrics_lm['mse']:,.2f}")
            with sc4:
                st.info(f"MAE: {self.metrics_lm['mae']:,.2f}")

        st.write("---")
        df_display = self.df.copy()
        df_display.columns = df_display.columns.str.upper()
        st.markdown("### Raw Data for Training")
        st.dataframe(df_display)

    def display_metrics(self):
        st.header("General Metrics")
        co1, co2, co3, co4 = st.columns(4)

        co1.metric("Mean Age", int(np.mean(self.df["age"])))
        
        m_count = self.df[self.df["gender"] == "M"].shape[0]
        f_count = self.df[self.df["gender"] == "F"].shape[0]
        total = self.df.shape[0]

        co2.metric("Male", m_count, f"{(m_count / total) * 100:.1f}%")
        co3.metric("Female", f_count, f"{(f_count / total) * 100:.1f}%")
        co4.metric("Total Insurance Cost", f"${self.df.insurance_cost.sum():,}")

    def scatter_age_insurance(self):
        fig = px.scatter(self.df, x="age", y="insurance_cost", color="gender",
                         title="<b>Scatter Plot of Age vs Insurance Cost</b>")
        
        st.plotly_chart(fig, use_container_width=True)

    def bar_gender_insurance(self):
        fig = px.bar(self.df, x="gender", y="insurance_cost", color="gender",
                     title="<b>Bar Plot of Gender / Insurance Cost</b>")
        
        st.plotly_chart(fig, use_container_width=True)

    def scatter_sigmoid_insurance(self):
        # Muestra la sigmoide calculada en tiempo real por la regresión logística
        ages_range = np.linspace(18, 60, 300)
        synth_df = pd.DataFrame({"age": ages_range, "gender": ["M"] * 300})
        probs = self.model.predict_proba(synth_df)[:, 1]

        fig = px.scatter(self.df, x="age", y="insurance",
                         labels={"insurance": "Insurance (1/0)", "age": "Age"},
                         title="<b>Real Logistic Regression Sigmoidal Curve</b>")
        
        fig.add_trace(go.Scatter(
            x=ages_range,
            y=probs,
            mode='lines',
            name='Logistic Curve',
            line=dict(color='Red', width=3)
        ))
        
        st.plotly_chart(fig, use_container_width=True)

    def scatter_model_dynamic(self, probability):
        progress_val = int(probability * 100)
        
        st.progress(
            progress_val,
            text="You are too young for insurance!" if progress_val < 50 else "Alright, take your insurance!"
        )
        st.write(f"Transform/Approval Probability: **{probability * 100:.2f}%**")


if __name__ == "__main__":
    raw_df = generate_data()
    app = InsuranceProjectApp(raw_df)
    app.config()
    app.header()
    app.body()
    app.footer()