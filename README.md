# Insurance Cost & Eligibility Prediction App

An interactive web application built with Streamlit, scikit-learn, and Plotly to model insurance coverage eligibility and estimate premium costs based on demographic data.

---

## Overview

This application demonstrates an end-to-end Machine Learning pipeline integrated into a real-time web dashboard. It uses synthetic demographic data to showcase two distinct modeling tasks:

1. **Classification (Eligibility)**: A Logistic Regression model predicts whether an individual qualifies for insurance coverage based on age and gender.
2. **Regression (Cost Estimation)**: A Polynomial Regression model (3rd-degree polynomial) predicts the expected insurance cost based on age, gender, and insurance status.

---

## Key Features

- **In-Memory Synthetic Data Generation**: Generates structured benchmark data and caches it for rapid processing using `@st.cache_data`.
- **Cached Model Training**: Trains scikit-learn pipelines with standard preprocessing (One-Hot Encoding, Feature Scaling, Mean Imputation) cached via `@st.cache_resource`.
- **Interactive Visualizations**: Powered by Plotly, including live scatter plots, probability sigmoid curves, and demographic distribution bar charts.
- **Real-Time Predictions**: Allows users to input custom age and gender parameters via sliders and selectors to receive dynamic eligibility probabilities and cost estimates.
- **Model Evaluation Metrics**: Reports regression metrics ($R^2$, Mean Squared Error, and Mean Absolute Error) directly within the UI.

---

## Technical Stack

- **Frontend / Dashboard**: Streamlit
- **Data Manipulation**: Pandas, NumPy
- **Machine Learning**: scikit-learn (`LogisticRegression`, `LinearRegression`, `PolynomialFeatures`, `Pipeline`, `ColumnTransformer`)
- **Data Visualization**: Plotly Express, Plotly Graph Objects

---

## Project Structure

```text
.
├── app.py              # Main application logic, UI, and ML pipelines
├── requirements.txt    # Python dependencies
└── README.md           # Documentation
```

---

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/your-repository-name.git
   cd your-repository-name
   ```

2. **Create and activate a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install numpy pandas streamlit plotly scikit-learn
   ```

4. **Run the Streamlit application:**
   ```bash
   streamlit run app.py
   ```

---

## Pipeline Architecture

```text
Raw Input Data
   │
   ├── Categorical Features (Gender) ──> OneHotEncoder ──────────────┐
   │                                                                 ├──> ColumnTransformer ──> Model
   └── Numerical Features (Age/Status) ──> Imputer ──> StandardScaler ─┘
```

- **Logistic Pipeline**: Preprocesses age and gender to output class probability via Logistic Regression.
- **Regression Pipeline**: Transforms numerical features through 3rd-degree polynomial expansion before passing data to Linear Regression.

---

## Author

Created by **Luck888**.