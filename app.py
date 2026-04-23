import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Title
# -----------------------------
st.title("📊 NPA Risk Prediction System")
st.write("Predict whether the banking system is under High NPA Stress")

# -----------------------------
# Dataset (same as your code)
# -----------------------------
data = {
    'Year': [2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,
             2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,
             2020,2021,2022,2023],
    'GNPA_pct': [12.4,11.4,10.4,8.8,7.2,5.2,3.3,2.5,2.3,2.2,
                 2.4,2.7,3.4,4.0,4.5,5.9,9.2,10.2,11.2,9.1,
                 7.5,6.9,5.0,3.9],
    'GDP_growth': [5.4,4.8,3.8,8.0,7.9,9.3,9.3,9.8,3.9,8.4,
                   10.3,6.6,5.5,6.4,7.4,8.0,8.3,6.8,6.5,4.0,
                   -6.6,8.7,7.2,6.3],
    'Inflation_CPI':[4.0,3.7,4.3,3.8,3.8,4.4,6.7,6.4,8.3,10.8,
                     12.0,8.9,9.3,10.9,6.4,4.9,4.5,3.6,3.4,4.8,
                     6.2,5.5,6.7,5.4],
    'Repo_rate':[8.0,7.0,6.5,6.0,6.0,6.25,7.25,7.75,7.5,5.0,
                 5.25,6.5,8.0,7.75,8.0,6.75,6.25,6.0,6.5,5.15,
                 4.0,4.0,6.25,6.5],
    'Credit_growth':[17,16,15,12,10,27,31,28,25,17,
                     21,19,17,15,9,9,5,6,13,7,
                     6,7,15,16]
}

df = pd.DataFrame(data)

# -----------------------------
# Feature Engineering
# -----------------------------
df['GDP_Lag'] = df['GDP_growth'].shift(1)
df['Credit_Lag'] = df['Credit_growth'].shift(1)
df.dropna(inplace=True)

# Target variable
THRESHOLD = 7.0
df['High_NPA'] = (df['GNPA_pct'] > THRESHOLD).astype(int)

# Features
features = ['GDP_growth','Inflation_CPI','Repo_rate',
            'Credit_growth','GDP_Lag','Credit_Lag']

X = df[features]
y = df['High_NPA']

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Model
model = LogisticRegression(max_iter=1000)
model.fit(X_scaled, y)

# -----------------------------
# USER INPUT
# -----------------------------
st.subheader("Enter Economic Indicators")

gdp = st.number_input("GDP Growth (%)", value=6.0)
inflation = st.number_input("Inflation (%)", value=5.0)
repo = st.number_input("Repo Rate (%)", value=6.5)
credit = st.number_input("Credit Growth (%)", value=10.0)
gdp_lag = st.number_input("Previous GDP Growth (%)", value=6.0)
credit_lag = st.number_input("Previous Credit Growth (%)", value=10.0)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict NPA Risk"):
    input_data = np.array([[gdp, inflation, repo, credit, gdp_lag, credit_lag]])
    input_scaled = scaler.transform(input_data)

    prob = model.predict_proba(input_scaled)[0][1]
    pred = model.predict(input_scaled)[0]

    if pred == 1:
        st.error(f"⚠ HIGH NPA RISK ({prob*100:.2f}%)")
    else:
        st.success(f"✅ LOW NPA RISK ({prob*100:.2f}%)")

# -----------------------------
# Show Dataset
# -----------------------------
if st.checkbox("Show Dataset"):
    st.dataframe(df)