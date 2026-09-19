import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(
    page_title="Predicción de Consumo & Mitigación de Falsas Alarmas",
    layout="wide"
)

st.title("⚡ Monitoreo Predictivo & Mitigación de Alarmas")
st.markdown("Análisis multiventana para optimización operativa y control de desvíos.")

# 1. Carga de datos
@st.cache_data
def load_data(filepath):
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    return df

# Reemplazá con el nombre exacto de tu archivo si difiere
FILE_NAME = "electricity_consumption_based_weather.csv"

try:
    df = load_data(FILE_NAME)
except FileNotFoundError:
    st.error(f"No se encontró el archivo '{FILE_NAME}'. Verificá la ruta en tu proyecto.")
    st.stop()

# 2. Configuración en la barra lateral
st.sidebar.header("⚙️ Parámetros de Ventana & Alerta")
window_choice = st.sidebar.selectbox(
    "Ventana móvil de análisis (días):",
    options=[3, 7, 14, 30],
    index=1
)
threshold_sigma = st.sidebar.slider(
    "Umbral de Desvío (Sigmas / Desvíos Estándar):",
    min_value=1.5,
    max_value=3.5,
    value=2.0,
    step=0.1
)

# 3. Feature Engineering (Ventanas temporales)
df['temp_mean'] = (df['TMAX'] + df['TMIN']) / 2.0
df[f'rolling_mean_{window_choice}'] = df['daily_consumption'].shift(1).rolling(window=window_choice).mean()
df[f'rolling_std_{window_choice}'] = df['daily_consumption'].shift(1).rolling(window=window_choice).std()
df['rolling_temp_mean'] = df['temp_mean'].shift(1).rolling(window=window_choice).mean()

# Lags directos
df['lag_1'] = df['daily_consumption'].shift(1)
df['lag_2'] = df['daily_consumption'].shift(2)

# Features temporales
df['dayofweek'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month

df_model = df.dropna().copy()

features = [
    'AWND', 'PRCP', 'TMAX', 'TMIN', 'temp_mean',
    f'rolling_mean_{window_choice}', f'rolling_std_{window_choice}',
    'rolling_temp_mean', 'lag_1', 'lag_2', 'dayofweek', 'month'
]
target = 'daily_consumption'

# 4. División Train / Test (80% / 20% temporal)
split_idx = int(len(df_model) * 0.8)
train_df = df_model.iloc[:split_idx]
test_df = df_model.iloc[split_idx:].copy()

X_train, y_train = train_df[features], train_df[target]
X_test, y_test = test_df[features], test_df[target]

# 5. Modelo XGBoost
model = XGBRegressor(n_estimators=120, learning_rate=0.05, max_depth=4, random_state=42)
model.fit(X_train, y_train)

test_df['prediction'] = model.predict(X_test)
test_df['error'] = test_df[target] - test_df['prediction']

# 6. Detección de Desvíos / Falsas Alarmas
error_mean = test_df['error'].mean()
error_std = test_df['error'].std()
test_df['anomaly'] = np.abs(test_df['error'] - error_mean) > (threshold_sigma * error_std)

# Métricas operativas
mae = mean_absolute_error(y_test, test_df['prediction'])
r2 = r2_score(y_test, test_df['prediction'])
anomalies_count = test_df['anomaly'].sum()

col1, col2, col3, col4 = st.columns(4)
col1.metric("MAE (Error Absoluto Medio)", f"{mae:.2f}")
col2.metric("R² Score", f"{r2:.3f}")
col3.metric("Alarmas Detectadas", f"{anomalies_count}")
col4.metric("Total Puntos Evaluados", f"{len(test_df)}")

# 7. Gráfico Interactivo (Plotly)
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=test_df['date'], y=test_df[target],
    mode='lines', name='Consumo Real',
    line=dict(color='#1f77b4', width=2)
))

fig.add_trace(go.Scatter(
    x=test_df['date'], y=test_df['prediction'],
    mode='lines', name='Predicción Modelo',
    line=dict(color='#2ca02c', dash='dash')
))

anomalies = test_df[test_df['anomaly']]
fig.add_trace(go.Scatter(
    x=anomalies['date'], y=anomalies[target],
    mode='markers', name='Alerta Operativa (Desvío)',
    marker=dict(color='red', size=8, symbol='x')
))

fig.update_layout(
    title=f"Evaluación de Series Temporales (Ventana: {window_choice} días | Umbral: {threshold_sigma}σ)",
    xaxis_title="Fecha",
    yaxis_title="Consumo Diario",
    hovermode="x unified",
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)

# 8. Exportación de Resultados
csv_data = test_df[['date', target, 'prediction', 'error', 'anomaly']].to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Descargar Reporte de Alarmas (CSV)",
    data=csv_data,
    file_name="reporte_alarmas_consumo.csv",
    mime="text/csv"
)