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

# 1. Carga de datos optimizada
@st.cache_data
def load_data(filepath):
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    return df

FILE_NAME = "electricity_consumption_based_weather.csv"

try:
    df_raw = load_data(FILE_NAME)
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

# Selector directo por botones para evitar encolamiento de peticiones
threshold_sigma = st.sidebar.radio(
    "Umbral de Desvío (Sigmas / Desvíos Estándar):",
    options=[1.5, 2.0, 2.5, 3.0, 3.5],
    index=1,
    format_func=lambda x: f"{x} σ",
    horizontal=True
)

# 3. Pipeline de Entrenamiento y Predicción en Caché
@st.cache_data
def train_and_predict(data, window):
    df = data.copy()
    
    # Feature Engineering
    df['temp_mean'] = (df['TMAX'] + df['TMIN']) / 2.0
    df[f'rolling_mean_{window}'] = df['daily_consumption'].shift(1).rolling(window=window).mean()
    df[f'rolling_std_{window}'] = df['daily_consumption'].shift(1).rolling(window=window).std()
    df['rolling_temp_mean'] = df['temp_mean'].shift(1).rolling(window=window).mean()

    df['lag_1'] = df['daily_consumption'].shift(1)
    df['lag_2'] = df['daily_consumption'].shift(2)

    df['dayofweek'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month

    df_model = df.dropna().copy()

    features = [
        'AWND', 'PRCP', 'TMAX', 'TMIN', 'temp_mean',
        f'rolling_mean_{window}', f'rolling_std_{window}',
        'rolling_temp_mean', 'lag_1', 'lag_2', 'dayofweek', 'month'
    ]
    target = 'daily_consumption'

    split_idx = int(len(df_model) * 0.8)
    train_df = df_model.iloc[:split_idx]
    test_df = df_model.iloc[split_idx:].copy()

    X_train, y_train = train_df[features], train_df[target]
    X_test, y_test = test_df[features], test_df[target]

    model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42)
    model.fit(X_train, y_train)

    test_df['prediction'] = model.predict(X_test)
    test_df['error'] = test_df[target] - test_df['prediction']
    
    mae_val = mean_absolute_error(y_test, test_df['prediction'])
    r2_val = r2_score(y_test, test_df['prediction'])

    return test_df, mae_val, r2_val, target

# Obtención de resultados precalculados
test_df, mae, r2, target = train_and_predict(df_raw, window_choice)

# 4. Clasificación dinámica de anomalías (Instantánea en milisegundos)
error_mean = test_df['error'].mean()
error_std = test_df['error'].std()
test_df['anomaly'] = np.abs(test_df['error'] - error_mean) > (threshold_sigma * error_std)
anomalies_count = int(test_df['anomaly'].sum())

# Métricas de cabecera
col1, col2, col3, col4 = st.columns(4)
col1.metric("MAE (Error Absoluto)", f"{mae:.2f}")
col2.metric("R² Score", f"{r2:.3f}")
col3.metric("Alarmas Detectadas", f"{anomalies_count}")
col4.metric("Total Puntos", f"{len(test_df)}")

# 5. Gráfico Plotly
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
    template="plotly_dark",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, width='stretch')

# 6. Exportación de Auditoría
csv_data = test_df[['date', target, 'prediction', 'error', 'anomaly']].to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Descargar Reporte de Alarmas (CSV)",
    data=csv_data,
    file_name=f"reporte_alarmas_{window_choice}d_{threshold_sigma}sigma.csv",
    mime="text/csv"
)