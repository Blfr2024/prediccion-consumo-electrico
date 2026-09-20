import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(
    page_title="Predicción de Consumo & Mitigación de Falsas Alarmas",
    layout="wide"
)

st.title("⚡ Monitoreo Predictivo & Mitigación de Alarmas")
st.markdown("Análisis multiventana y filtro de persistencia temporal para optimización operativa.")

# ---------------------------------------------------------
# 1. CARGA DE DATOS
# ---------------------------------------------------------
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
    st.error(f"No se encontró el archivo '{FILE_NAME}'. Verificá la ruta en el directorio.")
    st.stop()

# ---------------------------------------------------------
# 2. CONFIGURACIÓN EN LA BARRA LATERAL
# ---------------------------------------------------------
st.sidebar.header("⚙️ Parámetros de Ventana & Alerta")

window_choice = st.sidebar.selectbox(
    "Ventana móvil de análisis (días):",
    options=[3, 7, 14, 30],
    index=1
)

threshold_sigma = st.sidebar.radio(
    "Umbral de Desvío (Sigmas / Desvíos Estándar):",
    options=[1.5, 2.0, 2.5, 3.0, 3.5],
    index=1,
    format_func=lambda x: f"{x} σ",
    horizontal=True
)

# MEJORA PEDIDA POR VÍCTOR: Filtro de persistencia por ciclos
ciclos_persistencia = st.sidebar.slider(
    "Persistencia requerida (ciclos consecutivos):",
    min_value=1,
    max_value=10,
    value=3,
    help="Número de periodos continuos fuera de umbral para confirmar la alarma y descartar ruido transitorio."
)

# ---------------------------------------------------------
# 3. ENTRENAMIENTO DEL MODELO XGBOOST Y RESIDUOS
# ---------------------------------------------------------
# Variables predictoras típicas del dataset
features = [col for col in ['temperature', 'humidity', 'wind_speed', 'pressure'] if col in df_raw.columns]
target_col = [col for col in ['electricity_consumption', 'consumption', 'power_consumption'] if col in df_raw.columns]

if not target_col:
    # Si las columnas tienen nombres alternativos, tomamos la última numérica
    target = df_raw.select_dtypes(include=[np.number]).columns[-1]
else:
    target = target_col[0]

if not features:
    features = [c for c in df_raw.select_dtypes(include=[np.number]).columns if c != target]

df = df_raw.copy()

# Modelo XGBoost
X = df[features]
y = df[target]
model = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.08, random_state=42)
model.fit(X, y)
df['prediccion'] = model.predict(X)

# Cálculo de residuo: diferencia entre consumo real y estimado
df['residuo'] = df[target] - df['prediccion']

# Umbral dinámico adaptativo (ventana móvil en registros)
std_movil = df['residuo'].rolling(window=window_choice, min_periods=2).std().bfill()
df['umbral_adaptativo'] = threshold_sigma * std_movil

# ---------------------------------------------------------
# 4. ALARMAS: INSTANTÁNEA VS PERSISTENTE
# ---------------------------------------------------------
df['alarma_instantanea'] = df['residuo'].abs() > df['umbral_adaptativo']

def aplicar_filtro_persistencia(residuos, umbrales, k_ciclos):
    contador = 0
    estado = []
    for r, u in zip(residuos, umbrales):
        if abs(r) > u:
            contador += 1
        else:
            contador = 0
        estado.append(contador >= k_ciclos)
    return estado

df['alarma_persistente'] = aplicar_filtro_persistencia(
    df['residuo'], df['umbral_adaptativo'], ciclos_persistencia
)

# ---------------------------------------------------------
# 5. TARJETAS DE MÉTRICAS OPERATIVAS
# ---------------------------------------------------------
total_inst = int(df['alarma_instantanea'].sum())
total_pers = int(df['alarma_persistente'].sum())
falsas_descartadas = total_inst - total_pers

mae = mean_absolute_error(y, df['prediccion'])
r2 = r2_score(y, df['prediccion'])

col1, col2, col3, col4 = st.columns(4)
col1.metric("Precisión Modelo (R²)", f"{r2:.3f}")
col2.metric("MAE de Predicción", f"{mae:.2f}")
col3.metric("Alarmas Instantáneas (Brutas)", total_inst)
col4.metric(
    f"Alarmas Confirmadas (N≥{ciclos_persistencia})",
    total_pers,
    delta=f"-{falsas_descartadas} falsas/espurias",
    delta_color="inverse"
)

# ---------------------------------------------------------
# 6. GRÁFICOS INTERACTIVOS (PLOTLY EN STREAMLIT)
# ---------------------------------------------------------
fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.08,
    subplot_titles=(
        f"Consumo Real vs. Modelo Predictivo ({target})",
        "Residuos Absolutos, Umbral Adaptativo y Supresión de Ruido"
    )
)

# Panel 1: Curva real vs esperada
fig.add_trace(go.Scatter(x=df['date'], y=df[target], name="Consumo Real", line=dict(color="#00D2FF", width=1.5)), row=1, col=1)
fig.add_trace(go.Scatter(x=df['date'], y=df['prediccion'], name="Estimación XGBoost", line=dict(color="#FFB300", dash="dash")), row=1, col=1)

# Panel 2: Residuos y umbrales
fig.add_trace(go.Scatter(x=df['date'], y=df['residuo'].abs(), name="|Residuo|", line=dict(color="#888888", width=1)), row=2, col=1)
fig.add_trace(go.Scatter(x=df['date'], y=df['umbral_adaptativo'], name=f"Umbral ({threshold_sigma}σ)", line=dict(color="#FF5252", dash="dot")), row=2, col=1)

# Puntos de alarma instantánea (cruces amarillas)
df_inst = df[df['alarma_instantanea']]
fig.add_trace(go.Scatter(
    x=df_inst['date'], y=df_inst['residuo'].abs(),
    mode="markers", name="Alarma Instantánea (Pico)",
    marker=dict(color="#FFD600", symbol="x", size=7)
), row=2, col=1)

# Puntos confirmados por persistencia (círculos rojos)
df_pers = df[df['alarma_persistente']]
fig.add_trace(go.Scatter(
    x=df_pers['date'], y=df_pers['residuo'].abs(),
    mode="markers", name=f"Alarma Confirmada (N≥{ciclos_persistencia})",
    marker=dict(color="#FF1744", symbol="circle-open", size=10, line=dict(width=2))
), row=2, col=1)

fig.update_layout(
    template="plotly_dark",
    height=650,
    hovermode="x unified",
    margin=dict(l=20, r=20, t=40, b=20)
)

st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# 7. TABLA DE AUDITORÍA OPERATIVA
# ---------------------------------------------------------
st.subheader("📋 Registro de Eventos Confirmados")
eventos_reales = df[df['alarma_persistente']][['date', target, 'prediccion', 'residuo', 'umbral_adaptativo']]
st.dataframe(eventos_reales.tail(15), use_container_width=True)