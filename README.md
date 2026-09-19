# ⚡ Monitoreo Predictivo & Mitigación de Falsas Alarmas en Consumo Eléctrico

Plataforma analítica e interactiva diseñada para la supervisión del consumo eléctrico industrial, reducción de fatiga de alarmas mediante ventanas móviles de desviación estadística ($N\sigma$) y modelado predictivo con **XGBoost**.

🔗 **Demo en vivo:** [https://prediccion-consumo-electrico.streamlit.app/](https://prediccion-consumo-electrico.streamlit.app/)

---

## 🎯 Problema de Negocio y Operativo

En centros de monitoreo y plantas industriales, los umbrales fijos de telemetría generan un exceso de alertas espurias debidas a estacionalidad, clima y picos normales de operación. Esto deriva en:

- **Fatiga de alarmas:** Los operadores desestiman avisos críticos reales.
- **Costos por paradas innecesarias:** Despacho de mantenimiento preventivo sin anomalías físicas.

**Solución implementada:** Un modelo de Machine Learning ajustado a variables exógenas (temperatura y variables climáticas) combinado con bandas móviles adaptativas que aíslan únicamente desvíos estadísticamente representativos.

---

## 🛠️ Stack Tecnológico

- **Lenguaje:** Python 3.10+
- **Machine Learning:** Scikit-Learn, XGBoost
- **Análisis de Datos:** Pandas, NumPy
- **Visualización & Dashboard:** Streamlit, Plotly
- **Control de Versiones & Despliegue:** Git, GitHub, Streamlit Community Cloud

---

## 📊 Arquitectura de la Solución

1. **Ingeniería de Características:** Integración de patrones de estacionalidad temporal y variables climáticas exógenas.
2. **Modelo Predictivo:** Regresión supervisada evaluada bajo métricas de negocio ($MAE$ y $R^2$).
3. **Filtro Estadístico Multiventana:** Monitoreo dinámico del error residual ($|Y_{real} - \hat{Y}|$) evaluado en ventanas móviles parametrizables (días) y umbrales en sigmas ($\sigma$).
4. **Exportación Operativa:** Generación de reportes de auditoría en formato CSV para los puntos anómalos clasificados.

---

## 🚀 Ejecución Local
