import inspect
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# ═════════════ Idioma ═════════════
# Mismo estado que usa Interfaz.py (st.session_state["idioma"]): "es" o "en".
TX = {
    "titulo": ("Predicción de Detecciones (15 Días)", "Detection Forecast (15 Days)"),
    "escala": ("Escala de análisis:", "Analysis scale:"),
    "region": ("Región de", "Region of"),
    "proyectando": ("Proyectando desde:", "Forecasting from:"),
    "sin_datos": ("No hay datos históricos para entrenar en esta selección.",
                  "There is no historical data to train on in this selection."),
    "procesando": ("Procesando datos continuos y entrenando modelo...",
                   "Processing continuous data and training model..."),
    "pocos_datos": ("No hay suficientes datos históricos en esta selección para entrenar el modelo (se requieren al menos 30 días).",
                    "Not enough historical data in this selection to train the model (at least 30 days are required)."),
    "eval": ("Evaluación del Modelo", "Model Evaluation"),
    "mae": ("Margen de Error Medio (MAE)", "Mean Absolute Error (MAE)"),
    "mae_unidad": ("{v} focos/día", "{v} detections/day"),
    "mae_delta": ("Evaluado en últimos 15 días", "Evaluated on the last 15 days"),
    "total": ("Total Focos Proyectados (15d)", "Total Projected Detections (15d)"),
    "total_unidad": ("{v} focos", "{v} detections"),
    "riesgo": ("Día de mayor riesgo futuro", "Highest-risk future day"),
    "tr_hist": ("Histórico Real", "Actual History"),
    "tr_eval": ("Predicción (Evaluación)", "Prediction (Evaluation)"),
    "tr_proy": ("Proyección 15 Días", "15-Day Projection"),
    "graf_titulo": ("Detecciones Diarias: Histórico, Evaluación y Proyección",
                    "Daily Detections: History, Evaluation and Projection"),
    "detalle": ("Detalle de Proyección", "Projection Detail"),
    "col_fecha": ("Fecha", "Date"),
    "col_focos": ("Focos Estimados", "Estimated Detections"),
}

STRETCH = ({"width": "stretch"} if "width" in inspect.signature(st.button).parameters
           else {"use_container_width": True})


def idioma() -> str:
    v = st.session_state.get("idioma", "es")
    return v if v in ("es", "en") else "es"


def t(clave: str, **valores) -> str:
    par = TX.get(clave)
    if par is None:
        return clave
    texto = par[0] if idioma() == "es" else par[1]
    return texto.format(**valores) if valores else texto


def fmt_fecha(ts) -> str:
    return ts.strftime("%d-%m-%Y") if idioma() == "es" else ts.strftime("%Y-%m-%d")


def fmt_dec(n, d=1) -> str:
    s = f"{n:,.{d}f}"
    return s.replace(",", "§").replace(".", ",").replace("§", ".") if idioma() == "es" else s


def fmt_int(n) -> str:
    s = f"{int(round(n)):,}"
    return s.replace(",", ".") if idioma() == "es" else s


# ═════════════ Modelo ═════════════
def preparar_datos_diarios(df, fecha_base):
    """Agrupa por día, rellena los días sin focos con 0 y extrae características."""
    diario = df.groupby(df['fecha'].dt.date).size().reset_index(name='detecciones')
    diario['fecha'] = pd.to_datetime(diario['fecha'])

    # Crear un calendario continuo hasta la fecha base elegida
    rango_fechas = pd.date_range(start=diario['fecha'].min(), end=pd.to_datetime(fecha_base))
    diario_completo = pd.DataFrame({'fecha': rango_fechas})

    # Unir datos y rellenar con ceros los días sin incendios
    diario = pd.merge(diario_completo, diario, on='fecha', how='left').fillna(0)

    diario['dia_año'] = diario['fecha'].dt.dayofyear
    diario['mes'] = diario['fecha'].dt.month
    diario['año'] = diario['fecha'].dt.year
    return diario


def entrenar_y_predecir(diario, fecha_base, horizonte=15):
    if len(diario) < 30:
        return None, None, None, None

    train = diario.iloc[:-horizonte]
    test = diario.iloc[-horizonte:]

    X_train, y_train = train[['dia_año', 'mes', 'año']], train['detecciones']
    X_test, y_test = test[['dia_año', 'mes', 'año']], test['detecciones']

    modelo = RandomForestRegressor(n_estimators=100, random_state=42)
    modelo.fit(X_train, y_train)

    test_pred = modelo.predict(X_test)
    error_mae = mean_absolute_error(y_test, test_pred)

    # Predicción futura estrictamente a partir de la Fecha Base
    fechas_futuras = [pd.to_datetime(fecha_base) + pd.Timedelta(days=i) for i in range(1, horizonte + 1)]
    df_futuro = pd.DataFrame({'fecha': fechas_futuras})
    df_futuro['dia_año'] = df_futuro['fecha'].dt.dayofyear
    df_futuro['mes'] = df_futuro['fecha'].dt.month
    df_futuro['año'] = df_futuro['fecha'].dt.year

    prediccion_futura = modelo.predict(df_futuro[['dia_año', 'mes', 'año']])
    df_futuro['prediccion'] = np.round(prediccion_futura).astype(int)

    return test, test_pred, error_mae, df_futuro


# ═════════════ Página ═════════════
def renderizar_pagina(df, region_nombre, fecha_base):
    st.markdown(f'<div class="ptitle" style="font-size:24px; font-weight:bold; margin-bottom: 20px;">{t("titulo")}</div>',
                unsafe_allow_html=True)
    st.write(f"**{t('escala')}** {region_nombre} | **{t('proyectando')}** {fmt_fecha(pd.Timestamp(fecha_base))}")

    if df.empty:
        st.warning(t("sin_datos"))
        return

    with st.spinner(t("procesando")):
        diario = preparar_datos_diarios(df, fecha_base)
        test, test_pred, error, df_futuro = entrenar_y_predecir(diario, fecha_base, horizonte=15)

    if df_futuro is None:
        st.warning(t("pocos_datos"))
        return

    st.markdown(f"### {t('eval')}")
    c1, c2, c3 = st.columns(3)
    c1.metric(t("mae"), t("mae_unidad", v=fmt_dec(error, 1)), t("mae_delta"))
    c2.metric(t("total"), t("total_unidad", v=fmt_int(df_futuro['prediccion'].sum())))
    c3.metric(t("riesgo"), fmt_fecha(df_futuro.loc[df_futuro['prediccion'].idxmax(), 'fecha']))

    fig = go.Figure()

    contexto = diario.iloc[-45:]
    fig.add_trace(go.Scatter(x=contexto['fecha'], y=contexto['detecciones'], mode='lines',
                             name=t("tr_hist"), line=dict(color='#ff9f1c')))
    fig.add_trace(go.Scatter(x=test['fecha'], y=test_pred, mode='lines',
                             name=t("tr_eval"), line=dict(color='#4cc9f0', dash='dash')))
    fig.add_trace(go.Scatter(x=df_futuro['fecha'], y=df_futuro['prediccion'], mode='lines+markers',
                             name=t("tr_proy"), line=dict(color='#f72585', width=3)))

    fig.update_layout(
        title=t("graf_titulo"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c9d4ec"),
        hovermode="x unified",
        xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#1c2c4a")
    )
    st.plotly_chart(fig, **STRETCH)

    st.markdown(f"### {t('detalle')}")
    tabla = df_futuro[['fecha', 'prediccion']].rename(
        columns={'fecha': t("col_fecha"), 'prediccion': t("col_focos")})
    st.dataframe(tabla, **STRETCH)
