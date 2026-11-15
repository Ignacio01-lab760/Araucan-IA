import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

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

def renderizar_pagina(df, region_nombre, fecha_base):
    st.markdown('<div class="ptitle" style="font-size:24px; font-weight:bold; margin-bottom: 20px;">Predicción de Detecciones (15 Días)</div>', unsafe_allow_html=True)
    st.write(f"**Escala de análisis:** Región de {region_nombre} | **Proyectando desde:** {fecha_base.strftime('%d-%m-%Y')}")
    
    if df.empty:
        st.warning("No hay datos históricos para entrenar en esta selección.")
        return

    with st.spinner("Procesando datos continuos y entrenando modelo..."):
        diario = preparar_datos_diarios(df, fecha_base)
        test, test_pred, error, df_futuro = entrenar_y_predecir(diario, fecha_base, horizonte=15)
        
    if df_futuro is None:
        st.warning("No hay suficientes datos históricos en esta selección para entrenar el modelo (se requieren al menos 30 días).")
        return
        
    st.markdown("### Evaluación del Modelo")
    c1, c2, c3 = st.columns(3)
    c1.metric("Margen de Error Medio (MAE)", f"{error:.1f} focos/día", "Evaluado en últimos 15 días")
    c2.metric("Total Focos Proyectados (15d)", f"{df_futuro['prediccion'].sum()} focos")
    c3.metric("Día de mayor riesgo futuro", df_futuro.loc[df_futuro['prediccion'].idxmax(), 'fecha'].strftime('%d-%m-%Y'))
    
    fig = go.Figure()
    
    contexto = diario.iloc[-45:]
    fig.add_trace(go.Scatter(x=contexto['fecha'], y=contexto['detecciones'], mode='lines', name='Histórico Real', line=dict(color='#ff9f1c')))
    fig.add_trace(go.Scatter(x=test['fecha'], y=test_pred, mode='lines', name='Predicción (Evaluación)', line=dict(color='#4cc9f0', dash='dash')))
    fig.add_trace(go.Scatter(x=df_futuro['fecha'], y=df_futuro['prediccion'], mode='lines+markers', name='Proyección 15 Días', line=dict(color='#f72585', width=3)))
    
    fig.update_layout(
        title="Detecciones Diarias: Histórico, Evaluación y Proyección",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c9d4ec"),
        hovermode="x unified",
        xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#1c2c4a")
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### Detalle de Proyección")
    st.dataframe(df_futuro[['fecha', 'prediccion']].rename(columns={'fecha': 'Fecha', 'prediccion': 'Focos Estimados'}), use_container_width=True)