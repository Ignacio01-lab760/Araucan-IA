import streamlit as st

st.set_page_config(page_title="Predicción · FireWatch", page_icon="🔥", layout="wide")

from firewatch.ui import render_prediccion

render_prediccion()
