import streamlit as st
import joblib

size = st.number_input("bedroom size")
n = st.number_input("number bedroom")
has_garden = st.checkbox("has garden")
