import streamlit as st
import joblib
import pandas as pd
from train_model import build_model,model_file


model = joblib.load(model_file)

size = st.number_input("bedroom size", 1, 100)
nb_rooms = st.number_input("number bedroom", 1, 5, 1)
garden = st.checkbox("has garden")

input = [[size, nb_rooms, garden]]
price = model.predict(input)

st.metric("Price", price)

