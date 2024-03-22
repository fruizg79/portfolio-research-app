import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import os
#"Global Land and Ocean January - December Average Temperature Anomalies"
#Anomalies are with respect to the 20th century average (1901-2000). Monthly and annual global anomalies are available through the most recent complete month and year, respectively.
st.title('Average Temperature Anomalies')
df_temperature = pd.read_csv("\\data\\co2\\1850-2023.csv")

st.line_chart(df_temperature.Anomaly,
              x="Year",
              y="Anomaly")