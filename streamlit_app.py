import altair as alt
import math
import numpy as np
import os
import pandas as pd
import streamlit as st
import numpy as np
import cblacklitterman


st.title('Black-Litterman')
st.title('_Streamlit_ is :blue[cool] :sunglasses:')

asset_names = ['asset1','asset2','asset3', 'asset4','asset5', 'asset6','asset7']
sigma = np.array([[0.00002,0.00001,-0.00010,-0.00017,-0.00018,-0.00020,-0.00007],
[0.00001,0.00299,0.00294,0.00071,0.00236,-0.00065,0.00604],
[-0.00010,0.00294,0.04131,0.02452,0.02094,0.02071,0.03069],
[-0.00017,0.00071,0.02452,0.02293,0.01863,0.01843,0.02524],
[-0.00018,0.00236,0.02094,0.01863,0.02181,0.01606,0.02722],
[-0.00020,-0.00065,0.02071,0.01843,0.01606,0.04705,0.02230],
[-0.00007,0.00604,0.03069,0.02524,0.02722,0.02230,0.05693]])

p_matrix = np.array([[ 1.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  1.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  1., -1.,  0.,  0.],
       [ 0.,  0., -1.,  1.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  1.,  0.],
       [ 0.,  0.,  0.,  1.,  0.,  0.,  0.]])


nu_vector = np.array([-0.25,0.00,0.25,0.25,-0.50,0.25])
reco_vector =['sell','neutral','buy','buy','strong_sell','buy']

c_coef  = 1

tau = 0.40
risk_free_rate = 0.0020
conf_level = np.array([0.5,0.5,0.5,0.5,0.5,0.5])
lambda_param = 7
port_wts_eq =np.array([0.200,0.330,0.050,0.200,0.150,0.020,0.050])

bl = cblacklitterman.CBlack_litterman(risk_free_rate,tau,
                sigma,p_matrix,lambda_param,
                port_wts_eq, conf_level,c_coef, reco_vector)

bl.get_eq_risk_premium()
eq_ret = bl.get_eq_returns()
bl.get_view_returns()
bl.get_omega()
bl_ret = bl.get_bl_returns()

chart_data = pd.DataFrame(eq_ret, columns=asset_names)
st.bar_chart(chart_data)