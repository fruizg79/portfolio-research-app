import altair as alt
import math
import numpy as np
import os
import pandas as pd
import streamlit as st
import numpy as np
import cblacklitterman


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


