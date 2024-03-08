
import math
import numpy as np
import pandas as pd


class CBlack_litterman(object):
    def __init__(self,risk_free_rate,tau,
                sigma,p_matrix,lambda_param,
                port_wts_eq, conf_level,c_coef, reco_vector):
        self.tau = tau
        self.risk_free_rate=risk_free_rate
        self.sigma = sigma
        self.p_matrix = p_matrix
        self.lambda_param = lambda_param
        self.port_wts_eq = port_wts_eq
        self.conf_level = conf_level
        self.c_coef = c_coef
        self.reco_vector=reco_vector
        self.num_recos = len(reco_vector)
        self.nu_vector=[]
        self.reco_dict ={'strong_sell':-0.5,
                'sell':-0.25,
                 'neutral':0.0,
                 'buy':0.25,
                 'strong_buy':0.5}
        self.view_returns = []
        self.eq_returns = []
        self.omega = []
        self.bl_returns = []
    def get_bl_returns(self):
        self.get_eq_risk_premium()
        self.get_eq_returns()
        self.get_omega()
        self.get_view_returns()
        inv_tau_sigma =np.linalg.inv(self.tau*self.sigma)
        inv_omega =np.linalg.inv(self.omega)
        bl_returns_1 = np.linalg.inv(inv_tau_sigma + np.dot(np.dot(self.p_matrix.T,inv_omega),self.p_matrix))
        bl_returns_2 = np.dot(inv_tau_sigma,self.eq_returns) + np.dot(np.dot(self.p_matrix.T,inv_omega),self.view_returns)
        self.bl_returns = np.dot(bl_returns_1,bl_returns_2)
        return self.bl_returns
    def get_eq_risk_premium(self):
        self.eq_risk_premium = np.dot(self.sigma,self.port_wts_eq.T)*self.lambda_param
        return self.eq_risk_premium
    def get_eq_returns(self):
        self.eq_returns = self.get_eq_risk_premium() +self.risk_free_rate
        return self.eq_returns
    def get_omega(self):
        p_sigma_p = np.dot(np.dot(self.p_matrix,self.sigma),self.p_matrix.T)
        u = np.diag(self.conf_level)/self.c_coef
        self.omega = np.diag(np.diag(np.dot(u,np.dot(p_sigma_p,u))))
        return self.omega
    def get_view_returns(self):
        p_sigma_p = np.dot(np.dot(self.p_matrix,self.sigma),self.p_matrix.T)
        nu_vector = []
        for reco in  self.reco_vector:
            nu_vector.append( self.reco_dict[reco])
        nu_vector = np.array(nu_vector)
        view_mean = np.dot(self.p_matrix, self.eq_returns)
        self.view_returns = view_mean + np.multiply(np.sqrt(np.diag(p_sigma_p)),nu_vector)
        return self.view_returns
    