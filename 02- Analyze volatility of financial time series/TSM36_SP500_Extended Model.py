# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 14:14:38 2021

@author: Aram
"""

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import math
from scipy import stats
from scipy.stats import norm
import scipy
import statsmodels.api as sm

#######################################################################
data_path=r'D:\One Drive\Vrije Universiteit Amsterdam\Mostowfi, A. - TSM_assignment\Assignment2\Data\SP500.csv'
export_path=r'D:\One Drive\Vrije Universiteit Amsterdam\Mostowfi, A. - TSM_assignment\Assignment2\Figure'+'\\'
#######################################################################

def data_import():
    #######################################################################
    ## Data Importing
    #######################################################################
    from dateutil.relativedelta import relativedelta
    ## data import
    data=pd.read_csv(data_path)
    data=data.rename(columns={'time':'t','rv10':'RV'})
    data['t']=pd.to_datetime(data.t,format= '%d/%m/%Y')
    ## transforming the RV data
    data['ln_RV']=np.log(data.RV)
    for i in data.index:
        if i>0:
            data.loc[i,'y']=np.log(data.loc[i,'close_price']/data.loc[i-1,'close_price'])
    ## select a specific 5 year period of data
    starting_date=datetime.strptime('2010-01-01', '%Y-%m-%d')
    ending_date=starting_date+ relativedelta(years=5)
    date_condition=(data.t>=starting_date) & (data.t<ending_date)
    # selected_row=data[date_condition]
    selected_columns=['t','ln_RV','y']
    return data.loc[date_condition,selected_columns].reset_index()


#######################################################################
#####QML transformation (b)
#######################################################################
def QML(y,miu):
    x= np.log((y-miu)**2)
    return x

def QML_function(data):
    ## calculate the mean of y as miu
    miu=data.y.mean()
    for i in data.index:    
        
        data.loc[i,'x']=QML(data.loc[i,'y'],miu) 
    
    return data



#######################################################################
## Kalman Filter and Smoother
#######################################################################
## v_t
def cal_vt(y,a,beta,ln_RV_bar):
    ## add 1.27 for the mean of sig2_eps
    ## Modification for model in Part e: "-beta*ln_RV_bar"
    return y-a+1.27-beta*ln_RV_bar

## update a_t+1
def update_at(a_t,p_t,sig2_eps_in,v_t,phi_in,omega_in,k_t):
    return omega_in+phi_in*a_t+k_t*v_t

## update P_t+1
def update_pt(p_t,k_t,f_t,sig2_eta_in,phi_in):
    return (phi_in**2) *p_t -(k_t**2) *f_t +sig2_eta_in

## K_t
def update_kt(pt,sig2_eps_in,phi_in,f_t):
    return (phi_in*pt)/f_t

## F_t
def update_ft(pt,sig2_eps_in,beta,ln_RV_var):
    return (beta**2)*ln_RV_var+pt+sig2_eps_in

## L_t
def update_lt(kt,phi_in):
    return phi_in-kt

def update_rt_m1(F_t,v_t,L_t,r_t):
    return v_t/F_t+L_t*r_t

def update_Nt_m1(F_t,L_t,N_t):
    return 1/F_t+(L_t**2)*N_t

def update_Vt(P_t,N_tm1):
    return P_t-(P_t**2)*N_tm1

def update_alphat_hat(a_t,P_t,r_tm1):
    return a_t+P_t*r_tm1

def Kalman_Filter(data_in,omega_in,phi_in,beta_in,sig2_eps_in,sig2_eta_in):
    # print('Kalman Filter')
    #######################################################################
    ## Data initialization
    #######################################################################
    ## initialization of a1,p1 : because state equation is stationary, so uses these two as initialization
    a1=omega_in/(1-phi_in)
    p1=sig2_eta_in/(1-phi_in**2)
    data_in.loc[0,'a']=a1
    data_in.loc[0,'P']=p1
    ## calculate mean and variance for ln_RV -> for further use in v_t and f_t
    lnRV_bar=data_in.ln_RV.mean()
    lnRV_var=data_in.ln_RV.var()
    
    ######################################################################
    for i in data_in.index:    
        data_in.loc[i,'F']=update_ft(data_in.loc[i,'P'],sig2_eps_in,beta_in,lnRV_var)
        data_in.loc[i,'K']=update_kt(data_in.loc[i,'P'],sig2_eps_in,phi_in,data_in.loc[i,'F'])
        data_in.loc[i,'L']=update_lt(data_in.loc[i,'K'],phi_in)
        data_in.loc[i,'v']=cal_vt(data_in.loc[i,'x'],data_in.loc[i,'a'],beta_in,lnRV_bar)
        data_in.loc[i+1,'a']=update_at(data_in.loc[i,'a'],data_in.loc[i,'P'],sig2_eps_in,data_in.loc[i,'v'],phi_in,omega_in,data_in.loc[i,'K'])
        data_in.loc[i+1,'P']=update_pt(data_in.loc[i,'P'],data_in.loc[i,'K'],data_in.loc[i,'F'],sig2_eta_in,phi_in)
    return data_in[:len(data_in.index)-1]

def Smoothing(data_in,sig2_eps,sig2_eta):
    print ('Smoothing')
    total_length=len(data_in.index)
    
    ## run a reverse loop for r_t
    for i in data_in.index[::-1]:
        if i+1 == total_length:
            data_in.loc[i,'r']=0
            data_in.loc[i,'N']=0
        # if i-1>=0 :
        data_in.loc[i-1,'r']=update_rt_m1(data_in.loc[i,'F'],data_in.loc[i,'v'],data_in.loc[i,'L'],data_in.loc[i,'r'])
        data_in.loc[i-1,'N']=update_Nt_m1(data_in.loc[i,'F'],data_in.loc[i,'L'],data_in.loc[i,'N'])
        data_in.loc[i,'V']=update_Vt(data_in.loc[i,'P'],data_in.loc[i-1,'N'])
        data_in.loc[i,'alpha_hat']=update_alphat_hat(data_in.loc[i,'a'],data_in.loc[i,'P'],data_in.loc[i-1,'r'])
    return data_in[:len(data_in.index)-1]
   
######################################################################
# QMLE parameter estimation
######################################################################
def log_lik(parameters,this_sig2_eps):
    ## unpacking the parameters
    this_omega=parameters[0]
    this_phi=parameters[1]
    this_beta=parameters[2]
    this_sig2_eta=parameters[3]
    ## Initializing the data
    data_MLE=data_import()
    data_MLE=QML_function(data_MLE)
    n=len(data_MLE.index)
    ## Running KF with this new set of parameter
    data_MLE=Kalman_Filter(data_MLE,this_omega,this_phi,this_beta,this_sig2_eps,this_sig2_eta)
    ## Calculating components of the log-likelihood 
    data_MLE.loc[:,'v_square']=np.square(data_MLE.v)
    data_MLE.loc[:,'f_inverse']=1/(data_MLE.F)
    data_MLE.loc[:,'f_ln']=np.log(data_MLE.F)
    ## definition of log-likelihod
    data_MLE.loc[:,'MLE_last_term']=data_MLE.f_ln+data_MLE.v_square*data_MLE.f_inverse
    log_lik_value= -(n/2)* np.log(2*np.pi)-(1/2)*data_MLE.MLE_last_term.sum()
    print(parameters,log_lik_value) ##Debug: to track the parameter
    return -log_lik_value ## minus sign because of maximization problem

def hyperparameter_estimation():
    print('Parameter Estimation')
    ## Initialization of parameters
    omega_ini=0        
    phi_ini=0.9          
    beta_ini=0          
    sig2_eps=np.square(np.pi)/2      ## variance of log-chi square(1)
    sig2_eta_ini=1    
    ini_para=np.array((omega_ini,phi_ini,beta_ini,sig2_eta_ini))
    options ={'eps':1e-04,  # argument convergence criteria
             'disp': True,  # display iterations
             'maxiter':100} # maximum number of iterations
    results = scipy.optimize.minimize(log_lik, 
                                      x0=ini_para,
                                      args=(sig2_eps),
                                      options = options,
                                      method='L-BFGS-B' ,
                                      # Parameter Space Bounds
                                      bounds=( (-10,  10), ## omega bound -> no special
                                              (-0.9999, 0.9999), ## phi_bound -> (-1,1) to ensure stationarity
                                              (-10, 10), ## beta bound -> no special
                                              (0.01, 10) ## sig2_eta bound -> non-zero positive value
                                                            )
                                      ) #restrictions in parameter space
    ## Debug Only : For printing max result
    print(results.x)
    print(results.fun)
    print(results.success)
    omega_est=results.x[0]        
    phi_est=results.x[1]          
    beta_est=results.x[2]          
    sig2_eta_est=results.x[3] 
    print('estimated omega:',omega_est)
    print('estimated phi:',phi_est)
    print('estimated beta:',beta_est)
    print('estimated sig2_eta_est:',sig2_eta_est)

    return omega_est,phi_est,beta_est,sig2_eta_est

######################################################################
# Graph Creation
######################################################################

def graph_export_d(data_in,xi,export_path):
    x_value=data_in.x
    smoothed_ht=data_in.alpha_hat
    filter_Ht=data_in.a-xi
    smoothed_Ht=data_in.alpha_hat-xi
    lnRV_value=data_in.ln_RV
    time_axis=data_in.t
    standard_figsize=(10,5)
    ## Figure 1 - x_t with smoothed h_t
    fig1=plt.figure(figsize=standard_figsize)
    ax1=fig1.add_axes([0,0,1,1])
    ax1.scatter(time_axis, x_value, color='b',s=5)
    ax1.plot(time_axis, smoothed_ht, color='r')
    ax1.plot(time_axis, lnRV_value, color='m')
    ax1.set_xlabel('time')
    ax1.set_title('Extended Model: xt, ln_RV and smoothed ht')
    ax1.legend(labels = ('ht_hat','ln_RV','x_t'))
    
    ## Figure 2 - filtered and smoothed H_t
    fig2=plt.figure(figsize=standard_figsize)
    ax2=fig2.add_axes([0,0,1,1])
    ax2.plot(time_axis[1:], filter_Ht[1:], color='r')
    ax2.plot(time_axis[1:], smoothed_Ht[1:], color='b')
    ax2.set_title('filtered and smoothed Ht')
    ax2.legend(labels = ('filtered Ht','smoothed Ht'))
    
    ## Graph Export
    # fig1.savefig(export_path+'SP500_withRV_Graph1_x_and_ht_hat.png',dpi=300,bbox_inches='tight')
    # fig2.savefig(export_path+'SP500_withRV_Graph2_Ht.png',dpi=300,bbox_inches='tight')
    
#######################################################################
## parameter estimation
#######################################################################
## To run parameter estimation,uncomment the next line
# omega,phi,beta,sig2_eta=hyperparameter_estimation()

## In order to not run MLE everytime,i put the estimation result below
## estimation result from MLE
omega=-0.09485303443439598        
phi=0.9373868174303424          
beta=0.8315061715897409          
sig2_eps=np.square(np.pi)/2      ## variance of log-chi square(1)
sig2_eta=0.12121729678443706  
xi= omega/(1-phi) 

#######################################################################
## Main
#######################################################################
# 1.This file is only for the extended model with SP500 data and the RV data
# 2.The extended model include beta and lnRV. Hence KF and KS are changed accordingly
# 3.Parameters are also estimated with QML and the updated KF,KS
#######################################################################

## Data Import and Calculate x
data = data_import()
data=QML_function(data)

## Extended Model , Part E
data=Kalman_Filter(data,omega,phi,beta,sig2_eps,sig2_eta)
data=Smoothing(data,sig2_eps,sig2_eta)

## Graph Drawing
graph_export_d(data,xi,export_path)

