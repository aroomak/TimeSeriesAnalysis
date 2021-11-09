# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 17:48:25 2021

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
import random
import seaborn as sns

#######################################################################
data_path=r'D:\Box\Box\2020 Master\VU\Study\P4 Time Series Model\Assignment\Assignment2\Data\sv.csv'
export_path=r'D:\One Drive\Vrije Universiteit Amsterdam\Mostowfi, A. - TSM_assignment\Assignment2\Figure'+'\\'
plot_path=export_path
#######################################################################
## Data Importing
#######################################################################
def data_import():
    
    data=pd.read_csv(data_path)
    return data

#######################################################################
#####QML transformation (b)
#######################################################################
def QML(y):
    x= np.log(y**2)
    return x
def QML_function(data):
    for i in data.index:    
        data.loc[i,'x']=QML(data.loc[i,'y']) 
    
    return data


#######################################################################
## Kalman Filter and Smoother
#######################################################################
## v_t
def cal_vt(y,a):
    ## add 1.27 for the mean of sig2_eps
    return y-a+1.27

## update a_t+1
def update_at(a_t,p_t,sig2_eps_in,v_t,phi_in,omega_in):
    return omega_in+phi_in*a_t+update_kt(p_t,sig2_eps_in,phi_in)*v_t

## update P_t+1
def update_pt(p_t,k_t,f_t,sig2_eta_in,phi_in):
    return (phi_in**2) *p_t -(k_t**2) *f_t +sig2_eta_in

## K_t
def update_kt(pt,sig2_eps_in,phi_in):
    return (phi_in*pt)/update_ft(pt,sig2_eps_in)

## F_t
def update_ft(pt,sig2_eps_in):
    return pt+sig2_eps_in

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

def Kalman_Filter(data_in,omega_in,phi_in,sig2_eps_in,sig2_eta_in):
    # print('Kalman Filter')
    #######################################################################
    ## Data initialization
    #######################################################################
    a1=omega_in/(1-phi_in)
    p1=sig2_eta_in/(1-phi_in**2)
    data_in.loc[0,'a']=a1
    data_in.loc[0,'P']=p1
    ######################################################################
    for i in data_in.index:    
        data_in.loc[i,'F']=update_ft(data_in.loc[i,'P'],sig2_eps_in)
        data_in.loc[i,'K']=update_kt(data_in.loc[i,'P'],sig2_eps_in,phi_in)
        data_in.loc[i,'L']=update_lt(data_in.loc[i,'K'],phi_in)
        data_in.loc[i,'v']=cal_vt(data_in.loc[i,'x'],data_in.loc[i,'a'])
        data_in.loc[i+1,'a']=update_at(data_in.loc[i,'a'],data_in.loc[i,'P'],sig2_eps_in,data_in.loc[i,'v'],phi_in,omega_in)
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

#######################################################################
## ML Estimation
######################################################################

# loglikelihood for q with the input as psi
def q_lik(theta_ini):
    sig2_eps_in=np.square(np.pi)/2 
    print(theta_ini)
    data_MLE=data_import()
    data_MLE=QML_function(data_MLE)
    n=len(data_MLE.index)
    data_MLE=Kalman_Filter(data_MLE,theta_ini[0],theta_ini[1],sig2_eps_in,theta_ini[2])
    data_MLE.loc[:,'v_square']=np.square(data_MLE.v)
    sum_v_over_f=(data_MLE.v_square/data_MLE.F)[1:].sum()
    sum_f=np.log(data_MLE.F[1:]).sum()
    sum_ll=(np.log(data_MLE.F)+(data_MLE.v_square/data_MLE.F)).sum()
    sig2_eps_hat=sum_v_over_f/(n-1)
    log_lik_value= -(n/2)* np.log(2*np.pi)-(0.5)*sum_ll
    print(log_lik_value)#Change this to 2.58
    return -log_lik_value ## minus sign because of maximization problem
    

def hyperparameter_estimation():
    omega_in = 0.1  # initial value for omega
    phi_in = 0.2  # initial value for alpha
    sig2_eta_in = 0.8
    
    theta_ini = np.array([omega_in,
                      phi_in,
                      sig2_eta_in

                     ])
    options ={'eps':1e-5,  # argument convergence criteria
             'disp': True,  # display iterations
             'maxiter':200} # maximum number of iterations
    results = scipy.optimize.minimize(q_lik, 
                                      x0=theta_ini, 
                                      options = options,
                                      method='L-BFGS-B', bounds= ((-10,10),(-0.999,0.999),(0.000001,10))
                                      ) #restrictions in parameter space
    ## Debug Only : For printing max result
    print(results.x)
    print(results.fun)
    print(results.success)
    
    omega_est=results.x[0]
    phi_est=results.x[1]
    sig2_eta_est=results.x[2]
    print('estimated omega:',omega_est)
    print('estimated phi:',phi_est)
    print('estimated sig2_eta_hat:',sig2_eta_est)
    return omega_est,phi_est,sig2_eta_est

#######################################################################
## Particle Filtering
#######################################################################

def sigma2_from_Ht(xi_in,Ht):
    return np.exp(xi_in+Ht)

def normalize_weight(weight):
    total=weight.sum()
    return weight/total

def resampling(wt,Ht,N):
    return np.array(random.choices(Ht,weights=wt,k=N))

def particle_filtering(data_in,phi_in,xi_in,sig2_eta_in):
    print('Particle Filtering')
    N=10000
    #######################################################################
    ## Data initialization
    #######################################################################
    a1=0
    p1=sig2_eta_in/(1-phi_in**2)
    data_in.loc[0,'a']=a1
    data_in.loc[0,'P']=p1
    miu=data_in.y.mean()
    for i in data_in.index:
        
        if i==1:
            this_draw=np.random.normal(data_in.loc[i-1,'a'], np.sqrt(data_in.loc[i-1,'P']), N)
        elif i>1:
            this_draw=np.random.normal(phi_in*this_draw_rs, np.sqrt(sig2_eta_in), N)
        else:
            continue
        this_sigma2=sigma2_from_Ht(xi_in,this_draw)
        weight=norm.pdf(data_in.loc[i,'y'],miu,np.sqrt(this_sigma2))
        nor_weight=normalize_weight(weight)
        at_t=(nor_weight*this_draw).sum()
        pt_t=(nor_weight*np.square(this_draw)).sum()-np.square(at_t)
        data_in.loc[i,'a']=at_t
        data_in.loc[i,'P']=pt_t
        this_draw_rs=resampling(nor_weight,this_draw,N)
    return data_in

######################################################################
# Graph Creation
######################################################################

def graph_export_d(data_in,data_pf_in,xi,export_path):
    x_value=data_in.x
    smoothed_ht=data_in.alpha_hat
    filter_Ht=data_in.a-xi
    smoothed_Ht=data_in.alpha_hat-xi
    time_axis=data_in.t
    pf_Ht=data_pf.a

    standard_figsize=(10,5)
    ## Figure 1 - x_t with smoothed h_t
    fig1=plt.figure(figsize=standard_figsize)
    ax1=fig1.add_axes([0,0,1,1])
    ax1.scatter(time_axis, x_value, color='b',s=5)
    ax1.plot(time_axis, smoothed_ht, color='r')
    ax1.set_xlabel('time')
    ax1.set_title('xt and smoothed ht')
    ax1.legend(labels = ('ht_hat','x_t'))
    
    ## Figure 2 - filtered and smoothed H_t
    fig2=plt.figure(figsize=standard_figsize)
    ax2=fig2.add_axes([0,0,1,1])
    ax2.plot(time_axis, filter_Ht, color='r')
    ax2.plot(time_axis, smoothed_Ht, color='b')
    ax2.set_title('filtered and smoothed Ht')
    ax2.legend(labels = ('filtered Ht','smoothed Ht'))
    
    ## Figure 3 - PF vs QML Ht
    fig3=plt.figure(figsize=(10,5))
    ax3=fig3.add_axes([0,0,1,1])
    ax3.plot(time_axis, filter_Ht, color='b',label='QML')
    ax3.plot(time_axis, pf_Ht, color='r',label='PF')
    ax3.legend()
    ax3.set_title('Filtered Ht : QML and Particle Filter')
    
    ## Graph Export
    # fig1.savefig(export_path+'SV_Graph1_x_and_ht_hat.png',dpi=300,bbox_inches='tight')
    # fig2.savefig(export_path+'SV_Graph2_Ht.png',dpi=300,bbox_inches='tight')
    # fig3.savefig(export_path+'SV_Graph3_PF_Ht.png',dpi=300,bbox_inches='tight')

#(a)############
#plot1- Fig. 14.5 (I)
def dataplot1():
    fig, ax = plt.subplots()
    plt.style.use('seaborn-darkgrid')
    ax.plot(data.t, data.y/100)
    ax.set(xlabel='Time', ylabel='Exchange Rate Returns')
    ax.set_xticks(range(0, 1000, 100))
    ax.set_xlim(0,1000)
    #ax.set_yticks(range(-0.025, 0.050, 0.025))
    ax.set_ylim(-0.025, 0.050, 0.025)
    #plt.figure(figsize=(30,10))
    #ax.grid()
    ax.axline([0,0],[1000,0], color='k')
    fig.set_figheight(5)
    fig.set_figwidth(20)
    plt.show() 
    # fig.savefig(plot_path+'a1-dataplot.png', dpi=300)

#plot2- Histogram and correlogram of data 
def dataplot2():
    sns.set(style="darkgrid")
    fig, axs = plt.subplots(2, 2, figsize=(15, 15))
    sns.regplot(x=data.t, y=data.y, color="grey", ax=axs[0, 0])
    sns.scatterplot(data=data, x="t", y="y",  legend=False, sizes=(20, 2000), ax=axs[1, 0])
    sns.distplot(a=data.y, hist=True, kde=False, rug=False, color="red", ax=axs[0, 1]) 
    sns.distplot(data["y"],
             kde=True,
             kde_kws={"color": "g", "alpha": 0.3, "linewidth": 5, "shade": True}, 
             ax=axs[1, 1])
    plt.show()
    # fig.savefig(plot_path+'a2-dataplot.png', dpi=300)

#Descriptive of the data 
def describeData():
    print("Data Descibe")
    print(data['y'].describe())
    
    
#(b)############
def dataplot3():
    fig, ax = plt.subplots()
    plt.style.use('seaborn-darkgrid')
    ax.plot(data.x, color='mediumvioletred')
    fig.set_figheight(5)
    fig.set_figwidth(20)
    plt.show() 
    # fig.savefig(plot_path+'b1-dataplot.png', dpi=300)

#######################################################################
## parameter estimation
#######################################################################
## To run parameter estimation,uncomment the next line
# omega,phi,sig2_eta=hyperparameter_estimation()

## In order to not run MLE everytime,i put the estimation result below
## estimation result from MLE
omega=-0.00915306515109392        
phi=0.9887807790978227          
sig2_eps=np.square(np.pi)/2      ## variance of log-chi square(1)
sig2_eta=0.00876976171160163
xi= omega/(1-phi) 
    
#######################################################################
## Main
#######################################################################
# 1. Part C&D uses QML for the transformed linear model
# 2. Part F uses Particle Filter 
# 3. dataplot1-3() create graphs for part A-B
# 3. graph_export_d() create graphs for part C-F
#######################################################################

## Data Import and Calculate x
data = data_import()
data=QML_function(data)

#### Part A and B
dataplot1()
dataplot2()
dataplot3()
describeData()

## Part C & D : QML
data=Kalman_Filter(data,omega,phi,sig2_eps,sig2_eta)
data=Smoothing(data,sig2_eps,sig2_eta)

## Part F : Particle Filter
data_pf = data_import()
data_pf=particle_filtering(data_pf,phi,xi,sig2_eta)

## Graph Drawing
graph_export_d(data,data_pf,xi,export_path)


