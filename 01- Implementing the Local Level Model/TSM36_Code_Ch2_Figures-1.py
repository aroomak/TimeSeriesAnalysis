# -*- coding: utf-8 -*-
"""
Created on Wed Feb 5 14:56:34 2021

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
## Paths
#######################################################################
data_path=r'D:\Box\Box\2020 Master\VU\Study\P4 Time Series Model\Assignment\Assignment1\Data\Nile.dat'
graph_export_path=r"D:\One Drive\Vrije Universiteit Amsterdam\Mostowfi, A. - TSM_assignment\Assignment1\Figure"+"\\"

def data_import():
    #######################################################################
    ## Data Importing
    #######################################################################
    data=pd.read_csv(data_path,header=None, skiprows=1)
    data=data.rename(columns={0:'volume'})
    data['year']=np.array(range(1871, 1971))
    
    #######################################################################
    ## Data initialization
    #######################################################################
    a1=0
    p1=10**7
    data.loc[0,'a']=a1
    data.loc[0,'P']=p1
    return data

sig2_eps=15099
sig2_eta=1469.1
data=data_import()
#######################################################################
## Figure 2.1 Kalman Filter
#######################################################################
## v_t
def cal_vt(y,a):
    return y-a

## a_t given t
def update_at(a_t,p_t,sig2_eps,v_t):
    at_givent=a_t+update_kt(p_t,sig2_eps)*v_t
    return at_givent

## P_t given t
def update_pt(p_t,sig2_eps):
    return (update_kt(p_t,sig2_eps)*sig2_eps)

## K_t
def update_kt(pt,sig2_eps):
    return pt/(pt+sig2_eps)

## F_t
def update_ft(pt,sig2_eps):
    return pt+sig2_eps

## L_t
def update_lt(kt):
    return 1-kt

def Kalman_Filter(data_in,sig2_eps,sig2_eta):
    print('Kalman Filter')
    for i in data_in.index:    
        data_in.loc[i,'F']=update_ft(data_in.loc[i,'P'],sig2_eps)
        data_in.loc[i,'K']=update_kt(data_in.loc[i,'P'],sig2_eps)
        data_in.loc[i,'L']=update_lt(data_in.loc[i,'K'])
        data_in.loc[i,'v']=cal_vt(data_in.loc[i,'volume'],data_in.loc[i,'a'])
        data_in.loc[i,'a_t']=update_at(data_in.loc[i,'a'],data_in.loc[i,'P'],sig2_eps,data_in.loc[i,'v'])
        data_in.loc[i+1,'a']=data_in.loc[i,'a_t']
        data_in.loc[i,'P_t']=update_pt(data_in.loc[i,'P'],sig2_eps)
        data_in.loc[i+1,'P']=data_in.loc[i,'P_t']+sig2_eta
    return data_in[:len(data_in.index)-1]
#######################################################################

data=Kalman_Filter(data,sig2_eps,sig2_eta)
## Fig 2.1
fig1,ax1 =  plt.subplots(2,2,figsize=(15,10))
## [0][0] plot : filter a plot
## calculate 90% CI
u90=stats.norm.ppf(0.95,loc=data.a,scale=np.sqrt(data.P)) 
l90=stats.norm.ppf(0.05,loc=data.a,scale=np.sqrt(data.P))
ax1[0][0].scatter(data.year,data.volume, color='b',s=15)    ## volume data
ax1[0][0].plot(data.year[1:],data.a[1:], color='r')         ## at value
ax1[0][0].plot(data.year[1:],u90[1:], color='k',alpha=0.5)  ## 90% upper bound
ax1[0][0].plot(data.year[1:],l90[1:], color='k',alpha=0.5)  ## 90% lower bound
ax1[0][0].set_yticks(range(400,1500,100))
ax1[0][0].set_ylim(400,1400)
## [0][1] plot : Pt plot
ax1[0][1].plot(data.year[1:],data.P[1:], color='r')         ## P value
ax1[0][1].set_yticks(range(5000,17501,2500))
## [1][0] plot : v plot
ax1[1][0].plot(data.year[1:],data.v[1:], color='r')         ## v value
ax1[1][0].plot(data.year[1:],np.zeros(len(data.v[1:])), color='k') ## plotting the y=0 line
ax1[1][0].set_xlim(1870,1970)
ax1[1][0].set_yticks(range(-500,500,250))
## [1][1] plot : F plot
ax1[1][1].plot(data.year[1:],data.F[1:], color='r')         ## F value
ax1[1][1].set_yticks(range(20000,32501,2500))

## Exporting Graph
# fig1.savefig(graph_export_path+'fig2.1.png',dpi=300)
#######################################################################
## Figure 2.2 Smoothing
#######################################################################
def update_rt_m1(F_t,v_t,L_t,r_t):
    return v_t/F_t+L_t*r_t

def update_Nt_m1(F_t,L_t,N_t):
    return 1/F_t+(L_t**2)*N_t

def update_Vt(P_t,N_tm1):
    return P_t-(P_t**2)*N_tm1

def update_alphat_hat(a_t,P_t,r_tm1):
    return a_t+P_t*r_tm1

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
    return data_in

#######################################################################
data=Smoothing(data,sig2_eps,sig2_eta)
## Fig 2.2
fig2,ax2 =  plt.subplots(2,2,figsize=(15,10))
## [0][0] plot: filter a plot
## calculate 90% CI
u90=stats.norm.ppf(0.95,loc=data.alpha_hat,scale=np.sqrt(data.V)) 
l90=stats.norm.ppf(0.05,loc=data.alpha_hat,scale=np.sqrt(data.V))
ax2[0][0].scatter(data.year,data.volume, color='b',s=15)    ## volume data
ax2[0][0].plot(data.year[1:],data.alpha_hat[1:], color='r') ## alpha_hat value
ax2[0][0].plot(data.year[1:],u90[1:], color='k',alpha=0.5)  ## 90% upper bound
ax2[0][0].plot(data.year[1:],l90[1:], color='k',alpha=0.5)  ## 90% lower bound
ax2[0][0].set_yticks(range(400,1500,100))
ax2[0][0].set_ylim(400,1400)
## [0][1] plot: Vt plot
ax2[0][1].plot(data.year[:],data.V[:], color='r')         ## V value
# ax2[0][1].set_yticks(range(2500,4001,500))
## [1][0] plot: r plot
ax2[1][0].plot(data.year,data.r, color='r')         ## r value
ax2[1][0].plot(data.year[1:],np.zeros(len(data.v[1:])), color='k') ## plotting the y=0 line
ax2[1][0].set_xlim(1870,1970)
# ax2[1][0].set_yticks(range(-500,500,250))
## [1][1] plot: N plot
ax2[1][1].plot(data.year[1:98],data.N[1:98], color='r')         ## N value
# ax2[1][1].set_yticks(range(20000,32501,2500))

## Exporting Graph
# fig2.savefig(graph_export_path+'fig2.2.png',dpi=300)

#################################
#Figure 2.3 Disturbance smoothing
#################################
def update_u_t(F_t,v_t,K_t,r_t):
    u_t = 1/F_t * v_t - K_t * r_t
    return u_t

def update_ehat_t(sig2_eps,u_t):
    return sig2_eps*u_t

def update_D_t(F_t,K_t, N_t):
    D_t = (1/F_t) + (K_t**2) * N_t
    return D_t

def update_var_egivenY(sigma2_eps,D_t):
    return math.sqrt(sig2_eps - sig2_eps**2 * D_t)

def update_eta_t(sig2_eta,r_t):
    return sig2_eta * r_t

def update_var_etagivenY(sigma2_eta, N_t):
    return math.sqrt(sig2_eta - sig2_eta**2 * N_t)

def Disturbance_Smoothing(data_in,sig2_eps,sig2_eta):
    print('Disturbance Smoothing')
    for i in data_in.index:    
        data_in.loc[i,'U']=update_u_t(data_in.loc[i,'F'],data_in.loc[i, 'v'], data_in.loc[i,'K'], data_in.loc[i,'r'])
        data_in.loc[i,'ehat']=update_ehat_t(sig2_eps,data_in.loc[i,'U'])
        data_in.loc[i,'D']=update_D_t(data_in.loc[i,'F'], data_in.loc[i,'K'],data_in.loc[i,'N'])
        data_in.loc[i,'var_egivenY']=update_var_egivenY(sig2_eps,data_in.loc[i,'D'],)
        data_in.loc[i,'Eta']=update_eta_t(sig2_eta,data_in.loc[i,'r'])
        data_in.loc[i,'var_etagivenY']=update_var_etagivenY(sig2_eta, data_in.loc[i,'N'])
    return data_in[:len(data_in.index)-1]

#######################################################################
data_disturbance= Disturbance_Smoothing(data,sig2_eps,sig2_eta)
## Fig 2.3
fig3,ax3 =  plt.subplots(2,2,figsize=(15,10))
## [0][0] plot: filter a plot
## calculate 90% CI
y_year=data_disturbance.year
y_ehat=data_disturbance.ehat
y_var_egY=data_disturbance.var_egivenY[:]
y_eta=data_disturbance.Eta
y_var_etagY=data_disturbance.var_etagivenY

ax3[0][0].plot(y_year,y_ehat, color='b')    ## observation error
ax3[0][0].plot(y_year[1:],np.zeros(len(y_ehat[1:])), color='k')

## [0][1] plot: Vt plot
ax3[0][1].plot(y_year,y_var_egY, color='r')         ## observation error variance
# ax2[0][1].set_yticks(range(2500,4001,500))
## [1][0] plot: r plot
ax3[1][0].plot(y_year,y_eta, color='r')         ## state error
ax3[1][0].plot(y_year[1:],np.zeros(len(y_eta[1:])), color='k') ## plotting the y=0 line
ax3[1][0].set_xlim(1870,1970)
# ax2[1][0].set_yticks(range(-500,500,250))
## [1][1] plot: N plot
ax3[1][1].plot(y_year[1:98],y_var_etagY[1:98], color='r')         ## state error variance
# ax2[1][1].set_yticks(range(20000,32501,2500))

## Exporting Graph
# fig3.savefig(graph_export_path+'fig2.3.png',dpi=300)

#######################################################################
## Figure 2.5 Missing Data 
#######################################################################

## Setting up the missing data
missing_data1=list(range(21,41))
missing_data2=list(range(61,81))
missing_data1.extend(missing_data2)
missing_data=np.array(missing_data1)

def Kalman_Filter_Missing(data_in,row_missing_data,sig2_eps,sig2_eta):
    print('Kalman Filter with Missing Data')
    for i in data_in.index:    
        data_in.loc[i,'F']=update_ft(data_in.loc[i,'P'],sig2_eps)
        
        ## different method for the missing value
        if i in row_missing_data:
            data_in.loc[i,'K']=0
        else:
            data_in.loc[i,'K']=update_kt(data_in.loc[i,'P'],sig2_eps)
            
        data_in.loc[i,'L']=update_lt(data_in.loc[i,'K'])
        
        
        ## different method for the missing value
        if i in row_missing_data:
            data_in.loc[i,'a_t']=data_in.loc[i,'a']
            data_in.loc[i,'P_t']=data_in.loc[i,'P']
            data_in.loc[i,'v']=np.NaN
        else:
            data_in.loc[i,'v']=cal_vt(data_in.loc[i,'volume'],data_in.loc[i,'a'])
            data_in.loc[i,'a_t']=update_at(data_in.loc[i,'a'],data_in.loc[i,'P'],sig2_eps,data_in.loc[i,'v'])
            data_in.loc[i,'P_t']=update_pt(data_in.loc[i,'P'],sig2_eps)

        data_in.loc[i+1,'a']=data_in.loc[i,'a_t']
        data_in.loc[i+1,'P']=data_in.loc[i,'P_t']+sig2_eta
    return data_in[:len(data_in.index)-1]

def Smoothing_Missing(data_in,row_missing_data,sig2_eps,sig2_eta):
    print ('Smoothing with Missing Data')
    total_length=len(data_in.index)
    
    ## run a reverse loop for r_t
    for i in data_in.index[::-1]:
        if i+1 == total_length:
            data_in.loc[i,'r']=0
            data_in.loc[i,'N']=0
        if i in row_missing_data:
            data_in.loc[i-1,'r']=data_in.loc[i,'r']
            data_in.loc[i-1,'N']=data_in.loc[i,'N']
        else:
            data_in.loc[i-1,'r']=update_rt_m1(data_in.loc[i,'F'],data_in.loc[i,'v'],data_in.loc[i,'L'],data_in.loc[i,'r'])
            data_in.loc[i-1,'N']=update_Nt_m1(data_in.loc[i,'F'],data_in.loc[i,'L'],data_in.loc[i,'N'])
        data_in.loc[i,'V']=update_Vt(data_in.loc[i,'P'],data_in.loc[i-1,'N'])
        data_in.loc[i,'alpha_hat']=update_alphat_hat(data_in.loc[i,'a'],data_in.loc[i,'P'],data_in.loc[i-1,'r'])
    return data_in.loc[range(0,len(data_in.index)-1),:]


def missing_data_run(data_in,list_missing_data,sig2_eps,sig2_eta):
    ## importing the raw data and data initialization
    data_in=data_import()
    row_missing_data=list_missing_data-1
    data_in.loc[row_missing_data,'volume']=np.NaN
    print('Missing Data')
    data_in=Kalman_Filter_Missing(data_in,row_missing_data,sig2_eps,sig2_eta)
    data_in=Smoothing_Missing(data_in,row_missing_data,sig2_eps,sig2_eta)
    return data_in

#######################################################################
data_m=missing_data_run(data,missing_data,sig2_eps,sig2_eta)
## Fig 2.5
fig5,ax5 =  plt.subplots(2,2,figsize=(15,10))
year_label=data_m.year
y_volume=data_m.volume
y_a=data_m.a
y_P=data_m.P
y_alpha_hat=data_m.alpha_hat
y_V=data_m.V
## [0][0] plot : filter a plot
## calculate 90% CI
ax5[0][0].plot(year_label,y_volume, color='b')    ## volume data
ax5[0][0].plot(year_label[1:],y_a[1:], color='r')         ## a value
ax5[0][0].set_yticks(range(400,1500,100))
ax5[0][0].set_ylim(400,1400)
## [0][1] plot : Pt plot
ax5[0][1].plot(year_label[1:],y_P[1:], color='r')         ## P value
# ax5[0][1].set_yticks(range(5000,17501,2500))
# ## [1][0] plot : v plot
ax5[1][0].plot(year_label,y_volume, color='b')    ## volume data
ax5[1][0].plot(year_label[1:],y_alpha_hat[1:], color='r')         ## alpha_hat value
ax5[1][0].set_yticks(range(400,1500,100))
ax5[1][0].set_ylim(400,1400)
# ax5[1][0].set_yticks(range(-500,500,250))
# ## [1][1] plot : F plot
ax5[1][1].plot(year_label[1:],y_V[1:], color='r')         ## V value
# ax5[1][1].set_yticks(range(20000,32501,2500))

## Exporting Graph
# fig5.savefig(graph_export_path+'fig2.5.png',dpi=300)

#######################################################################
## Fig 2.6 Forecasting 
#######################################################################
def update_pBar_nPj(pBar_nPj, sig2_eps):
    return (pBar_nPj + sig2_eps) 

def update_FBar_nPj(PBar_nP1, sig2_eps):
    return (PBar_nP1 + sig2_eps)

"""
yBar_npj = aBar_n+j
fBar_npj = PBar_n+j + sig2_eps
"""
list0 = np.array(data.index)
list1 = np.array(range(0,len(list0)-1))
j = np.array(range(len(list1), 130))
list2 = np.concatenate((list1, j))


def Forecasting(data_in):
    print('Forecasting')
    for i in (list2):
        if i in (list1):
            data_in.loc[i,'aBar_nP1'] = data_in.loc[i,'a']
            data_in.loc[i,'PBar_nP1'] = data_in.loc[i,'P']
        else:
            data_in.loc[i,'aBar_nP1'] = data_in.loc[i-1,'aBar_nP1']
            data_in.loc[i,'PBar_nP1'] = data_in.loc[i-1,'PBar_nP1'] + sig2_eta
            # increament year
            data_in.loc[i,'year'] = (data_in.loc[i-1,'year'])+1
        
        #yBar_nPj
        data_in.loc[i,'yBar_nPj'] = data_in.loc[i,'aBar_nP1']
        # FBar_n+j = PBar_n+j + sigma2_epsilon
        data_in.loc[i,'FBar_nPj'] = update_FBar_nPj(data_in.loc[i,'PBar_nP1'], sig2_eps)
        
    return data_in.loc[list2,:]

#######################################################################
## Plotting

data_forecast = Forecasting(data)
## Fig 2.6
fig6,ax6 =  plt.subplots(2,2,figsize=(15,10))
fig6.suptitle("Forecasting - fig.2.6")
## [0][0] plot : filter a plot
## calculate 50% 
u75=stats.norm.ppf(0.75, loc=data_forecast.aBar_nP1, scale=np.sqrt(data_forecast.PBar_nP1)) 
l25=stats.norm.ppf(0.25, loc=data_forecast.aBar_nP1, scale=np.sqrt(data_forecast.PBar_nP1))

ax6[0][0].scatter(data_forecast.year, data_forecast.volume, color='b',s=15)    ## volume data
ax6[0][0].plot(data_forecast.year[1:], data_forecast.aBar_nP1[1:], color='r')  ## aBar_nP1 value
ax6[0][0].plot(data_forecast.year[len(list0):], u75[len(list0):], color='k',alpha=0.5)  ## 75% upper bound
ax6[0][0].plot(data_forecast.year[len(list0):], l25[len(list0):], color='k',alpha=0.5)  ## 25% lower bound
ax6[0][0].set_yticks(range(500,1350,250))
ax6[0][0].set_ylim(450,1400)
ax6[0][0].set_xlim(1870,2000)
ax6[0][0].set_title("forecast a_t")
## [0][1] plot : Pt plot
ax6[0][1].plot(data_forecast.year[1:],data_forecast.PBar_nP1[1:], color='r')         ## P value
ax6[0][1].set_yticks(range(10000,55000,10000))
ax6[0][1].set_ylim(5000,60000)
ax6[0][1].set_xlim(1870,2000)
ax6[0][1].set_title("variance P_t")
## [1][0] plot : v plot
ax6[1][0].plot(data_forecast.year[1:],data_forecast.yBar_nPj[1:], color='r')         ## v value
ax6[1][0].plot(data_forecast.year[1:],np.zeros(len(data_forecast.yBar_nPj[1:])), color='k') ## plotting the y=0 line
ax6[1][0].set_xlim(1870,2000)
ax6[1][0].set_yticks(range(800,1200,100))
ax6[1][0].set_ylim(700,1200)
ax6[1][0].set_title("observation forecast")
## [1][1] plot : F plot
ax6[1][1].plot(data_forecast.year[1:],data_forecast.FBar_nPj[1:], color='r')         ## F value
ax6[1][1].set_yticks(range(20000,65000,10000))
ax6[1][1].set_ylim(20000,65000)
ax6[1][1].set_xlim(1870,2000)
ax6[1][1].set_title("observation forecast variance F_t")

## Exporting Graph
# fig6.savefig(graph_export_path+'fig2.6.png',dpi=300)


#######################################################################
## Maximum likelihood Estimation for sigma_eps and sigma_eta
#######################################################################
def update_ft_star(P_t_star):
    return P_t_star+1
def update_kt_estimation(P_t_star,F_t_star):
    return P_t_star/F_t_star

## Adjusted Kalman Filter for the estimation process
def Kalman_Filter_MLE(data_in,q):
    for i in data_in.index:
        data_in.loc[i,'F']=update_ft_star(data_in.loc[i,'P'])
        data_in.loc[i,'K']=update_kt_estimation(data_in.loc[i,'P'],data_in.loc[i,'F'])
        data_in.loc[i,'L']=update_lt(data_in.loc[i,'K'])
        data_in.loc[i,'v']=cal_vt(data_in.loc[i,'volume'],data_in.loc[i,'a'])
        #data_in.loc[i-1,'N']=update_Nt_m1(data_in.loc[i,'F'],data_in.loc[i,'L'],data_in.loc[i,'N'])
        #data_in.loc[i,'D']=update_D_t(data_in.loc[i,'F'], data_in.loc[i,'K'],data_in.loc[i,'N'])
        data_in.loc[i+1,'a']=data_in.loc[i,'a']+data_in.loc[i,'K']*data_in.loc[i,'v']
        if i==0:
            data_in.loc[i+1,'P']=1+q
        else:
            data_in.loc[i+1,'P']=data_in.loc[i,'P']*(1-data_in.loc[i,'K'])+q
    return data_in[:len(data_in.index)-1]

# loglikelihood for q with the input as psi
def q_lik(psi):
    q=np.exp(psi)
    data_MLE=data_import()
    n=len(data_MLE.index)
    data_MLE=Kalman_Filter_MLE(data_MLE, q)
    data_MLE.loc[:,'v_square']=np.square(data_MLE.v)
    sum_v_over_f=(data_MLE.v_square/data_MLE.F)[1:].sum()
    sum_f=np.log(data_MLE.F[1:]).sum()
    sig2_eps_hat=sum_v_over_f/(n-1)
    log_lik_value= -(n/2)* np.log(2*np.pi)-(n-1)/2-((n-1)/2)*np.log(sig2_eps_hat)-sum_f/2
    return -log_lik_value ## minus sign because of maximization problem

## to obtain the sigma_square_eps after finding q
def estimate_sig2_eps(psi):
    q=np.exp(psi)
    data_MLE=data_import()
    n=len(data_MLE.index)
    data_MLE=Kalman_Filter_MLE(data_MLE, q)
    data_MLE.loc[:,'v_square']=np.square(data_MLE.v)
    sum_v_over_f=(data_MLE.v_square/data_MLE.F)[1:].sum()
    sig2_eps_hat=sum_v_over_f/(n-1)
    return sig2_eps_hat

## the main function for the estimation of two sigma
def hyperparameter_estimation():
    init_psi=0
    options ={'eps':1e-09,  # argument convergence criteria
             'disp': True,  # display iterations
             'maxiter':200} # maximum number of iterations
    results = scipy.optimize.minimize(q_lik, 
                                      x0=init_psi, 
                                      options = options,
                                      method='BFGS' ,
                                      ) #restrictions in parameter space
    ## Debug Only : For printing max result
    # print(results.x)
    # print(results.fun)
    # print(results.success)
    best_qsi=results.x[0]
    est_q=np.exp(best_qsi)
    sig2_eps_hat=estimate_sig2_eps(best_qsi)
    sig2_eta_hat=est_q*sig2_eps_hat
    print('estimated q:',est_q)
    print('estimated sig2_eps_hat:',sig2_eps_hat)
    print('estimated sig2_eta_hat:',sig2_eta_hat)
    return est_q,sig2_eps_hat,sig2_eta_hat

#######################################################################
## Running the estimation function
est_q,sig2_eps_hat,sig2_eta_hat=hyperparameter_estimation()
data_MLE=Kalman_Filter(data_import(),sig2_eps_hat,sig2_eta_hat)
data_MLE=Smoothing(data_MLE,sig2_eps_hat,sig2_eta_hat)
data_MLE=Disturbance_Smoothing(data_MLE,sig2_eps_hat,sig2_eta_hat)

#######################################################################
## Fig 2.7 (Diagnostic Checks for standardised residuals) 
#######################################################################
def calculate_et(v,F):
    return v/np.sqrt(F)

def calculate_m_value(e):
    n=e.count()
    m1=e.sum()/n
    diff=e-m1
    m2=np.power(diff,2).sum()/n
    m3=np.power(diff,3).sum()/n
    m4=np.power(diff,4).sum()/n
    return m1,m2,m3,m4
    
def fig2_7(data_in):
    data_in.loc[:,'e']=calculate_et(data_in.v,data_in.F)
    m1,m2,m3,m4=calculate_m_value(data_in.e[:-1])
    return data_in

#######################################################################
data_MLE=fig2_7(data_MLE)
## Fig 2.7
fig7,ax7 =  plt.subplots(2,2,figsize=(15,10))
year_label=data_MLE.year
y_e=data_MLE.e
## for the historgram and density
y_e_wo_na=y_e
kde_axis=np.linspace(-3, 3, 1000)
kde = stats.gaussian_kde(y_e_wo_na)

##Data for QQ Plot
y_e_sorted=sorted(y_e)
N=len(y_e_sorted)
actual_dist_quantiles = []
quantiles_percent = []
for i, val in enumerate(y_e_sorted[:-1]):
    actual_dist_quantiles.append((val + y_e_sorted[i+1])/2)
    quantiles_percent.append((i+1)/N)
theoretical_quantiles=norm.ppf(quantiles_percent)
lim=[actual_dist_quantiles[0],actual_dist_quantiles[N-2]]

## [0][0] plot : filter a plot
ax7[0][0].plot(year_label,y_e, color='b')    ## e
ax7[0][0].plot(year_label,np.zeros(len(y_e)), color='k') ## plotting the y=0 line

## [0][1] plot : Pt plot
ax7[0][1].hist(y_e,bins=10, density = True) ## Histogram
ax7[0][1].plot(kde_axis,kde(kde_axis))      ## the kde line

# # ## [1][0] plot : v plot
ax7[1][0].plot(theoretical_quantiles,actual_dist_quantiles, color='b')    ## quantile-quantile plot
ax7[1][0].plot(lim,lim, color='k')         ## y=x line
ax7[1][0].grid(True)
# # ax5[1][0].set_yticks(range(-500,500,250))
# # ## [1][1] plot : F plot
ax7[1][1].acorr(y_e,maxlags=10,linewidth=4)         ## auto-correlation
ax7[1][1].set_xlim([0,11])
ax7[1][1].set_ylim([-1,1])
# ax5[1][1].set_yticks(range(20000,32501,2500))

## Exporting Graph
# fig7.savefig(graph_export_path+'fig2.7.png',dpi=300)

#######################################################################
## Figure 2.8 Diagnostic checks for auxilliary residuals
#######################################################################

def update_ustar_t(D_t, u_t):
     ustar_t = (1/np.sqrt(D_t)) * u_t
     return ustar_t

def update_rstar_t(N_t, r_t):
    rstar_t = (1/np.sqrt(N_t)) * r_t
    return rstar_t

def calculate_k_value_ustar(ustar_t):
    n=ustar_t.count()
    k1=ustar_t.sum()/n
    diff=ustar_t-k1
    k2=np.power(diff,2).sum()/n
    k3=np.power(diff,3).sum()/n
    k4=np.power(diff,4).sum()/n
    return k1,k2,k3,k4

def calculate_m_value_rstar(rstar_t):
    n=rstar_t.count()
    m1=rstar_t.sum()/n
    diff=rstar_t-m1
    m2=np.power(diff,2).sum()/n
    m3=np.power(diff,3).sum()/n
    m4=np.power(diff,4).sum()/n
    return m1,m2,m3,m4

def fig2_8(data_in):
    data_in.loc[:,'ustar']=update_ustar_t(data_in.D,data_in.U)
    data_in.loc[:,'rstar']=update_rstar_t(data_in.N,data_in.r)
    k1,k2,k3,k4=calculate_k_value_ustar(data_in.ustar[:-1])
    m1,m2,m3,m4=calculate_m_value_rstar(data_in.rstar[:-1])
    return data_in
#######################################################################
data_MLE=fig2_8(data_MLE)

fig8,ax8 =  plt.subplots(2,2,figsize=(15,10))
year_label=data_MLE.year
y_ustar=data_MLE.ustar
y_rstar=data_MLE.rstar
y_ustar_wo_na=y_ustar
y_rstar_wo_na=y_rstar.loc[0:98]
kdustar_axis=np.linspace(-3, 3, 1000)
kdustar = stats.gaussian_kde(y_ustar_wo_na)
kdrstar_axis=np.linspace(-4, 4, 1000)
bandwith = 10 ** np.linspace(-3,3,1000)
kdrstar = stats.gaussian_kde(y_rstar_wo_na)
y_rstarm1 = y_rstar.loc[0:98] 
## [0][0] plot : filter a plot
ax8[0][0].plot(year_label,y_ustar, color='b')    ## ustar
ax8[0][0].plot(year_label,np.zeros(len(y_ustar)), color='k') ## plotting the y=0 line

## [0][1] plot : Pt plot
ax8[0][1].hist(y_ustar,bins=10, density = True) ## Histogram u star
ax8[0][1].plot(kdustar_axis,kdustar(kdustar_axis))      ## the kde line

# # ## [1][0] plot : v plot
ax8[1][0].plot(year_label,y_rstar, color='b')    ## r star
ax8[1][0].plot(year_label,np.zeros(len(y_rstar)), color='k')
# # ax5[1][0].set_yticks(range(-500,500,250))
# # ## [1][1] plot : F plot
ax8[1][1].hist(y_rstarm1,bins=10, density = True) ## Histogram r star
ax8[1][1].plot(kdrstar_axis,kdrstar(kdrstar_axis))      ## the kde line

## Exporting Graph
# fig8.savefig(graph_export_path+'fig2.8.png',dpi=300)