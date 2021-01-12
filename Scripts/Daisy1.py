# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 18:16:01 2020
@author: T Y van der Duim
"""

'''
------------------------------PACKAGES-----------------------------------------
'''

import numpy as np
import matplotlib.pyplot as plt
import math as m

'''
---------------------------plotting preferences--------------------------------
'''
plt.style.use('seaborn-dark')
plt.rc('text', usetex=False)
plt.rc('font', family='times')
plt.rc('xtick', labelsize=15) 
plt.rc('ytick', labelsize=15) 
plt.rc('font', size=15) 
plt.rc('figure', figsize = (12, 5))

'''
-------------------------------PARAMS------------------------------------------
'''

S_0 = 1366 # solar constant, W/m^2
alpha_g = 0.5 # albedo ground
alpha_w = 0.75 # albbedo white daisies
alpha_b = 0.25 # albedo black daisies
gamma = 0.3 # death rate daisies per unit time
p = 1 # proportion of the planets area which is fertile ground
beta = 16 # what is this? (W m-2 K-1)
b = 2.2 # what is this? (W m-2 K-1)
I_0 = 220 # W m-2
L = 1 # Percentage of the current solar luminosity
T_opt = 22.5 # optimum temperature daisies
T_min = 5 # mimimum temperature daisies
T_max = 40 # maximum temperature daisies

'''
------------------------------FUNCTIONS----------------------------------------
'''

def alpha_p(A_w, A_b):
    A_g = p - A_w  - A_b
    return alpha_g * A_g + alpha_w * A_w + alpha_b * A_b

def avg_T_g(L, A_w, A_b):
    alphap = alpha_p(A_w, A_b)
    return (0.25 * S_0 * L * (1 - alphap) - I_0) / b

def T_daisy(L, A_w, A_b, daisy_type):
    if daisy_type == "black":
        alpha_i = alpha_b
    elif daisy_type == "white":
        alpha_i = alpha_w
    else:
        print("daisy type not recognized")
        alpha_i = np.nan
    alphap = alpha_p(A_w, A_b)
    T_g = avg_T_g(L, A_w, A_b)
    return 0.25 * S_0 * L * (alphap - alpha_i) / (b + beta) + T_g
    
def growth_rate(L, A_w, A_b, daisy_type):
    Tdaisy = T_daisy(L, A_w, A_b, daisy_type)
    return 1 - 4 * (T_opt - Tdaisy)**2 / (T_max - T_min)**2

def dA_dt(L, A_w, A_b, daisy_type):
    A_g = p - A_w  - A_b
    if daisy_type == "black":
        A = A_b
    elif daisy_type == "white":
        A = A_w
    else:
        print("daisy type not recognized")
        A = np.nan
    beta = growth_rate(L, A_w, A_b, daisy_type)
    return A * (A_g * beta - gamma)

'''
--------------------------TIME INTEGRATION-------------------------------------
'''

t_init = 0 # initial time
t_end = 10 # end time of simulation in seconds
dt = 0.01 # time step in seconds
time = np.arange(t_init, t_end + dt, dt) # time array

lums = np.concatenate([np.arange(0.6, 2.4, 0.05), np.arange(2.35, 0.55, -0.05)])
temps = []
aws = []
abss = []

for L in lums:
    print(L)
    A_w = np.zeros((len(time),)) # area white daisies
    A_b = np.zeros((len(time),)) # area black daisies
    temperatures = []
    A_w_max = 0
    A_b_max = 0
    
    # initial conditions
    idx = 0
    if L == lums[0]:
        A_w[idx] = 0.5 # start with half of the available area white daisies
        A_b[idx] = 0.5 # start with half of the available area black daisies
    else:
        if A_w_max < 0.01 and A_b_max < 0.01:
            A_w[idx] = 0.01
            A_b[idx] = 0.01
        elif A_w_max < 0.01 and A_b_max >= 0.01:
            A_w[idx] = 0.01
            A_b[idx] = A_b_max
        elif A_w_max >= 0.01 and A_b_max < 0.01:
            A_w[idx] = A_w_max
            A_b[idx] = 0.01
        else:
            A_w[idx] = A_w_max # start with steady state soln previous iteration white daisies
            A_b[idx] = A_b_max # start with steady state soln previous iteration black daisies
    # print(A_w[idx], A_b[idx])
    for idx in range(len(time) - 1):
        
        X_0 = A_w[idx]
        Y_0 = A_b[idx]
        X_1 = X_0 + dA_dt(L, X_0, Y_0, daisy_type = "white") * dt / 2
        Y_1 = Y_0 + dA_dt(L, X_0, Y_0, daisy_type = "black") * dt / 2
        X_2 = X_0 + dA_dt(L, X_1, Y_1, daisy_type = "white") * dt / 2
        Y_2 = Y_0 + dA_dt(L, X_1, Y_1, daisy_type = "black") * dt / 2
        X_3 = X_0 + dA_dt(L, X_2, Y_2, daisy_type = "white") * dt
        Y_3 = Y_0 + dA_dt(L, X_2, Y_2, daisy_type = "black") * dt
        X_4 = X_0 - dA_dt(L, X_3, Y_3, daisy_type = "white") * dt / 2
        Y_4 = Y_0 - dA_dt(L, X_3, Y_3, daisy_type = "black") * dt / 2
        A_w[idx + 1] = (X_1 + 2 * X_2 + X_3 - X_4) / 3
        A_b[idx + 1] = (Y_1 + 2 * Y_2 + Y_3 - Y_4) / 3
        #temperatures.append(avg_T_g(L, A_w[idx + 1], A_b[idx + 1]))
    A_w_max = A_w[-1]
    A_b_max = A_b[-1]
    aws.append(A_w_max)
    abss.append(A_b_max)
    temps.append(avg_T_g(L, A_w[-1], A_b[-1]))
    
  
    
plt.figure()
plt.plot(lums,aws)
plt.plot(lums,abss)
plt.xlabel("Solar luminosity")
plt.ylabel("Temp (deg C)")
plt.grid()


    
    