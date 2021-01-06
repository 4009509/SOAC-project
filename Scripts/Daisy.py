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
plt.style.use('seaborn-darkgrid')
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
alpha_g = 0.25 # albedo ground
alpha_w = 0.75 # albedo white daisies
alpha_b = 0.15 # albedo black daisies
gamma = 0.3 # death rate daisies per unit time
p = 1 # proportion of the planets area which is fertile ground
beta = 16 # Meridional heat transport (W m-2 K-1)
b = 2.2 # Net outgoing longwave radiation due to daisies (W m-2 K-1)
I_0 = 220 # Cnstant outgoing radiation due to planet
L = 1 # Percentage of the current solar luminosity
T_opt = 22.5 # optimum temperature daisies
T_min = 5 # mimimum temperature daisies
T_max = 40 # maximum temperature daisies

'''
------------------------------FUNCTIONS----------------------------------------
'''

def alpha_p(alpha_g, A_w, A_b):
    A_g = p - A_w  - A_b
    return alpha_g * A_g + alpha_w * A_w + alpha_b * A_b

def sol_irradiance(phi):
    d_m = 1.495978770e9 # distance Sun-Earth
    d = d_m # assume dm / d to be 1 (perfect circle)
    delta = 0 # Sun inclination assumed to be zero
    delta_H = np.arccos(-np.tan(phi) * np.tan(delta)) # daylength
    return S_0 / m.pi * (d_m / d)**2  * (delta_H * np.sin(phi) * np.sin(delta) + \
                                          np.cos(phi) * np.cos(delta) * np.sin(delta_H))

def avg_T_g(L, A_w, A_b):
    alphap = alpha_p(alpha_g, A_w, A_b)
    return (S_0 / 4 * L * (1 - alphap) - I_0) / b

def avg_T_lat(lat, L, A_w, A_b):
    if abs(lat) >= 0 and abs(lat) < 60:
        alpha_g = 0.32
    elif abs(lat) >= 60 and abs(lat) < 80:
        alpha_g = 0.50
    else:
        alpha_g = 0.62
    alphap = alpha_p(alpha_g, A_w, A_b)
    lat = np.radians(lat)
    Q = sol_irradiance(lat)
    T_p = avg_T_g(L, A_w, A_b)
    return (Q * L * (1 - alphap) - I_0) / b, \
        (Q * L * (1 - alphap) - I_0 + beta * T_p) / (b + beta)

lats = np.arange(-90, 91, 1)

T_transfer = [avg_T_lat(lat = lat, L = 1, A_w = 0.1, A_b = 0.75)[1] for lat in lats]
T_notransfer = [avg_T_lat(lat = lat, L = 1, A_w = 0.1, A_b = 0.75)[0] for lat in lats]

plt.figure()
ax = plt.gca()
ax.set_facecolor('darkgrey')
plt.plot(lats, T_transfer, label = "including meridional heat transfer")
plt.plot(lats, T_notransfer, label = "excluding meridional heat transfer")
plt.xlabel("latitude (deg)")
plt.ylabel("temperature (deg C)")
plt.grid(color = 'grey')
plt.legend()
plt.show()


def T_daisy(L, A_w, A_b, daisy_type):
    if daisy_type == "black":
        alpha_i = alpha_b
    elif daisy_type == "white":
        alpha_i = alpha_w
    else:
        print("daisy type not recognized")
        alpha_i = np.nan
    alphap = alpha_p(alpha_g, A_w, A_b)
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
    beta_g = growth_rate(L, A_w, A_b, daisy_type)
    return A * (A_g * beta_g - gamma)

#%%
'''
--------------------------TIME INTEGRATION-------------------------------------
'''

t_init = 0 # initial time
t_end = 1e1 # end time of simulation in seconds
dt = 0.01 # time step in seconds
time = np.arange(t_init, t_end + dt, dt) # time array

lums = np.arange(1.90, 0.55, -0.05)
#lums = np.arange(0.6, 2, 0.05)
#lums = np.concatenate([np.arange(0.6, 2, 0.05), np.arange(1.90, 0.55, -0.05)])
temps = []
aws = []
abss = []
growth = []
Tempdaisy = []

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
        if A_w_max < 1e-3 and A_b_max < 1e-3:
            A_w[idx] = 0.01
            A_b[idx] = 0.01
        elif A_w_max < 1e-3 and A_b_max >= 1e-3:
            A_w[idx] = 0.01
            A_b[idx] = A_b_max
        elif A_w_max >= 1e-3 and A_b_max < 1e-3:
            A_w[idx] = A_w_max
            A_b[idx] = 0.01
        else:
            A_w[idx] = A_w_max # start with steady state soln previous iteration white daisies
            A_b[idx] = A_b_max # start with steady state soln previous iteration black daisies
    for idx in range(len(time) - 1):
        
        X_0 = A_w[idx]
        Y_0 = 0#A_b[idx]
        X_1 = X_0 + dA_dt(L, X_0, Y_0, daisy_type = "white") * dt / 2
        Y_1 = 0#Y_0 + dA_dt(L, X_0, Y_0, daisy_type = "black") * dt / 2
        X_2 = X_0 + dA_dt(L, X_1, Y_1, daisy_type = "white") * dt / 2
        Y_2 = 0#Y_0 + dA_dt(L, X_1, Y_1, daisy_type = "black") * dt / 2
        X_3 = X_0 + dA_dt(L, X_2, Y_2, daisy_type = "white") * dt
        Y_3 = 0#Y_0 + dA_dt(L, X_2, Y_2, daisy_type = "black") * dt
        X_4 = X_0 - dA_dt(L, X_3, Y_3, daisy_type = "white") * dt / 2
        Y_4 = 0#Y_0 - dA_dt(L, X_3, Y_3, daisy_type = "black") * dt / 2
        A_w[idx + 1] = (X_1 + 2 * X_2 + X_3 - X_4) / 3
        A_b[idx + 1] = (Y_1 + 2 * Y_2 + Y_3 - Y_4) / 3
        temperatures.append(avg_T_g(L, A_w[idx + 1], A_b[idx + 1]))
    growth.append(growth_rate(L, A_w[-1], A_b[-1], daisy_type = "white"))
    Tempdaisy.append(T_daisy(L, A_w[-1], A_b[-1], daisy_type = "white"))
    A_w_max = A_w[-1]
    A_b_max = A_b[-1]
    aws.append(A_w_max)
    abss.append(A_b_max)
    temps.append(avg_T_g(L, A_w[-1], A_b[-1]))

#%%
plt.figure()
ax = plt.gca()
ax.set_facecolor('darkgrey')
plt.plot(lums,temps, color = 'darkblue', label = 'White daisies')
#plt.plot(lums,abss, color = 'black', label = 'Black daisies')
plt.xlabel("Solar luminosity")
#plt.xlim([0.5,1.7])
plt.ylabel("Temperature (deg C)")
plt.title("White daisies, adjusting initial conditions")
#plt.legend()
#plt.ylim([0,70])
plt.grid(color = 'grey')    
    
plt.figure()
ax = plt.gca()
ax.set_facecolor('darkgrey')
plt.plot(temps, aws, color = 'white', label = 'White daisies')
plt.xlabel("Solar luminosity")
#plt.xlim([0.5,1.7])
plt.ylabel("Area (-)")
plt.title("White daisies, adjusting initial conditions")
#plt.legend()
#plt.ylim([0,70])
plt.grid(color = 'grey')    

plt.figure()
ax = plt.gca()
ax.set_facecolor('darkgrey')
plt.plot(Tempdaisy, growth, color = 'white', label = 'White daisies')
plt.axhline(gamma)
plt.xlabel("Solar luminosity")
#plt.xlim([0.5,1.7])
plt.ylabel("Area (-)")
plt.title("White daisies, adjusting initial conditions")
#plt.legend()
#plt.ylim([0,70])
plt.grid(color = 'grey') 