
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
alpha_g = 0.5 # albedo ground
alpha_w = 0.75 # albedo white daisies
alpha_b = 0.25 # albedo black daisies
gamma = 0.3 # death rate daisies per unit time
p = 1 # proportion of the planets area which is fertile ground
beta = 16 # Meridional heat transport (W m-2 K-1)
b = 2.2 # Net outgoing longwave radiation due to daisies (W m-2 K-1)
I_0 = 220 # Constant outgoing radiation due to planet (W m-2)
L = 1 # Percentage of the current solar luminosity
T_opt = 22.5 # optimum temperature daisies
T_min = 5 # mimimum temperature daisies
T_max = 40 # maximum temperature daisies

#%%
'''
------------------------------MAIN CLASS----------------------------------------
'''

class Daisies:
    
    def __init__(self, A_w, A_b, L, *args): # optionally add a specific latitude
        self.A_w = A_w
        self.A_b = A_b
        self.L = L
        self.phi = args
        if self.phi:
            self.phi = np.radians(args)[0]

    '''
    ---------------------------------GLOBAL------------------------------------
    '''
        
    def A_g(self):
        return p - self.A_w  - self.A_b
        
    def alpha_p(self):
        return alpha_g * self.A_g() + alpha_w * self.A_w + alpha_b * self.A_b
            
    def avg_T_g(self):
        return (S_0 / 4 * self.L * (1 - self.alpha_p()) - I_0) / b
    
    '''
    ---------------------LATITUDE DEPENDENCE (LOCAL)---------------------------
    '''
    
    def sol_irradiance(self):
        d_m = 1.495978770e9 # distance Sun-Earth
        d = d_m # assume dm / d to be 1 (perfect circle)
        delta = 0 # Sun inclination assumed to be zero
        delta_H = np.arccos(-np.tan(self.phi) * np.tan(delta)) # daylength
        return S_0 / m.pi * (d_m / d)**2  * (delta_H * np.sin(self.phi) * np.sin(delta) + \
                                              np.cos(self.phi) * np.cos(delta) * np.sin(delta_H))
    
    def avg_T_lat(self):
        if abs(self.phi) >= 0 and abs(self.phi) < np.radians(60):
            alpha_g = 0.32
        elif abs(self.phi) >= np.radians(60) and abs(self.phi) < np.radians(80):
            alpha_g = 0.50
        else:
            alpha_g = 0.62
        return (self.sol_irradiance() * self.L * (1 - self.alpha_p()) - I_0) / b, \
            (self.sol_irradiance() * self.L * (1 - self.alpha_p()) - I_0 + beta * self.avg_T_g()) / (b + beta)
    
    '''
    ---------------------------DAISY DYNAMICS----------------------------------
    '''
    
    def T_daisy(self, daisytype):
        if daisytype == "black":
            alpha_i = alpha_b
        elif daisytype == "white":
            alpha_i = alpha_w
        else:
            print("daisy type not recognized")
            alpha_i = np.nan
        if self.phi:
            return 0.25 * S_0 * self.L * (self.alpha_p() - alpha_i) / (b + beta) + self.avg_T_lat()
        else:
            return 0.25 * S_0 * self.L * (self.alpha_p() - alpha_i) / (b + beta) + self.avg_T_g()
    
    def growth_rate(self, daisytype):
        return max(1 - 4 * (T_opt - self.T_daisy(daisytype))**2 / (T_max - T_min)**2, 0)
    
    def dA_dt(self, daisytype):
        if daisytype == "black":
            A = self.A_b
        elif daisytype == "white":
            A = self.A_w
        else:
            print("daisy type not recognized")
            A = np.nan
        return A * (self.A_g() * self.growth_rate(daisytype) - gamma)
        
    def RK4(self, include_daisy):
        X_1 = self.A_w + self.dA_dt(daisytype = "white") * dt / 2
        Y_1 = self.A_b + self.dA_dt(daisytype = "black") * dt / 2
        X_2 = self.A_w + self.dA_dt(daisytype = "white") * dt / 2
        Y_2 = self.A_b + self.dA_dt(daisytype = "black") * dt / 2
        X_3 = self.A_w + self.dA_dt(daisytype = "white") * dt
        Y_3 = self.A_b + self.dA_dt(daisytype = "black") * dt
        X_4 = self.A_w - self.dA_dt(daisytype = "white") * dt / 2
        Y_4 = self.A_b - self.dA_dt(daisytype = "black") * dt / 2
        white_daisies = {"white" : (X_1 + 2 * X_2 + X_3 - X_4) / 3, "black" : 0,
                         "white & black" : (X_1 + 2 * X_2 + X_3 - X_4) / 3}
        black_daisies = {"white" : 0, "black" : (Y_1 + 2 * Y_2 + Y_3 - Y_4) / 3,
                         "white & black" : (Y_1 + 2 * Y_2 + Y_3 - Y_4) / 3}
        return white_daisies[include_daisy], black_daisies[include_daisy]

lats = np.arange(-90, 91, 1)

lat_temps = [Daisies(0.1, 0.7, 1, lat).avg_T_lat()[1] for lat in lats]
#T_transfer = [avg_T_lat(lat = lat, L = 1, A_w = 0.1, A_b = 0.75)[1] for lat in lats]
#T_notransfer = [avg_T_lat(lat = lat, L = 1, A_w = 0.1, A_b = 0.75)[0] for lat in lats]

#%%
'''
--------------------------TIME INTEGRATION-------------------------------------
'''

t_init = 0 # initial time
t_end = 10 # end time of simulation in seconds
maxstep = 1000 # maximum nr of steps
time = np.linspace(t_init, t_end, maxstep + 1) # time array
dt = (t_end - t_init) / maxstep

luminosities = np.concatenate([np.arange(0.6, 3, 0.05), np.arange(2.95, 0.5, -0.05)])
A_w_steady = 0.5
A_b_steady = 0.5

area_white_steady = np.zeros((len(luminosities),))
area_black_steady = np.zeros((len(luminosities),))
area_total = np.zeros((len(luminosities),))
growth_white = np.zeros((len(luminosities),))
growth_black = np.zeros((len(luminosities),))
Temp_white_daisy = np.zeros((len(luminosities),))
Temp_black_daisy = np.zeros((len(luminosities),))
temperatures = np.zeros((len(luminosities),))

daisy_setting = "white & black"

for idx, L in enumerate(luminosities):
    print("computing steady state solution for luminosity #{0} out of {1}.".format(idx + 1, len(luminosities)))
    A_w = np.zeros((len(time),)) # area white daisies
    A_b = np.zeros((len(time),)) # area black daisies

    # initial conditions
    it = 0
    if idx == 0:
        A_w[it] = 0.5 # start with half of the available area white daisies
        A_b[it] = 0.5 # start with half of the available area black daisies
    else:
        A_w[it] = A_w_steady # start with steady state white daisies
        A_b[it] = A_b_steady # start with steady state black daisies

    # solve Runge-Kutta scheme
    for it in range(len(time) - 1):
        X_0 = A_w[it]
        Y_0 = A_b[it]
        Daisy = Daisies(X_0, Y_0, L)
        A_w[it + 1] = Daisy.RK4(include_daisy = daisy_setting)[0]
        A_b[it + 1] = Daisy.RK4(include_daisy = daisy_setting)[1]
        if A_w[it + 1] < 0:
            A_w[it + 1] = 0
        if A_b[it + 1] < 0:
            A_b[it + 1] = 0
        
    # save solutions
    A_w_steady = A_w[-1]
    A_b_steady = A_b[-1]
    area_white_steady[idx] = A_w_steady
    area_black_steady[idx] = A_b_steady
    area_total[idx] = A_w_steady + A_b_steady
    growth_white[idx] = Daisies(A_w_steady, A_b_steady, L).growth_rate(daisytype = "white")
    growth_black[idx] = Daisies(A_w_steady, A_b_steady, L).growth_rate(daisytype = "black")
    Temp_white_daisy[idx] = Daisies(A_w_steady, A_b_steady, L).T_daisy(daisytype = "white")
    Temp_black_daisy[idx] = Daisies(A_w_steady, A_b_steady, L).T_daisy(daisytype = "black")
    temperatures[idx] = Daisies(A_w_steady, A_b_steady, L).avg_T_g()
    if A_w_steady < 1e-3:
        A_w_steady = 0.01
    if A_b_steady < 1e-3:
        A_b_steady = 0.01


#%%
'''
-------------------------------FIGURES-----------------------------------------
'''

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

fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)

ax1.set_facecolor('darkgrey')
ax1.plot(luminosities[:int(len(luminosities) / 2)], area_white_steady[:int(len(luminosities) / 2)],\
         color = 'white', label = 'White daisies (increasing L)')
ax1.plot(luminosities[int(len(luminosities) / 2):], area_white_steady[int(len(luminosities) / 2):],\
         color = 'white', linestyle = 'dashed', label = 'White daisies (decreasing L)')
ax1.plot(luminosities[:int(len(luminosities) / 2)], area_total[:int(len(luminosities) / 2)],\
         color = 'blue', label = 'All daisies (increasing L)')
ax1.plot(luminosities[int(len(luminosities) / 2):], area_total[int(len(luminosities) / 2):],\
         color = 'blue', linestyle =  'dashed', label = 'All daisies (decreasing L)')
ax1.plot(luminosities[:int(len(luminosities) / 2)], area_black_steady[:int(len(luminosities) / 2)],\
         color = 'black', label = 'Black daisies (increasing L)')
ax1.plot(luminosities[int(len(luminosities) / 2):], area_black_steady[int(len(luminosities) / 2):],\
        color = 'black', linestyle = 'dashed', label = 'Black daisies (decreasing L)')
ax1.set_ylabel("Area (-)")
ax1.legend()
ax1.grid(color = 'grey')

ax2.set_facecolor('darkgrey')
ax2.plot(luminosities[:int(len(luminosities) / 2)], temperatures[:int(len(luminosities) / 2)],\
         color = 'darkblue', label = "increasing L")
ax2.plot(luminosities[int(len(luminosities) / 2):], temperatures[int(len(luminosities) / 2):],\
         color = 'darkblue', linestyle = 'dashed', label = "decreasing L")
ax2.legend()
ax2.set_ylabel("Temperature (deg C)")
ax2.set_xlabel("Solar luminosity")
ax2.grid(color = 'grey')

fig.suptitle("Run for {0} daisies, adjusting initial conditions".format(daisy_setting))

plt.figure()
ax = plt.gca()
ax.set_facecolor('darkgrey')
plt.plot(luminosities[:int(len(luminosities) / 2)], growth_white[:int(len(luminosities) / 2)],\
         color = 'white', label = 'Growth rate white daisies (increasing L)')
plt.plot(luminosities[int(len(luminosities) / 2):], growth_white[int(len(luminosities) / 2):],\
         color = 'white', linestyle = 'dashed', label = 'Growth rate white daisies (decreasing L)')
plt.plot(luminosities[:int(len(luminosities) / 2)], growth_black[:int(len(luminosities) / 2)],\
         color = 'black', label = 'Growth rate black daisies (increasing L)')
plt.plot(luminosities[int(len(luminosities) / 2):], growth_black[int(len(luminosities) / 2):],\
        color = 'black', linestyle = 'dashed', label = 'Growth rate black daisies (decreasing L)')
#plt.plot(Temp_white_daisy, growth_white, color = 'white', label = 'Growth rate white daisies')
#plt.plot(Temp_black_daisy, growth_black, color = 'black', label = 'Death rate black daisies')
plt.axhline(gamma, label = "Death rate")
plt.xlabel("Daisy temperature (deg C)")
#plt.ylim([-2, 2])
plt.ylabel("Growth/death rate (-)")
plt.title("Run for {0} daisies, adjusting initial conditions".format(daisy_setting))
plt.legend()
plt.grid(color = 'grey') 