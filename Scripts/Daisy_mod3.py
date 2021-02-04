
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 18:16:01 2020

@authors: T Y van der Duim & F.Y.J. Drijfhout
"""

'''
------------------------------PACKAGES-----------------------------------------
'''

import numpy as np
import matplotlib.pyplot as plt
import math as m
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from random import randint, choice
from matplotlib import gridspec
import os
# ADJUST TO PATH
os.environ['PROJ_LIB'] = r"C:\Users\fafri\anaconda3\pkgs\proj4-6.1.1-hc2d0af5_1\Library\share"
from mpl_toolkits.basemap import Basemap

'''
---------------------------PLOTTING PREFERENCES--------------------------------
'''

plt.style.use('seaborn-darkgrid')
plt.rc('text', usetex=False)
plt.rc('font', family='times')
plt.rc('xtick', labelsize=20) 
plt.rc('ytick', labelsize=20) 
plt.rc('font', size=20) 
plt.rc('figure', figsize = (12, 5))

'''
---------------------------AUXILIARY FUNCTIONS---------------------------------
'''

def global_landf():
    '''
    Parameters
    ----------
    R : Radius Earth (m)
    lat : Latitude (deg)
    
    Returns
    -------
    global land fraction, computing by a weighting function using ground area
    as weights    
    '''
    R = 6.371e6 # radius Earth (m)
    labels = land_frac.keys()
    circums = [2 * m.pi * R * np.cos(np.radians(lat)) for lat in latitudes]
    weighted_landfrac = [land_frac[label] * circum for label, circum in zip(labels,circums)]
    return sum(weighted_landfrac) / sum(circums)

'''
-------------------------------PARAMS------------------------------------------
'''

land_frac = {"-90" : 1, "-85" : 1, "-80" : 0.85, "-75" : 0.7, "-70" : 0.45, "-65" : 0.1,
              "-60" : 0, "-55" : 0, "-50" : 0.05, "-45" : 0.05, "-40" : 0.05, "-35" : 0.1,
              "-30" : 0.2, "-25" : 0.25, "-20" : 0.25, "-15" : 0.2, "-10" : 0.2, "-5" : 0.25,
              "0" : 0.2, "5" : 0.2, "10" : 0.25, "15" : 0.25, "20" : 0.3, "25" : 0.35,
              "30" : 0.4,  "35" : 0.4, "40" : 0.4, "45" : 0.5, "50" : 0.55, "55" : 0.55,
              "60" : 0.6, "65" : 0.8, "70" : 0.6, "75" : 0.25, "80" : 0.15, "85" : 0.05,
              "90" : 0} # land fraction per latitude band as determined from Hartmann (1994)

latitudes = np.arange(-90, 91, 5) # latitudes defined in bands of 5 deg
S_0 = 1366 # solar constant, W/m^2
alpha_g = 0.335 # albedo ground
alpha_o = 0.28 # albedo ocean
alpha_w = 0.75 # albedo white daisies
alpha_b = 0.25 # albedo black daisies
gamma = 0.3 # death rate daisies per unit time
p = global_landf() # proportion of the planets area which is fertile ground
beta = 6 # Meridional heat transport (W m-2 K-1)
b = 2.2 # Net outgoing longwave radiation due to daisies (W m-2 K-1)
I_0 = 220 # Constant outgoing radiation due to planet (W m-2)
L = 1 # Percentage of the current solar luminosity
T_opt = 22.5 # optimum temperature daisies
T_min = 5 # mimimum temperature daisies
T_max = 40 # maximum temperature daisies
t_init = 0 # initial time
t_end = 10 # end time of simulation in seconds
maxstep = 100 # maximum nr of steps
time = np.linspace(t_init, t_end, maxstep + 1) # time array
dt = (t_end - t_init) / maxstep # time step (for 4th order RK scheme)
daisy_setting = "white & black" # choose what type of daisies can grow on the planet

print("In the Daisy World ocean model, the planet is covered for {0}% by oceans.\n".format(round((1 - p) * 100)),
      "Correcting for 50% cloud cover, the albedo for ocean grounds is {0}, for white daisies {1},".format(alpha_o, alpha_w),
          "for black daisies {0} and for the remaining fertile ground {1}.\n".format(alpha_b, alpha_g),
              "A latitude-variant solar irradiance is included, and the relaxation parameter",
                  "\u03B2 equals {0} W/m$^2$/K.".format(beta))
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
        
    def A_g(self):
        '''
        Returns
        -------
        area fertile ground (fraction)    
        '''
        label = str(int(round(np.degrees(self.phi))))
        return max(land_frac[label] - self.A_w  - self.A_b, 0)
    
    def A_o(self):
        '''
        Returns
        -------
        area ocean (fraction)
        '''
        label = str(int(round(np.degrees(self.phi))))
        return 1 - land_frac[label]
        
    def alpha_p(self):
        '''
        Returns
        -------
        net albedo of region     
        '''
        return alpha_g * self.A_g() + alpha_w * self.A_w + alpha_b * self.A_b + alpha_o * self.A_o()
            
    def avg_T_g(self):
        '''
        Returns
        -------
        global temperature   
        '''
        return (S_0 / 4 * self.L * (1 - self.alpha_p()) - I_0) / b
    
    '''
    ---------------------LATITUDE DEPENDENCE (LOCAL)---------------------------
    '''
    
    def sol_irradiance(self):
        '''
        Parameters
        ----------
        d_m : mean distance Sun-Earth
        d : current distance Sun-Earth
        delta : Sun inclination (assumed zero)
        delta_H : Daylength (in rad)
        
        Returns
        -------
        solar irradiance on specified latitude  
        '''
        d_m = 1.495978770e9
        d = d_m
        delta = 0
        delta_H = np.arccos(-np.tan(self.phi) * np.tan(delta))
        return S_0 / m.pi * (d_m / d)**2  * (delta_H * np.sin(self.phi) * np.sin(delta) + \
                                              np.cos(self.phi) * np.cos(delta) * np.sin(delta_H))
    
    def avg_T_lat(self):
        '''
        Returns
        -------
        temperature at specified latitude   
        '''
        return (self.sol_irradiance() * self.L * (1 - self.alpha_p()) - I_0) / b, \
            (self.sol_irradiance() * self.L * (1 - self.alpha_p()) - I_0 + beta * self.avg_T_g()) / (b + beta)
    
    '''
    ---------------------------DAISY DYNAMICS----------------------------------
    '''
    
    def T_daisy(self, daisytype):
        '''
        Parameters
        ----------
        daisytype : can be either black or white
        
        Returns
        -------
        local (daisy) temperature   
        '''
        if daisytype == "black":
            alpha_i = alpha_b
        elif daisytype == "white":
            alpha_i = alpha_w
        else:
            print("daisy type not recognized")
            alpha_i = np.nan
        if self.phi or self.phi == 0:
            return self.sol_irradiance() * self.L * (self.alpha_p() - alpha_i) / (b + beta) + self.avg_T_lat()[1]
        else:
            return 0.25 * S_0 * self.L * (self.alpha_p() - alpha_i) / (b + beta) + self.avg_T_g()
    
    def growth_rate(self, daisytype):
        '''
        Parameters
        ----------
        daisytype : can be either black or white
        
        Returns
        -------
        daisy growth rate
        '''
        return max(1 - 4 * (T_opt - self.T_daisy(daisytype))**2 / (T_max - T_min)**2, 0)
    
    def dA_dt(self, daisytype):
        '''
        Parameters
        ----------
        daisytype : can be either black or white
        
        Returns
        -------
        daisy population growth (per time unit)
        '''
        if daisytype == "black":
            A = self.A_b
        elif daisytype == "white":
            A = self.A_w
        else:
            print("daisy type not recognized")
            A = np.nan
        return A * (self.A_g() * self.growth_rate(daisytype) - gamma)
        
    def RK4(self, include_daisy):
        '''
        Parameters
        ----------
        include_daisy : include either no daisies, only black, only white, or black & white
        
        Returns
        -------
        RK scheme solution for white and black daisies
        '''
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

    def steady_state_sol(self, include_daisy):
        '''
        Parameters
        ----------
        include_daisy : include either no daisies, only black, only white, or black & white
        A_w : area white daisies
        A_b : area black daisies
        
        Returns
        -------
        steady state solution using the RK scheme
        '''
        A_w = np.zeros((len(time),))
        A_b = np.zeros((len(time),))

        # initial conditions
        it = 0
        A_w[it] = self.A_w
        A_b[it] = self.A_b
        # solve Runge-Kutta scheme
        for it in range(len(time) - 1):
            self.A_w = A_w[it]
            self.A_b = A_b[it]
            A_w[it + 1] = self.RK4(include_daisy)[0]
            A_b[it + 1] = self.RK4(include_daisy)[1]
            if A_w[it + 1] < 0:
                A_w[it + 1] = 0
            if A_b[it + 1] < 0:
                A_b[it + 1] = 0
        self.A_w, self.A_b = A_w[-1], A_b[-1]
        if self.A_w < 1e-3:
            self.A_w = 0.01
        if self.A_b < 1e-3:
            self.A_b = 0.01
        return self.A_w, self.A_b

#%%
'''
---------------------------------MAIN------------------------------------------
'''
print("---------------\n SIMULATION\n ---------------")
luminosities = np.arange(1, 3.05, 0.05) # vary luminosity

'''
Parameters
----------
A_w_steady_lat : steady state solution for white daisy area at specific latitude, for specific luminosity
A_b_steady_lat : steady state solution for black daisy area at specific latitude, for specific luminosity
A_g_steady_lat : steady state solution for fertile ground area at specific latitude, for specific luminosity
Temp_notrans : steady state temperature at location when there (possibly) are daisies, but no meridional heat transport
Temp_nodaisiestrans : steady state temperature at location when there are no daisies, but there is meridional heat transport
Temp_nodaisiesnotrans : steady state temperature at location when there are no daisies, and no meridional heat transport
Temp_trans : steady state temperature at location when there (possibly) are daisies and meridional heat transport
'''
        
A_w_steady_lat = np.zeros((len(luminosities), len(latitudes))) 
A_b_steady_lat = np.zeros((len(luminosities), len(latitudes)))
A_g_steady_lat = np.zeros((len(luminosities), len(latitudes)))
Temp_notrans = np.zeros((len(luminosities), len(latitudes)))
Temp_nodaisiestrans = np.zeros((len(luminosities), len(latitudes)))
Temp_nodaisiesnotrans = np.zeros((len(luminosities), len(latitudes)))
Temp_trans = np.zeros((len(luminosities), len(latitudes)))

for idx, L in enumerate(luminosities):
    for idy, lat in enumerate(latitudes):
        print("Steady state sol for L #{0} out of {1}, lat {2} out of {3}.".format(idx + 1, len(luminosities), idy + 1, len(latitudes)))
        
        if idx == 0 and lat != -60 and lat != -55:
            A_w_init = 0.01 # start with 0.01 available area white daisies
            A_b_init = 0.01 # start with 0.01 available area black daisies
        elif idx == 0 and lat == -60 and lat == -55:
            A_w_init = 0 # start with 0 available area white daisies (as there is no land for those lats)
            A_b_init = 0 # start with 0 available area black daisies (as there is no land for those lats)
        else:
            A_w_init = A_w_steady_lat[idx - 1, idy] # start with steady state previous run white daisies
            A_b_init = A_b_steady_lat[idx - 1, idy] # start with steady state previous run black daisies
            
        [A_w_steady_lat[idx, idy], A_b_steady_lat[idx, idy]] = Daisies(A_w_init, A_b_init, L, lat).steady_state_sol(include_daisy = daisy_setting)
        # steady state sols
        Daisy_steady = Daisies(A_w_steady_lat[idx, idy], A_b_steady_lat[idx, idy], L, lat)
        [Temp_notrans[idx, idy], Temp_trans[idx, idy]] = Daisy_steady.avg_T_lat()
        A_g_steady_lat[idx, idy] = Daisy_steady.A_g()
        # temperature without daisies
        [Temp_nodaisiesnotrans[idx, idy], Temp_nodaisiestrans[idx, idy]] = Daisies(0, 0, L, lat).avg_T_lat()

#%%
'''
------------------------------FIG 8 REPORT-------------------------------------
'''

X, Y = np.meshgrid(luminosities,latitudes)
fig, (axs) = plt.subplots(nrows = 2, ncols = 1, sharex = True)

cs = axs[0].contourf(X, Y, Temp_nodaisiestrans.T, levels=np.arange(-20,201,1),
                     extend = 'both', cmap=cm.jet)
axs[0].set_ylabel('latitude')
axs[0].set_yticks(np.arange(-90,91,45))
axs[0].set_title("(a)", fontsize = 20)

cs3 = axs[1].contourf(X, Y, Temp_trans.T, levels=np.arange(-20,201,1),
                      extend = 'both', cmap=cm.jet)
axs[1].set_xlabel('luminosity')
axs[1].set_ylabel('latitude')
axs[1].set_yticks(np.arange(-90,91,45))
axs[1].set_title("(b)", fontsize = 20)
col = fig.colorbar(cs3, ax = axs[[0,1]], label = "temperature ($\degree$C)")
plt.show()

#%%
'''
------------------------------FIG 9 REPORT-------------------------------------
'''

fig = plt.figure()
plt.tight_layout()
ax = fig.add_subplot(121,projection='3d')
ax.xaxis.labelpad=20
ax.yaxis.labelpad=20
ax.zaxis.labelpad=20
X, Y = np.meshgrid(latitudes, luminosities)
surf = ax.plot_surface(X, Y, A_w_steady_lat * 100, cmap=cm.jet, linewidth=0.1, alpha=0.8)
col = fig.colorbar(surf, shrink=0.5, aspect=5)
ax.set_yticks(np.arange(1, 3.5, 0.5))
col.mappable.set_clim(0, 60)
col.remove()
cset = ax.contour(X, Y, np.tile([land_frac[key] * 100 for key in land_frac.keys()], (41,1)),
                  zdir='y', offset=3, cmap=cm.coolwarm)
ax.set_xlabel("latitude", fontweight='bold')
ax.set_ylabel("luminosity", fontweight='bold')
ax.set_zlabel("area (%)", fontweight='bold')

ax = fig.add_subplot(122,projection='3d')
ax.xaxis.labelpad=20
ax.yaxis.labelpad=20
ax.zaxis.labelpad=20
X, Y = np.meshgrid(latitudes, luminosities)
surf = ax.plot_surface(X, Y, A_b_steady_lat * 100, cmap=cm.jet, linewidth=0.1, alpha=0.8)
col = fig.colorbar(surf, shrink=0.5, aspect=5)
col.mappable.set_clim(0, 60)
cset = ax.contour(X, Y, np.tile([land_frac[key] * 100 for key in land_frac.keys()], (41,1)),
                  zdir='y', offset=3, cmap=cm.coolwarm, clabel = "Fertile ground (%)")
ax.clabel(cset, "Fertile ground (%)")
ax.set_xlabel("latitude", fontweight='bold')
ax.set_ylabel("luminosity", fontweight='bold')
ax.set_zlabel("area (%)", fontweight='bold')
ax.set_yticks(np.arange(1, 3.5, 0.5))
ax.legend()
plt.show()

#%%
'''
-------------------------FIG 10 REPORT (SPLIT IN 2)----------------------------
'''
lon_pos = {"-90" : [-85,85,-85,85,-85,85],
               "-85" : [-85,85,-85,85,-85,85],
               "-80" : [-85,85,-85,85,-85,85],
               "-75" : [-10,85,-10,85,-10,85],
               "-70" : [-10,85,-10,85,-10,85],
               "-65" : [-65,-62,50,55,50,55],
               "-60" : [-100]*6,
               "-55" : [-100]*6,
               "-50" : [-74,-69]*3,
               "-45" : [-72,-67]*3,
               "-40" : [-70,-66]*3,
               "-35" : [-71,-59]*3,
               "-30" : [-70,-53,18,28,18,28],
               "-25" : [-69,-50,16,31,16,31],
               "-20" : [-69,-42,15,33,45,48],
               "-15" : [-70,-40,14,37,47,49],
               "-10" : [-77,-35,13,40,13,40],
               "-5" : [-80,-35,12,39,12,39],
               "0" : [-80,-50,10,42,10,42],
               "5" : [-75,-57,-10,48,-10,48],
               "10" : [-75,-62,-15,50,-15,43],
               "15" : [-15,37,43,48,73,81],
               "20" : [-16,34,41,56,73,86],
               "25" : [-13,33,38,48,62,90],
               "30" : [-9,13,34,90,34,90],
               "35" : [0,10,37,90,37,90],
               "40" : [-90,-76,-8,0,28,90],
               "45" : [-90,-71,0,27,55,90],
               "50" : [-90,-68,1,90,1,90],
               "55" : [-90,-64,-3,-1,22,90],
               "60" : [-90,-67,5,17,23,90],
               "65" : [-90,-69,-52,-41,14,90],
               "70" : [-90,-69,-50,-27,70,90],
               "75" : [-55,-23]*3,
               "80" : [-64,-23]*3,
               "85" : [-38,-30]*3,
               "90" : [-100]*6} # land locations on each latitude

plot_daisy_nr = 50

idx_lum = 6

fig = plt.figure(figsize=(24, 12))
ax0 = fig.add_subplot(231)
ax1 = fig.add_subplot(232)
ax2 = fig.add_subplot(233)

ax0.plot(A_w_steady_lat[idx_lum] * 100, latitudes, color = 'white', label = "white")
ax0.plot(A_b_steady_lat[idx_lum] * 100, latitudes, color = 'black', label = "black")
ax0.invert_xaxis()
ax0.set_ylabel("latitude")
ax0.set_xlabel("Daisy area (% of total)")
ax0.set_xlim([0,70])
ax0.set_yticks(np.arange(-90,91,45))
ax0.legend(frameon = True, facecolor='silver', loc='center right')
ax0.set_facecolor('darkgrey')
ax0.grid(color = 'grey')

map = Basemap(projection='ortho', lat_0=-0, lon_0=-0, resolution='c', ax=ax1)
map.drawlsmask(land_color='sandybrown', ocean_color='blue')

#create and draw meridians and parallels grid lines
parallels = np.arange( -90., 100.,10.)
map.drawparallels(parallels, labels=[1,0,0,0], fontsize=10)

# plot daisies as points in map
for idx, lat in enumerate(latitudes):
    plot_white_nr = int(plot_daisy_nr * A_w_steady_lat[idx_lum][idx])
    plot_black_nr = int(plot_daisy_nr * A_b_steady_lat[idx_lum][idx])
    
    plot_list = ["white"] * plot_white_nr + ["black"] * plot_black_nr
    
    for daisy in plot_list:
        items = lon_pos[str(lat)]
        x, y = map(choice([randint(items[0],items[1]), randint(items[2],items[3]), randint(items[4],items[5])]),
                          lat + np.random.normal(0, 0.5))
        map.plot(x, y, 'X', markersize=6, color = daisy)

ax1.set_title("(a)\nL = {0}".format(round(luminosities[idx_lum],1)))
white_legend = ax1.plot(np.nan, np.nan, 'X', color = "white", label = "white daisy")
black_legend = ax1.plot(np.nan, np.nan, 'X', color = "black", label = "black daisy")
ax1.legend(frameon = True, facecolor='silver', markerscale=1.2, bbox_to_anchor=(0.75, 0.1),
           fontsize=16)

ax2.plot(Temp_trans[idx_lum], latitudes, color = 'blue', label = "with daisies")
ax2.plot(Temp_nodaisiestrans[idx_lum], latitudes, color = 'red', label = "without daisies")
ax2.set_ylabel("latitude")
ax2.set_xlabel("Temperature (deg C)")
ax2.set_yticks(np.arange(-90,91,45))
ax2.yaxis.set_ticks_position("right")
ax2.yaxis.set_label_position("right")
ax2.legend(frameon = True, facecolor='silver', loc='center left')
ax2.set_facecolor('darkgrey')
ax2.grid(color = 'grey')

ax3 = fig.add_subplot(234)
ax4 = fig.add_subplot(235)
ax5 = fig.add_subplot(236)
idx_lum = 9

ax3.plot(A_w_steady_lat[idx_lum] * 100, latitudes, color = 'white', label = "white")
ax3.plot(A_b_steady_lat[idx_lum] * 100, latitudes, color = 'black', label = "black")
ax3.invert_xaxis()
ax3.set_ylabel("latitude")
ax3.set_xlabel("Daisy area (% of total)")
ax3.set_xlim([0,70])
ax3.set_yticks(np.arange(-90,91,45))
ax3.set_facecolor('darkgrey')
ax3.grid(color = 'grey')

map = Basemap(projection='ortho', lat_0=-0, lon_0=-0, resolution='c', ax=ax4)
map.drawlsmask(land_color='sandybrown', ocean_color='blue')

#create and draw meridians and parallels grid lines
parallels = np.arange( -90., 100.,10.)
map.drawparallels(parallels, labels=[1,0,0,0], fontsize=10)

# plot daisies as points in map
for idx, lat in enumerate(latitudes):
    plot_white_nr = int(plot_daisy_nr * A_w_steady_lat[idx_lum][idx])
    plot_black_nr = int(plot_daisy_nr * A_b_steady_lat[idx_lum][idx])
    
    plot_list = ["white"] * plot_white_nr + ["black"] * plot_black_nr
    
    for daisy in plot_list:
        items = lon_pos[str(lat)]
        x, y = map(choice([randint(items[0],items[1]), randint(items[2],items[3]), randint(items[4],items[5])]),
                          lat + np.random.normal(0, 0.5))
        map.plot(x, y, 'X', markersize=6, color = daisy)

ax4.set_title("(b)\nL = {0}".format(round(luminosities[idx_lum],2)))
white_legend = ax4.plot(np.nan, np.nan, 'X', color = "white", label = "white daisy")
black_legend = ax4.plot(np.nan, np.nan, 'X', color = "black", label = "black daisy")

ax5.plot(Temp_trans[idx_lum], latitudes, color = 'blue', label = "with daisies")
ax5.plot(Temp_nodaisiestrans[idx_lum], latitudes, color = 'red', label = "without daisies")
ax5.set_ylabel("latitude")
ax5.set_xlabel("Temperature (deg C)")
ax5.set_yticks(np.arange(-90,91,45))
ax5.yaxis.set_ticks_position("right")
ax5.yaxis.set_label_position("right")
ax5.set_facecolor('darkgrey')
ax5.grid(color = 'grey')

fig = plt.figure(figsize=(24, 12))
ax6 = fig.add_subplot(231)
ax7 = fig.add_subplot(232)
ax8 = fig.add_subplot(233)
idx_lum = 12

ax6.plot(A_w_steady_lat[idx_lum] * 100, latitudes, color = 'white', label = "white")
ax6.plot(A_b_steady_lat[idx_lum] * 100, latitudes, color = 'black', label = "black")
ax6.invert_xaxis()
ax6.set_ylabel("latitude")
ax6.set_xlabel("Daisy area (% of total)")
ax6.set_yticks(np.arange(-90,91,45))
ax6.set_xlim([0,70])
ax6.set_facecolor('darkgrey')
ax6.grid(color = 'grey')

map = Basemap(projection='ortho', lat_0=-0, lon_0=-0, resolution='c', ax=ax7)
map.drawlsmask(land_color='sandybrown', ocean_color='blue')

#create and draw meridians and parallels grid lines
parallels = np.arange( -90., 100.,10.)
map.drawparallels(parallels, labels=[1,0,0,0], fontsize=10)

# plot daisies as points in map
for idx, lat in enumerate(latitudes):
    plot_white_nr = int(plot_daisy_nr * A_w_steady_lat[idx_lum][idx])
    plot_black_nr = int(plot_daisy_nr * A_b_steady_lat[idx_lum][idx])
    
    plot_list = ["white"] * plot_white_nr + ["black"] * plot_black_nr
    
    for daisy in plot_list:
        items = lon_pos[str(lat)]
        x, y = map(choice([randint(items[0],items[1]), randint(items[2],items[3]), randint(items[4],items[5])]),
                          lat + np.random.normal(0, 0.5))
        map.plot(x, y, 'X', markersize=6, color = daisy)

ax7.set_title("(c)\nL = {0}".format(round(luminosities[idx_lum],1)))
white_legend = ax7.plot(np.nan, np.nan, 'X', color = "white", label = "white daisy")
black_legend = ax7.plot(np.nan, np.nan, 'X', color = "black", label = "black daisy")

ax8.plot(Temp_trans[idx_lum], latitudes, color = 'blue', label = "with daisies")
ax8.plot(Temp_nodaisiestrans[idx_lum], latitudes, color = 'red', label = "without daisies")
ax8.set_ylabel("latitude")
ax8.set_xlabel("Temperature (deg C)")
ax8.set_yticks(np.arange(-90,91,45))
ax8.yaxis.set_ticks_position("right")
ax8.yaxis.set_label_position("right")
ax8.set_facecolor('darkgrey')
ax8.grid(color = 'grey')

ax9 = fig.add_subplot(234)
ax10 = fig.add_subplot(235)
ax11 = fig.add_subplot(236)
idx_lum = 30

ax9.plot(A_w_steady_lat[idx_lum] * 100, latitudes, color = 'white', label = "white")
ax9.plot(A_b_steady_lat[idx_lum] * 100, latitudes, color = 'black', label = "black")
ax9.invert_xaxis()
ax9.set_ylabel("latitude")
ax9.set_xlabel("Daisy area (% of total)")
ax9.set_xlim([0,70])
ax9.set_yticks(np.arange(-90,91,45))
ax9.set_facecolor('darkgrey')
ax9.grid(color = 'grey')

map = Basemap(projection='ortho', lat_0=-0, lon_0=-0, resolution='c', ax=ax10)
map.drawlsmask(land_color='sandybrown', ocean_color='blue')

#create and draw meridians and parallels grid lines
parallels = np.arange( -90., 100.,10.)
map.drawparallels(parallels, labels=[1,0,0,0], fontsize=10)

# plot daisies as points in map
for idx, lat in enumerate(latitudes):
    plot_white_nr = int(plot_daisy_nr * A_w_steady_lat[idx_lum][idx])
    plot_black_nr = int(plot_daisy_nr * A_b_steady_lat[idx_lum][idx])
    
    plot_list = ["white"] * plot_white_nr + ["black"] * plot_black_nr
    
    for daisy in plot_list:
        items = lon_pos[str(lat)]
        x, y = map(choice([randint(items[0],items[1]), randint(items[2],items[3]), randint(items[4],items[5])]),
                          lat + np.random.normal(0, 0.5))
        map.plot(x, y, 'X', markersize=6, color = daisy)

ax10.set_title("(d)\nL = {0}".format(round(luminosities[idx_lum],2)))
white_legend = ax10.plot(np.nan, np.nan, 'X', color = "white", label = "white daisy")
black_legend = ax10.plot(np.nan, np.nan, 'X', color = "black", label = "black daisy")

ax11.plot(Temp_trans[idx_lum], latitudes, color = 'blue', label = "with daisies")
ax11.plot(Temp_nodaisiestrans[idx_lum], latitudes, color = 'red', label = "without daisies")
ax11.set_ylabel("latitude")
ax11.set_xlabel("Temperature (deg C)")
ax11.set_yticks(np.arange(-90,91,45))
ax11.yaxis.set_ticks_position("right")
ax11.yaxis.set_label_position("right")
ax11.set_facecolor('darkgrey')
ax11.grid(color = 'grey')