
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
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import random
from matplotlib import gridspec
import os
os.environ['PROJ_LIB'] = r"C:\Users\fafri\anaconda3\pkgs\proj4-6.1.1-hc2d0af5_1\Library\share"
from mpl_toolkits.basemap import Basemap

'''
---------------------------plotting preferences--------------------------------
'''
plt.style.use('seaborn-darkgrid')
plt.rc('text', usetex=False)
plt.rc('font', family='times')
plt.rc('xtick', labelsize=20) 
plt.rc('ytick', labelsize=18) 
plt.rc('font', size=20) 
plt.rc('figure', figsize = (12, 5))

'''
-------------------------------PARAMS------------------------------------------
'''
latitudes = np.arange(-90, 91, 5) # latitudes
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
t_init = 0 # initial time
t_end = 10 # end time of simulation in seconds
maxstep = 100 # maximum nr of steps
time = np.linspace(t_init, t_end, maxstep + 1) # time array
dt = (t_end - t_init) / maxstep
daisy_setting = "white & black"

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
        return max(p - self.A_w  - self.A_b, 0)
        
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
        if abs(self.phi) >= np.radians(0) and abs(self.phi) < np.radians(60):
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
        if self.phi or self.phi == 0:
            return self.sol_irradiance() * self.L * (self.alpha_p() - alpha_i) / (b + beta) + self.avg_T_lat()[1]
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

    def steady_state_sol(self, include_daisy):
        A_w = np.zeros((len(time),)) # area white daisies
        A_b = np.zeros((len(time),)) # area black daisies

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
--------------------------VARYING LUMINOSITY-----------------------------------
'''

#luminosities = np.concatenate([np.arange(0.9, 2, 0.1), np.arange(1.9, 0.8, -0.1)])
luminosities = np.arange(1, 2.85, 0.05)

chosen_lat = 0

A_w_steady_lat = np.zeros((len(luminosities), len(latitudes)))
A_b_steady_lat = np.zeros((len(luminosities), len(latitudes)))
A_g_steady_lat = np.zeros((len(luminosities), len(latitudes)))
Temp_notrans = np.zeros((len(luminosities), len(latitudes)))
Temp_nodaisiestrans = np.zeros((len(luminosities), len(latitudes)))
Temp_nodaisiesnotrans = np.zeros((len(luminosities), len(latitudes)))
Temp_trans = np.zeros((len(luminosities), len(latitudes)))
growth_white = np.zeros((len(luminosities), len(latitudes)))
growth_black = np.zeros((len(luminosities), len(latitudes)))

for idx, L in enumerate(luminosities):
    print("computing steady state solution for luminosity #{0} out of {1}.".format(idx + 1, len(luminosities)))
    for idy, lat in enumerate(latitudes):
        print("computing steady state solution for latitude #{0} out of {1}.".format(idy + 1, len(latitudes)))
        
        if idx == 0:
            A_w_init = 0.01 # start with half of the available area white daisies
            A_b_init = 0.01 # start with half of the available area black daisies
        else:
            A_w_init = A_w_steady_lat[idx - 1, idy] # start with steady state white daisies
            A_b_init = A_b_steady_lat[idx - 1, idy] # start with steady state black daisies
            
        [A_w_steady_lat[idx, idy], A_b_steady_lat[idx, idy]] = Daisies(A_w_init, A_b_init, L, lat).steady_state_sol(include_daisy = daisy_setting)
        # steady state sols
        Daisy_steady = Daisies(A_w_steady_lat[idx, idy], A_b_steady_lat[idx, idy], L, lat)
        [Temp_notrans[idx, idy], Temp_trans[idx, idy]] = Daisy_steady.avg_T_lat()
        [growth_white[idx, idy], growth_black[idx, idy]] = Daisy_steady.growth_rate(daisytype = "white"), Daisy_steady.growth_rate(daisytype = "black")
        A_g_steady_lat[idx, idy] = Daisy_steady.A_g()
        # temperature without daisies
        [Temp_nodaisiesnotrans[idx, idy], Temp_nodaisiestrans[idx, idy]] = Daisies(0, 0, L, lat).avg_T_lat()

#%%
plot_daisy_nr = 50

idx_lum = 6

fig = plt.figure(figsize=(24, 12))
#ax = fig.add_subplot(122)
#gs = gridspec.GridSpec(1, 3, width_ratios=[1, 2, 1]) 
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
ax0.legend(frameon = True, facecolor='silver', loc='upper right')
ax0.set_facecolor('darkgrey')
ax0.grid(color = 'grey')

map = Basemap(projection='ortho', lat_0=-0, lon_0=-0, resolution='c', ax=ax1)
map.drawlsmask(land_color='sandybrown', ocean_color='sandybrown')

#create and draw meridians and parallels grid lines
parallels = np.arange( -90., 100.,10.)
map.drawparallels(parallels, labels=[1,0,0,0], fontsize=10)

# plot daisies as points in map
for idx, lat in enumerate(latitudes):
    plot_white_nr = int(plot_daisy_nr * A_w_steady_lat[idx_lum][idx])
    plot_black_nr = int(plot_daisy_nr * A_b_steady_lat[idx_lum][idx])
    
    plot_list = ["white"] * plot_white_nr + ["black"] * plot_black_nr
    
    for daisy in plot_list:
            x, y = map(random.randint(-85,85), lat + np.random.normal(0, 1))
            map.plot(x, y, 'X', markersize=8, color = daisy)

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
ax2.legend(frameon = True, facecolor='silver', loc='upper right')
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
#ax3.legend(frameon = True, facecolor='silver', loc='upper left')
ax3.set_facecolor('darkgrey')
ax3.grid(color = 'grey')

map = Basemap(projection='ortho', lat_0=-0, lon_0=-0, resolution='c', ax=ax4)
map.drawlsmask(land_color='sandybrown', ocean_color='sandybrown')

#create and draw meridians and parallels grid lines
parallels = np.arange( -90., 100.,10.)
map.drawparallels(parallels, labels=[1,0,0,0], fontsize=10)

# plot daisies as points in map
for idx, lat in enumerate(latitudes):
    plot_white_nr = int(plot_daisy_nr * A_w_steady_lat[idx_lum][idx])
    plot_black_nr = int(plot_daisy_nr * A_b_steady_lat[idx_lum][idx])
    
    plot_list = ["white"] * plot_white_nr + ["black"] * plot_black_nr
    
    for daisy in plot_list:
            x, y = map(random.randint(-85,85), lat + np.random.normal(0, 1))
            map.plot(x, y, 'X', markersize=8, color = daisy)

ax4.set_title("(b)\nL = {0}".format(round(luminosities[idx_lum],2)))
white_legend = ax4.plot(np.nan, np.nan, 'X', color = "white", label = "white daisy")
black_legend = ax4.plot(np.nan, np.nan, 'X', color = "black", label = "black daisy")
#ax4.legend(frameon = True, facecolor='silver', markerscale=1.5, bbox_to_anchor=(0.8, 0.1))

ax5.plot(Temp_trans[idx_lum], latitudes, color = 'blue', label = "with daisies")
ax5.plot(Temp_nodaisiestrans[idx_lum], latitudes, color = 'red', label = "without daisies")
ax5.set_ylabel("latitude")
ax5.set_xlabel("Temperature (deg C)")
ax5.set_yticks(np.arange(-90,91,45))
ax5.yaxis.set_ticks_position("right")
ax5.yaxis.set_label_position("right")
#ax5.legend(frameon = True, facecolor='silver', loc='center left')
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
#ax6.legend(frameon = True, facecolor='silver', loc='upper left')
ax6.set_facecolor('darkgrey')
ax6.grid(color = 'grey')

map = Basemap(projection='ortho', lat_0=-0, lon_0=-0, resolution='c', ax=ax7)
map.drawlsmask(land_color='sandybrown', ocean_color='sandybrown')

#create and draw meridians and parallels grid lines
parallels = np.arange( -90., 100.,10.)
map.drawparallels(parallels, labels=[1,0,0,0], fontsize=10)

# plot daisies as points in map
for idx, lat in enumerate(latitudes):
    plot_white_nr = int(plot_daisy_nr * A_w_steady_lat[idx_lum][idx])
    plot_black_nr = int(plot_daisy_nr * A_b_steady_lat[idx_lum][idx])
    
    plot_list = ["white"] * plot_white_nr + ["black"] * plot_black_nr
    
    for daisy in plot_list:
            x, y = map(random.randint(-85,85), lat + np.random.normal(0, 1))
            map.plot(x, y, 'X', markersize=8, color = daisy)

ax7.set_title("(c)\nL = {0}".format(round(luminosities[idx_lum],1)))
white_legend = ax7.plot(np.nan, np.nan, 'X', color = "white", label = "white daisy")
black_legend = ax7.plot(np.nan, np.nan, 'X', color = "black", label = "black daisy")
#ax7.legend(frameon = True, facecolor='silver', markerscale=1.5, bbox_to_anchor=(0.8, 0.1))

ax8.plot(Temp_trans[idx_lum], latitudes, color = 'blue', label = "with daisies")
ax8.plot(Temp_nodaisiestrans[idx_lum], latitudes, color = 'red', label = "without daisies")
ax8.set_ylabel("latitude")
ax8.set_xlabel("Temperature (deg C)")
ax8.set_yticks(np.arange(-90,91,45))
ax8.yaxis.set_ticks_position("right")
ax8.yaxis.set_label_position("right")
#ax8.legend(frameon = True, facecolor='silver', loc='center left')
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
#ax9.legend(frameon = True, facecolor='silver', loc='upper left')
ax9.set_facecolor('darkgrey')
ax9.grid(color = 'grey')

map = Basemap(projection='ortho', lat_0=-0, lon_0=-0, resolution='c', ax=ax10)
map.drawlsmask(land_color='sandybrown', ocean_color='sandybrown')

#create and draw meridians and parallels grid lines
parallels = np.arange( -90., 100.,10.)
map.drawparallels(parallels, labels=[1,0,0,0], fontsize=10)

# plot daisies as points in map
for idx, lat in enumerate(latitudes):
    plot_white_nr = int(plot_daisy_nr * A_w_steady_lat[idx_lum][idx])
    plot_black_nr = int(plot_daisy_nr * A_b_steady_lat[idx_lum][idx])
    
    plot_list = ["white"] * plot_white_nr + ["black"] * plot_black_nr
    
    for daisy in plot_list:
            x, y = map(random.randint(-85,85), lat + np.random.normal(0, 1))
            map.plot(x, y, 'X', markersize=8, color = daisy)

ax10.set_title("(d)\nL = {0}".format(round(luminosities[idx_lum],2)))
white_legend = ax10.plot(np.nan, np.nan, 'X', color = "white", label = "white daisy")
black_legend = ax10.plot(np.nan, np.nan, 'X', color = "black", label = "black daisy")
#ax10.legend(frameon = True, facecolor='silver', markerscale=1.5, bbox_to_anchor=(0.8, 0.1))

ax11.plot(Temp_trans[idx_lum], latitudes, color = 'blue', label = "with daisies")
ax11.plot(Temp_nodaisiestrans[idx_lum], latitudes, color = 'red', label = "without daisies")
ax11.set_ylabel("latitude")
ax11.set_xlabel("Temperature (deg C)")
ax11.set_yticks(np.arange(-90,91,45))
ax11.yaxis.set_ticks_position("right")
ax11.yaxis.set_label_position("right")
#ax11.legend(frameon = True, facecolor='silver', loc='center left')
ax11.set_facecolor('darkgrey')
ax11.grid(color = 'grey')
#%%
X, Y = np.meshgrid(luminosities,latitudes)
fig, (axs) = plt.subplots(nrows = 4, ncols = 1, sharex = True)
#fig = plt.figure()
#ax1 = fig.add_subplot(211)

cs1 = axs[0].contourf(X, Y, A_w_steady_lat.T * 100, levels=np.arange(0,75,5),
                      cmap=cm.jet)
axs[0].set_ylabel('latitude')
axs[0].set_yticks(np.arange(-90,91,45))
axs[0].set_title("(a)", fontsize = 20)
#col = fig.colorbar(cs1, ax = axs[0])

cs2 = axs[1].contourf(X, Y, A_b_steady_lat.T * 100, levels=np.arange(0,75,5),
                      cmap=cm.jet)
axs[1].set_ylabel('latitude')
axs[1].set_yticks(np.arange(-90,91,45))
axs[1].set_title("(b)", fontsize = 20)
col = fig.colorbar(cs2, ax = axs[[0,1]], label = "area of total (%)")

cs = axs[2].contourf(X, Y, Temp_nodaisiestrans.T, levels=np.arange(-20,101,5),
                     extend = 'both', cmap=cm.jet)
axs[2].set_ylabel('latitude')
axs[2].set_yticks(np.arange(-90,91,45))
axs[2].set_title("(c)", fontsize = 20)
#cs.set_clim(-30, 120)
#axs[0].clabel(cs, inline=True, fontsize=8)
#col = fig.colorbar(cs, ax = axs[2])
#col.mappable.set_clim(0, 80)
#col.remove()

#ax1.set_xlabel('latitude')
#ax1.set_ylabel('latitude')
#ax2 = fig.add_subplot(212)
cs3 = axs[3].contourf(X, Y, Temp_trans.T, levels=np.arange(-20,101,5),
                      extend = 'both', cmap=cm.jet)
#axs[1].clabel(cs1, inline=True, fontsize=8)
#cs1.set_clim(-30, 120)
#col = fig.colorbar(cs, extend = 'both')
#col.mappable.set_clim(0, 80)
#col.remove()
#cbar_ax = fig.add_axes([1.05, 0.15, 0.05, 0.7])
#col = fig.colorbar(cs, ax=axs, label = "temperature$^{ab}$ (difference$^{c}$) ($\degree$C)")
#axs[1].set_xlabel('luminosity')
axs[3].set_xlabel('luminosity')
axs[3].set_ylabel('latitude')
axs[3].set_yticks(np.arange(-90,91,45))
axs[3].set_title("(d)", fontsize = 20)
col = fig.colorbar(cs3, ax = axs[[2,3]], label = "temperature ($\degree$C)")
plt.show()


#%%
fig = plt.figure()
plt.tight_layout()
ax = Axes3D(fig)
ax.xaxis.labelpad=20
ax.yaxis.labelpad=20
ax.zaxis.labelpad=20
X, Y = np.meshgrid(latitudes, luminosities[:10])
surf = ax.plot_surface(X, Y, Temp_trans[:][:10], cmap=cm.jet, linewidth=0.1, alpha=0.8)
col = fig.colorbar(surf, shrink=0.5, aspect=5)
#col.mappable.set_clim(-30, 200)
#col.remove()
#cset = ax.contour(X, Y, A_w_steady_lat, zdir='y', offset=2.9, cmap=cm.coolwarm)
ax.set_xlabel("latitude", fontweight='bold')
ax.set_ylabel("luminosity", fontweight='bold')
ax.set_zlabel("temperature (deg C)", fontweight='bold', color='r')
plt.title("Temp excluding meridional heat transport, no daisies")
#%%
fig = plt.figure()
plt.tight_layout()
ax = fig.add_subplot(121,projection='3d')
#ax = Axes3D(fig)
ax.xaxis.labelpad=20
ax.yaxis.labelpad=20
ax.zaxis.labelpad=20
X, Y = np.meshgrid(latitudes, luminosities)
surf = ax.plot_surface(X, Y, A_w_steady_lat * 100, cmap=cm.jet, linewidth=0.1, alpha=0.8)
col = fig.colorbar(surf, shrink=0.5, aspect=5)
col.mappable.set_clim(0, 70)
col.remove()
#cset = ax.contour(X, Y, A_w_steady_lat, zdir='y', offset=2.9, cmap=cm.coolwarm)
ax.set_xlabel("latitude", fontweight='bold')
ax.set_ylabel("luminosity", fontweight='bold')
ax.set_zlabel("area (%)", fontweight='bold', color='r')
plt.title("Area white daisies")

ax = fig.add_subplot(122,projection='3d')
#ax = Axes3D(fig)
ax.xaxis.labelpad=20
ax.yaxis.labelpad=20
ax.zaxis.labelpad=20
X, Y = np.meshgrid(latitudes, luminosities)
surf = ax.plot_surface(X, Y, A_b_steady_lat * 100, cmap=cm.jet, linewidth=0.1, alpha=0.8)
col = fig.colorbar(surf, shrink=0.5, aspect=5)
col.mappable.set_clim(0, 70)
#col.remove()
#cset = ax.contour(X, Y, A_w_steady_lat, zdir='y', offset=2.9, cmap=cm.coolwarm)
ax.set_xlabel("latitude", fontweight='bold')
ax.set_ylabel("luminosity", fontweight='bold')
ax.set_zlabel("area (%)", fontweight='bold', color='r')
plt.title("Area black daisies")
plt.show()

#%%
#from matplotlib import animation
for idx_lum in range(len(luminosities)):
    plot_daisy_nr = 50
    #idx_lum = 15
            
    fig = plt.figure(figsize=(24, 12)) 
    fig.suptitle("Daisy World", fontsize = 50)
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 2, 1]) 
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    ax2 = plt.subplot(gs[2])
    
    ax0.plot(A_w_steady_lat[idx_lum] * 100, latitudes, color = 'white', label = "white")
    ax0.plot(A_b_steady_lat[idx_lum] * 100, latitudes, color = 'black', label = "black")
    ax0.invert_xaxis()
    ax0.set_ylabel("latitude")
    ax0.set_xlabel("Daisy area (% of total)")
    ax0.set_xlim([0,70])
    ax0.legend(frameon = True, facecolor='silver', loc='upper left')
    ax0.set_facecolor('darkgrey')
    ax0.grid(color = 'grey')
    
    map = Basemap(projection='ortho', lat_0=-0, lon_0=-0, resolution='c', ax=ax1)
    map.drawlsmask(land_color='sandybrown', ocean_color='sandybrown')
    
    #create and draw meridians and parallels grid lines
    parallels = np.arange( -90., 100.,10.)
    map.drawparallels(parallels, labels=[1,0,0,0], fontsize=10)
    #for i in np.arange(len(parallels)):
    #    ax1.annotate(np.str(int(parallels[i])) + "$\degree$",xy=map(-90, int(parallels[i])),
    #                 size = 10, xycoords='data')
    
    # plot daisies as points in map
    for idx, lat in enumerate(latitudes):
        plot_white_nr = int(plot_daisy_nr * A_w_steady_lat[idx_lum][idx])
        plot_black_nr = int(plot_daisy_nr * A_b_steady_lat[idx_lum][idx])
        
        plot_list = ["white"] * plot_white_nr + ["black"] * plot_black_nr
        
        for daisy in plot_list:
            x, y = map(random.randint(-85,85), lat + np.random.normal(0, 1))
            map.plot(x, y, 'X', markersize=8, color = daisy)
    
    ax1.set_title("Luminosity = {0}".format(round(luminosities[idx_lum],1)), fontsize = 30)
    white_legend = ax1.plot(np.nan, np.nan, 'X', color = "white", label = "white daisy")
    black_legend = ax1.plot(np.nan, np.nan, 'X', color = "black", label = "black daisy")
    ax1.legend(frameon = True, facecolor='silver', markerscale=1.5, bbox_to_anchor=(0.8, 0.1), fontsize=12)
    
    ax2.plot(Temp_trans[idx_lum], latitudes, color = 'blue', label = "with daisies")
    ax2.plot(Temp_nodaisiestrans[idx_lum], latitudes, color = 'red', label = "without daisies")
    ax2.set_ylabel("latitude")
    ax2.set_xlabel("Temperature (deg C)")
    ax2.yaxis.set_ticks_position("right")
    ax2.yaxis.set_label_position("right")
    ax2.legend(frameon = True, facecolor='silver', loc='center left')
    ax2.set_facecolor('darkgrey')
    ax2.grid(color = 'grey')
    plt.savefig('figsmod2/fig{0}.png'.format(idx_lum))
#%%
'''
-------------------------------FIGURES-----------------------------------------
'''

# fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)

# ax1.set_facecolor('darkgrey')
# ax1.plot(luminosities[:int(len(luminosities) / 2)], area_white_steady[:int(len(luminosities) / 2)],\
#          color = 'white', label = 'White daisies (increasing L)')
# ax1.plot(luminosities[int(len(luminosities) / 2):], area_white_steady[int(len(luminosities) / 2):],\
#          color = 'white', linestyle = 'dashed', label = 'White daisies (decreasing L)')
# ax1.plot(luminosities[:int(len(luminosities) / 2)], area_total[:int(len(luminosities) / 2)],\
#          color = 'blue', label = 'All daisies (increasing L)')
# ax1.plot(luminosities[int(len(luminosities) / 2):], area_total[int(len(luminosities) / 2):],\
#          color = 'blue', linestyle =  'dashed', label = 'All daisies (decreasing L)')
# ax1.plot(luminosities[:int(len(luminosities) / 2)], area_black_steady[:int(len(luminosities) / 2)],\
#          color = 'black', label = 'Black daisies (increasing L)')
# ax1.plot(luminosities[int(len(luminosities) / 2):], area_black_steady[int(len(luminosities) / 2):],\
#         color = 'black', linestyle = 'dashed', label = 'Black daisies (decreasing L)')
# ax1.set_ylabel("Area (-)")
# ax1.legend()
# ax1.grid(color = 'grey')

# ax2.set_facecolor('darkgrey')
# ax2.plot(luminosities[:int(len(luminosities) / 2)], temperatures[:int(len(luminosities) / 2)],\
#          color = 'darkblue', label = "increasing L")
# ax2.plot(luminosities[int(len(luminosities) / 2):], temperatures[int(len(luminosities) / 2):],\
#          color = 'darkblue', linestyle = 'dashed', label = "decreasing L")
# ax2.legend()
# ax2.set_ylabel("Temperature (deg C)")
# ax2.set_xlabel("Solar luminosity")
# ax2.grid(color = 'grey')

# fig.suptitle("Run for {0} daisies, adjusting initial conditions".format(daisy_setting))

# plt.figure()
# ax = plt.gca()
# ax.set_facecolor('darkgrey')
# plt.plot(luminosities[:int(len(luminosities) / 2)], growth_white[:int(len(luminosities) / 2)],\
#          color = 'white', label = 'Growth rate white daisies (increasing L)')
# plt.plot(luminosities[int(len(luminosities) / 2):], growth_white[int(len(luminosities) / 2):],\
#          color = 'white', linestyle = 'dashed', label = 'Growth rate white daisies (decreasing L)')
# plt.plot(luminosities[:int(len(luminosities) / 2)], growth_black[:int(len(luminosities) / 2)],\
#          color = 'black', label = 'Growth rate black daisies (increasing L)')
# plt.plot(luminosities[int(len(luminosities) / 2):], growth_black[int(len(luminosities) / 2):],\
#         color = 'black', linestyle = 'dashed', label = 'Growth rate black daisies (decreasing L)')
# plt.axhline(gamma, label = "Death rate")
# plt.xlabel("Solar luminosity")
# plt.ylabel("Growth/death rate (-)")
# plt.title("Run for {0} daisies, adjusting initial conditions".format(daisy_setting))
# plt.legend()
# plt.grid(color = 'grey') 
