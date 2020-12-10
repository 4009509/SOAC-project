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
-------------------------------PARAMS------------------------------------------
'''

S_0 = 1366 # W/m^2
alpha_g = 0.25 # albedo ground
alpha_w = 0.75 # albbedo white daisies
alpha_b = 0.15 # albedo black daisies
gamma = 0.3 # death rate daisies per unit time
p = 1 # proportion of the planets area which is fertile ground
beta = 16 # what is this? (W m-2 K-1)
b = 2.2 # what is this? (W m-2 K-1)
I_0 = 220 # W m-2
L = 1 # Percentage of the current solar luminosity
'''
------------------------------FUNCTIONS----------------------------------------
'''

def alpha_p(A_g, A_w, A_b):
    return alpha_g * A_g + alpha_w * A_w + alpha_b * A_b

def avg_T_g(L, A_g, A_w, A_b):
    return (0.25 * S_0 * L * (1 - alpha_p(A_g, A_w, A_b)) - I_0) / b

def T_daisy(L, A_g, A_w, A_b, daisy_type):
    if daisy_type == "black":
        alpha_i = alpha_b
    elif daisy_type == "white":
        alpha_i = alpha_w
    else:
        print("daisy type not recognized")
        alpha_i = np.nan
    return 0.25 * S_0 * L * (alpha_p(A_g, A_w, A_b) - alpha_i) / (b + beta)
    
