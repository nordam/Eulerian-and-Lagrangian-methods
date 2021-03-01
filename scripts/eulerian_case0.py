#!/usr/bin/env python
# -*- coding: utf-8 -*-

#Importing standard packages
import time
import argparse
import logging

# Numerical packages
import numpy as np
from numba import jit

# import stuff from .py files in notebooks folder
import sys
sys.path.append('.')
from eulerian_functions import EulerianSystemParameters, Crank_Nicolson_FVM_TVD_advection_diffusion_reaction


####################################
####   Command line arguments   ####
####################################

parser = argparse.ArgumentParser()
parser.add_argument('--dt', dest = 'dt', type = int, default = 600, help = 'Timestep')
parser.add_argument('--NJ', dest = 'NJ', type = int, default = 1000, help = 'Number of grid cells')
parser.add_argument('--NK', dest = 'NK', type = int, default = 3, help = 'Number of speed classes')
args = parser.parse_args()


########################################
#### Scenario parameters for Case 1 ####
########################################

#### Hard-coded parameters for this case ####
# Total depth
Zmax = 50
# Simulation time
Tmax = 72*3600

# For this test, we use three speeds, one positive, one zero, one negative
mean_speed = 0 * 1e-3
std_dev_speed = 0.38 * 1e-3
Vmin = mean_speed - 2*std_dev_speed
Vmax = mean_speed + 2*std_dev_speed
speed_distribution = lambda v: np.exp(-0.5*((v - mean_speed)/std_dev_speed)**2) / (std_dev_speed*np.sqrt(2*np.pi))
speed_distribution = lambda v: np.ones_like(v) / (4*std_dev_speed)

# Initial condition:
# Normal distribution with mean mu and standard deviation sigma
sigma_IC = 4
mu_IC = 25
pdf_IC = lambda z: np.exp(-0.5*((z - mu_IC)/sigma_IC)**2) / (sigma_IC*np.sqrt(2*np.pi))


##################################
####   Diffusivity profiles   ####
##################################

# Constant diffusivity
K_A = lambda z: 1e-3*np.ones(len(z))

# Fitted to results of GOTM simulation
alpha, beta, zeta, z0 = (0.00636, 0.088, 1.54, 1.3)
K_B = lambda z: alpha*(z+z0)*np.exp(-(beta*(z+z0))**zeta)

K_B = lambda z: 1e-3 + 1e-2*np.sin(z*np.pi/Zmax)


####################################################
####   Populate object with system parameters   ####
####################################################

params = EulerianSystemParameters(
        Zmax = Zmax, # Max depth of water column
        Nz = args.NJ, # Number of cells in z-direction
        Tmax = Tmax, # Simulation time
        dt = args.dt, # timestep
        Vmin = Vmin, # Minimum speed
        Vmax = Vmax, # maximum speed
        Nclasses = args.NK, # Number of speed classes
        speed_distribution = speed_distribution, # speed density
    )


###########################################################
####    Run simulation for both diffusivity profiles   ####
###########################################################

for K, label in zip((K_A, K_B), ('A', 'B')):

    # Initial concentration array for all cells and time levels
    C0 = pdf_IC(params.z_cell)[:,None] * params.mass_fractions[None,:]

    start = time.time()
    c = Crank_Nicolson_FVM_TVD_advection_diffusion_reaction(C0, K, params)
    end = time.time()

    np.save(f'../data/Case0_K_{label}_block_Nclasses={params.Nclasses}_NJ={params.Nz}_dt={params.dt}.npy', c)

