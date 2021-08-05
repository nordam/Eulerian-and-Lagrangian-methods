#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing standard packages
import os
import time
import argparse
#from tqdm import trange

# Numerical packages
from scipy import stats
import numpy as np
from numba import jit

# import stuff from .py files in local folder
import sys
sys.path.append('.')
from particlefunctions import *


###########################################
#### Main function to run a simulation ####
###########################################

def experiment_case1(Z0, V0, Np, Tmax, dt, save_dt, K, randomstep):
    '''
    Run the model. 
    Returns the number of submerged particles, the histograms (concentration profile),
    the depth and diameters of the particles.
    
    Z0: initial depth of particles, positive downwards (m)
    V0: initial rising/settling speed of particles, positive downwards (m/s)
    Np: Maximum number of particles
    Tmax: total duration of the simulation (s)
    dt: timestep (s)
    bins: bins for histograms (concentration profiles)
    K: diffusivity-function on form K(z, t)
    randomstep: random walk scheme
    '''
    # Number of timesteps
    Nt = int(Tmax / dt)
    # Arrays for z-position (depth) and droplet size
    Z  = Z0.copy()
    V  = V0.copy()
    # Calculate size of output arrays
    N_skip = int(save_dt/dt)
    N_out = 1 + int(Nt / N_skip)
    # Array to store output
    Z_out = np.zeros((N_out, Np)) - 999

    # Time loop
    t = 0
    for n in range(Nt):
        # Store output once every N_skip steps
        if n % N_skip == 0:
            i = int(n / N_skip)
            Z_out[i,:len(Z)] = Z
        # Random displacement
        Z = randomstep(K, Z, t, dt)
        # Reflect from surface
        Z = reflect(Z)
        # Rise due to buoyancy
        Z = advect(Z, V, dt)
        # Ensure fish eggs are not above the surface
        Z = np.maximum(0.0, Z)
        # Increment time
        t = dt*i
    return Z_out


####################################
####   Command line arguments   ####
####################################

parser = argparse.ArgumentParser()
parser.add_argument('--dt', dest = 'dt', type = int, default = 30, help = 'Timestep')
parser.add_argument('--save_dt', dest = 'save_dt', type = int, default = 1800, help = 'Interval at which to save results')
parser.add_argument('--Np', dest = 'Np', type = int, default = 10000, help = 'Number of particles')
parser.add_argument('--run_id', dest = 'run_id', type = int, default = 0, help = 'Run ID (used to differentiate runs when saving')
parser.add_argument('--profile', dest = 'profile', type = str, default = 'A', choices = ['A', 'B'], help = 'Diffusivity profiles')
#parser.add_argument('--checkpoint', dest = 'checkpoint', type = bool, default = False, help = 'Save results for checkpointing at every output timestep?')
args = parser.parse_args()

# Consistency check of arguments

if (args.save_dt / args.dt) != int(args.save_dt / args.dt):
    print('dt does not evenly divide save_dt, output times will not be as expected')
    sys.exit()


########################################
#### Scenario parameters for Case 1 ####
########################################

#### Hard-coded parameters for this case ####
# Total depth
Zmax = 50
# Simulation time
Tmax = 12*3600

# For this case, we use a speed distribution directly, taken from
# Table 3 in Sundby (1983).
# Mean speed = 0.96 mm/s
# Standard deviation = 0.38 mm/s
# Truncated at +/- 2*sigma
mean_speed = -0.96 * 1e-3
std_dev_speed = 0.38 * 1e-3
Vmin = mean_speed - 2*std_dev_speed
Vmax = mean_speed + 2*std_dev_speed
speed_distribution = stats.norm(loc = mean_speed, scale = std_dev_speed)
V0 = speed_distribution.rvs(size = args.Np)
# Re-draw until all samples are within bounds
mask = (V0 < Vmin) | (Vmax < V0)
while np.any(mask):
    V0[mask] = speed_distribution.rvs(size = sum(mask))
    mask = (V0 < Vmin) | (Vmax < V0)

# Initial condition:
# Normal distribution with mean mu and standard deviation sigma
sigma_IC = 4
mu_IC = 20
position_distribution = stats.norm(loc = mu_IC, scale = sigma_IC)
Z0 = position_distribution.rvs(size = args.Np)
# Re-draw until all samples are within bounds
mask = (Z0 < 0.0) | (Zmax < Z0)
while np.any(mask):
    Z0[mask] = position_distribution.rvs(size = sum(mask))
    mask = (Z0 < 0.0) | (Zmax < Z0)



##################################
####   Diffusivity profiles   ####
##################################

# Constant diffusivity
K_A = lambda z, t: 1e-2*np.ones(len(z))

# Fitted to results of GOTM simulation
alpha, beta, zeta, z0 = (0.00636, 0.088, 1.54, 1.3)
K_B = lambda z, t: alpha*(z+z0)*np.exp(-(beta*(z+z0))**zeta)


#########################################################
####    Run simulation for one diffusivity profile   ####
#########################################################

if args.profile == 'A':
    K = K_A
    label = 'A'
else:
    K = K_B
    label = 'B'

datafolder = '../results/'
datafolder = '/work6/torn/EulerLagrange/'

tic = time.time()
Z_out = experiment_case1(Z0, V0, args.Np, Tmax, args.dt, args.save_dt, K, correctstep)
toc = time.time()
print(f'Simulation took {toc - tic:.1f} seconds, Np = {args.Np}, dt = {args.dt}, run = {args.run_id}')

outputfilename = os.path.join(datafolder, f'Case1_K_{label}_lagrangian_Nparticles={args.Np}_dt={args.dt}_Z_{args.run_id:04}.npy')
np.save(outputfilename, Z_out)

