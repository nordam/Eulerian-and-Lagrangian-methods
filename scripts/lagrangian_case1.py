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

# import stuff from .py files in local folder
import sys
sys.path.append('src')
from particlefunctions import *
from logger import lagrangian_logger as logger


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
    N_out = int(Nt / N_skip) + 1 # Add 1 to store initial state
    # Array to store output
    Z_out = np.zeros((N_out, Np)) - 999

    # Time loop
    t = 0
    for n in range(Nt+1):
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
parser.add_argument('--save_dt', dest = 'save_dt', type = int, default = 3600, help = 'Interval at which to save results')
parser.add_argument('--Np', dest = 'Np', type = int, default = 10000, help = 'Number of particles')
parser.add_argument('--run_id', dest = 'run_id', type = int, default = 0, help = 'Run ID (used to differentiate runs when saving')
parser.add_argument('--profile', dest = 'profile', type = str, default = 'B', choices = ['A', 'B'], help = 'Diffusivity profiles')
parser.add_argument('--progress', dest = 'progress', action = 'store_true', help = 'Display progress bar?')
parser.add_argument('--overwrite', dest = 'overwrite', action = 'store_true', help = 'Overwrite existing file?')
parser.add_argument('-v', '--verbose', dest = 'verbose', action = 'store_true', help = 'Produce lots of status updates?')
parser.add_argument('--statusfile', dest = 'statusfilename', default = None, help = 'Filename to write log messages to')
args = parser.parse_args()

# Consistency check of arguments

if (args.save_dt / args.dt) != int(args.save_dt / args.dt):
    print('dt does not evenly divide save_dt, output times will not be as expected')
    sys.exit()


########################################
#### Scenario parameters for Case 1 ####
########################################

# Total depth
Zmax = 50
# Simulation time
Tmax = 12*3600


############################
#### Initial conditions ####
############################

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

# Constant diffusivity (not used in paper)
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

resultsfolder = '../results/'
outputfilename_Z = os.path.join(resultsfolder, f'Case1_K_{label}_lagrangian_Nparticles={args.Np}_dt={args.dt}_save_dt={args.save_dt}_Z_{args.run_id:04}.npy')

if (not os.path.exists(outputfilename_Z)) or args.overwrite:
    tic = time.time()
    Z_out = experiment_case1(Z0, V0, args.Np, Tmax, args.dt, args.save_dt, K, correctstep)
    toc = time.time()
    np.save(outputfilename_Z, Z_out)
    logger(f'Simulation took {toc - tic:.1f} seconds, output written to {outputfilename_Z}', args, error = True)
else:
    logger(f'File exists, skipping: {outputfilename_Z}', args, error = True)

if args.statusfilename is not None:
    args.statusfile.close()
