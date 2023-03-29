#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing standard packages
import os
import time
import argparse
from tqdm import trange

# Numerical packages
from scipy import stats
import numpy as np
from numba import jit

# import stuff from .py files in local folder
import sys
sys.path.append('.')
from particlefunctions import *
from logger import lagrangian_logger as logger

# Function to generate random speed distribution for microplastics
from case2_speed_generator import draw_random_speeds


###########################################
#### Main function to run a simulation ####
###########################################

def experiment_case2(Z0, V0, Np, Zmax, Tmax, dt, save_dt, K, randomstep, args = None):
    '''
    Run the model. 
    Returns the number of submerged particles, the histograms (concentration profile),
    the depth and diameters of the particles.

    Z0: initial depth of particles, positive downwards (m)
    V0: initial rising/settling speed of particles, positive downwards (m/s)
    Np: Maximum number of particles
    Tmax: total duration of the simulation (s)
    dt: timestep (s)
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

    # Use trange (progress bar) if instructed
    iterator = range
    if args is not None:
        if args.progress:
            iterator = trange

    # Time loop
    t = 0
    for n in iterator(Nt+1):
        # Store output once every N_skip steps
        if n % N_skip == 0:
            i = int(n / N_skip)
            Z_out[i,:len(Z)] = Z
        # Random displacement
        Z = randomstep(K, Z, t, dt)
        # Reflect from surface and sea floor
        Z = reflect(Z, zmax = Zmax)
        # Rise/sink due to buoyancy
        Z = advect(Z, V, dt)
        # Remove sinking particles that reached the sea floor
        Z, V = settle(Z, V, zmax = Zmax)
        # Ensure buoyant particles are not above the surface
        Z = np.maximum(0.0, Z)
        # Increment time
        t = dt*n

    # Store output also after final step
    Z_out[-1,:len(Z)] = Z

    return Z_out


##############################
#### Numerical parameters ####
##############################

parser = argparse.ArgumentParser()
parser.add_argument('--dt', dest = 'dt', type = float, default = 10, help = 'Timestep')
parser.add_argument('--save_dt', dest = 'save_dt', type = int, default = 3600, help = 'Interval at which to save results')
parser.add_argument('--Np', dest = 'Np', type = int, default = 100000, help = 'Number of particles')
parser.add_argument('--run_id', dest = 'run_id', type = int, default = 0, help = 'Run ID (used to differentiate runs when saving')
parser.add_argument('--profile', dest = 'profile', type = str, default = 'A', choices = ['A', 'B'], help = 'Diffusivity profiles')
parser.add_argument('--checkpoint', dest = 'checkpoint', type = bool, default = False, help = 'Save results for checkpointing at every output timestep?')
parser.add_argument('--progress', dest = 'progress', action = 'store_true', help = 'Display progress bar?')
parser.add_argument('--overwrite', dest = 'overwrite', action = 'store_true', help = 'Overwrite existing file?')
parser.add_argument('-v', '--verbose', dest = 'verbose', action = 'store_true', help = 'Produce lots of status updates?')
parser.add_argument('--statusfile', dest = 'statusfilename', default = None, help = 'Filename to write log messages to')
args = parser.parse_args()



# Open file for writing statusmessages if required
if args.statusfilename is not None:
    args.statusfile = open(args.statusfilename, 'w')

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


############################
#### Initial conditions ####
############################

# Rising/settling speeds, generated randomly
V0 = draw_random_speeds(args.Np)

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


##############################
#### Diffusivity profiles ####
##############################

# Constant diffusivity
K0  = 1e-2
K_A = lambda z, t: K0 * np.ones_like(z)

# Analytical function which has been fitted to
# simulation results from the GOTM turbulence model
# with a wind stress corresponding to wind speed of about 9 m/s.
a, b, c, z0 = (0.00636, 0.088, 1.54, 1.3)
K_B = lambda z, t: a*(z+z0)*np.exp(-(b*(z+z0))**c)


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
datafolder = '/media/torn/SSD/EulerLagrange/'
#datafolder = '/work6/torn/EulerLagrange/'
outputfilename_Z = os.path.join(datafolder, f'Case2_K_{label}_lagrangian_Nparticles={args.Np}_dt={args.dt}_save_dt={args.save_dt}_Z_{args.run_id:04}.npy')


print(outputfilename_Z, os.path.exists(outputfilename_Z), args.overwrite)
if (not os.path.exists(outputfilename_Z)) or args.overwrite:
    tic = time.time()
    Z_out = experiment_case2(Z0, V0, args.Np, Zmax, Tmax, args.dt, args.save_dt, K, correctstep, args)
    toc = time.time()
    logger(f'Simulation took {toc - tic:.1f} seconds, Case 2, Np = {args.Np}, dt = {args.dt}, run = {args.run_id}, profile = {label}', args, error=True)
    #np.savez_compressed(outputfilename_Z, Z=Z_out)
    np.save(outputfilename_Z, Z_out)
else:
    logger(f'File exists, skipping: {outputfilename_Z}', args, error = True)

