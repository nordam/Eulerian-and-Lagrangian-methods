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

# import stuff from .py files in local folder
import sys
sys.path.append('src')
from particlefunctions import *
from wavefunctions import jonswap
from webernaturaldispersion import weber_natural_dispersion
from logger import lagrangian_logger as logger


###########################################
#### Main function to run a simulation ####
###########################################

def experiment_case3(Z0, D0, Np, Tmax, dt, save_dt, K, windspeed, h0, mu, ift, rho, randomstep, surfacing = True, entrainment = True, args = None):
    '''
    Run the model. 
    Returns the number of submerged particles, the histograms (concentration profile),
    the depth and diameters of the particles.

    Z0: initial depth of particles (m)
    D0: initial diameter of particles (m)
    Np: Maximum number of particles
    Tmax: total duration of the simulation (s)
    dt: timestep (s)
    save_dt: time interval between saves (s)
    K: diffusivity-function on form K(z, t)
    windspeed: windspeed (m/s)
    h0: initial oil film thickness (m)
    mu: dynamic viscosity of oil (kg/m/s)
    ift: interfacial tension (N/m)
    rho: oil density (kg/m**3)
    randomstep: random walk scheme
    '''
    # Number of timesteps
    Nt = int(Tmax / dt)
    # Calculate size of output arrays
    N_skip = int(save_dt/dt)
    N_out = int(Nt / N_skip) + 1 # Add 1 to store initial state
    # Array to store output
    Z_out = np.zeros((N_out, Np)) - 999
    V_out = np.zeros((N_out, Np)) - 999

    # Arrays for z-position (depth) and droplet size
    Z = Z0.copy()
    D = D0.copy()
    V = rise_speed(D, rho)

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
            V_out[i,:len(V)] = V

        # Random displacement
        Z = randomstep(K, Z, t, dt)
        # Reflect from surface
        Z = reflect(Z, zmax = 50)
        # Rise due to buoyancy
        Z = advect(Z, V, dt)

        if surfacing:
            # Remove surfaced (applies to depth, size and velocity arrays)
            Z, D, V = surface(Z, D, V)
        else:
            # Ensure droplets are not above the surface
            Z = np.maximum(0.0, Z)

        if entrainment:
            # Calculate oil film thickness
            h = h0 * (Np - len(Z)) / Np
            # Entrain
            Z, D, V = entrain(Z, D, V, Np, dt, windspeed, h, mu, ift, rho)

        # Increment time
        t = dt*n

    return Z_out, V_out

##############################
#### Numerical parameters ####
##############################

parser = argparse.ArgumentParser()
parser.add_argument('--dt', dest = 'dt', type = int, default = 30, help = 'Timestep')
parser.add_argument('--save_dt', dest = 'save_dt', type = int, default = 3600, help = 'Interval at which to save results')
parser.add_argument('--Np', dest = 'Np', type = int, default = 100000, help = 'Number of particles')
parser.add_argument('--run_id', dest = 'run_id', type = int, default = 0, help = 'Run ID (used to differentiate runs when saving')
parser.add_argument('--profile', dest = 'profile', type = str, default = 'B', choices = ['A', 'B'], help = 'Diffusivity profiles')
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
    logger('dt does not evenly divide save_dt, output times will not be as expected', args, error = True)
    sys.exit()


########################################
#### Scenario parameters for Case 3 ####
########################################

# Total depth
Zmax = 50
# Simulation time
Tmax = 12*3600

# Oil parameters
## Dynamic viscosity of oil (kg/m/s)
mu     = 1.51
## Interfacial tension (N/m)
ift    = 0.013
## Oil density (kg/m**3)
rho    = 992
## Intial oil film thickness (m)
h0     = 3e-3

# Environmental parameters
## Windspeed (m/s)
windspeed = 9
# Significant wave height and peak wave period
Hs, Tp = jonswap(windspeed, fetch = 143233)
print('Hs = ', Hs)
sys.exit()


############################
#### Initial conditions ####
############################


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


# Size distribution parameters
sigma = 0.4 * np.log(10)
D50n  = weber_natural_dispersion(rho, mu, ift, Hs, h0)
D50v  = np.exp(np.log(D50n) + 3*sigma**2)
D0  = np.random.lognormal(mean = np.log(D50v), sigma = sigma, size = args.Np)


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

resultsfolder = '../results/'
outputfilename_Z = os.path.join(resultsfolder, f'Case3_K_{label}_lagrangian_Nparticles={args.Np}_dt={args.dt}_save_dt={args.save_dt}_Z_{args.run_id:04}.npy')

if (not os.path.exists(outputfilename_Z)) or args.overwrite:
    tic = time.time()
    Z_out, V_out = experiment_case3(Z0, D0, args.Np, Tmax, args.dt, args.save_dt, K, windspeed, h0, mu, ift, rho, correctstep, surfacing = True, entrainment = True, args = args)
    toc = time.time()
    np.save(outputfilename_Z, Z_out)
    logger(f'Simulation took {toc - tic:.1f} seconds, output written to {outputfilename_Z}', args, error = True)

else:
    logger(f'File exists, skipping: {outputfilename_Z}', args, error = True)

if args.statusfilename is not None:
    args.statusfile.close()
