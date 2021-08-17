#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing standard packages
import os
import time
import argparse
from tqdm import trange

# Numerical packages
from scipy import stats
from scipy.stats import lognorm
import numpy as np
from numba import jit

# import stuff from .py files in local folder
import sys
sys.path.append('.')
from particlefunctions import *
from wavefunctions import *
from coagulation_functions import *
from logger import logger

################################
#### Lagrangian coagulation ####
################################


@njit
def coagulate_dense(Z, D, dt, bins, Np):
    # For particles in the same bin, allow pairwise collisions
    # and coagulation with some probability
    # Scale with bin size to go from number to concentration
    bin_size = bins[1:] - bins[:-1]
    count = 0
    for n in range(len(bins)-1):
        # For each cell, apply coagulation to a fraction of particles
        # calculated from the rate p0 and the number of particle-pairs
        mask = (bins[n] <= Z) & (Z < bins[n+1])
        Np_cell = np.sum(mask)
        D_masked = D[mask].copy()
        reacted = np.zeros(Np_cell)
        # Skip rest if there are no particles in cell
        if Np_cell > 0:
            # Check all combinations of particles in this cell
            for i in range(Np_cell):
                if not reacted[i]:
                    for j in range(i):
                        # Probability of reacting
                        #p = 1 - np.exp(-dt*alpha*beta*scale_factor[mask][i]*scale_factor[mask][j]/bin_size[n])
                        p = dt*coagulation_rate_prefactor(D_masked[i]/2, D_masked[j]/2)/(bin_size[n]*Np)
                        # Uniform random number
                        r = np.random.random()
                        # If reaction occurs, calculate new particle size
                        if r < p:
                            D_masked[i], D_masked[j] = np.ones(2)*get_new_radius(D_masked[i], D_masked[j])
                            # Count recations
                            count += 1
                            reacted[i] = 1
                            reacted[j] = 1
        # Put updated values back into global array
        D[mask] = D_masked
    return Z, D

###########################################
#### Main function to run a simulation ####
###########################################

def experiment_case4(Z0, D0, Np, Zmax, Tmax, dt, save_dt, K, rho, randomstep, Nbins = 101, args = None):
    '''
    Run the model.

    Z0:    initial depth of particles [m]
    D0:    initial diameter of particles [m]
    Np:    Maximum number of particles
    Zmax:  Maximum depth [m]
    Tmax:  Total duration of the simulation [s]
    dt:    Timestep [s]
    save_dt:    Interval between storing output [s]
    K:     Diffusivity-function on form K(z, t) [m^2/s]
    randomstep: Random walk step function
    Nbins: Number of concentration bins (flocculation happens internally in each bin)
    '''

    # Number of timesteps
    Nt = int(Tmax / dt)
    # Calculate size of output arrays
    N_skip = int(save_dt/dt)
    N_out = 1 + int(Nt / N_skip)
    # Array to store output
    Z_out = np.zeros((N_out, Np)) - 999
    D_out = np.zeros((N_out, Np)) - 999
    # Bins for coagulation calculations
    bins = np.linspace(0, Zmax, Nbins)

    # Arrays for z-position (depth) and droplet size
    Z = Z0.copy()
    D = D0.copy()

    # Use trange (progress bar) if instructed
    iterator = range
    if args is not None:
        if args.progress:
            iterator = trange

    # Time loop
    t = 0
    for n in iterator(Nt):

        # Store output once every N_skip steps
        if n % N_skip == 0:
            i = int(n / N_skip)
            Z_out[i,:len(Z)] = Z
            D_out[i,:len(D)] = D

        # Reaction
        Z, D = coagulate_dense(Z, D, dt, bins, Np)
        # Random displacement
        Z = randomstep(K, Z, t, dt)
        # Reflect from surface and sea floor
        Z = reflect(Z, zmax = Zmax)
        # Rise or sink due to buoyancy
        V = rise_speed(D, rho)
        Z = advect(Z, V, dt)
        # Settle onto seabed
        Z, D = settle(Z, D, Zmax)
        # Increment time
        t = dt*i
    return Z_out, D_out


##############################
#### Numerical parameters ####
##############################

parser = argparse.ArgumentParser()
parser.add_argument('--dt', dest = 'dt', type = int, default = 10, help = 'Timestep')
parser.add_argument('--save_dt', dest = 'save_dt', type = int, default = 1800, help = 'Interval at which to save results')
parser.add_argument('--Np', dest = 'Np', type = int, default = 10000, help = 'Number of particles')
parser.add_argument('--Nbins', dest = 'Nbins', type = int, default = 51, help = 'Number of bins to use in flocculation')
parser.add_argument('--run_id', dest = 'run_id', type = int, default = 0, help = 'Run ID (used to differentiate runs when saving')
parser.add_argument('--profile', dest = 'profile', type = str, default = 'A', choices = ['A', 'B'], help = 'Diffusivity profiles')
parser.add_argument('--checkpoint', dest = 'checkpoint', action = 'store_true', help = 'Save results for checkpointing at every output timestep?')
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
#### Scenario parameters for Case 4 ####
########################################

#### Hard-coded parameters for this case ####
# Total depth
Zmax = 50
# Simulation time
Tmax = 24*3600
# Particle density [kg/m3]
rho  = 1500
# Size distribution
size_distribution = lognorm(0.25, scale = 2.5e-6)


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


# Size distribution
D0  = size_distribution.rvs(size = args.Np)


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
#datafolder = '/work6/torn/EulerLagrange/'

outputfilename_Z = os.path.join(datafolder, f'Case4_K_{label}_lagrangian_Nparticles={args.Np}_dt={args.dt}_Z_{args.run_id:04}.npy')
outputfilename_D = os.path.join(datafolder, f'Case4_K_{label}_lagrangian_Nparticles={args.Np}_dt={args.dt}_D_{args.run_id:04}.npy')

if (not os.path.exists(outputfilename_Z)) or args.overwrite:
    tic = time.time()
    Z_out, D_out = experiment_case4(Z0, D0, args.Np, Zmax, Tmax, args.dt, args.save_dt, K, rho, correctstep, Nbins = args.Nbins, args = args)
    toc = time.time()
    np.save(outputfilename_Z, Z_out)
    np.save(outputfilename_Z, D_out)
    logger(f'Simulation took {toc - tic:.1f} seconds, output written to {outputfilename_Z} and {outputfilename_D}', args, error = True)
else:
    logger(f'File exists, skipping: {outputfilename_Z}', args, error = True)


