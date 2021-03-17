#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing standard packages
import os
import time
import argparse
from tqdm import trange

# Numerical packages
import numpy as np
from numba import jit

# import stuff from .py files in local folder
import sys
sys.path.append('.')
from eulerian_functions import EulerianSystemParameters, Crank_Nicolson_FVM_TVD_advection_diffusion_reaction


####################################
####   Command line arguments   ####
####################################

parser = argparse.ArgumentParser()
parser.add_argument('--dt', dest = 'dt', type = int, default = 600, help = 'Timestep')
parser.add_argument('--NJ', dest = 'NJ', type = int, default = 1000, help = 'Number of grid cells')
parser.add_argument('--NK', dest = 'NK', type = int, default = 8, help = 'Number of speed classes')
parser.add_argument('--profile', dest = 'profile', type = str, default = 'A', choices = ['A', 'B'], help = 'Diffusivity profiles')
parser.add_argument('--checkpoint', dest = 'checkpoint', type = bool, default = False, help = 'Save results for checkpointing at every output timestep?')
args = parser.parse_args()


########################################
#### Scenario parameters for Case 2 ####
########################################

#### Hard-coded parameters for this case ####
# Total depth
Zmax = 50
# Simulation time
Tmax = 12*3600

# For this case, we use a speed distribution taken from
# a random sample of microplastics properties
bin_spacing = 'logarithmic'
Nclasses = args.NK
speeds_filename = f'../data/Case2_speeds_{bin_spacing}_Nclasses={Nclasses}.npy'
mass_fractions_filename = f'../data/Case2_mass_fractions_{bin_spacing}_Nclasses={Nclasses}.npy'

if os.path.exists(speeds_filename) and os.path.exists(mass_fractions_filename):
    # If speeds are already calculated and stored, load those
    speeds = np.load(speeds_filename)
    mass_fractions = np.load(mass_fractions_filename)

else:
    # Read random samples and create histogram of speeds
    if bin_spacing == 'linear':
        bins_positive = np.linspace(0, 0.3, 3*int(Nclasses/4)+1)
        mids_positive = bins_positive[:-1] + width_positive/2

        bins_negative = np.linspace(-0.1, 0, int(Nclasses/4)+1)
        mids_negative = bins_negative[:-1] + width_negative/2

        bins = np.concatenate((bins_negative, bins_positive[1:]))
        mids = np.concatenate((mids_negative, mids_positive))
        counts = np.zeros(len(bins)-1)

    elif bin_spacing == 'logarithmic':
        bins_positive = np.logspace(-4, np.log10(0.3), 3*int(Nclasses/4) + 1)
        bins_negative = np.logspace(-4, np.log10(0.1), 1*int(Nclasses/4) + 1)

        mids_positive =    np.sqrt(bins_positive[1:]*bins_positive[:-1])
        mids_negative = -( np.sqrt(bins_negative[1:]*bins_negative[:-1]) )[::-1]

        bins_negative = -bins_negative[::-1]
        bins = np.concatenate((bins_negative[:-1], [0.0], bins_positive[1:]))
        mids = np.concatenate((mids_negative, mids_positive))
        counts = np.zeros(len(bins)-1)

    else:
        print('Invalid bin spacing: ', bin_spacing)
        import sys
        sys.exit()

    #for filename in tqdm(glob('speeds_nonfibre_*.npy')[:100]):
    for i in trange(500):
        speeds_fibre = np.load(f'../data/speeds_fibre_{i:04}.npy')
        speeds_nonfibre = np.load(f'../data/speeds_nonfibre_{i:04}.npy')
        cf, _ = np.histogram(speeds_fibre, bins = bins)
        cnf, _ = np.histogram(speeds_nonfibre, bins = bins)
        counts += cf*0.485 + cnf*0.465

    # Normalised mass fractions
    mass_fractions = counts / np.sum(counts)
    # midpoints representing speed of each class
    speeds = mids

    np.save(speeds_filename, speeds)
    np.save(mass_fractions_filename, mass_fractions)


# Initial condition:
# Normal distribution with mean mu and standard deviation sigma
sigma_IC = 4
mu_IC = 20
pdf_IC = lambda z: np.exp(-0.5*((z - mu_IC)/sigma_IC)**2) / (sigma_IC*np.sqrt(2*np.pi))



##################################
####   Diffusivity profiles   ####
##################################

K_A = lambda z: 1e-2*np.ones(len(z))

alpha, beta, zeta, z0 = (0.00636, 0.088, 1.54, 1.3)
K_B = lambda z: alpha*(z+z0)*np.exp(-(beta*(z+z0))**zeta)


####################################################
####   Populate object with system parameters   ####
####################################################


params = EulerianSystemParameters(
        Zmax = Zmax, # Max depth of water column
        Nz = args.NJ, # Number of cells in z-direction
        Tmax = Tmax, # Simulation time
        dt = args.dt, # timestep
        Nclasses = args.NK, # Number of speed classes
        speeds = speeds, # speed class values
        mass_fractions = mass_fractions, # fraction of mass in each speed class
        eta_bottom = 1, # Absorbing boundary in advection at bottom
        checkpoint = args.checkpoint, # save results underway?
    )


#########################################################
####    Run simulation for one diffusivity profile   ####
#########################################################

if args.profile == 'A':
    K = K_A
    label = 'A'
else:
    K = K_B
    label = 'B'

# Initial concentration array for all cells and time levels
C0 = pdf_IC(params.z_cell)[None,:] * params.mass_fractions[:,None]

datafolder = '/work6/torn/EulerLagrange'
datafolder = '../results/'
outputfilename = os.path.join(datafolder, f'Case2_K_{label}_block_Nclasses={params.Nclasses}_NJ={params.Nz}_dt={params.dt}.npy')


tic = time.time()
c = Crank_Nicolson_FVM_TVD_advection_diffusion_reaction(C0, K, params, outputfilename = outputfilename)
toc = time.time()
print(f'Simulation took {toc - tic:.1f} seconds, output written to {outputfilename}')

np.save(outputfilename, c)

