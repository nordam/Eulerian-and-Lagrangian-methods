#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing standard packages
import os
import time
import argparse
from tqdm import trange

# Numerical packages
import numpy as np
from scipy.stats import lognorm
from scipy.integrate import romb

# import stuff from .py files in local folder
import sys
sys.path.append('.')
from eulerian_functions import EulerianSystemParameters, Crank_Nicolson_FVM_TVD_advection_diffusion_reaction
from particlefunctions import rise_speed


####################################
####   Command line arguments   ####
####################################

parser = argparse.ArgumentParser()
parser.add_argument('--dt', dest = 'dt', type = float, default = 600, help = 'Timestep')
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
Tmax = 6*3600
# Particle density [kg/m3]
rho  = 2000
# Smallest and largest particle size
Dmin = 1e-5
Dmax = 1e-3

# For this case, we define log-spaced size classes, and calculate
# the corresponding settling speeds from Stokes' law with a transition
# to constant drag coefficient at high Re

# Size class "cell" faces
Rf = np.logspace(np.log10(Dmin), np.log10(Dmax), args.NK+1)
# Size class "cell" centers
Rc = np.sqrt(Rf[:-1]*Rf[1:])
# Size distribution
pdf = lognorm(0.2, scale = 2e-5).pdf
# Calculate fractions in each cell
mass_fractions = np.zeros(args.NK)
# Parameter for romberg integration
Nsub = 2**10 + 1
for j in range(args.NK):
    evaluation_points = np.linspace(Rf[j], Rf[j+1], Nsub)
    dx = evaluation_points[1] - evaluation_points[0]
    mass_fractions[j] = romb(pdf(evaluation_points), dx = dx)
# Normalise mass fractions
mass_fractions = mass_fractions / np.sum(mass_fractions)

# Speed class centers
speeds = -rise_speed(2*Rc, rho)

print('speeds: ', speeds)

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
        radii = Rc,
        coagulate = True,
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
outputfilename = os.path.join(datafolder, f'Case4_K_{label}_block_Nclasses={params.Nclasses}_NJ={params.Nz}_dt={params.dt}.npy')


tic = time.time()
c = Crank_Nicolson_FVM_TVD_advection_diffusion_reaction(C0, K, params, outputfilename = outputfilename)
toc = time.time()
print(f'Simulation took {toc - tic:.1f} seconds, output written to {outputfilename}')

np.save(outputfilename, c)

