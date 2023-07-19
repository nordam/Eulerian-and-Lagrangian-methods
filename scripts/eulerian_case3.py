#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing standard packages
import os
from time import time
import argparse

# Numerical packages
import numpy as np

# import stuff from .py files in local folder
import sys
sys.path.append('src')
from eulerian_functions import *
from fractionator import Fractionator
from particlefunctions import rise_speed
from constants import CONST
from wavefunctions import jonswap, Fbw
from oilfunctions import entrainmentrate
from webernaturaldispersion import weber_natural_dispersion
from logger import eulerian_logger as logger


####################################
####   Command line arguments   ####
####################################

parser = argparse.ArgumentParser()
parser.add_argument('--dt', dest = 'dt', type = float, default = 300, help = 'Timestep')
parser.add_argument('--save_dt', dest = 'save_dt', type = int, default = 3600, help = 'Interval at which to save results')
parser.add_argument('--NJ', dest = 'NJ', type = int, default = 1000, help = 'Number of grid cells')
parser.add_argument('--NK', dest = 'NK', type = int, default = 8, help = 'Number of speed classes')
parser.add_argument('--profile', dest = 'profile', type = str, default = 'B', choices = ['A', 'B'], help = 'Diffusivity profiles')
parser.add_argument('--tol', dest = 'tol', type = float, default = 1e-6, help = 'Tolerance to use in the iterative procedure')
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
    logger('dt does not evenly divide save_dt, output times will not be as expected', args, error = True)
    sys.exit()



############################################
#### Main functions to run a simulation ####
############################################

def iterative_solver_case3(params, C0, L_AD,  R_AD, K_vec, v_minus, v_plus, args):

    logger('Entering iterative solver...', args)

    # Make a copy, to avoid overwriting input
    c_now = C0.copy()

    # Set up flux-limeter matrices
    L_FL, R_FL = setup_FL_matrices(params, v_minus, v_plus, c_now)

    # Calculate right-hand side (does not change with iterations)
    R = add_sparse(R_FL, R_AD, overwrite = False)
    RHS = (R).dot(c_now.flatten())

    # Calculate reaction term for entrainment
    #reaction_term_now = (0.5*params.dt*entrainment_reaction_term_function(params, c_now)).flatten()
    reaction_term_now_tmp, fractions_now = entrainment_reaction_term_function(params, c_now)
    reaction_term_now = (0.5*params.dt*reaction_term_now_tmp).flatten()

    # Max number of iterations
    maxiter = 50
    # Tolerance
    tol = args.tol
    # List to store norms for later comparison
    norms = []

    for n in range(maxiter):

        # Calculate reaction term for entrainment
        #reaction_term_next = (0.5*params.dt*entrainment_reaction_term_function(params, c_now)).flatten()
        reaction_term_next_tmp, fractions_next = entrainment_reaction_term_function(params, c_now)
        reaction_term_next = (0.5*params.dt*reaction_term_next_tmp).flatten()

        # In cases where the full left-hand matrix isn't diagonally dominant,
        # we cannot use the Thomas algorithm.
        # Hence, we move the flux-limiter part to the RHS.
        FL_RHS = L_FL.dot(c_now.flatten())
        # Then solve with the Thomas algorithm
        c_next = thomas(L_AD, RHS - FL_RHS + reaction_term_next + reaction_term_now).reshape((params.Nclasses, params.Nz))

        # Calculate norm and check for convergence
        if n == 0:
            first_guess = c_next.copy()
        elif n == 1:
            # Calculate norm
            norms.append( np.amax(np.sqrt(params.dz*np.sum((first_guess - c_next)**2, axis=0))) )
        else:
            # Calculate norm
            norms.append( np.amax(np.sqrt(params.dz*np.sum((c_now - c_next)**2, axis=0))) )
            # Exit iterative loop if tolerance is met, or if norm is unchanged from previous iteration
            if len(norms) >= 2:
                if (norms[-1] < tol*norms[0]) or np.any(norms[-1] >= norms[:-1]) or (n >= (maxiter - 1)):
                    break

        # Recalculate the left-hand side flux-limiter matrix using new concentration estimate
        L_FL = setup_FL_matrices(params, v_minus, v_plus, c_next, return_both = False)

        # Copy concentration
        c_now[:] = c_next.copy()

    logger(f'iterations = {n}', args)
    return c_next

def experiment_case3(C0, K, params, outputfilename, args):

    # Evaluate diffusivity function at cell faces
    K_vec = K(params.z_face)
    # Arrays of velocities for all cells and all classes
    v_plus = np.maximum(velocity_vector_function(params), 0)
    v_minus = np.minimum(velocity_vector_function(params), 0)

    # Set up matrices encoding advection and diffusion
    # (these are tri-diagonal, and constant in time)
    L_AD, R_AD = setup_AD_matrices(params, K_vec, v_minus, v_plus)

    # Shorthand variables
    NJ = params.Nz
    NK = params.Nclasses

    # Array to hold one timestep, to avoid allocating too much memory
    C_now  = np.zeros_like(C0)
    C_now[:] = C0.copy()
    # Array for output, store once every save_dt seconds
    N_skip = int(args.save_dt/params.dt)
    N_out = 1 + int(params.Nt / N_skip)
    C_out = np.zeros((N_out, NK, NJ)) - 999

    # Use trange (progress bar) if instructed
    if args.progress:
        iterator = trange
    else:
        iterator = range

    tic = time()
    for n in iterator(0, params.Nt):

        # Store output once every N_skip steps
        if n % N_skip == 0:
            i = int(n / N_skip)
            C_out[i,:,:] = C_now[:]
            if args.checkpoint:
                if outputfilename is not None:
                    logger(f'Saving temporary results to file: {outputfilename}', args)
                    np.save(outputfilename, C_out)

        # Iterative procedure
        C_now = iterative_solver_case3(params, C_now, L_AD,  R_AD, K_vec, v_minus, v_plus, args)

        # Estimate remaining time
        if n >= 1:
            toc = time()
            ETA = ((toc - tic) / n) * (params.Nt - n)
            logger(f'ETA = {ETA:.4f}', args)

    # Finally, store last timestep to output array
    C_out[-1,:,:] = C_now
    return C_out


########################################
#### Scenario parameters for Case 3 ####
########################################

#### Hard-coded parameters for this case ####
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
Hs, Tp = jonswap(windspeed)
# Entrainment rate
gamma = entrainmentrate(rho, mu, ift, Hs, Tp, windspeed)

# Size distribution parameters
sigma = 0.4 * np.log(10)
D50n  = weber_natural_dispersion(rho, mu, ift, Hs, h0)
D50v  = np.exp(np.log(D50n) + 3*sigma**2)


# bin edges
speed_class_edges = np.logspace(-6, 0, args.NK+1)
fractionator = Fractionator(speed_class_edges, rho)
# Fractions
mass_fractions = fractionator.evaluate(D50v, sigma)
# Normalise mass fractions
mass_fractions = mass_fractions / np.sum(mass_fractions)

# Initial condition:
# Normal distribution with mean mu and standard deviation sigma
sigma_IC = 4
mu_IC = 20
pdf_IC = lambda z: np.exp(-0.5*((z - mu_IC)/sigma_IC)**2) / (sigma_IC*np.sqrt(2*np.pi))


##################################
####   Diffusivity profiles   ####
##################################

# Constant diffusivity (not used in paper)
K_A = lambda z: 1e-2*np.ones(len(z))

# Fitted to results of GOTM simulation
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
        speeds = -fractionator.speeds, # speed class values
        mass_fractions = mass_fractions, # fraction of mass in each speed class
        eta_top = 1, # Absorbing boundary in advection at top
        checkpoint = args.checkpoint, # save results underway?
        fractionator = fractionator, # Object to calculate speed class froctions of newly entrained oil droplets
        # Oil related properties
        gamma = gamma, # Entrainment rate
        h0 = h0, # Initial oil thickness
        mu = mu, # Dynamic viscosity of oil
        ift = ift, # Oil-water interfacial tension
        rho = rho, # Oil density
        Hs = Hs, # significant wave height
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

resultsfolder = '../results/'
outputfilename = os.path.join(resultsfolder, f'Case3_K_{label}_block_Nclasses={params.Nclasses}_NJ={params.Nz}_dt={params.dt}_save_dt={args.save_dt}.npy')


if (not os.path.exists(outputfilename)) or args.overwrite:
    tic = time()
    c = experiment_case3(C0, K, params, outputfilename, args)
    toc = time()
    np.save(outputfilename, c)
    logger(f'Simulation took {toc - tic:.1f} seconds, output written to {outputfilename}', args, error = True)
else:
    logger(f'File exists, skipping: {outputfilename}', args, error = True)

if args.statusfilename is not None:
    args.statusfile.close()
