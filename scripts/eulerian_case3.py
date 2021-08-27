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
sys.path.append('.')
from eulerian_functions import EulerianSystemParameters, Crank_Nicolson_FVM_TVD_advection_diffusion_reaction
from fractionator import Fractionator
from particlefunctions import CONST, rise_speed
from wavefunctions import jonswap
from eulerian_functions import *
from webernaturaldispersion import weber_natural_dispersion
from logger import logger


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
    R = add_sparse(R_AD, R_FL)
    RHS = (R).dot(c_now.flatten())

    # Calculate reaction term for entrainment
    reaction_term_now = (0.5*params.dt*entrainment_reaction_term_function(params, c_now)).flatten()

    # Max number of iterations
    maxiter = 50
    # Tolerance
    tol = args.tol
    # List to store norms for later comparison
    norms = []

    for n in range(maxiter):

        # Calculate reaction term for entrainment
        reaction_term_next = (0.5*params.dt*entrainment_reaction_term_function(params, c_now)).flatten()

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
        # Recalculate concentration-dependent coagulation matrix

        # Copy concentration
        c_now[:] = c_next.copy()

    print(f'[{datetime.datetime.now()}] dt = {params.dt}, NK = {params.Nclasses}, NJ = {params.Nz}, iterations = {n}')
    sys.stdout.flush()
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

        Z_mean = np.sum(C_now*params.z_cell)*params.dz
        print(f't = {n*params.dt} mean(Z) = {Z_mean}')

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


###########################################################
#### Functions used once to calculate entrainment rate ####
###########################################################

def Fbw(windspeed, Tp):
    '''
    Fraction of breaking waves per second.
    See Holthuijsen and Herbers (1986)
    and Delvigne and Sweeney (1988) for details.

    windspeed: windspeed (m/s)
    Tp: wave period (s)
    '''
    if windspeed > 5:
        return 0.032 * (windspeed - 5)/Tp
    else:
        return 0

def entrainmentrate(windspeed, Tp, Hs, rho, ift):
    '''
    Entrainment rate (s**-1).
    See Li et al. (2017) for details.

    windspeed: windspeed (m/s)
    Tp: wave period (s)
    Hs: wave height (m)
    rho: density of oil (kg/m**3)
    ift: oil-water interfacial tension (N/m)
    '''
    # Physical constants
    g = CONST.g         # Acceleration of gravity (m/s**2)
    rho_w = CONST.rho_w # Density of seawater (kg/m**3)

    # Model parameters (empirical constants from Li et al. (2017))
    a = 4.604 * 1e-10
    b = 1.805
    c = -1.023
    # Rayleigh-Taylor instability maximum diameter:
    d0 = 4 * np.sqrt(ift / ((rho_w - rho)*g))
    # Ohnesorge number
    Oh = mu / np.sqrt(rho * ift * d0)
    # Weber number
    We = d0 * rho_w * g * Hs / ift
    return a * (We**b) * (Oh**c) * Fbw(windspeed, Tp)


####################################
####   Command line arguments   ####
####################################

parser = argparse.ArgumentParser()
parser.add_argument('--dt', dest = 'dt', type = float, default = 300, help = 'Timestep')
parser.add_argument('--save_dt', dest = 'save_dt', type = int, default = 1800, help = 'Interval at which to save results')
parser.add_argument('--NJ', dest = 'NJ', type = int, default = 1000, help = 'Number of grid cells')
parser.add_argument('--NK', dest = 'NK', type = int, default = 8, help = 'Number of speed classes')
parser.add_argument('--profile', dest = 'profile', type = str, default = 'A', choices = ['A', 'B'], help = 'Diffusivity profiles')
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
    print('dt does not evenly divide save_dt, output times will not be as expected')
    sys.exit()

########################################
#### Scenario parameters for Case 3 ####
########################################

#### Hard-coded parameters for this case ####
# Total depth
Zmax = 50
# Simulation time
Tmax = 1*3600

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
# Entrainment rate
gamma = entrainmentrate(windspeed, Tp, Hs, rho, ift)

# Size distribution parameters
sigma = 0.4 * np.log(10)
D50n  = weber_natural_dispersion(rho, mu, ift, Hs, h0)
D50v  = np.exp(np.log(D50n) + 3*sigma**2)
# bin edges
speed_class_edges = np.logspace(-6, 0, args.NK+1)
fractionator = Fractionator(speed_class_edges, rho)
speeds = -fractionator.speeds
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

datafolder = '/work6/torn/EulerLagrange'
datafolder = '../tmp_results'
outputfilename = os.path.join(datafolder, f'Case3_K_{label}_block_Nclasses={params.Nclasses}_NJ={params.Nz}_dt={params.dt}.npy')

print(params.speeds)
sys.exit()

if (not os.path.exists(outputfilename)) or args.overwrite:
    tic = time()
    c = experiment_case3(C0, K, params, outputfilename, args)
    toc = time()
    logger(f'Simulation took {toc - tic:.1f} seconds, output written to {outputfilename}', args, error = True)
    np.save(outputfilename, c)
else:
    logger(f'File exists, skipping: {outputfilename}', args, error = True)

if args.statusfilename is not None:
    args.statusfile.close()
