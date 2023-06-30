#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing standard packages
import os
import time
import argparse
from tqdm import trange

# Numerical packages
import numpy as np

# import stuff from .py files in local folder
import sys
sys.path.append('src')
from eulerian_functions import *
from logger import eulerian_logger as logger
# Function to generate random speed distribution for microplastics
from microplastics_speed_generator import draw_random_speeds




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

def iterative_solver_case2(params, C0, L_AD,  R_AD, K_vec, v_minus, v_plus, args):

    logger('Entering iterative solver...', args)

    # Make a copy, to avoid overwriting input
    c_now = C0.copy()

    # Set up flux-limeter matrices
    L_FL, R_FL = setup_FL_matrices(params, v_minus, v_plus, c_now)

    # Calculate right-hand side (does not change with iterations)
    R = add_sparse(R_FL, R_AD, overwrite = False)
    RHS = (R).dot(c_now.flatten())

    # Max number of iterations
    maxiter = 50
    # Tolerance
    tol = args.tol
    # List to store norms for later comparison
    norms = []

    for n in range(maxiter):

        # In cases where the full left-hand matrix isn't diagonally dominant,
        # we cannot use the Thomas algorithm.
        # Hence, we move the flux-limiter part to the RHS.
        FL_RHS = L_FL.dot(c_now.flatten())
        # Then solve with the Thomas algorithm
        c_next = thomas(L_AD, RHS - FL_RHS).reshape((params.Nclasses, params.Nz))

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

def experiment_case2(C0, K, params, outputfilename, args):

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
        C_now = iterative_solver_case2(params, C_now, L_AD,  R_AD, K_vec, v_minus, v_plus, args)

        # Estimate remaining time
        if n >= 1:
            toc = time()
            ETA = ((toc - tic) / n) * (params.Nt - n)
            logger(f'ETA = {ETA:.4f}', args)

    # Finally, store last timestep to output array
    C_out[-1,:,:] = C_now
    return C_out


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
Nclasses = args.NK
speeds_filename = f'../data/Case2_speeds_Nclasses={Nclasses}.npy'
mass_fractions_filename = f'../data/Case2_mass_fractions_Nclasses={Nclasses}.npy'

if os.path.exists(speeds_filename) and os.path.exists(mass_fractions_filename):
    # If speeds are already calculated and stored, load those
    speeds = np.load(speeds_filename)
    mass_fractions = np.load(mass_fractions_filename)

else:
    # Read random samples and create histogram of speeds
    # and store historgram for later use

    # Create positive and negative log-spaced bins
    bins_positive = np.logspace(-8, np.log10(0.1), 2*int(Nclasses/4) + 1)
    bins_negative = np.logspace(-8, np.log10(0.3), 2*int(Nclasses/4) + 1)

    mids_positive =    np.sqrt(bins_positive[1:]*bins_positive[:-1])
    mids_negative = -( np.sqrt(bins_negative[1:]*bins_negative[:-1]) )[::-1]
    bins_negative = -bins_negative[::-1]

    counts = np.zeros(Nclasses)
    mids = np.concatenate([mids_negative, mids_positive])


    #for filename in tqdm(glob('speeds_nonfibre_*.npy')[:100]):
    logger('Generating random speeds...', args, error=True)
    outside = 0
    for i in trange(100):
        # 100 times 10,000,000 speeds, total 1 billion
        v = draw_random_speeds(10000000)
        #print(np.amax(v), np.amin(v[v>0]), np.amax(v[v<0]), np.amin(v))
        counts_negative = np.histogram(v, bins=bins_negative)[0]
        counts_positive = np.histogram(v, bins=bins_positive)[0]
        outside += 10000000 - (np.sum(counts_negative) + np.sum(counts_positive))
        counts[:len(counts_negative)] += counts_negative
        counts[len(counts_negative):] += counts_positive

    logger(f'{outside} speeds fell outside histogram range', args, error=True)
    # Normalised mass fractions
    mass_fractions = counts / np.sum(counts)
    # midpoints representing speed of each class
    speeds = mids

    np.save(speeds_filename, speeds)
    np.save(mass_fractions_filename, mass_fractions)
    logger('Done generating speeds', args, error=True)


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

resultsfolder = '../results'
resultsfolder = '/media/torn/SSD/EulerLagrange'
outputfilename = os.path.join(resultsfolder, f'Case2_K_{label}_block_Nclasses={params.Nclasses}_NJ={params.Nz}_dt={params.dt}_save_dt={args.save_dt}.npy')


if (not os.path.exists(outputfilename)) or args.overwrite:
    tic = time()
    c = experiment_case2(C0, K, params, outputfilename, args)
    toc = time()
    np.save(outputfilename, c)
    logger(f'Simulation took {toc - tic:.1f} seconds, output written to {outputfilename}', args, error = True)
else:
    logger(f'File exists, skipping: {outputfilename}', args, error = True)

if args.statusfilename is not None:
    args.statusfile.close()

