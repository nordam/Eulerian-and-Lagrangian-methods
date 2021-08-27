#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing standard packages
import os
from time import time
import argparse
from tqdm import trange

# Numerical packages
import numpy as np
from scipy.stats import lognorm
from scipy.integrate import romb

# import stuff from .py files in local folder
import sys
sys.path.append('.')
from eulerian_functions import *
from particlefunctions import rise_speed
from logger import logger


############################################
#### Main functions to run a simulation ####
############################################


def iterative_solver_case4(params, C0, L_AD,  R_AD, K_vec, v_minus, v_plus, args):

    logger('Entering iterative solver...', args)

    # Make a copy, to avoid overwriting input
    c_now = C0.copy()

    # Set up flux-limeter matrices
    L_FL, R_FL = setup_FL_matrices(params, v_minus, v_plus, c_now)

    # Set up coagulation reaction matrices
    L_Co, R_Co = setup_coagulation_matrices(params, c_now)
    R = add_sparse(R_Co, R_AD, R_FL)

    # Calculate right-hand side (does not change with iterations)
    RHS = (R).dot(c_now.flatten())

    # Max number of iterations
    maxiter = 50
    # Tolerance
    tol = args.tol
    # List to store norms for later comparison
    norms = []

    for n in range(maxiter):

        # Add together matrices on left-hand side
        L = add_sparse(L_Co, L_AD)
        if n == 0:
            # Create preconditioner only at first iteration, since it is anyway only an
            # approximate inverse of L. This can save a lot of time.
            logger('Creating preconditioner...', args)
            tic = time()
            try:
                ILU = spilu(csc_matrix(L))
                toc = time()
                logger(f'Creating preconditioner took {toc - tic:.4f} seconds', args)
                preconditioner = LinearOperator(L.shape, lambda x : ILU.solve(x))
            except RuntimeError as e:
                logger('Creating preconditioner failed with error:', args, error = True)
                logger(f'{e}', args, error = True)
                logger('Trying without preconditioner instead', args, error = True)
                preconditioner = None

        # Move the flux-limiter matrix to the RHS
        # This has been found to improve condition of the LHS matrix
        FL_RHS = L_FL.dot(c_now.flatten())

        # Solve with iterative solver
        logger('Solving with BiCGStab...', args)
        tic = time()
        c_next, status = bicgstab(L, RHS - FL_RHS, x0 = c_now.flatten(), tol = 1e-9, M = preconditioner, maxiter = 1000)
        toc = time()

        if status == 0:
            logger(f'Solving with BiCGStab took {toc - tic:.4f} seconds', args)
        else:
            logger(f'BiCGStab failed with error: {status}, switching to GMRES', args, error = True)
            c_next, status = gmres(L, RHS - FL_RHS, x0 = c_now.flatten(), tol = 1e-9, M = preconditioner, maxiter = 1000)
            if status == 0:
                logger(f'Solving with GMRES took {toc - tic:.4f} seconds', args)
            else:
                logger(f'GMRES failed with error: {status}, exiting', args, error = True)
                sys.exit()

        # Reshape from vector to NK x NJ array
        c_next = c_next.reshape((params.Nclasses, params.Nz))

        # Calculate norm
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
        L_Co = setup_coagulation_matrices(params, c_next, return_both = False)

        # Copy concentration
        c_now[:] = c_next.copy()

    logger(f'Iterative solver finished with {n+1} iterations', args)
    return c_next


def experiment_case4(C0, K, params, outputfilename, args):

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
        C_now = iterative_solver_case4(params, C_now, L_AD,  R_AD, K_vec, v_minus, v_plus, args)

        # Estimate remaining time
        if n >= 1:
            toc = time()
            ETA = ((toc - tic) / n) * (params.Nt - n)
            logger(f'ETA = {ETA:.4f}', args)

    # Finally, store last timestep to output array
    C_out[-1,:,:] = C_now
    return C_out

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
#### Scenario parameters for Case 4 ####
########################################

#### Hard-coded parameters for this case ####
# Total depth
Zmax = 50
# Simulation time
Tmax = 24*3600
# Particle density [kg/m3]
rho  = 1500
# Smallest and largest particle size
Dmin = 1e-6
Dmax = 2e-4
# Size distribution
size_distribution = lognorm(0.25, scale = 2.5e-6)

# For this case, we define log-spaced size classes, and calculate
# the corresponding settling speeds from Stokes' law with a transition
# to constant drag coefficient at high Re

# Size class "cell" faces
Rf = np.logspace(np.log10(Dmin), np.log10(Dmax), args.NK+1)
# Size class "cell" centers
Rc = np.sqrt(Rf[:-1]*Rf[1:])
# Size distribution
pdf = size_distribution.pdf
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
speeds = rise_speed(2*Rc, rho)

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

#datafolder = '/work6/torn/EulerLagrange'
datafolder = '../results/'
outputfilename = os.path.join(datafolder, f'Case4_K_{label}_block_Nclasses={params.Nclasses}_NJ={params.Nz}_dt={params.dt}.npy')

if (not os.path.exists(outputfilename)) or args.overwrite:
    tic = time()
    c = experiment_case4(C0, K, params, outputfilename, args)
    toc = time()
    logger(f'Simulation took {toc - tic:.1f} seconds, output written to {outputfilename}', args, error = True)
    np.save(outputfilename, c)
else:
    logger(f'File exists, skipping: {outputfilename}', args, error = True)

if args.statusfilename is not None:
    args.statusfile.close()
