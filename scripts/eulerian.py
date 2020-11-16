#!/usr/bin/env python
# -*- coding: utf-8 -*-

#Importing packages
import time
import numpy as np
from numba import jit
# import stuff from .py files in notebooks folder
import sys
sys.path.append('.')
from fractionator import Fractionator
from particlefunctions import CONST, rise_speed
from wavefunctions import jonswap
from webernaturaldispersion import weber_natural_dispersion



#########################################################
####   Functions used by the Crank-Nicolson solver   ####
#########################################################

def velocity_function(z, NK):
    # Generating a vector containing the velocity at position z, for number of components NK
    global depth
    global eta
    global v
    vel = v*np.ones([len(z), NK], order='F')
    vel[z == 0, :] = eta*vel[0, :]
    vel[z == depth, :] = 0
    return vel

def diffusivity_vector_function(z_face, dz, NJ, NK):
    # Diffusivities on cell faces, calculated directly using the positions of the cell faces, z_face.
    # It is noted that D_{1-0.5} = D_{0+0.5}, meaning that only one vector is needed. The surface boundary
    # is notated by z_{a} = z_{-1/2}, while the bottom boundary is notated by z_{b} = z_{NJ+1/2}.
    # The diffusivities are the same for all components (no interaction), thus only one vector is needed.

    global diffusivity_profile
    # Allocate vector, one element longer than the number of cells to coincide with the number of cell faces
    diffusivity_vector = np.zeros([NJ+1, NK], order='F')
    # All cell faces within domain, including boundaries D_{-1/2} and D_{NJ+1/2}
    for k in range(0, NK):
        diffusivity_vector[:, k] = diffusivity_profile(z_face[:])

    return diffusivity_vector

def velocity_vector_function(z_face, NJ, NK):
    # Velocities on cell faces for all components NK, given directly by the already defined velocity_function.
    # It is noted that v_{1-0.5} = v_{0+0.5}, meaning that only one vector is needed. The surface boundary
    # is notated by z_{a} = z_{-1/2}, while the bottom boundary is notated by z_{b} = z_{NJ+1/2}. 

    # Allocate vector, one element longer than the number of cells to coincide with the number of cell faces
    velocity_vector = np.zeros([NJ+1, NK], order='F')
    # All cell faces within domain, including boundaries v_{-1/2} and v_{NJ+1/2}
    #for k in range(0, NK):
    velocity_vector = velocity_function(z_face[:], NK)

    return velocity_vector


def reaction_term_function(c, dz, NJ, NK, N_sub, sub_cells, dt, gamma):
    # Creation of concentration uniformly distributed in the top 1m, that is, N_sub grid points, at time step n.
    # Inputs c for all z, k, at time n
    
    global rho, mu, ift, sigma, Hs, h0, speed_class_edges, fractionator

    # Allocate vector
    reaction_term = np.zeros([NJ, NK], order='F')
    
    # Generate fractions of surfaced mass for each component
    h = (1 - max(0, min(1, np.sum(c[:, :])*dz)))*h0
    if h > 0:
        D50n_h  = weber_natural_dispersion(rho, mu, ift, Hs, h)
        D50v_h  = np.exp(np.log(D50n_h) + 3*sigma**2)
        #f = Fractionator(speed_class_edges)
        fractions = fractionator.evaluate(D50v_h, sigma)*np.ones([N_sub, NK], order='F')
        
        #print(h, D50n_h, D50v_h, np.sum(fractions[0, :]), gamma, N_sub)
        
        # Reaction term for the N_sub top cellss
        #gamma = 1 - np.exp(-dt/taus)
        reaction_term[sub_cells, :] = gamma*(1 - np.sum(c[:, :])*dz)*fractions[0:N_sub, :]/((dt*N_sub*dz))
        #print(np.sum(reaction_term[sub_cells, :])*dz)

    return reaction_term
    # CHANGE TO USE TOTAL INITIAL CONCENTRATION MINUS CURRENT INSTEAD OF "1 - sum(c)"
    # CHANGE TO DISTRIBUTE BETWEEN ALL COMPONENTS



def rho_vector_plus_function(c, NJ, NK):
    # Vector containing all values of rho^{+} for the flux limiting for positive (downward) velocity,
    # for all components NK
    
    # Global epsilon to prevent division by zero
    global epsilon
    
    # Allocate vector coinciding with the cell faces
    rho_vector_plus = np.zeros([NJ+1, NK], order='F')
    # Cell faces that include only interior cells
    rho_vector_plus[2:NJ, :] = (c[1:NJ-1, :] - c[0:NJ-2, :])/(c[2:NJ, :] - c[1:NJ-1, :] + epsilon) # epsilon in numerator as well?
    
    # First (index 0) cell face (-1/2) (previously stored in rho_plus_min1 function), reduces to upwind, but the correction cancels (no flux into the system from the boundary)
    #rho_vector_plus[0, :] = 0
    # Second cell face (+1/2) set to zero due to boundary condition for diffusive flux
    #rho_vector_plus[1, :] = 0 
    # Last cell face(NJ-1/2) set to zero and reduces to upwind, does not appear for zero total flux BC
    #rho_vector_vector[NJ, :] = 0
    return rho_vector_plus
    # CHECK AT BOUNDARIES, + epsilon above and below to ensure no division by zero

def rho_vector_minus_function(c, NJ, NK):
    # Vector containing all values of rho^{-} for the flux limiting for negative (upward) velocity,
    # for all components NK

    # Global epsilon to prevent division by zero
    global epsilon
    
    # Allocate vector coinciding with the cell faces
    rho_vector_minus = np.zeros([NJ+1, NK], order='F')
    # Cell faces that include only interior cells
    rho_vector_minus[1:NJ-1, :] = (c[2:NJ, :] - c[1:NJ-1, :])/(c[1:NJ-1, :] - c[0:NJ-2, :] + epsilon) # epsilon in numerator as well?
    
    # First (index 0) cell face (-1/2) (previously stored in rho_minus_min1 function), reduces to upwind
    #rho_vector_minus[0, :] = 0
    # Second-to-last cell set to zero, thus upwind, since c[NJ] is unknown. Extrapolation? Generation by BC?
    #rho_vector_minus[NJ-1, :] = 0
    # Last cell set to zero, no flux into the system from the boundary
    #rho_vector_minus[NJ, :] = 0
    return rho_vector_minus
    # CHECK AT BOUNDARIES, + epsilon above and below to ensure no division by zero


# REMOVE
def rho_plus_min1(NK):
    # Rho for ghost cell at j=-1
    return np.zeros(NK)
# REMOVE
def rho_minus_min1(NK):
    # Rho for ghost cell at j=-1
    return np.zeros(NK)




def psi_vector_function(rho_vec, NJ, NK):
    
    # First-order upwind scheme
    #return np.zeros(NJ, NK)

    # Second-order central scheme
    #return np.ones(NJ, NK)
    
    # Linear (second-order) upwind scheme
    #return rho_vec[0:NJ, :]

    # (Third-order) QUICK scheme
    #return 0.25*(3 + rho_vec[:, :])
    
    # Third-order upwind-biased (Hundsdorfer)
    #return (2/3 + 2/6*rho_vec[:, :])
    
    # Van Leer flux limiter
    #return (rho_vec + np.abs(rho_vec))/(1 + rho_vec)
    
    # Van Albada flux limiter
    #return (rho_vec + rho_vec**2)/(1 + rho_vec**2)
    
    # Sweby flux limiter
    #beta = 1.5
    #return np.maximum(np.maximum(0, np.minimum(beta*rho_vec, 1)), np.minimum(rho_vec, beta))

    # Koren flux limiter
    #return np.maximum(0, np.minimum(2*rho_vec, np.minimum((1 + 2*rho_vec)/(3), 2)))
    
    # UMIST flux limiter
    #return np.maximum(0, np.minimum(2*rho_vec, np.minimum(0.25 + 0.75*rho_vec, np.minimum(0.75 + 0.25*rho_vec, 2))))

    # QUICK flux limiter
    return np.maximum(0, np.minimum(2*rho_vec, np.minimum(0.75 + 0.25*rho_vec, 2)))




def flux_limiter_reaction_term_function(c, NJ, NK, r_sans_D, CFL_sans_v, velocity_vector_minus, velocity_vector_plus):
    # Deferred correction reaction term for the Crank-Nicolson scheme, i.e. 0.5*R_n^{DC}.
    # By using the deferred correction method, the higher-order extension(flux limiter) is considered as a reaction term,
    # such that the system of equations easily may be written as a tridiagonal matrix with an added correction.
    
    # Allocating
    flux_limiter_reaction_term = np.zeros([NJ, NK], order='F')
    
    # Global epsilon to prevent division by zero
    global epsilon
    
    # Vectors for the flux limiter functions
    psi_vector_plus = psi_vector_function(rho_vector_plus_function(c, NJ, NK), NJ, NK)
    #psi_vector_plus_min1 = psi_vector_function(rho_plus_min1(NK), 1, NK)
    psi_vector_minus = psi_vector_function(rho_vector_minus_function(c, NJ, NK), NJ, NK)
    #psi_vector_minus_min1 = psi_vector_function(rho_minus_min1(), 1, NK)
    
    # Interior cells, j = 1, ..., NJ-2
    flux_limiter_reaction_term[1:NJ-1, :] = 0.5*CFL_sans_v*(-velocity_vector_plus[2:NJ, :]*psi_vector_plus[2:NJ, :]*(c[2:NJ, :] - c[1:NJ-1, :]) + 
                                                        velocity_vector_minus[2:NJ, :]*psi_vector_minus[2:NJ, :]*(c[2:NJ, :] - c[1:NJ-1, :]) +
                                                        velocity_vector_plus[1:NJ-1, :]*psi_vector_plus[1:NJ-1, :]*(c[1:NJ-1, :] - c[0:NJ-2, :])  - 
                                                        velocity_vector_minus[1:NJ-1, :]*psi_vector_minus[1:NJ-1, :]*(c[1:NJ-1, :] - c[0:NJ-2, :]))
    
    # First cell, j = 0. Adjacent to surface boundary with absorbing BC - only advective flux allowed through
    flux_limiter_reaction_term[0, :] = 0.5*CFL_sans_v*(-velocity_vector_plus[1, :]*psi_vector_plus[1, :]*(c[1, :] - c[0, :]) + 
                                                        velocity_vector_minus[1, :]*psi_vector_minus[1, :]*(c[1, :] - c[0, :]) +
                                                        velocity_vector_plus[0, :]*psi_vector_plus[0, :]*(c[0, :] - c[0, :])  - 
                                                        velocity_vector_minus[0, :]*psi_vector_minus[0, :]*(c[0, :] - c[0, :]))
    
    #NB! CHECK LIMITER AT SURFACE, REDUCES TO UPWIND HERE IN THE LAST TWO ROWS DUE TO BC FOR DIFFUSIVE FLUX!
    
    
    # Last cell, j = NJ-1. Adjacent to bottom boundary with reflecting BC - no flux allowed through
    flux_limiter_reaction_term[NJ-1, :] = 0.5*CFL_sans_v*(velocity_vector_plus[NJ-1, :]*psi_vector_plus[NJ-1, :]*(c[NJ-1, :] - c[NJ-2, :])  - 
                                                        velocity_vector_minus[NJ-1, :]*psi_vector_minus[NJ-1, :]*(c[NJ-1, :] - c[NJ-2, :]))
    
    # NB! CHECK LIMITER AT BOTTOM, REFLECTIVE BC DICTATES NO FLUX IN/OUT OF BOTTOM
    
    return flux_limiter_reaction_term




def Crank_Nicolson_RHS(c, z_cell, dz, NJ, NK, r_sans_D, CFL_sans_v, diffusivity_vector, velocity_vector_minus, velocity_vector_plus, reaction_term_n, flux_limiter_reaction_term):
    # Calculates RHS of tridiagonal (Crank-Nicolson) matrix system, that is B*c_n + 0.5*R_n + 0.5*R_n^{DC}, by array slicing,
    # using reflecting BC at cell j=NJ-1 adjacent to bottom and an absorbing BC at cell j=0 adjacent to surface.
    
    # ALlocating
    Bc_n_plus_R_n_half = np.zeros([NJ, NK], order='F')
    # Vector for deferred correction reaction term on RHS
    #flux_limiter_reaction_term = flux_limiter_reaction_term_function(c, NJ, r_sans_D, CFL_sans_v, velocity_vector_minus, velocity_vector_plus)
    
    # Interior points, j = 1, ..., NJ-2
    Bc_n_plus_R_n_half[1:NJ-1, :] = (0.5*r_sans_D*diffusivity_vector[2:NJ, :] - 0.5*CFL_sans_v*velocity_vector_minus[2:NJ, :])*c[2:NJ, :] + (
                                1 - 0.5*r_sans_D*(diffusivity_vector[2:NJ, :] + diffusivity_vector[1:NJ-1, :]) - 0.5*CFL_sans_v*(
                                velocity_vector_plus[2:NJ, :] - velocity_vector_minus[1:NJ-1, :]))*c[1:NJ-1, :] + (
                                0.5*r_sans_D*diffusivity_vector[1:NJ-1, :] + 0.5*CFL_sans_v*velocity_vector_plus[1:NJ-1, :])*c[0:NJ-2, :] + (
                                0.5*dt*reaction_term_n[1:NJ-1, :]) + 0.5*flux_limiter_reaction_term[1:NJ-1, :]
    
    # First cell, j = 0, cell adjacent to surface boundary. 
    # Absorbing boundary condition J_T = J_A + J_D = J_A => J_D = 0
    Bc_n_plus_R_n_half[0, :] = (0.5*r_sans_D*diffusivity_vector[1, :] - 0.5*CFL_sans_v*velocity_vector_minus[1, :])*c[1, :] + (
                            1 - 0.5*r_sans_D*diffusivity_vector[1, :] - 0.5*CFL_sans_v*(
                            velocity_vector_plus[1, :] - velocity_vector_minus[0, :] - velocity_vector_plus[0, :]))*c[0, :] + (
                            0.5*dt*reaction_term_n[0, :]) + 0.5*flux_limiter_reaction_term[0, :]
    
    # Last cell, j = NJ-1, cell adjacent to bottom boundary.
    # Reflecting boundary condition J_T = J_A + J_D = 0
    Bc_n_plus_R_n_half[NJ-1, :] = (1 - 0.5*r_sans_D*diffusivity_vector[NJ-1, :] + 0.5*CFL_sans_v*velocity_vector_minus[NJ-1, :])*c[NJ-1, :] + (
                                0.5*r_sans_D*diffusivity_vector[NJ-1, :] + 0.5*CFL_sans_v*velocity_vector_plus[NJ-1, :])*c[NJ-2, :] + (
                                0.5*dt*reaction_term_n[NJ-1, :]) + 0.5*flux_limiter_reaction_term[NJ-1, :]

    return Bc_n_plus_R_n_half


def Crank_Nicolson_LHS(z_cell, dz, NJ, NK, r_sans_D, CFL_sans_v, diffusivity_vector, velocity_vector_minus, velocity_vector_plus):
    # Setting up diagonals of tridiagonal matrix A on LHS for system A*c_{n+1} = B*c_n
    
    # Superdiagonal for all components, with zeros between the superdiagonals for each component
    A_sup = np.zeros(NJ*NK-1)
    # Main diagonal for all components
    A_diag = np.zeros(NJ*NK)
    # Subdiagonal for all components, with zeros between the subdiagonals for each component
    A_sub = np.zeros(NJ*NK-1)
    
    for k in range(0, NK):
        
        # Superdiagonal
        A_sup[1+k*NJ:NJ-1+k*NJ] = (-0.5*r_sans_D*diffusivity_vector[2:NJ, k] + 0.5*CFL_sans_v*velocity_vector_minus[2:NJ, k])
        # Absorbing boundary condition at surface (J_D = 0)
        A_sup[0+k*NJ] = (-0.5*r_sans_D*diffusivity_vector[1, k] + 0.5*CFL_sans_v*velocity_vector_minus[1, k])

        
        # Superdiagonal
        A_diag[1+k*NJ:NJ-1+k*NJ] = (1 + 0.5*r_sans_D*(diffusivity_vector[2:NJ, k] + diffusivity_vector[1:NJ-1, k]) + 0.5*CFL_sans_v*(
                            velocity_vector_plus[2:NJ, k] - velocity_vector_minus[1:NJ-1, k]))
        # Absorbing boundary condition at surface (J_D = 0)
        A_diag[0+k*NJ] = (1 + 0.5*r_sans_D*diffusivity_vector[1, k] + 0.5*CFL_sans_v*(velocity_vector_plus[1, k] - 
                                                                              velocity_vector_minus[0, k] - velocity_vector_plus[0, k]))
        # Reflecting boundary condition at bottom (J_T = J_A + J_D = 0)
        A_diag[NJ-1+k*NJ] = (1 + 0.5*r_sans_D*diffusivity_vector[NJ-1, k] - 0.5*CFL_sans_v*velocity_vector_minus[NJ-1, k])


        # Subdiagonal
        A_sub[0+k*NJ:NJ-2+k*NJ] = -(0.5*r_sans_D*diffusivity_vector[1:NJ-1, k] + 0.5*CFL_sans_v*velocity_vector_plus[1:NJ-1, k]) # check indices
        # Reflecting boundary condition at bottom (J_T = J_A + J_D = 0)
        A_sub[NJ-2+k*NJ] = -(0.5*r_sans_D*diffusivity_vector[NJ-1, k] + 0.5*CFL_sans_v*velocity_vector_plus[NJ-1, k])

    return A_sub, A_diag, A_sup



# Based on Wikipedia article about TDMA, written by Tor Nordam
@jit(nopython = True)
def thomas(a, b, c, d):
    # Solves Ax = d,
    # where layout of matrix A is
    # b1 c1 ......... 0
    # a2 b2 c2 ........
    # .. a3 b3 c3 .....
    # .................
    # .............. cN-1
    # 0 ..........aN bN
    # Note index offset of a
    N = len(d)
    c_ = np.zeros(N-1)
    d_ = np.zeros(N)
    x  = np.zeros(N)
    c_[0] = c[0]/b[0]
    d_[0] = d[0]/b[0]
    for i in range(1, N-1):
        c_[i] = c[i]/(b[i] - a[i-1]*c_[i-1])
    for i in range(1, N):
        d_[i] = (d[i] - a[i-1]*d_[i-1])/(b[i] - a[i-1]*c_[i-1])
    x[-1] = d_[-1]
    for i in range(N-2, -1, -1):
        x[i] = d_[i] - c_[i]*x[i+1]
    return x


def Iterative_Solver(c_n, c_nplus1, A_sub, A_diag, A_sup, Bc_n_plus_R_n_half, z_cell, dz, NJ, NK, N_sub, sub_cells, n, r_sans_D, CFL_sans_v, diffusivity_vector, velocity_vector_minus, velocity_vector_plus):
    # Iterative process for approximating the non-linearities that occur on the LHS of the matrix system due to
    # the reaction term and the higher-order flux limiting in the deferred correction reaction term

    c_nplus1_guess = np.zeros([NJ, NK], order='F')
    # Initial guess calculated using previous concentration to approximate source term R_{n+1} (lagging)
    c_nplus1_guess = c_nplus1

    # Approximate the reaction terms using the initial guess
    reaction_term_lagged = reaction_term_function(c_nplus1_guess, dz, NJ, NK, N_sub, sub_cells, dt, gamma)
    flux_limiter_reaction_term_lagged = flux_limiter_reaction_term_function(c_nplus1_guess, NJ, NK, r_sans_D, CFL_sans_v, velocity_vector_minus, velocity_vector_plus)


    # Compute new approximation of solution c[:, n+1] for first iteration, for each component k = 0, ..., NK
    #for k in range(0, NK):
    c_nplus1[:, :] = thomas(A_sub[:], A_diag[:], A_sup[:], Bc_n_plus_R_n_half.ravel(order='F')[:] + (0.5*dt*(reaction_term_lagged.ravel(order='F')[:])) +  
                         0.5*flux_limiter_reaction_term_lagged.ravel(order='F')[:]).reshape((NJ, NK), order='F')

    # Computing 2-norm between first iteration and initial guess to use as a measure of the error
    norm_2_first = np.amax(np.sqrt(dz*np.sum((c_nplus1 - c_nplus1_guess)**2, axis=0)))
    # NB! CHECK! MAXIMUM OF ALL 2-NORMS?

    # Initialize 2-norm to be calculated between consecutive iterations for c_n+1 until a condition is satisified
    norm_2 = 1

    # Iteration number
    kappa = 2
    # Max number of iterations
    kappa_max = 1000

    # Tolerance
    tol = 1e-20

    # Save previous iteration to compute 2-norm
    c_nplus1_old = c_nplus1

    # Iterate until condition is satisfied
    while norm_2 > (tol)*norm_2_first and kappa < kappa_max:

        # Approximate reaction terms R_{n+1} for new iteration
        reaction_term_lagged = reaction_term_function(c_nplus1, dz, NJ, NK, N_sub, sub_cells, dt, gamma)
        flux_limiter_reaction_term_lagged = flux_limiter_reaction_term_function(c_nplus1_guess, NJ, NK, r_sans_D, CFL_sans_v, velocity_vector_minus, velocity_vector_plus)

        # Compute new approximation of solution c[:, n+1] for new iteration, for each component
        #for k in range(0, NK):
        c_nplus1[:, :] = thomas(A_sub[:], A_diag[:], A_sup[:], Bc_n_plus_R_n_half.ravel(order='F')[:] + (0.5*dt*(reaction_term_lagged.ravel(order='F')[:])) + 
                             0.5*flux_limiter_reaction_term_lagged.ravel(order='F')[:]).reshape((NJ, NK), order='F')

        # Compute 2-norm between new iteration and previous iteration
        norm_2 = np.amax(np.sqrt(dz*np.sum((c_nplus1 - c_nplus1_old)**2, axis=0)))

        # Update iterator
        kappa += 1
        # Update previous iteration
        c_nplus1_old = c_nplus1

    #np.save('kappa.npy', kappa)
    return c_nplus1



def Crank_Nicolson_FVM_TVD_advection_diffusion_reaction(C0, z_cell, z_face, dz, NJ, NK, NN, dt, N_sub, sub_cells, gamma):

    # von Neumann number, without diffusivity D
    r_sans_D = dt/(dz**2)
    # CFL number, without velocity v
    CFL_sans_v = dt/dz

    diffusivity_vector = diffusivity_vector_function(z_face, dz, NJ, NK)    
    velocity_vector_plus = np.maximum(velocity_vector_function(z_face, NJ, NK), 0)
    velocity_vector_minus = np.minimum(velocity_vector_function(z_face, NJ, NK), 0)

    # Initialize diagonals on LHS
    A_sub = np.zeros(NJ*NK-1) #<---- was NJ, but receives NJ-1 from C_N_LHS-function
    A_diag = np.zeros(NJ*NK) #<---- was NJ+1, but receives NJ from C_N_LHS-function
    A_sup = np.zeros(NJ*NK-1) #<---- was NJ, but receives NJ-1 from C_N_LHS-function

    # Set up matrix A on LHS
    A_sub, A_diag, A_sup = Crank_Nicolson_LHS(z_cell, dz, NJ, NK, r_sans_D, CFL_sans_v, diffusivity_vector, velocity_vector_minus, velocity_vector_plus)

    # Arrays to hold two timesteps, to avoid allocating too much memory
    C_next = np.zeros_like(C0)
    C_now  = np.zeros_like(C0)
    C_now[:] = C0
    # Array for output, store once every 300 seconds
    N_skip = int(300/dt)
    N_out = 1 + int(NN / N_skip)
    C = np.zeros((NJ, N_out, NK))

    starttime = time.time()

    for n in range(0, NN):
        # Simple progress bar
        Nsteps = int(NN/50)
        if n % Nsteps == 0:
            if n == 0:
                eta = '-'
            else:
                eta = int((time.time() - starttime) * ((NN / n) - 1))
            print(f'Running simulation with {Nclasses:>3} classes... | {"#" * int(n/Nsteps):<50} | ETA {eta} seconds      ', end = '\r')
            sys.stdout.flush()

        # Store output once every N_skip steps
        if n % N_skip == 0:
            i = int(n / N_skip)
            C[:,i,:] = C_now[:]

        # RHS
        # Calculating reaction terms at current time step
        reaction_term_n = reaction_term_function(C_now, dz, NJ, NK, N_sub, sub_cells, dt, gamma)
        flux_limiter_reaction_term_n = flux_limiter_reaction_term_function(C_now, NJ, NK, r_sans_D, CFL_sans_v, velocity_vector_minus, velocity_vector_plus)

        # Compute RHS of system
        Bc_n_plus_R_n_half = Crank_Nicolson_RHS(C_now, z_cell, dz, NJ, NK, r_sans_D, CFL_sans_v, diffusivity_vector, velocity_vector_minus, velocity_vector_plus, reaction_term_n, flux_limiter_reaction_term_n)

        # LHS
        # Iterative approach to approximate the concentration at n+1 used in reaction terms
        # First do one computation of system with lagged values, to compute 2-norm ||c_1 - c_0||

        # Check diagonal dominance
        # diagonal_dominance_checker(A_sub, A_diag, A_sup)

        # Compute first iteration, using the reaction terms at the previous time level to obtain an initial guess for
        # the concentration at time level (n+1), for each component
        #for k in range(0, NK):
        C_next = thomas(A_sub[:], A_diag[:], A_sup[:], Bc_n_plus_R_n_half.ravel(order='F')[:] + (0.5*dt*(reaction_term_n.ravel(order='F')[:])) + 0.5*flux_limiter_reaction_term_n.ravel(order='F')[:]).reshape((NJ, NK), order='F')

        # Iterative procedure
        C_next = Iterative_Solver(C_now, C_next, A_sub, A_diag, A_sup, Bc_n_plus_R_n_half[:], z_cell, dz, NJ, NK, N_sub, sub_cells, n, r_sans_D, CFL_sans_v, diffusivity_vector, velocity_vector_minus, velocity_vector_plus)
        # Copy next to now
        C_now[:] = C_next[:]

    # Finally, store last timestep to output array
    C[:,-1,:] = C_next
    return C


#############################
#### Scenario parameters ####
#############################

# Oil parameters
## Dynamic viscosity of oil (kg/m/s)
mu     = 1.51
## Interfacial tension (N/m)
ift    = 0.013
## Oil density (kg/m**3)
rho    = 992
## Intial oil film thickness (m)
h0     = 3e-3
# Environmental parameter
## Windspeed (m/s)
windspeed = 9

# Size distribution (using, somewhat arbitrarily, the distribution that would be created by some wave height)
# Significant wave height and peak wave period
Hs, Tp = jonswap(windspeed)
# Assign new sizes from Johansen distribution
sigma = 0.4 * np.log(10)
D50n  = weber_natural_dispersion(rho, mu, ift, Hs, h0)
D50v  = np.exp(np.log(D50n) + 3*sigma**2)



##################################
####   Numerical parameters   ####
##################################

# Small number to prevent division by zero
epsilon = 1e-20
# Total depth
depth = 50
# Number of cells
NJ = 1000
# Uniform grid spacing between neighbouring cell centres/cell faces
dz = (abs(depth - 0))/NJ
# Position of cell centres
z_cell = np.linspace(0, depth, NJ + 1)[:-1] + dz/2
# Position of cell faces
z_face = np.linspace(0, depth, NJ + 1)
# Indices of entrainment region [1.15*Hs, 1.85*Hs]
sub_cells = np.where((z_cell >= 1.15*Hs) & (z_cell <= 1.85*Hs))[0]
# Number of cells in the entrainment region
N_sub = len(sub_cells)
# Time step
dt = 300
# Start time
t_start = 0
# End time
t_end = 72*3600
# Time gridpoints
NN = int((t_end - t_start)/dt)



##################################
####   Diffusivity profiles   ####
##################################

K_A = lambda z: 1e-2*np.ones(len(z))

alpha, beta, zeta, z0 = (0.00636, 0.088, 1.54, 1.3)
K_B = lambda z: alpha*(z+z0)*np.exp(-(beta*(z+z0))**zeta)



##############################################################
####    Run simulations for different number of classes   ####
##############################################################

diffusivity_profile = K_A

# Eta for reduced velocity at surface
eta = 0

# Probability for resubmersion
gamma = 0


Nclass_values = np.array([1, 2, 3, 4, 6, 8, 10, 12, 16, 24, 32, 40, 48, 64, 72, 96])
moments = np.zeros_like(Nclass_values, dtype = np.float64)


for i, Nclasses in enumerate(Nclass_values):
    # bin edges
    speed_class_edges = np.logspace(-6, 0, Nclasses+1)
    # Create object to calculate fractions
    fractionator = Fractionator(speed_class_edges, rho)
    # Use the evaluate method of the fractionator class
    fractions = fractionator.evaluate(D50v, sigma)
    # The speeds for each class are stored in the object
    speeds = fractionator.speeds

    # Mass fractions
    mass_fractions = fractions
    print(f'Running simulation with {Nclasses:>3} classes... ', end = '\r')

    # Velocities
    v = speeds*(-1)

    # Number of components
    NK = len(v)

    # Concentration array for all cells and time levels
    C0 = np.zeros([NJ, NK], order='F')

    # Initial condition:
    # Normal distribution with mean mu and standard deviation sigma
    sigma_IC = 4
    mu_IC = 20
    for k in range(0, NK):
        C0[:, k] = mass_fractions[k]*np.exp(-(z_cell - mu_IC)**2/(2*sigma_IC**2))/(sigma_IC*np.sqrt(2*np.pi))


    start = time.time()
    c = Crank_Nicolson_FVM_TVD_advection_diffusion_reaction(C0, z_cell, z_face, dz, NJ, NK, NN, dt, N_sub, sub_cells, gamma)
    end = time.time()
    print(f'Running simulation with {Nclasses:>3} classes... | {"#" * 50} | Elapsed: {(end-start):.2f} seconds')
    moments[i] = dz*np.sum(z_cell * np.sum(c, axis = 2)[:,-1])

    np.save(f'../data2/Case1_K_A_block_Nclasses={Nclasses}_NJ={NJ}_dt={dt}.npy', c)

np.save(f'../data2/moments_NJ={NJ}_dt={dt}.npy', moments)
np.save(f'../data2/z_cell_NJ={NJ}_dt={dt}.npy', z_cell)
