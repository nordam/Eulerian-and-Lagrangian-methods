#!/usr/bin/env python
# -*- coding: utf-8 -*-

#Importing packages
import numpy as np
from numba import jit
import argparse
from scipy.integrate import romb
from scipy.sparse import diags
# Progress bar
from tqdm import trange


# Based on Wikipedia article about TDMA, written by Tor Nordam
@jit(nopython = True)
def thomas_solver(a, b, c, d):
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
    c_ = np.empty(N-1)
    d_ = np.empty(N)
    x  = np.empty(N)
    c_[0] = c[0]/b[0]
    d_[0] = d[0]/b[0]
    for i in range(1, N-1):
        tmp = (b[i] - a[i-1]*c_[i-1])
        c_[i] = c[i]/tmp
        d_[i] = (d[i] - a[i-1]*d_[i-1])/tmp
    d_[-1] = (d[-1] - a[-2]*d_[-2])/(b[-1] - a[-2]*c_[-2])
    x[-1] = d_[-1]
    for i in range(N-2, -1, -1):
        x[i] = d_[i] - c_[i]*x[i+1]
    return x


def thomas(A, b):
    # Solves Ax = b to find x
    # This is a wrapper function, which unpacks
    # A from a sparse array structure into separate diagonals,
    # and passes them to the numba-compiled solver defined above.
    # Note, this method needs A to be diagonally dominant.
    x = thomas_solver(A.diagonal(-1), A.diagonal(0), A.diagonal(1), b)
    return x


class EulerianSystemParameters():

    def __init__(self, Zmax, Nz, Tmax, dt, Vmin, Vmax, Nclasses, speed_distribution, logspaced = False, eta_top = 0, eta_bottom = 0, gamma = 0):
        self.Z0 = 0.0
        self.Zmax = Zmax
        self.Nz = Nz
        self.T0 = 0.0
        self.Tmax = Tmax
        self.dt = dt
        self.Vmin = Vmin
        self.Vmax = Vmax
        self.Nclasses = Nclasses
        self.speed_distribution = speed_distribution
        self.logspaced = logspaced
        self.eta_top = eta_top
        self.eta_bottom = eta_bottom
        self.gamma = gamma

        # Inferred parameters
        # Position of cell faces, and spacing
        self.z_face, self.dz = np.linspace(self.Z0, self.Zmax, Nz + 1, retstep = True)
        # Position of cell centres
        self.z_cell = np.linspace(self.Z0 + self.dz/2, self.Zmax - self.dz/2, self.Nz)
        # Number of timesteps
        self.Nt = int( (self.Tmax - self.T0) / self.dt)

        # Calculate speed for each class
        if logspaced:
            # speed class edges (i.e., the values at the border between classes)
            self.speed_class_edges = np.logspace(np.log10(self.Vmin), np.log10(self.Vmax), self.Nclasses + 1)
            # The center of each class on a log scale
            self.speeds = np.sqrt(self.speed_class_edges[1:]*self.speed_class_edges[:-1])
        else:
            # speed class edges (i.e., the values at the border between classes)
            self.speed_class_edges = np.linspace(self.Vmin, self.Vmax, self.Nclasses + 1)
            # The center of each class on a linear scale
            self.speeds = (self.speed_class_edges[1:] + self.speed_class_edges[:-1]) / 2
        self.mass_fractions = np.zeros(Nclasses)

        # Parameter for romberg integration
        Nsub = 2**10 + 1
        for j in range(Nclasses):
            evaluation_points = np.linspace(self.speed_class_edges[j], self.speed_class_edges[j+1], Nsub)
            dx = evaluation_points[1] - evaluation_points[0]
            self.mass_fractions[j] = romb(self.speed_distribution(evaluation_points), dx = dx)
        # Normalise mass fractions
        self.mass_fractions = self.mass_fractions / np.sum(self.mass_fractions)


#########################################################
####   Functions used by the Crank-Nicolson solver   ####
#########################################################


def velocity_vector_function(params):
    # Velocities on cell faces for all components Nclasses.
    # It is noted that v_{1-0.5} = v_{0+0.5}, meaning that only one vector is needed. The surface boundary
    # is notated by z_{a} = z_{-1/2}, while the bottom boundary is notated by z_{b} = z_{NJ+1/2}. 

    # All cell faces within domain, including boundaries v_{-1/2} and v_{NJ+1/2}
    vel = params.speeds*np.ones((params.Nz+1, params.Nclasses))
    # Optionally reduced velocty at surface
    vel[ 0, :] = params.eta_top*vel[0, :]
    # Zero velocity at bottom boundary
    vel[-1, :] = params.eta_bottom*vel[-1,:]
    return vel


def get_rho_vectors(c):
    # Vector containing all values of rho^{+} for the flux limiting for
    # positive (downward) velocity, for all components NK
    # and
    # Vector containing all values of rho^{-} for the flux limiting for
    # negative (upward) velocity, for all components NK

    # Small number to prevent division by zero
    epsilon = 1e-20
    # array shape
    NJ, NK = c.shape

    # Allocate vector coinciding with the cell faces
    rho_vector_plus = np.zeros([NJ+1, NK])
    rho_vector_minus = np.zeros([NJ+1, NK])
    # Cell faces that include only interior cells
    # epsilon in numerator as well?
    rho_vector_plus[2:NJ, :] = (c[1:NJ-1, :] - c[0:NJ-2, :])/(c[2:NJ, :] - c[1:NJ-1, :] + epsilon)
    rho_vector_minus[1:NJ-1, :] = (c[2:NJ, :] - c[1:NJ-1, :])/(c[1:NJ-1, :] - c[0:NJ-2, :] + epsilon)

    # Elements that are not explicitly set should remain zero:
    # First (index 0) cell face (-1/2) reduces to upwind, but the correction cancels (no flux into the system from the boundary)
    # Second (index 1) cell face (+1/2) set to zero due to boundary condition for diffusive flux
    # Last (index -1) cell face(NJ-1/2) set to zero and reduces to upwind, does not appear for zero total flux BC
    return rho_vector_plus, rho_vector_minus


def psi_vector_function(rho_vec):
    # UMIST flux limiter
    return np.maximum(0, np.minimum(2*rho_vec, np.minimum(0.25 + 0.75*rho_vec, np.minimum(0.75 + 0.25*rho_vec, 2))))


def setup_FL_matrices(params, v_minus, v_plus, c):

    # CFL number, without velocity v
    b = params.dt/(2*params.dz)
    # Shorthand variables
    NJ = params.Nz
    NK = params.Nclasses

    # Rho vectors
    rho_plus, rho_minus = get_rho_vectors(c)
    # Vectors for the flux limiter functions
    psi_plus = psi_vector_function(rho_plus)
    psi_minus = psi_vector_function(rho_minus)

    # Superdiagonal for all components, with zeros between the superdiagonals for each component
    sup = np.zeros(NJ*NK-1)
    # Main diagonal for all components
    diag = np.zeros(NJ*NK)
    # Subdiagonal for all components, with zeros between the subdiagonals for each component
    sub = np.zeros(NJ*NK-1)

    for k in range(0, NK):

        # Superdiagonal
        sup[1+k*NJ:NJ-1+k*NJ] = b*(-v_plus[2:NJ, k]*psi_plus[2:NJ, k] + v_minus[2:NJ, k]*psi_minus[2:NJ, k])
        # Absorbing boundary condition at surface (J_D = 0)
        sup[0+k*NJ] = b*(-v_plus[1, k]*psi_plus[1, k] + v_minus[1, k]*psi_minus[1, k])

        # Main diagonal
        diag[1+k*NJ:NJ-1+k*NJ] = b*(v_plus[1:NJ-1, k]*psi_plus[1:NJ-1, k] - v_minus[1:NJ-1, k]*psi_minus[1:NJ-1, k] + v_plus[2:NJ, k]*psi_plus[2:NJ, k] - v_minus[2:NJ, k]*psi_minus[2:NJ, k])
        # Absorbing boundary condition at surface (J_D = 0)
        diag[0+k*NJ] = b*(v_plus[1, k]*psi_plus[1, k] - v_minus[1, k]*psi_minus[1, k])
        # Reflecting boundary condition at bottom (J_T = J_A + J_D = 0)
        diag[NJ-1+k*NJ] = b*(v_plus[NJ-1, k]*psi_plus[NJ-1, k] - v_minus[NJ-1, k]*psi_minus[NJ-1, k])

        # Subdiagonal
        sub[0+k*NJ:NJ-2+k*NJ] = b*(-v_plus[1:NJ-1, k]*psi_plus[1:NJ-1, k] + v_minus[1:NJ-1, k]*psi_minus[1:NJ-1, k])
        # Reflecting boundary condition at bottom (J_T = J_A + J_D = 0)
        sub[NJ-2+k*NJ] = b*(-v_plus[NJ-1, k]*psi_plus[NJ-1, k] + v_minus[NJ-1, k]*psi_minus[NJ-1, k])

    # Return diagonal sparse matrices
    L_FL = 0.5*diags([ sup,  diag,  sub], offsets = [1, 0, -1])
    R_FL = 0.5*diags([-sup, -diag, -sub], offsets = [1, 0, -1])
    return L_FL, R_FL


def setup_AD_matrices(params, K_vec, v_minus, v_plus):
    # Setting up tridiagonal matrices L_AD and R_AD for the Crank-Nicolson solver
    # In this implementation, the matrices L_AD and R_AD are tridiagonal, and constant in time
    # (note that the higher-order correction flux-limiter for the advection is found in another matrix)

    # von Neumann number, without diffusivity D
    a = params.dt/(2*params.dz**2)
    # CFL number, without velocity v
    b = params.dt/(2*params.dz)
    # Shorthand variables
    NJ = params.Nz
    NK = params.Nclasses

    # Superdiagonal for all components, with zeros between the superdiagonals for each component
    sup = np.zeros(NJ*NK-1)
    # Main diagonal for all components
    diag = np.zeros(NJ*NK)
    # Subdiagonal for all components, with zeros between the subdiagonals for each component
    sub = np.zeros(NJ*NK-1)

    for k in range(0, NK):

        # Superdiagonal
        sup[1+k*NJ:NJ-1+k*NJ] = -a*K_vec[2:NJ] + b*v_minus[2:NJ, k]
        # Absorbing boundary condition at surface (J_D = 0)
        sup[0+k*NJ] = -a*K_vec[1] + b*v_minus[1, k]

        # Main diagonal
        diag[1+k*NJ:NJ-1+k*NJ] = a*(K_vec[2:NJ] + K_vec[1:NJ-1]) + b*(v_plus[2:NJ, k] - v_minus[1:NJ-1, k])
        # Absorbing boundary condition at surface (J_D = 0)
        diag[0+k*NJ] = a*K_vec[1] + b*(v_plus[1, k] - v_minus[0, k] - v_plus[0, k])
        # Reflecting boundary condition at bottom (J_T = J_A + J_D = 0)
        diag[NJ-1+k*NJ] = a*K_vec[NJ-1] + b*(v_plus[NJ, k] + v_minus[NJ, k] - v_minus[NJ-1, k])

        # Subdiagonal
        sub[0+k*NJ:NJ-2+k*NJ] = -a*K_vec[1:NJ-1] - b*v_plus[1:NJ-1, k]
        # Reflecting boundary condition at bottom (J_T = J_A + J_D = 0)
        sub[NJ-2+k*NJ] = -a*K_vec[NJ-1] - b*v_plus[NJ-1, k]

    # Return diagonal sparse matrices
    L_AD = diags([ sup, 1 + diag,  sub], offsets = [1, 0, -1])
    R_AD = diags([-sup, 1 - diag, -sub], offsets = [1, 0, -1])
    return L_AD, R_AD


def Iterative_Solver(params, C0, L_AD,  R_AD, K_vec, v_minus, v_plus):

    # Make a copy, to avoid overwriting input
    c_now = C0.copy()

    # Max number of iterations
    maxiter = 20
    # Tolerance
    tol = 1e-9

    # Set up flux-limeter matrices
    L_FL, R_FL = setup_FL_matrices(params, v_minus, v_plus, c_now)

    # Iterate up to kappa_max times
    for n in range(maxiter):

        # Compute new approximation of solution c[:, n+1] for new iteration, for each component
        RHS = (R_AD + R_FL).dot(C0.T.flatten())
        c_next = thomas(L_AD + L_FL, RHS).reshape((params.Nclasses, params.Nz)).T

        # Calculate norm
        norm = np.amax(np.sqrt(params.dz*np.sum((c_now - c_next)**2, axis=0)))
        print(f'This is iterative_solver, iteration {n}, norm = {norm}')
        if norm < tol:
            return c_next

        # Recalculate the left-hand side flux-limiter matrix using new concentration estimate
        L_FL, _ = setup_FL_matrices(params, v_minus, v_plus, c_next)

        # Copy concentration
        c_now[:] = c_next.copy()

    return c_next


def Crank_Nicolson_FVM_TVD_advection_diffusion_reaction(C0, K, params):

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
    # Array for output, store once every 600 seconds
    N_skip = int(600/params.dt)
    N_out = 1 + int(params.Nt / N_skip)
    C_out = np.zeros((NJ, N_out, NK))

    for n in trange(0, params.Nt):

        # Store output once every N_skip steps
        if n % N_skip == 0:
            i = int(n / N_skip)
            C_out[:,i,:] = C_now[:]

        # Iterative procedure
        C_now = Iterative_Solver(params, C_now, L_AD,  R_AD, K_vec, v_minus, v_plus)

    # Finally, store last timestep to output array
    C_out[:,-1,:] = C_now
    return C_out
