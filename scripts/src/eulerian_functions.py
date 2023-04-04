#!/usr/bin/env python
# -*- coding: utf-8 -*-

#Importing packages
import sys
import numpy as np
from numba import njit
import argparse
from scipy.integrate import romb
from scipy.sparse import diags, csc_matrix, dia_matrix
from scipy.sparse.linalg import spilu, LinearOperator, bicgstab, gmres
import datetime

# Progress bar
from tqdm import trange
from time import time

from webernaturaldispersion import weber_natural_dispersion

# Based on Wikipedia article about TDMA
@njit
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
    c_ = np.zeros(N-1)
    d_ = np.zeros(N)
    x  = np.zeros(N)
    c_[0] = c[0]/b[0]
    d_[0] = d[0]/b[0]
    for i in range(1, N-1):
        q = (b[i] - a[i-1]*c_[i-1])
        c_[i] = c[i]/q
        d_[i] = (d[i] - a[i-1]*d_[i-1])/q
    d_[N-1] = (d[N-1] - a[N-2]*d_[N-2])/(b[N-1] - a[N-2]*c_[N-2])
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

    def __init__(self, Zmax, Nz, Tmax, dt, Nclasses, Vmin = None, Vmax = None, speed_distribution = None, speeds = None, mass_fractions = None, checkpoint = False, logspaced = False, eta_top = 0, eta_bottom = 0, gamma = 0, fractionator = None, h0 = None, mu = None, ift = None, rho = None, Hs = None, coagulate = False, radii = None):
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
        self.checkpoint = checkpoint
        self.fractionator = fractionator
        self.h0 = h0
        self.mu = mu
        self.ift = ift
        self.rho = rho
        self.Hs = Hs
        self.coagulate = coagulate
        self.radii = radii

        # Inferred parameters
        # Position of cell faces, and spacing
        self.z_face, self.dz = np.linspace(self.Z0, self.Zmax, Nz + 1, retstep = True)
        # Position of cell centres
        self.z_cell = np.linspace(self.Z0 + self.dz/2, self.Zmax - self.dz/2, self.Nz)
        # Number of timesteps
        self.Nt = int( (self.Tmax - self.T0) / self.dt)

        if self.gamma > 0:
            # Indices of entrainment region [1.15*Hs, 1.85*Hs]
            self.sub_cells = np.where((self.z_cell >= 2) & (self.z_cell <= 3.2))[0]
            # Number of cells in the entrainment region
            self.N_sub = len(self.sub_cells)

        if speed_distribution is not None:
            assert Vmin is not None
            assert Vmax is not None
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
            self.speeds = -self.speeds
        else:
            assert speeds is not None
            assert mass_fractions is not None
            assert len(mass_fractions) == len(speeds)
            assert len(mass_fractions) == self.Nclasses
            self.speeds = speeds
            self.mass_fractions = mass_fractions
            # Normalise mass fractions
            self.mass_fractions = self.mass_fractions / np.sum(self.mass_fractions)

        if self.coagulate:
            assert self.radii is not None
            assert len(self.radii) == len(self.speeds)



#########################################################
####   Functions used by the Crank-Nicolson solver   ####
#########################################################


def entrainment_reaction_term_function(params, C):
    # Creation of concentration uniformly distributed in the top 1m, that is, N_sub grid points, at time step n.
    # Inputs c for all z, k, at time n
    # Allocate vector
    NK, NJ = C.shape
    reaction_term = np.zeros((NK, NJ))
    fractions = np.zeros((NK, NJ))
    # Width parameter of distribution (constant)
    sigma = 0.4*np.log(10)

    # Find surface slick thicness as fraction of initial thickness,
    # with some safeguards to prevent roundoff error leading to numbers outside [0, 1]
    surfaced_fraction = (1 - max(0, min(1, np.sum(C[:, :])*params.dz)))
    h = surfaced_fraction*params.h0
    if h > 0:
        D50n_h  = weber_natural_dispersion(params.rho, params.mu, params.ift, params.Hs, h)
        D50v_h  = np.exp(np.log(D50n_h) + 3*sigma**2)
        #f = Fractionator(speed_class_edges)
        fractions = params.fractionator.evaluate(D50v_h, sigma)[:,None]*np.ones([NK, params.N_sub])
        # Reaction term for the N_sub top cellss
        reaction_term[:,params.sub_cells] = params.gamma*surfaced_fraction*fractions/((params.N_sub*params.dz))

    return reaction_term, fractions[:,0]


def velocity_vector_function(params):
    # Velocities on cell faces for all components Nclasses.
    # It is noted that v_{1-0.5} = v_{0+0.5}, meaning that only one vector is needed. The surface boundary
    # is notated by z_{a} = z_{-1/2}, while the bottom boundary is notated by z_{b} = z_{NJ+1/2}. 

    # All cell faces within domain, including boundaries v_{-1/2} and v_{NJ+1/2}
    vel = params.speeds[:,None]*np.ones((params.Nclasses, params.Nz+1))
    # Optionally reduced velocty at surface
    vel[:, 0] = params.eta_top*vel[:,0]
    # Zero velocity at bottom boundary
    vel[:,-1] = params.eta_bottom*vel[:,-1]
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
    NK, NJ = c.shape

    # Allocate vector coinciding with the cell faces
    rho_vector_minus = np.zeros([NK, NJ+1])
    rho_vector_plus = np.zeros([NK, NJ+1])
    # Cell faces that include only interior cells
    # epsilon in numerator as well?
    rho_vector_minus[:, 1:NJ-1] = (c[:, 2:NJ] - c[:, 1:NJ-1])/(c[:, 1:NJ-1] - c[:, 0:NJ-2] + epsilon)
    rho_vector_plus[:, 2:NJ] = (c[:, 1:NJ-1] - c[:, 0:NJ-2])/(c[:, 2:NJ] - c[:, 1:NJ-1] + epsilon)

    # Elements that are not explicitly set should remain zero:
    # First (index 0) cell face (-1/2) reduces to upwind, but the correction cancels (no flux into the system from the boundary)
    # Second (index 1) cell face (+1/2) set to zero due to boundary condition for diffusive flux
    # Last (index -1) cell face(NJ-1/2) set to zero and reduces to upwind, does not appear for zero total flux BC
    return rho_vector_minus, rho_vector_plus


def psi_vector_function(rho_vec):
    # Sweby flux limiter
    #beta = 1.5
    #return np.maximum(np.maximum(0, np.minimum(beta*rho_vec, 1)), np.minimum(rho_vec, beta))
    # UMIST flux limiter
    return np.maximum(0, np.minimum(2*rho_vec, np.minimum(0.25 + 0.75*rho_vec, np.minimum(0.75 + 0.25*rho_vec, 2))))


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
        sup[1+k*NJ:NJ-1+k*NJ] = -a*K_vec[2:NJ] + b*v_minus[k, 2:NJ]
        # Absorbing boundary condition at surface (J_D = 0)
        sup[0+k*NJ] = -a*K_vec[1] + b*v_minus[k, 1]

        # Main diagonal
        diag[1+k*NJ:NJ-1+k*NJ] = a*(K_vec[2:NJ] + K_vec[1:NJ-1]) + b*(v_plus[k, 2:NJ] - v_minus[k, 1:NJ-1])
        # Absorbing boundary condition at surface (J_D = 0)
        diag[0+k*NJ] = a*K_vec[1] + b*(v_plus[k, 1] - v_minus[k, 0] - v_plus[k, 0])
        # Reflecting boundary condition at bottom (J_T = J_A + J_D = 0)
        diag[NJ-1+k*NJ] = a*K_vec[NJ-1] + b*(v_plus[k, NJ] + v_minus[k, NJ] - v_minus[k, NJ-1])

        # Subdiagonal
        sub[0+k*NJ:NJ-2+k*NJ] = -a*K_vec[1:NJ-1] - b*v_plus[k, 1:NJ-1]
        # Reflecting boundary condition at bottom (J_T = J_A + J_D = 0)
        sub[NJ-2+k*NJ] = -a*K_vec[NJ-1] - b*v_plus[k, NJ-1]

    # Return diagonal sparse matrices
    L_AD = diags([ sup, 1 + diag,  sub], offsets = [1, 0, -1])
    R_AD = diags([-sup, 1 - diag, -sub], offsets = [1, 0, -1])
    return L_AD, R_AD


def setup_FL_matrices(params, v_minus, v_plus, c, return_both = True):

    # CFL number, without velocity v
    b = params.dt/(2*params.dz)
    # Shorthand variables
    NJ = params.Nz
    NK = params.Nclasses

    # Rho vectors
    rho_minus, rho_plus = get_rho_vectors(c)
    # Vectors for the flux limiter functions
    psi_minus = psi_vector_function(rho_minus)
    psi_plus = psi_vector_function(rho_plus)

    # Superdiagonal for all components, with zeros between the superdiagonals for each component
    sup = np.zeros(NJ*NK-1)
    # Main diagonal for all components
    diag = np.zeros(NJ*NK)
    # Subdiagonal for all components, with zeros between the subdiagonals for each component
    sub = np.zeros(NJ*NK-1)

    for k in range(0, NK):

        # Superdiagonal
        sup[1+k*NJ:NJ-1+k*NJ] = b*(-v_plus[k, 2:NJ]*psi_plus[k, 2:NJ] + v_minus[k, 2:NJ]*psi_minus[k, 2:NJ])
        # Absorbing boundary condition at surface (J_D = 0)
        sup[0+k*NJ] = b*(-v_plus[k, 1]*psi_plus[k, 1] + v_minus[k, 1]*psi_minus[k, 1])

        # Main diagonal
        diag[1+k*NJ:NJ-1+k*NJ] = b*(v_plus[k, 1:NJ-1]*psi_plus[k, 1:NJ-1] - v_minus[k, 1:NJ-1]*psi_minus[k, 1:NJ-1] + v_plus[k, 2:NJ]*psi_plus[k, 2:NJ] - v_minus[k, 2:NJ]*psi_minus[k, 2:NJ])
        # Absorbing boundary condition at surface (J_D = 0)
        diag[0+k*NJ] = b*(v_plus[k, 1]*psi_plus[k, 1] - v_minus[k, 1]*psi_minus[k, 1])
        # Reflecting boundary condition at bottom (J_T = J_A + J_D = 0)
        diag[NJ-1+k*NJ] = b*(v_plus[k, NJ-1]*psi_plus[k, NJ-1] - v_minus[k, NJ-1]*psi_minus[k, NJ-1])

        # Subdiagonal
        sub[0+k*NJ:NJ-2+k*NJ] = b*(-v_plus[k, 1:NJ-1]*psi_plus[k, 1:NJ-1] + v_minus[k, 1:NJ-1]*psi_minus[k, 1:NJ-1])
        # Reflecting boundary condition at bottom (J_T = J_A + J_D = 0)
        sub[NJ-2+k*NJ] = b*(-v_plus[k, NJ-1]*psi_plus[k, NJ-1] + v_minus[k, NJ-1]*psi_minus[k, NJ-1])

    # Return diagonal sparse matrices
    if return_both:
        L_FL = 0.5*diags([-sup, -diag, -sub], offsets = [1, 0, -1])
        R_FL = 0.5*diags([ sup,  diag,  sub], offsets = [1, 0, -1])
        return L_FL, R_FL
    else:
        L_FL = 0.5*diags([-sup, -diag, -sub], offsets = [1, 0, -1])
        return L_FL


def add_sparse(*matrices, overwrite = False):
    if overwrite:
        result = matrices[0]
    else:
        result = matrices[0].copy()
    if len(matrices) > 1:
        for m in matrices[1:]:
            assert type(m) == dia_matrix
            for offset in m.offsets:
                result.setdiag(result.diagonal(offset) + m.diagonal(offset), offset)
    return result

