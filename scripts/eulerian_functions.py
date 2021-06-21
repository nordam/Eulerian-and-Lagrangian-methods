#!/usr/bin/env python
# -*- coding: utf-8 -*-

#Importing packages
import sys
import numpy as np
from numba import njit
import argparse
from scipy.integrate import romb
from scipy.sparse import diags, csc_matrix, dia_matrix
from scipy.sparse.linalg import spilu, LinearOperator, bicgstab

# Progress bar
from tqdm import trange
from time import time

from webernaturaldispersion import weber_natural_dispersion
from coagulation_functions import get_new_classes_and_weights, coagulation_rate_prefactor


# Based on Wikipedia article about TDMA, written by Tor Nordam
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

    return reaction_term


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


@njit
def setup_coagulation_matrices_core(C_now, dt, radii):
    Nclasses, Ncells = C_now.shape
    diagonals = []
    # Create empty diagonals, starting with main diagonal
    # and going downwards
    for i in range(Nclasses):
        diagonals.append(np.zeros(Ncells*(Nclasses - i)))
    # Offsets of those diagonals
    offsets = -Ncells*np.arange(Nclasses)
    # Loop over all classes twice
    for j in range(Nclasses):
        for k in range(Nclasses):
            # radii of the two classes in question:
            r_j = radii[j]
            r_k = radii[k]
            # Best matching new classes, and weighting between them
            m1, w1, m2, w2 = get_new_classes_and_weights(j,k,radii)
            # Reaction rate
            rate_prefactor = coagulation_rate_prefactor(r_j, r_k)
            # Fill diagonals with rate multiplied by concentration
            # First, handle main diagonal (representing lost mass from class j)
            diagonals[0][Ncells*j:Ncells*(j+1)] += rate_prefactor*C_now[k,:]*dt/2
            # Then, handle off-diagonals (representing gained mass is classes m1 and m2)
            diagonals[m1-j][Ncells*j:Ncells*(j+1)] -= w1*rate_prefactor*C_now[k,:]*dt/2
            # Only add here if m2 is not above range
            if m2 < Nclasses:
                diagonals[m2-j][Ncells*j:Ncells*(j+1)] -= w2*rate_prefactor*C_now[k,:]*dt/2
    # return diagonals and offsests to wrapper function,
    # since scipy cannot be used in a numba-compiled function
    return diagonals, offsets

def setup_coagulation_matrices(params, C_now, return_both = True):
    diagonals, offsets = setup_coagulation_matrices_core(C_now, params.dt, params.radii)
    # Create matrices encoding reaction part of the equation
    Lr = diags(diagonals, offsets)
    if return_both:
        Rr = -Lr.copy()
        return Lr, Rr
    else:
        return Lr

def add_sparse(*matrices):
    result = matrices[0].copy()
    if len(matrices) > 1:
        for m in matrices[1:]:
            assert type(m) == dia_matrix
            for offset in m.offsets:
                result.setdiag(result.diagonal(offset) + m.diagonal(offset), offset)
    return result

def check_diagonal_dominance(m):
    assert len(m.offsets) == 3
    sup  = np.abs(m.diagonal( 1))
    main = np.abs(m.diagonal( 0))
    sub  = np.abs(m.diagonal(-1))
    if sup[0] >= main[0]:
        return False
    elif np.any((sup[1:] + sub[:-1]) >= main[1:-1]):
        return False
    elif sub[-1] >= main[-1]:
        return False
    else:
        return True

#@profile
def Iterative_Solver(params, C0, L_AD,  R_AD, K_vec, v_minus, v_plus):

    # Make a copy, to avoid overwriting input
    c_now = C0.copy()

    # Set up flux-limeter matrices
    L_FL, R_FL = setup_FL_matrices(params, v_minus, v_plus, c_now)


    # Set up coagulation reaction matrices
    if params.coagulate:
        L_Co, R_Co = setup_coagulation_matrices(params, c_now)
        R = add_sparse(R_AD, R_FL, R_Co)
    else:
        R = add_sparse(R_AD, R_FL)

    # Calculate right-hand side (does not change with iterations)
    RHS = (R).dot(c_now.flatten())

    # Max number of iterations
    maxiter = 50
    # Tolerance
    tol = 1e-6
    # List to store norms for later comparison
    norms = []

    # Preconditioner (updated only in the first iteration)
    ILU = None

    for n in range(maxiter):

        # If there is entrainment, calculate reaction term
        if params.gamma > 0:
            reaction_term_next = (0.5*params.dt*entrainment_reaction_term_function(params, c_now)).flatten()
        else:
            reaction_term_next = 0.0

        # If there is coagulation, take that into account
        if params.coagulate:
            # Add together matrices on left-hand side
            L = add_sparse(L_AD, L_FL, L_Co)
            if ILU is None:
                # Create preconditioner only once, since it is anyway only an
                # approximate inverse of L. This saves a lot of time.
                ILU = spilu(csc_matrix(L))
                preconditioner = LinearOperator(L.shape, lambda x : ILU.solve(x))

            # Solve with iterative solver
            c_next, status = bicgstab(L, RHS + reaction_term_next.flatten(), x0 = c_now.flatten(), tol = 1e-12, M = preconditioner)
            if status != 0:
                print(f'Bicgstab failed, with error: {status}, switching to GMRES')
                c_next, status = gmres(L, RHS + reaction_term_next.flatten(), x0 = c_now.flatten(), tol = 1e-12, M = preconditioner)
                if status != 0:
                    print(f'GMRES failed, with error {status}, stopping')
                    sys.exit()
            c_next = c_next.reshape((params.Nclasses, params.Nz))
        else:
            # Add together matrices on left-hand side
            L = add_sparse(L_AD, L_FL)
            c_next = thomas(L, RHS + reaction_term_next).reshape((params.Nclasses, params.Nz))


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
        if params.coagulate:
            L_Co = setup_coagulation_matrices(params, c_next, return_both = False)

        # Copy concentration
        c_now[:] = c_next.copy()

    print(n, ' iterations')
    return c_next

#@profile
def Crank_Nicolson_FVM_TVD_advection_diffusion_reaction(C0, K, params, outputfilename = None):

    # Evaluate diffusivity function at cell faces
    K_vec = K(params.z_face)
    # Arrays of velocities for all cells and all classes
    v_plus = np.maximum(velocity_vector_function(params), 0)
    v_minus = np.minimum(velocity_vector_function(params), 0)

    # Set up matrices encoding advection and diffusion
    # (these are tri-diagonal, and constant in time)
    L_AD, R_AD = setup_AD_matrices(params, K_vec, v_minus, v_plus)

    #DEBUG np.save('A_sub_new.npy', L_AD.diagonal(-1))
    #DEBUG np.save('A_main_new.npy', L_AD.diagonal(0))
    #DEBUG np.save('A_sup_new.npy', L_AD.diagonal( 1))
    #sys.exit()

    # Shorthand variables
    NJ = params.Nz
    NK = params.Nclasses

    # Array to hold one timestep, to avoid allocating too much memory
    C_now  = np.zeros_like(C0)
    C_now[:] = C0.copy()
    # Array for output, store once every 3600 seconds
    N_skip = int(1800/params.dt)
    N_out = 1 + int(params.Nt / N_skip)
    C_out = np.zeros((N_out, NK, NJ))

    tic = time()
    for n in range(0, params.Nt):

        # Store output once every N_skip steps
        if n % N_skip == 0:
            i = int(n / N_skip)
            C_out[i,:,:] = C_now[:]

        # Iterative procedure
        C_now = Iterative_Solver(params, C_now, L_AD,  R_AD, K_vec, v_minus, v_plus)

        # Print status
        if (n > 1) and (n % N_skip == 0):
            toc = time()
            ETA = ((toc - tic) / n) * (params.Nt - n)
            print(f'dt = {params.dt}, NK = {params.Nclasses}, NJ = {params.Nz}, ETA = {ETA:.4f} seconds')

    # Finally, store last timestep to output array
    C_out[-1,:,:] = C_now
    return C_out
