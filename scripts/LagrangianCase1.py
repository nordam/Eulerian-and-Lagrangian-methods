#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import namedtuple
import numpy as np

############################
#### Physical constants ####
############################

PhysicalConstants = namedtuple('PhysicalConstants', ('g', 'rho_w', 'nu'))
CONST = PhysicalConstants(
        g     = 9.81,       # Acceleration due to gravity (m/s**2)
        rho_w = 1025,       # Density of sea water (kg/m**3)
        nu    = 1.358e-6,   # Kinematic viscosity of sea water (m**2/s)
)

###############################
#### Numerical derivatives ####
###############################

def ddz(K, z, t):
    '''
    Numerical derivative of K(z, t).

    This function calculates a numerical partial derivative
    of K(z, t), with respect to z using central finite difference.

    K: diffusivity as a function of depth (m**2/s)
    z: current particle depth (m)
    t: current time (s)
    '''
    dz = 1e-6 # Finite difference spacing
    return (K(z+dz/2, t) - K(z-dz/2, t)) / dz


############################
#### Random walk scheme ####
############################

def randomstep(K, z, t, dt):
    '''
    Solving the random walk equation with the Euler-Maruyama scheme:

    dz = K'(z, t)*dt + sqrt(2K(z,t))*dW

    See Visser (1997) and Gräwe (2011) for details.

    K:  diffusivity as a function of depth (m**2/s)
    z:  current particle depth (m)
    t:  current time (s)
    dt: timestep (s)
    '''
    dW = np.random.normal(loc = 0, scale = np.sqrt(dt), size = z.size)
    dKdz = ddz(K, z, t)
    return z + dKdz*dt + np.sqrt(2*K(z, t))*dW


####################
#### Rise speed ####
####################

def rise_speed(d, rho):
    '''
    Calculate the rise speed (m/s) of a droplet due to buoyancy.
    This scheme uses Stokes' law at small Reynolds numbers, with
    a harmonic transition to a constant drag coefficient at high
    Reynolds numbers.

    See Johansen (2000), Eq. (14) for details.

    d: droplet diameter (m)
    rho: droplet density (kg/m**3)
    '''
    # Physical constants
    pref  = 1.054       # Numerical prefactor
    nu    = CONST.nu    # Kinematic viscosity of seawater (m**2/s)
    rho_w = CONST.rho_w # Density of seawater (kg/m**3)
    g = CONST.g         # Acceleration of gravity (m/s**2)

    g_    = g*(rho_w - rho) / rho_w # g prime, reduced gravity
    if g_ == 0.0:
        return 0.0*d
    else:
        w1    = d**2 * g_ / (18*nu)
        w2    = np.sqrt(d*abs(g_)) * pref * sign(g_) # Last bracket sets sign
        return w1*w2/(w1+w2)


###########################
#### Utility functions ####
###########################

def advect(z, v, dt):
    '''
    Return the rise in meters due to buoyancy,
    assuming constant speed (at least within a timestep).

    z: current droplet depth (m)
    dt: timestep (s)
    d: droplet diameter (m)
    rho: droplet density (kg/m**3)
    '''
    return z + v*dt

def reflect(z, zmax = None):
    '''
    Reflect from surface.
    Depth is positive downwards.

    z: current droplet depth (m)
    '''
    # Reflect from surface
    z = np.abs(z)
    # Optionally reflect from bottom
    # (depth positive downwards)
    if zmax is not None:
        z[z > zmax] = 2*zmax - z[z > zmax]
    return z

def surface(z, v):
    '''
    Remove surfaced elements.
    This method shortens the array by removing surfaced particles.

    z: current droplet depth (m)
    d: droplet diameter (m)
    '''
    # create mask which is True for particles at or below surface
    # (depth positive downwards)
    mask = z >= 0.0
    return z[mask], v[mask]

def settle(z, v, zmax):
    '''
    Remove elements that have settled onto the sediment.
    This method shortens the array by removing settled particles.

    z: current droplet depth (m)
    d: droplet diameter (m)
    '''
    # create mask which is True for particles at or above bottom
    # (depth positive downwards)
    mask = z <= zmax
    return z[mask], v[mask]


###########################################
#### Main function to run a simulation ####
###########################################

def experiment(Z0, V0, Tmax, dt, bins, Zmax, K, surfacing=False, settling=False):
    '''
    Run the model.
    Returns histograms (concentration profile) and the first moment as functions of time,
    and the particle positions at the end of the simulation.

    Z0: initial depth of particles (m)
    V0: terminal velocity of particles (m/s)
    Tmax: total duration of the simulation (s)
    dt: timestep (s)
    bins: bins for histograms (concentration profiles)
    K: diffusivity-function on form K(z, t)
    '''
    # Number of particles
    Np = len(Z0)
    # Number of timesteps
    Nt = int(Tmax / dt) + 1
    # Arrays for z-position (depth) and rising/settling speed
    Z  = Z0.copy()
    V  = V0.copy()
    # Array to store histograms (concentration profiles)
    H  = np.zeros((Nt, len(bins)-1))
    # Array to store moment
    M  = np.zeros(Nt)

    # Time loop
    t = 0
    for i in range(Nt):
        # Store histogram
        H[i,:] = np.histogram(Z, bins)[0]
        # Store first moment
        M[i] = np.mean(Z)
        # Random displacement
        Z = randomstep(K, Z, t, dt)
        # Reflect from surface and seabed
        Z = reflect(Z, zmax = Zmax)
        # Rise/sink due to buoyancy
        Z = advect(Z, V, dt)

        if surfacing:
            # Remove surfaced (applies to depth and velocity arrays)
            Z, V = surface(Z, V)
        else:
            # Ensure droplets are not above the surface
            Z = np.maximum(Z, 0.0)

        if settling:
            # Remove settled (applies to depth and velocity arrays)
            Z, V = settle(Z, V)
        else:
            # Ensure droplets are not below the seafloor
            Z = np.minimum(Z, Zmax)

        # Increment time
        t = dt*i
    return H, M, Z



##############################
#### Numerical parameters ####
##############################

# Number of runs
Nruns  = 100
# Number of particles
Np     = 100000
# Total duration of the simulation in seconds
Tmax   = 8*3600
# timestep in seconds
dt     = 30
# Max depth
Zmax   = 50
# Bins for histograms (concentration profiles)
bins = np.linspace(0, Zmax, 101)


##############################
#### Diffusivity profiles ####
##############################

# Constant diffusivity
K0  = 1e-2
K_A = lambda z, t: K0 * np.ones_like(z)

# Analytical function which has been fitted to
# simulation results from the GOTM turbulence model
# with a wind stress corresponding to wind speed of about 9 m/s.
a, b, c, z0 = (0.00636, 0.088, 1.54, 1.3)
K_B = lambda z, t: a*(z+z0)*np.exp(-(b*(z+z0))**c)


############################
#### Initial conditions ####
############################

# Initial positions
Z0 = np.random.normal(loc=20, scale=4, size=Np)
# Rising speeds, drawn from Gaussian, then truncated by redrawing
# those speeds that are outside +/- 2*sigma
mu    = -0.96 * 1e-3 # z positive downwards, hence negative speed means rising particle
sigma =  0.38 * 1e-3
V0 = np.random.normal(loc = mu, scale = sigma, size = Np)
# Redraw speed for those particles that are outside range
for i in range(Np):
    while ( ((mu - 2*sigma) > V0[i]) or (V0[i] > (mu + 2*sigma)) ):
        V0[i] = np.random.normal(loc = mu, scale = sigma)


#########################
#### Run simulations ####
#########################

# Run repeatedly to obtain statistics on random variations
for i in range(Nruns):
    print(i)

    for diffusivityprofile, label in zip( (K_A, K_B), ('A', 'B')):

        # Run simulation
        H, M, Z = experiment(Z0, V0, Tmax, dt, bins, Zmax, diffusivityprofile, surfacing=False, settling=False)

        # Save output
        np.save(f'../../data/Case1_K_{label}_Lagrangian_concentration_Np={Np}_dt={dt}_{i:04}.npy', H)
        np.save(f'../../data/Case1_K_{label}_Lagrangian_moment_Np={Np}_dt={dt}_{i:04}.npy', M)
        np.save(f'../../data/Case1_K_{label}_Lagrangian_positions_Np={Np}_dt={dt}_{i:04}.npy', Z)
