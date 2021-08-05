#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from collections import namedtuple


############################
#### Physical constants ####
############################
PhysicalConstants = namedtuple('PhysicalConstants', ('g', 'rho_w', 'nu'))
CONST = PhysicalConstants(g=9.81,      # Acceleration due to gravity (m/s**2)
                          rho_w=1025,  # Density of sea water (kg/m**3)
                          nu=1.358e-6, # Kinematic viscosity of sea water (m**2/s)
                         )


###############################
#### Numerical derivatives ####
###############################

def ddz(K, z, t):
    '''
    Numerical derivative of K(z, t).

    This function calculates a numerical partial derivative
    of K(z, t), with respect to z using forward finite difference.

    K: diffusivity as a function of depth (m**2/s)
    z: current particle depth (m)
    t: current time (s)
    '''
    dz = 1e-6
    return (K(z+dz/2, t) - K(z-dz/2, t)) / dz


############################
#### Random walk scheme ####
############################

def correctstep(K, z, t, dt):
    '''
    Solving the corrected equation with the Euler-Maruyama scheme:

    dz = K'(z, t)*dt + sqrt(2K(z,t))*dW

    See Visser (1997) and Gräwe (2011) for details.

    K: diffusivity as a function of depth (m**2/s)
    z: current particle depth (m)
    t: current time (s)
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

    g_    = g*(rho_w - rho) / rho_w
    if g_ == 0.0:
        return 0.0*d
    else:
        w1    = d**2 * g_ / (18*nu)
        w2    = np.sqrt(d*abs(g_)) * pref * (g_/np.abs(g_)) # Last bracket sets sign
        return w1*w2/(w1+w2)


###########################
#### Utility functions ####
###########################

def advect(z, v, dt):
    '''
    Return the rise in meters due to buoyancy, 
    assuming constant speed (at least within a timestep).

    z: current droplet depth (m)
    v: droplet speed, positive downwards (m/s)
    dt: timestep (s)
    '''
    return z + dt*v

def reflect(z):
    '''
    Reflect from surface.
    Depth is positive downwards.

    z: current droplet depth (m)
    '''
    # Reflect from surface
    z = np.abs(z)
    return z

def surface(z, d):
    '''
    Remove surfaced elements.
    This method shortens the array by removing surfaced particles.

    z: current droplet depth (m)
    d: droplet diameter (m)
    '''
    # Keep only particles at depths greater than 0
    mask = z >= 0.0
    return z[mask], d[mask]

def settle(z, arr, Zmax):
    '''
    Remove elements that settle to the bottom.
    This method shortens the array by removing settled particles.

    z: current droplet depth (m)
    arr: droplet diameter, settling speed, or other per-particle property
    Zmax: Maximal depth
    '''
    # Keep only particles at depths smaller than Zmax
    mask = z <= Zmax
    return z[mask], arr[mask]
