#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing standard packages
import os
import time
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
from webernaturaldispersion import weber_natural_dispersion

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
parser.add_argument('--dt', dest = 'dt', type = float, default = 60, help = 'Timestep')
parser.add_argument('--NJ', dest = 'NJ', type = int, default = 1000, help = 'Number of grid cells')
parser.add_argument('--NK', dest = 'NK', type = int, default = 8, help = 'Number of speed classes')
parser.add_argument('--profile', dest = 'profile', type = str, default = 'A', choices = ['A', 'B'], help = 'Diffusivity profiles')
parser.add_argument('--checkpoint', dest = 'checkpoint', type = bool, default = False, help = 'Save results for checkpointing at every output timestep?')
args = parser.parse_args()


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
outputfilename = os.path.join(datafolder, f'Case3_K_{label}_block_Nclasses={params.Nclasses}_NJ={params.Nz}_dt={params.dt}.npy')

tic = time.time()
c = Crank_Nicolson_FVM_TVD_advection_diffusion_reaction(C0, K, params, outputfilename = outputfilename)
toc = time.time()
print(f'Simulation took {toc - tic:.1f} seconds, output written to {outputfilename}')

np.save(outputfilename, c)

