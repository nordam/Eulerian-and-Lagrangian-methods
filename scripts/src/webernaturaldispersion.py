#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

def weber_natural_dispersion(rho, mu, ift, Hs, h):
    '''
    Weber natural dispersion model. Predicts median droplet size D50 (m).
    Johansen, 2015.

    rho: oil density (kg/m**3)
    mu: dynamic viscosity of oil (kg/m/s)
    ift: oil-water interfacial tension (N/m, kg/s**2)
    Hs: free-fall height or wave amplitude (m)
    h: oil film thickness (m)
    '''
    # Physical parameters
    g = 9.81     # Acceleration of gravity (m/s**2)

    # Model parameters from Johansen 2015 (fitted to experiment)
    A = 2.251
    Bm = 0.027
    a = 0.6

    # Velocity scale
    U = np.sqrt(2*g*Hs)

    # Calculate relevant dimensionless numbers for given parameters
    We = rho * U**2 * h / ift
    # Note that Vi = We/Re
    Vi = mu * U / ift

    # Calculate D, characteristic (median) droplet size predicted by WMD model
    WeMod = We / (1 + Bm * Vi**a)**(1/a)
    D50n = h * A / WeMod**a

    return D50n
