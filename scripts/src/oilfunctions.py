#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from wavefunctions import Fbw
from constants import CONST

def entrainmentrate(rho, mu, ift, Hs, Tp, windspeed):
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
