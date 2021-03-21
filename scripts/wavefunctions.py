#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

################################
#### Wave-related functions ####
################################

def waveperiod(windspeed, fetch):
    '''
    Calculate the peak wave period (s)

    Based on the JONSWAP model and associated empirical relations.
    See Carter (1982) for details

    windspeed: windspeed (m/s)
    fetch: fetch (m)
    '''
    # Avoid division by zero
    if windspeed > 0:
        # Constants for the JONSWAP model:
        g       = 9.81    # Acceleration of gravity
        t_const = 0.286   # Nondimensional period constant.
        t_max   = 8.134   # Nondimensional period maximum.

        # Calculate wave period
        t_nodim = t_const * (g * fetch / windspeed**2)**(1./3.)
        waveperiod = np.minimum(t_max, t_nodim) * windspeed / g
    else:
        waveperiod = 1.0
    return waveperiod

def waveheight(windspeed, fetch):
    '''
    Calculate the significant wave height (m)

    Based on the JONSWAP model and associated empirical relations.
    See Carter (1982) for details

    windspeed: windspeed (m/s)
    fetch: fetch (m)
    '''
    # Avoid division by zero
    if windspeed > 0:
        # Constants for the JONSWAP model:
        g       = 9.81    # Acceleration of gravity
        h_max   = 0.243   # Nondimensional height maximum.
        h_const = 0.0016  # Nondimensional height constant.

        # Calculate wave height
        h_nodim = h_const * np.sqrt(g * fetch / windspeed**2)
        waveheight = np.minimum(h_max, h_nodim) * windspeed**2 / g
    else:
        waveheight = 0.0
    return waveheight

def jonswap(windspeed, fetch = 170000):
    '''
    Calculate wave height and wave period, from wind speed
    and fetch length. A large default fetch means assuming
    fully developed sea (i.e., not fetch-limited).

    See Carter (1982) for details.

    windspeed: windspeed (m/s)
    fetch: fetch length (m)
    '''
    Hs = waveheight(windspeed, fetch)
    Tp = waveperiod(windspeed, fetch)
    return Hs, Tp

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
