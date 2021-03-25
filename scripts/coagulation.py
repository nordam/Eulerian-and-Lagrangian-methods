#!/usr/bin/env python
# -*- coding: utf-8 -*-


from numba import njit

@njit
def kernel(r1, r2):
    # This function should return collision rate,
    # which is a function of particle sizes
    return 1e-2

@njit
def coagulation_rate_prefactor(r1, r2):
    # This function should return coagulation rate prefactor,
    # which consists of collision rate and probability of sticking
    alpha = 1
    beta = kernel(r1, r2)
    return alpha*beta

@njit
def get_new_radius(r1, r2):
    fdim = 3 # fractal dimension
    return (r1**fdim + r2**fdim)**(1/fdim)

@njit
def get_new_classes_and_weights(j, k, radii):
    r_new = get_new_radius(radii[j], radii[k])
    # Find closest matching classes
    m2 = np.searchsorted(radii, r_new)
    m1 = m2 - 1
    # If m2 is above the available range, put everything in class m1
    # (by construction, we can never end up below the smallest class)
    if m2 == len(radii):
        w2 = 0.0
        w1 = 1.0
    else:
        # Find weights for distribution between classes m1 and m2, on log scale
        w2 = (np.log(radii[m1]) - np.log(r_new)) / (np.log(radii[m1]) - np.log(radii[m2]))
        w1 = 1 - w2
    return m1, w1, m2, w2
