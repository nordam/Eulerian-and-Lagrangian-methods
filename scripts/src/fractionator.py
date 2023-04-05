#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.stats import lognorm
from particlefunctions import rise_speed

class Fractionator():
    '''
    Thinking:
    We use a pre-defined set of classes to represent rise speeds.
    When the speed classes are fixed, we can calculate a fixed mapping from
    speed class to size class.
    The mass fraction that goes into each size class can be calculated
    quasi-analytically from the log-normal distribution of sizes.
    '''

    def __init__(self, speed_class_edges, rho, romberg = True, Nsub = 2**5+1):
        '''
        Takes as input the bin edges for the speed classes.
        From this, the midpoints are calculated.
        '''

        from scipy.interpolate import UnivariateSpline

        self.Nsub = Nsub
        self.romberg = romberg
        self.speed_class_edges = speed_class_edges
        self.speeds = np.sqrt(speed_class_edges[1:]*speed_class_edges[:-1])

        # Use cubic splines to create numerical inverse
        # of rise speed function, to go from speed to size.
        sizes_tmp   = np.logspace(-9, 2, 100000)
        speeds_tmp  = rise_speed(sizes_tmp, rho)
        # k gives degree, s=0 ensures no smoothing
        inverse = UnivariateSpline(np.abs(speeds_tmp), sizes_tmp, k=3, s=0)

        # Find bins in size distribution, from bins in speed distribution
        size_class_edges     = inverse(speed_class_edges)
        size_class_midpoints = np.sqrt(size_class_edges[1:] * size_class_edges[:-1])
        size_class_widths    = size_class_edges[1:] - size_class_edges[:-1]

        if self.romberg:
            # Calculate mass fractions in each size class
            # by numerical integration (Romberg). Use a subdivision
            # of each class into Nsub _equally spaced_ intervals.
            # Calculate the evaluation points here, and store for later use.
            # Note that romberg integration uses evaluations at the end points,
            # not at the midpoints, as is the case for a Riemann sum.
            Nclasses = len(self.speeds)
            evaluation_points = np.zeros((Nclasses, Nsub))
            evaluation_widths = np.zeros( Nclasses )
            for i in range(Nclasses):
                evaluation_points[i,:] = np.linspace(size_class_edges[i], size_class_edges[i+1], Nsub)
                evaluation_widths[i]   = evaluation_points[i,1] - evaluation_points[i,0]

            self.evaluation_points = evaluation_points
            self.evaluation_widths = evaluation_widths
        else:
            # Calculate mass fractions in each size class
            # by numerical integration (Riemann sum). Use a subdivision
            # of each class into Nsub intervals. Calculate the evaluation
            # points here, and store for later use.
            Nclasses = len(self.speeds)
            evaluation_points = np.zeros((Nclasses, Nsub))
            evaluation_widths = np.zeros( Nclasses )
            for i in range(Nclasses):
                evaluation_point_edges = np.logspace(np.log10(size_class_edges[i]), np.log10(size_class_edges[i+1]), Nsub+1)
                evaluation_points[i,:] = np.sqrt(evaluation_point_edges[1:]*evaluation_point_edges[:-1])
                evaluation_widths[i,:] = evaluation_point_edges[1:] - evaluation_point_edges[:-1]

            self.evaluation_points = evaluation_points
            self.evaluation_widths = evaluation_widths

    def evaluate(self, D50v, sigma):
        '''
        Calculates mass fractions to go into each speed class,
        based on a given D50v and sigma for the droplet size distribution.
        '''
        from scipy.integrate import romb
        # Evaluate log-normal density in size class centers
        pdf = lambda d : np.exp(-(np.log(d) - np.log(D50v))**2 / (2*sigma**2)) / (sigma*d*np.sqrt(2*np.pi))
        if self.romberg:
            fractions = [romb(pdf(self.evaluation_points[i,:]), dx = self.evaluation_widths[i]) for i in range(len(self.speeds))]
        else:
            fractions = np.sum(pdf(self.evaluation_points)*self.evaluation_widths, axis = 1)
        # The sum of these fractions may have an error on the order of 1e-3.
        # Normalise fractions to sum to 1.
        fractions = fractions / np.sum(fractions)
        return fractions
