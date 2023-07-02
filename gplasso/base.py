from copy import deepcopy
from functools import partial
from itertools import product
from dataclasses import dataclass

import numpy as np
import pandas as pd
import jax.numpy as jnp
from scipy.stats import norm as normal_dbn
from scipy.stats import chi2

import jax
from jax import jacfwd

from .fit_gplasso import fit_gp_lasso

from .peaks import (get_gradient,
                    get_tangent_gradient,
                    get_normal_gradient,
                    get_hessian,
                    Peak,
                    Point,
                    extract_peaks,
                    extract_points)

DEBUG = False

@dataclass
class PointWithSlices(object):

    point: Point
    value_idx: int    # index into col of cov for value coords
    gradient_slice: slice # index into col of cov for gradient coords
    hessian_slice: slice # index into row/col of Hessian for each peak

    def get_value(self, arr):
        return arr[self.value_idx]

    def set_value(self, arr, val):
        arr[self.value_idx] = val

    def get_gradient(self, arr):
        return arr[self.gradient_slice]

    def set_gradient(self, arr, grad):
        arr[self.gradient_slice] =  grad

    def get_hessian_block(self, arr):
        return arr[self.hessian_slice]

class LASSOInference(object):

    def __init__(self,
                 penalty,
                 model_kernel,
                 randomizer_kernel,
                 inference_kernel=None):
       
        if inference_kernel is None:
            inference_kernel = model_kernel

        (self.penalty,
         self.model_kernel,
         self.randomizer_kernel,
         self.inference_kernel) = (penalty,
                                   model_kernel,
                                   randomizer_kernel,
                                   inference_kernel)

    def fit(self,
            Z,
            perturbation=None,
            rng=None):
        
        # fit the GP lasso
        if perturbation is None:
            perturbation = self.randomizer_kernel.sample(rng=rng)
        self.Z, self.perturbation_ = Z, perturbation
        MK, RK = self.model_kernel, self.randomizer_kernel
        E, soln, subgrad = fit_gp_lasso(self.Z + self.perturbation_,
                                        [MK, RK],
                                        self.penalty)

        self.E_, self.soln, self.subgrad = E, soln, subgrad
        
        return E, soln, subgrad

    def setup_inference(self,
                        inactive,
                        model_spec=[]):

        raise NotImplementedError('abstract method')

    def summary(self,
                level=0.90,
                param=None):

        raise ValueError('abstract method')
