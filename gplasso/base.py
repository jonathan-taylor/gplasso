from typing import Union, Optional, Any
from copy import deepcopy
from functools import partial
from itertools import product
from dataclasses import dataclass

import numpy as np
from numpy.random import Generator, RandomState
import pandas as pd
import jax.numpy as jnp
from scipy.stats import norm as normal_dbn
from scipy.stats import chi2

import jax
from jax import jacfwd

from .fit_gplasso import fit_gp_lasso
from .kernels import covariance_structure

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

    def get_value(self,
                  arr: np.ndarray):
        """
        Parameters
        ----------
        self : [Argument]
        arr : np.ndarray [Argument]

        """
        return arr[self.value_idx]

    def set_value(self,
                  arr: np.ndarray,
                  val: np.ndarray):
        """
        Parameters
        ----------
        self : [Argument]
        arr : np.ndarray [Argument]
        val : np.ndarray [Argument]

        """
        arr[self.value_idx] = val

    def get_gradient(self,
                     arr: np.ndarray):
        """
        Parameters
        ----------
        self : [Argument]
        arr : np.ndarray [Argument]

        """
        return arr[self.gradient_slice]

    def set_gradient(self,
                     arr: np.ndarray,
                     grad: np.ndarray):
        """
        Parameters
        ----------
        self : [Argument]
        arr : np.ndarray [Argument]
        grad : np.ndarray [Argument]

        """
        arr[self.gradient_slice] =  grad

    def get_hessian_block(self,
                          arr: np.ndarray):
        """
        Parameters
        ----------
        self : [Argument]
        arr : np.ndarray [Argument]

        """
        return arr[self.hessian_slice]

@dataclass
class LASSOInference(object):

    penalty: np.ndarray
    model_kernel: covariance_structure
    randomizer_kernel: covariance_structure
    inference_kernel: Optional[covariance_structure] = None
    
    def __post_init__(self):
        if self.inference_kernel is None:
            self.inference_kernel = self.model_kernel

    def fit(self,
            Z: np.ndarray,
            perturbation: Optional[np.ndarray] = None,
            rng: Optional[Union[Generator, RandomState]] = None):
        """
        Parameters
        ----------
        self : [Argument]
        Z : np.ndarray [Argument]
        perturbation : Optional[np.ndarray], optional, default: None [Argument]
        rng : Optional[], optional, default: None [Argument]

        """

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
                        inactive: np.ndarray,
                        model_spec: Optional[Any]=[],
                        inference_kernel: Optional[covariance_structure]=None):
        """
        Parameters
        ----------
        self : [Argument]
        inactive : np.ndarray [Argument]
        model_spec : Optional[Any] [Argument]

        """

        raise NotImplementedError('abstract method')

    def summary(self,
                level: float = 0.90,
                param: Optional[Any]=None):
        """
        Parameters
        ----------
        self : [Argument]
        level : float, optional, default: 0.9 [Argument]
        param : Optional[Any], optional, default: None [Argument]

        """

        raise ValueError('abstract method')
