from copy import deepcopy
from functools import partial
from itertools import product
from dataclasses import dataclass, asdict, astuple

import numpy as np
import pandas as pd
import jax.numpy as jnp

import jax
from jax import jacfwd

from .optimization_problem import barrier as barrier_nojax
from .optimization_problem import logdet as logdet_nojax
from .base import (LASSOInference,
                   PointWithSlices)
from .utils import (mle_summary,
                    regression_decomposition,
                    _compute_mle,
                    _obj_maker)
from .peaks import (get_gradient,
                    get_tangent_gradient,
                    get_normal_gradient,
                    get_hessian,
                    Peak,
                    Point,
                    extract_peaks,
                    extract_points)

DEBUG = False

class DiscreteLASSOInference(LASSOInference):

    def extract_peaks(self,
                      model_locations=[]): 

        Z, perturbation = self.Z, self.perturbation_
        E_nz = np.nonzero(self.E_)[0]

        if model_locations is None:
            model_locations = E_nz

        extra_points = set(model_locations).difference(E_nz)
        signs = np.sign(self.subgrad[self.E_])

        idx = 0
        data_peaks = []
        random_peaks = []
        extra_points_ = []
        
        for s, e in zip(signs, E_nz):
            data_peak = Peak(value=Z[e],
                             location=[e],
                             sign=s,
                             penalty=self.penalty[e])
            data_peaks.append(PointWithSlices(point=data_peak,
                                              value_idx=idx,
                                              gradient_slice=None,
                                              hessian_slice=None))
            idx += 1
            
            random_peak = Point(value=perturbation[e],
                                location=[e])
            random_peaks.append(PointWithSlices(point=random_peak,
                                                value_idx=idx,
                                                gradient_slice=None,
                                                hessian_slice=None))
            idx += 1
            

        for p in extra_points:
            extra_point = Point(value=Z[p],
                                location=[p])
            extra_points_.append(PointWithSlices(point=extra_point,
                                                 value_idx=idx,
                                                 gradient_slice=None,
                                                 hessian_slice=None))
            idx += 1
            
        self._data_peaks = data_peaks
        self._random_peaks = random_peaks
        self._extra_points = extra_points_
        self._model_points = data_peaks + extra_points_
        self._model_locations = [(l,) for l in model_locations] # for sorting summary df later
        
        return E_nz
    
    def setup_inference(self,
                        inactive):

        self.inactive = inactive

        cov = self.cov = self._compute_cov()
        self.prec = np.linalg.inv(self.cov)
        
        # form the T / N matrices for the decomposition
        # T is a selector matrix for all sufficient stats in the model
        # N is empty for discrete LASSO

        T = []
        for p in self._model_points:
            value_col = np.zeros((cov.shape[0], 1))
            p.set_value(value_col, 1)
            T.append(value_col)

        T = np.concatenate(T, axis=1).T

        # now specify the value the gradient is pinned at
        
        N = np.zeros((0, T.shape[1]))

        self.regress_decomp = regression_decomposition(cov, T, N) 

        # compute the first order data

        self.first_order = np.zeros(cov.shape[0])
        for p in self._model_points + self._random_peaks:
            p.set_value(self.first_order, p.point.value)

        locations = [p.point.location for p in self._random_peaks]
        self.barrier_info = self._form_barrier()

    def _solve_MLE(self):
        
        barrier, G_barrier, N_barrier = self.barrier_info

        (T,
         N,
         L_beta,
         L_NZ,
         est_matrix,
         sqrt_cov_R,
         cov_beta_T,
         cov_beta_TN) = astuple(self.regress_decomp)

        first_order = self.first_order
        offset = L_NZ @ N @ first_order
        beta_nosel = est_matrix @ first_order
        R = first_order - offset - L_beta @ beta_nosel
        initial_W = np.linalg.inv(sqrt_cov_R.T @ sqrt_cov_R) @ sqrt_cov_R.T @ R

        if initial_W.shape[0] != (0,):

            N_barrier_ = N_barrier + G_barrier @ (offset + L_beta @ beta_nosel)
            G_barrier_ = G_barrier @ sqrt_cov_R

            B_ = _obj_maker(barrier,
                            offset,
                            L_beta,
                            sqrt_cov_R)

            obj_jax = lambda beta, W: B_(beta, W) 
            grad_jax = jacfwd(obj_jax, argnums=(0,1))
            hess_jax = jacfwd(grad_jax, argnums=(0,1))

            val_ = lambda W: obj_jax(beta_nosel, W)
            grad_ = lambda W: grad_jax(beta_nosel, W)[1]
            hess_ = lambda W: hess_jax(beta_nosel, W)[1][1]

            W_star = _compute_mle(initial_W,
                                  val_,
                                  grad_,
                                  hess_)

            mle = beta_nosel + cov_beta_TN @ grad_jax(beta_nosel, W_star)[0]

            I = np.identity(W_star.shape[0])
            H_full = hess_jax(beta_nosel, W_star)
            observed_info = (np.linalg.inv(cov_beta_TN) +
                             H_full[0][0] - H_full[0][1] @ np.linalg.inv(I + H_full[1][1]) @ H_full[1][0])
            mle_cov = cov_beta_TN @ observed_info @ cov_beta_TN
        else:
            mle = beta_nosel
            mle_cov = cov_beta_TN

        return mle, mle_cov

    def summary(self,
                level=0.90,
                one_sided=True,
                location=False,
                param=None):

        if param is None:
            param = np.zeros(len(self._data_peaks) + len(self._extra_points))

        mle, mle_cov = self._solve_MLE()
            
        height_mle, loc_mle = mle[:len(self._data_peaks)], mle[len(self._data_peaks):]
        peaks = self._data_peaks
        height_param = param[:len(self._data_peaks)]
        height_SD = np.sqrt(np.diag(mle_cov)[:len(self._data_peaks)])
        height_Z = (height_mle - height_param) / height_SD

        signs = np.array([p.point.sign for p in self._data_peaks])

        df = pd.DataFrame({'Location':[tuple(p.point.location) for p in self._data_peaks],
                           'Estimate':height_mle,
                           'SD':height_SD,
                           'Param':height_param})
        if one_sided:
            df = mle_summary(mle,
                             np.sqrt(np.diag(mle_cov)),
                             param=param,
                             signs=signs,
                             level=level)
        else:
            df = mle_summary(mle,
                             np.sqrt(np.diag(mle_cov)),
                             param=param,
                             level=level)
        df['Location'] = [tuple(p.point.location) for p in self._model_points]

        df = df.set_index('Location')
        return df.loc[self._model_locations]
    
    # private methods

    def _compute_cov(self):
        cov_size = len(self._data_peaks) + len(self._random_peaks) + len(self._extra_points)
        cov = np.zeros((cov_size, cov_size))

        IK = self.inference_kernel
        RK = self.randomizer_kernel

        # first fill in values for the covariance of the data

        for peaks_l, peaks_r, K in [(self._model_points, self._model_points, IK),
                                    (self._random_peaks, self._random_peaks, RK)]:
            for p_l in peaks_l:
                for p_r in peaks_r:
                    cov[p_l.value_idx, p_r.value_idx] = K.C00([p_l.point.location],
                                                              [p_r.point.location])
           
        return cov

    def _form_barrier(self):

        # decompose the KKT conditions

        penalty, inactive = self.penalty, self.inactive

        ((L_inactive, offset_inactive),
         (L_active, offset_active)) = self._decompose_KKT()

        # active KKT condtion is L_active @ obs + offset_active >= 0

        L_inactive /= (penalty[self.inactive,None] / 2)
        offset_inactive /= (penalty[self.inactive] / 2)

        GI_barrier = np.concatenate([L_inactive,
                                     -L_inactive], axis=0)
        NI_barrier = np.hstack([2 + offset_inactive,
                                2 - offset_inactive])

        def barrierI(G_barrier,
                     N_barrier,
                     obs):

            arg = G_barrier @ obs + N_barrier
            if DEBUG:
                if jnp.any(arg < 0):
                    jax.debug.print('INACTIVE BARRIER {}', arg[arg<0])
            val = -jnp.mean(jnp.log(arg / (arg + 0.5)))
            return val

        def barrierA(G_barrier,
                     N_barrier,
                     obs):

            arg = G_barrier @ obs + N_barrier
            if DEBUG:
                if jnp.any(arg < 0):
                    jax.debug.print('ACTIVE BARRIER {}', arg[arg<0])
            return -jnp.sum(jnp.log(arg / (arg + 1)))

        V = L_active @ self.first_order + offset_active

        barrierI_ = partial(barrierI,
                            GI_barrier,
                            NI_barrier)

        barrierA_ = partial(barrierA,
                            L_active,
                            offset_active)

        def barrier(barrierA,
                    barrierI,
                    arg):
            return barrierA(arg) + barrierI(arg)

        return (partial(barrier, barrierI_, barrierA_),
                np.concatenate([GI_barrier,
                                L_active], axis=0),
                np.hstack([NI_barrier, offset_active]))

    # decompose the subgradient process in terms of all the targets in cov

    def _decompose_KKT(self):

        MK = self.model_kernel
        RK = self.randomizer_kernel
        IK = self.inference_kernel

        locations = [p.point.location for p in self._data_peaks]
        C00 = (MK.C00(locations, locations) +
               RK.C00(locations, locations))
        C00i = np.linalg.inv(C00)

        irrep = (MK.C00(None,
                        locations) +
                 RK.C00(None,
                        locations))[self.inactive] @ C00i

        pre_proj = np.zeros(self.cov.shape[:1] + self.subgrad[self.inactive].shape[:1])

        # fill in covariance with data and extra points

        for peaks, K in [(self._model_points, IK),
                         (self._random_peaks, RK)]:
            for p in peaks:
                p.set_value(pre_proj, K.C00([p.point.location],
                                            None)[0, self.inactive])

        L_inactive = self.prec @ pre_proj

        for i, (p_d, p_r) in enumerate(zip(self._data_peaks,
                                           self._random_peaks)):
            p_d.set_value(L_inactive, p_d.get_value(L_inactive) - irrep[:,i])
            p_r.set_value(L_inactive, p_r.get_value(L_inactive) - irrep[:,i])

        L_inactive = L_inactive.T
        offset_inactive = self.subgrad[self.inactive] - L_inactive @ self.first_order

        signs = np.array([p.point.sign for p in self._data_peaks])
        L_active = np.zeros((self.cov.shape[0], len(self._data_peaks)))

        for i, (p_d, p_r) in enumerate(zip(self._data_peaks,
                                           self._random_peaks)):
            p_d.set_value(L_active, C00i[i])
            p_r.set_value(L_active, C00i[i])

        L_active = L_active.T
        offset_active = -C00i @ np.array([p.point.penalty * p.point.sign for p in self._data_peaks])
        L_active = (np.diag(signs) @ L_active) / np.sqrt(np.diag(C00i))[:,None]
        offset_active = signs * offset_active / np.sqrt(np.diag(C00i))

        return ((L_inactive, offset_inactive),
                (L_active, offset_active))

