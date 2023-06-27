from copy import deepcopy
from functools import partial
from itertools import product
from typing import NamedTuple

import numpy as np
import pandas as pd
import jax.numpy as jnp
from scipy.stats import norm as normal_dbn
from scipy.stats import chi2

import jax
from jax import jacfwd

from .kernel_calcs import covariance_structure
from .optimization_problem import barrier as barrier_nojax
from .optimization_problem import logdet as logdet_nojax
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

class DisplacementEstimate(NamedTuple):

    location: np.ndarray
    segment: np.ndarray
    cov: np.ndarray
    factor: float
    quantile: float

class RegressionInfo(NamedTuple):

    T: np.ndarray
    N: np.ndarray
    L_beta: np.ndarray
    L_NZ: np.ndarray
    est_matrix: np.ndarray
    sqrt_cov_R: np.ndarray
    cov_beta_T: np.ndarray
    cov_beta_TN: np.ndarray
    
class PointWithSlices(NamedTuple):

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

class DiscreteLASSOInference(object):

    def __init__(self,
                 Z,
                 penalty,
                 model_kernel,
                 randomizer_kernel,
                 inference_kernel=None):
       
        if inference_kernel is None:
            inference_kernel = model_kernel

        (self.Z,
         self.penalty,
         self.model_kernel,
         self.randomizer_kernel,
         self.inference_kernel) = (Z,
                                   penalty,
                                   model_kernel,
                                   randomizer_kernel,
                                   inference_kernel)

    def fit(self,
            perturbation=None,
            rng=None):
        
        # fit the GP lasso
        if perturbation is None:
            perturbation = self.randomizer_kernel.sample(rng=rng)
        self.perturbation_ = perturbation
        MK, RK = self.model_kernel, self.randomizer_kernel
        E, soln, subgrad = fit_gp_lasso(self.Z + self.perturbation_,
                                        [MK, RK],
                                        self.penalty)

        return E, soln, subgrad

    def extract_peaks(self,
                      E,
                      signs,
                      Z,
                      perturbation): # rng for choosing a representative from a cluster

        E_nz = np.nonzero(E)
        npt = E.sum()
        second_order = []
        for i in E_nz[0]:
            second_order.append((np.array([Z[i], perturbation[i]]),
                                 np.zeros((2,0)),
                                 np.zeros((2,0,0))))

        tangent_bases = [np.identity(0) for _ in range(npt)]
        normal_info = [(np.zeros((0, 0)), np.zeros((0, 0))) for _ in range(npt)]
        clusters = np.arange(npt)

        peaks, idx = extract_peaks(E_nz,
                                   clusters,
                                   second_order,
                                   tangent_bases,
                                   normal_info,
                                   self.model_kernel,
                                   signs,
                                   self.penalty,
                                   rng=None)

        return peaks, idx

    def setup_inference(self,
                        peaks,
                        inactive,
                        subgrad,
                        extra_points=[]):

        (self.inactive,
         self.subgrad) = (inactive,
                          subgrad)

        self.data_peaks = []
        self.random_peaks = []
        self.extra_points = []
        idx = 0
        hess_idx = 0
        for peak in peaks:
            ngrad = peak.gradient.shape[1]
            data_peak = peak._replace(value=peak.value[0],
                                      gradient=peak.gradient[0],
                                      hessian=peak.hessian[0],
                                      n_obs=1)
            data_peak_slice = PointWithSlices(point=data_peak,
                                              value_idx=idx,
                                              gradient_slice=slice(idx + 1,
                                                                   idx + 1 + ngrad),
                                              hessian_slice=slice(hess_idx, hess_idx + ngrad))
            idx += (1 + ngrad)
            hess_idx += ngrad
            
            random_peak = peak._replace(value=peak.value[1],
                                        gradient=peak.gradient[1],
                                        hessian=peak.hessian[1],
                                        n_obs=1)

            random_peak_slice = PointWithSlices(point=random_peak,
                                                value_idx=idx,
                                                gradient_slice=slice(idx + 1,
                                                                    idx + 1 + ngrad),
                                                hessian_slice=data_peak_slice.hessian_slice)
            self.data_peaks.append(data_peak_slice)
            self.random_peaks.append(random_peak_slice)

            idx += (1 + ngrad)

        for point in extra_points: # extra points should have no randomization
            ngrad = point.gradient.shape[0]
            extra_point_slice = PointWithSlices(point=point,
                                                value_idx=idx,
                                                gradient_slice=slice(idx + 1,
                                                                    idx + 1 + ngrad),
                                                hessian_slice=None)
            self.extra_points.append(extra_point_slice)
            idx += (1 + ngrad)

        cov_size = idx
        hess_size = hess_idx
        cov = self.cov = self._compute_cov(cov_size)
        self.prec = np.linalg.inv(self.cov)
        
        # we use our regression decomposition to rewrite this in terms of
        # our "best case" unbiased estimator (ie when no selection effect present)
        # the affine term in the Kac Rice formula and the corresponding residual of
        # (f_E, \omega_E, \nabla f_E, \nabla \omega_E) 

        C00i, M, G_blocks = _compute_random_model_cov(self.data_peaks,
                                                      self.model_kernel, 
                                                      self.randomizer_kernel)

        # form the T / N matrices for the decomposition

        # T is a selector matrix for all sufficient stats in the model
        # N is represents the random vector in the "0" of the Kac-Rice formula

        T = []
        for p in self.data_peaks + self.extra_points:
            value_col = np.zeros((cov.shape[0], 1))
            p.set_value(value_col, 1)

            # TODO, each peak / extra point should have a flag as to whether
            # their displacement is a parameter or not

            ngrad = get_gradient(p.point).shape[0]
            gradient_cols = np.zeros((cov.shape[0], ngrad))
            p.set_gradient(gradient_cols, np.identity(ngrad))
            T.append(np.concatenate([value_col, gradient_cols], axis=1).T)

        T = np.concatenate(T, axis=0)

        # now specify the value the gradient is pinned at
        
        N = []

        idx = 0 # restart counter
        for i, (p_d, p_r) in enumerate(zip(self.data_peaks, self.random_peaks)):
            ngrad = get_tangent_gradient(p_d.point).shape[0]
            N_cols = np.zeros((cov.shape[0], ngrad))
            p_d.set_gradient(N_cols, np.identity(ngrad))
            p_r.set_gradient(N_cols, np.identity(ngrad))
            p_d.set_value(N_cols, -p_d.get_hessian_block(M[i]))
            p_r.set_value(N_cols, -p_r.get_hessian_block(M[i]))

            N.append(N_cols)

        N = np.hstack(N).T

        # why isn't this used? hmm...
        NZ_obs = -M.T @ np.array([p.sign * p.penalty for p in peaks])

        self.regress_decomp = regression_decomposition(cov, T, N) 

        # compute the first order data

        self.first_order = np.zeros(cov.shape[0])
        for p in self.data_peaks + self.random_peaks + self.extra_points:
            p.set_value(self.first_order, p.point.value)
            p.set_gradient(self.first_order, get_gradient(p.point))


        self.logdet_info = self._form_logdet(G_blocks,
                                             C00i)
        self.barrier_info = self._form_barrier(C00i)

    def summary(self,
                level=0.90,
                displacement_level=0.90,
                one_sided=True,
                location=False,
                param=None):

        barrier, G_barrier, N_barrier = self.barrier_info
        logdet, G_logdet, N_logdet = self.logdet_info

        if param is None:
            param = np.zeros(len(self.data_peaks))

        (T,
         N,
         L_beta,
         L_NZ,
         est_matrix,
         sqrt_cov_R,
         cov_beta_T,
         cov_beta_TN) = self.regress_decomp

        first_order = self.first_order
        offset = L_NZ @ N @ first_order
        beta_nosel = est_matrix @ first_order
        R = first_order - offset - L_beta @ beta_nosel
        initial_W = np.linalg.inv(sqrt_cov_R.T @ sqrt_cov_R) @ sqrt_cov_R.T @ R

        N_barrier_ = N_barrier + G_barrier @ (offset + L_beta @ beta_nosel)
        G_barrier_ = G_barrier @ sqrt_cov_R

        N_logdet_ = N_logdet + G_logdet @ (offset + L_beta @ beta_nosel)
        G_logdet_ = G_logdet @ sqrt_cov_R

        def obj_maker_(obj,
                       offset,
                       L_beta,
                       L_W):

            def _new(offset,
                     L_beta,
                     L_W,
                     beta,
                     W):
                arg = offset + L_W @ W + L_beta @ beta
                return obj(arg)

            return partial(_new,
                           offset,
                           L_beta,
                           L_W)

        B_ = obj_maker_(barrier,
                         offset,
                         L_beta,
                         sqrt_cov_R)

        LD_ = obj_maker_(logdet,
                         offset,
                         L_beta,
                         sqrt_cov_R)


        L = logdet_nojax()
        B = barrier_nojax(scale=1, shift=1)

        obj_jax = lambda beta, W: B_(beta, W) + LD_(beta, W)
        grad_jax = jacfwd(obj_jax, argnums=(0,1))
        hess_jax = jacfwd(grad_jax, argnums=(0,1))

        val_ = lambda W: (B.value(G_barrier_ @ W + N_barrier_) +
                          L.value(G_logdet_ @ W + N_logdet_))
        grad_ = lambda W: (B.gradient(G_barrier_ @ W + N_barrier_, G_barrier_) +
                           L.gradient(G_logdet_ @ W + N_logdet_, G_logdet_))
        hess_ = lambda W: (B.hessian(G_barrier_ @ W + N_barrier_, G_barrier_, G_barrier_) +
                           L.hessian(G_logdet_ @ W + N_logdet_, G_logdet_, G_logdet_))
        # other derivatives if we don't use jax

        grad0_ = lambda W: (B.gradient(G_barrier_ @ W + N_barrier_, G_barrier_) +
                            L.gradient(G_logdet_ @ W + N_logdet_, G_logdet @ L_beta))
        hess11_ = hess_
        hess10_ = lambda W: (B.hessian(G_barrier_ @ W + N_barrier_, G_barrier_, G_barrier @ L_beta) +
                             L.hessian(G_logdet_ @ W + N_logdet_, G_logdet_, G_logdet @ L_beta))
        hess01_ = lambda W: hess10_(W).T
        hess00_ = lambda W: (B.hessian(G_barrier_ @ W + N_barrier_, G_logdet @ L_beta, G_barrier @ L_beta) +
                             L.hessian(G_logdet_ @ W + N_logdet_, G_logdet @ L_beta, G_logdet @ L_beta))
        W = initial_W.copy()
        I = np.identity(W.shape[0])

        use_jax = True
        num_newton = 20

        if W.shape != (0,): # for data splitting W has shape (0,)
            for i in range(num_newton):
                if DEBUG:
                    print('newton iterate {}'.format(i))
                if use_jax:
                    H = I + hess_jax(beta_nosel, W)[1][1]
                    G = W + grad_jax(beta_nosel, W)[1]
                else:
                    H = I + hess_(W)
                    G = W + grad_(W)

                # do a line search

                factor = 1
                niter = 0
                cur_val = np.inf
                step = np.linalg.inv(H) @ G

                if DEBUG:
                    jax.debug.print('grad {}', G)
                while True:
                    W_new = W - factor * step
                    if use_jax:
                        new_val = obj_jax(beta_nosel, W_new) + (W_new**2).sum() * 0.5
                    else:
                        new_val = val_(W_new) + (W_new**2).sum() * 0.5                
                    if new_val < cur_val:
                        break

                    factor *= 0.5
                    niter += 1
                    if niter >= 30:
                        raise ValueError('no descent')

                if np.linalg.norm(W - W_new) < 1e-9 * np.linalg.norm(W):
                    break

                W = W_new
                cur_val = new_val

            if DEBUG:
                print('W', W)

            mle = beta_nosel + cov_beta_TN @ grad_jax(beta_nosel, W)[0]

            H_full = hess_jax(beta_nosel, W)
            observed_info = (np.linalg.inv(cov_beta_TN) +
                             H_full[0][0] - H_full[0][1] @ np.linalg.inv(I + H_full[1][1]) @ H_full[1][0])
            mle_cov = cov_beta_TN @ observed_info @ cov_beta_TN

        else:
            mle = beta_nosel
            mle_cov = cov_beta_TN

        height_mle, loc_mle = mle[:len(self.data_peaks)], mle[len(self.data_peaks):]
        peaks = self.data_peaks
        height_param = param[:len(self.data_peaks)]
        height_SD = np.sqrt(np.diag(mle_cov)[:len(self.data_peaks)])
        height_Z = (height_mle - height_param) / height_SD

        if DEBUG:
            print(mle, 'mle')
            print(param, 'param')
            print(beta_nosel, 'no selection')

        signs = np.array([p.point.sign for p in self.data_peaks])
        P = normal_dbn.sf(height_Z * signs)
        df = pd.DataFrame({'Location':[tuple(p.point.location) for p in self.data_peaks],
                           'Estimate':height_mle,
                           'SD':height_SD,
                           'Param':height_param})
        if one_sided:
            df = mle_summary(height_mle,
                             height_SD,
                             param=height_param,
                             signs=signs,
                             level=level)
        else:
            df = mle_summary(height_mle,
                             height_SD,
                             param=height_param,
                             level=level)
        df['Location'] = [tuple(p.point.location) for p in self.data_peaks]

        # # now confidence regions for the peaks

        # loc_cov = mle_cov[len(peaks):,len(peaks):]
        # loc_mle = mle[len(peaks):]
        # loc_results = []

        # # should only output this for peaks where requested...
        # for i, (s, p) in enumerate(zip(info.slice_info.slices, info.peaks)):
        #     if hasattr(p, 'n_tangent') and len(loc_mle) > 0:
        #         df_p = df.iloc[i]
        #         up_lab = 'U ({:.0%})'.format(level)
        #         low_lab = 'L ({:.0%})'.format(level)
        #         if df_p[up_lab] * df_p[low_lab] > 0:
        #             factor = 1 / np.fabs([df_p[up_lab], df_p[low_lab]]).min()
        #         else:
        #             factor = np.inf
        #         n_spatial = p.n_tangent
        #         if hasattr(p, 'n_normal'):
        #             n_spatial += p.n_normal
        #         q = chi2.ppf(displacement_level, n_spatial)
        #         loc_results.append(DisplacementEstimate(location=p.location,
        #                                                 segment=np.array([loc_mle[s] / df_p[low_lab],
        #                                                                   loc_mle[s] / df_p[up_lab]]),
        #                                                 cov=loc_cov[s,s],
        #                                                 quantile=q,
        #                                                 factor=factor))
        #     else:
        #         loc_results.append(DisplacementEstimate(p.location, None, None, None, None))

        loc_results = None
        return df.set_index('Location'), loc_results

    def _compute_cov(self, cov_size):
        cov = np.zeros((cov_size, cov_size))

        IK = self.inference_kernel
        RK = self.randomizer_kernel

        # first fill in values for the covariance of the data

        for peaks_l, peaks_r, K in [(self.data_peaks + self.extra_points,
                                     self.data_peaks + self.extra_points, IK),
                                    (self.random_peaks, self.random_peaks, RK)]:
            for p_l in peaks_l:
                for p_r in peaks_r:
                    cov[p_l.value_idx, p_r.value_idx] = K.C00([p_l.point.location],
                                                              [p_r.point.location])
                    cov[p_l.gradient_slice, p_r.value_idx] = K.C10([p_l.point.location],
                                                                   [p_r.point.location],
                                                                   basis_l=p_l.point.tangent_basis)[0,0]
                    cov[p_l.value_idx, p_r.gradient_slice] = K.C01([p_l.point.location],
                                                                   [p_r.point.location],
                                                                   basis_r=p_r.point.tangent_basis)[0,0]
                    cov[p_l.gradient_slice, p_r.gradient_slice] = K.C11([p_l.point.location],
                                                                        [p_r.point.location],
                                                                        basis_l=p_l.point.tangent_basis,
                                                                        basis_r=p_r.point.tangent_basis)[0,0]
            
        return cov

    def _form_logdet(self,
                     G_blocks,
                     C00i):

        # compute terms necessary for Jacobian

        self._compute_G_hess()
        G_hess, N_hess = self.G_hess, self.N_hess
        
        self._compute_G_regressMR(G_blocks) 
        G_regressMR, N_regressMR = self.G_regressMR, self.N_regressMR

        self._compute_shape_prods(C00i)
        G_shape, N_shape = self.G_shape, self.N_shape

        G_logdet = G_hess + G_regressMR + G_shape
        N_logdet = N_hess + N_regressMR + N_shape

        def logdet(G_logdet,
                   N_logdet,
                   first_order):
            hessian_mat = jnp.einsum('ijk,k->ij', G_logdet, first_order) + N_logdet
            if DEBUG:
                for (G, N, n) in zip([G_logdet, G_hess, G_regressMR, G_shape],
                                     [N_logdet, N_hess, N_regressMR, N_shape],
                                     ['logdet', 'hessian', 'proj', 'dot']):
                    _mat = jnp.einsum('ijk,k->ij', G, first_order) + N
                    if jnp.any(jnp.linalg.eigvalsh(hessian_mat) < 0):
                        label = 'LOGDET {n} {{}}'.format(n=n)
                        jax.debug.print(label, jnp.linalg.eigvalsh(_mat))
            return -jnp.log(jnp.linalg.det(hessian_mat))

        if G_logdet.shape[0] > 0:
            logdet_ = partial(logdet,
                              jnp.array(G_logdet),
                              jnp.array(N_logdet))
        else:
            logdet_ = lambda first_order: 1

        return logdet_, G_logdet, N_logdet

    def _form_barrier(self, C00i):

        # decompose the KKT conditions

        penalty, inactive = self.penalty, self.inactive

        ((L_inactive, offset_inactive),
         (L_active, offset_active)) = self._decompose_KKT(C00i)

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

    # private methods to decompose the derivative of the 0-counted process

    def _compute_G_hess(self):  

        # decompose each part of the Hessian into
        # terms dependent on gradient and field at E

        IK = self.inference_kernel
        RK = self.randomizer_kernel

        blocks = []
        for peak in self.data_peaks:
            block = np.zeros(get_gradient(peak.point).shape*2 + self.cov.shape[:1])

            for point in self.data_peaks + self.extra_points:
                C20_q = IK.C20([peak.point.location],
                               [point.point.location],
                               basis_l=peak.point.tangent_basis)[0,0] 
                C21_q = IK.C21([peak.point.location],
                               [point.point.location],
                              basis_l=peak.point.tangent_basis,
                              basis_r=point.point.tangent_basis)[0,0] 
                block[:,:,point.value_idx] = C20_q
                block[:,:,point.gradient_slice] = C21_q

            for point in self.random_peaks:
                C20_q = RK.C20([peak.point.location],
                               [point.point.location],
                               basis_l=peak.point.tangent_basis)[0,0] 
                C21_q = RK.C21([peak.point.location],
                               [point.point.location],
                              basis_l=peak.point.tangent_basis,
                              basis_r=point.point.tangent_basis)[0,0] 
                block[:,:,point.value_idx] = C20_q
                block[:,:,point.gradient_slice] = C21_q

            blocks.append(block)
            
        total_block_sizes = np.sum([block.shape[0] for block in blocks])
        G = np.zeros((total_block_sizes, total_block_sizes, self.cov.shape[0]))
        N = np.zeros((total_block_sizes, total_block_sizes))

        idx = 0
        for block, p_d, p_r in zip(blocks, self.data_peaks, self.random_peaks):
            H_slice = p_d.hessian_slice
            G[H_slice, H_slice, :] = block
            N[H_slice, H_slice] += -p_d.point.sign * get_hessian(p_d.point)
            N[H_slice, H_slice] += -p_d.point.sign * get_hessian(p_r.point) # p_d.point.sign should match p_r.point.sign
            
        N -= np.einsum('ijk,kl,l->ij',
                       G,
                       self.prec,
                       self.first_order)

        self.G_hess, self.N_hess = G, N
    
    def _compute_G_regressMR(self,
                             G_blocks):

        # let's compute the -S_E P(\nabla^2 f_E; f_E) term
        # this comes from differentiating the derivative of the irrepresentable matrix
        # this is why we use covariance terms that come from the model + randomizer covariance
        
        N_regressMR = np.zeros_like(self.N_hess)
        G_regressMR = np.zeros_like(self.G_hess)

        idx = 0            
        for q, block in zip(self.data_peaks, G_blocks):
            for i, (p_d, p_r) in enumerate(zip(self.data_peaks, self.random_peaks)):
                G_regressMR[q.hessian_slice,
                            q.hessian_slice,
                            p_d.value_idx] = block[:,:,i]
                G_regressMR[q.hessian_slice,  # p_d.hessian_slice should match p_r.hessian_slice
                            q.hessian_slice,
                            p_r.value_idx] = block[:,:,i]

        # G_regressMR is now the regression of -p.sign * p.hessian on to f_E (MK+RK form)
        # i.e. it is the corresponding irrepresentable matrix

        arg = np.zeros_like(self.first_order)
        for p_d in self.data_peaks:
            p_d.set_value(arg, -p_d.point.sign * p_d.point.penalty)

        N_regressMR = np.einsum('ijk,k->ij', G_regressMR, arg)

        self.G_regressMR, self.N_regressMR = G_regressMR, N_regressMR

    def _compute_shape_prods(self,
                           C00i):

        MK = self.model_kernel
        RK = self.randomizer_kernel

        D_ = np.zeros_like(self.N_hess)
        E_ = np.zeros(self.N_hess.shape + (len(self.data_peaks),))
        G_shape = np.zeros_like(self.G_hess)

        # matrix of inner products
        # these get scaled on right by diagonal
        # with blocks like s_j beta_j

        for i, peak in enumerate(self.data_peaks):
            if peak.point.n_ambient > 0:
                ngrad = get_tangent_gradient(peak.point).shape[0]
                E_[peak.hessian_slice,
                   peak.hessian_slice,
                   i] = np.identity(ngrad)

        locations = [q.point.location for q in self.data_peaks]

        for i, qs_l in enumerate(self.data_peaks):
            q_l = qs_l.point
            if q_l.n_ambient > 0:
                c10_l = (MK.C10([q_l.location],
                                locations,
                                basis_l=q_l.tangent_basis)[0].T +
                         RK.C10([q_l.location],
                                locations,
                                basis_l=q_l.tangent_basis)[0].T) 
                for j, qs_r in enumerate(self.data_peaks):
                    q_r = qs_r.point
                    if q_r.n_ambient > 0:
                        c11 = (MK.C11([q_l.location],
                                      [q_r.location],
                                      basis_l=q_l.tangent_basis,
                                      basis_r=q_r.tangent_basis)[0,0] +
                               RK.C11([q_l.location],
                                      [q_r.location],
                                      basis_l=q_l.tangent_basis,
                                      basis_r=q_r.tangent_basis)[0,0])
                        c10_r = (MK.C10([q_r.location],
                                        locations,
                                        basis_l=q_r.tangent_basis)[0].T +
                                 RK.C10([q_r.location],
                                        locations,
                                        basis_l=q_r.tangent_basis)[0].T)

                        D_[qs_l.hessian_slice,
                           qs_r.hessian_slice] = c11 - c10_l @ C00i @ c10_r.T
                        D_[qs_r.hessian_slice,
                           qs_l.hessian_slice] = D_[qs_l.hessian_slice,
                                                    qs_r.hessian_slice].T

        arg = np.zeros_like(self.first_order)

        for q_d in self.data_peaks:
            D_[q_d.hessian_slice] *= q_d.point.sign
            q_d.set_value(arg, q_d.get_value(arg) - q_d.point.sign * q_d.point.penalty)

        G_ = np.einsum('ij,jkl,lm->ijm',
                       D_,
                       E_,
                       C00i)

        for i, (q_d, q_r) in enumerate(zip(self.data_peaks,
                                           self.random_peaks)):

            G_shape[:,:,q_d.value_idx] = G_[:,:,i]
            G_shape[:,:,q_r.value_idx] = G_[:,:,i]

        N_shape = np.einsum('ijk,k->ij', G_shape, arg)
        self.G_shape, self.N_shape = G_shape, N_shape

    # decompose the subgradient process in terms of all the targets in cov

    def _decompose_KKT(self,
                       C00i):

        MK = self.model_kernel
        RK = self.randomizer_kernel
        IK = self.inference_kernel

        irrep = (MK.C00(None,
                        [p.point.location for p in self.data_peaks]) +
                 RK.C00(None,
                        [p.point.location for p in self.data_peaks]))[self.inactive] @ C00i

        pre_proj = np.zeros(self.cov.shape[:1] + self.subgrad[self.inactive].shape[:1])

        # fill in covariance with data and extra points

        for peaks, K in [(self.data_peaks + self.extra_points, IK),
                         (self.random_peaks, RK)]:
            for p in peaks:
                p.set_value(pre_proj, K.C00([p.point.location],
                                            None)[0, self.inactive])
                p.set_gradient(pre_proj, K.C10([p.point.location],
                                               None,
                                               basis_l=p.point.tangent_basis)[0, self.inactive].T)

        L_inactive = self.prec @ pre_proj

        for i, (p_d, p_r) in enumerate(zip(self.data_peaks,
                                           self.random_peaks)):
            p_d.set_value(L_inactive, p_d.get_value(L_inactive) - irrep[:,i])
            p_r.set_value(L_inactive, p_r.get_value(L_inactive) - irrep[:,i])

        L_inactive = L_inactive.T
        offset_inactive = self.subgrad[self.inactive] - L_inactive @ self.first_order

        signs = np.array([p.point.sign for p in self.data_peaks])
        L_active = np.zeros((self.cov.shape[0], len(self.data_peaks)))

        for i, (p_d, p_r) in enumerate(zip(self.data_peaks,
                                           self.random_peaks)):
            p_d.set_value(L_active, C00i[i])
            p_r.set_value(L_active, C00i[i])

        L_active = L_active.T
        offset_active = -C00i @ np.array([p.point.penalty * p.point.sign for p in self.data_peaks])
        L_active = (np.diag(signs) @ L_active) / np.sqrt(np.diag(C00i))[:,None]
        offset_active = signs * offset_active / np.sqrt(np.diag(C00i))

        return ((L_inactive, offset_inactive),
                (L_active, offset_active))

def _compute_random_model_cov(peaks,
                              model_kernel,
                              randomizer_kernel):
    """
    Compute C00, C01, C02 for a process with covariance MK+RK.

    This is the actual quadratic form used in the optimization problem.
    
    Used for gradient of irrepresentable path at E
    and irrepresentable matrix.

    """
    C_p = []
    C_g = []
    C_h = []

    for i, p in enumerate(peaks):
        p = p.point # no slices needed below
        c_p = []
        c_g = []
        c_h = []
        
        for j, q in enumerate(peaks):
            q = q.point # no slices needed below
            MK_00 = model_kernel.C00([p.location],
                                     [q.location])[0,0]
            RK_00 = randomizer_kernel.C00([p.location],
                                          [q.location])[0,0]
            c_p.append(MK_00 + RK_00)

            if p.n_ambient > 0:
                MK_10 = model_kernel.C10([p.location],
                                         [q.location],
                                         basis_l=p.tangent_basis)[0,0]
                RK_10 = randomizer_kernel.C10([p.location],
                                              [q.location],
                                              basis_l=p.tangent_basis)[0,0]
                MK_10 = MK_10.reshape(MK_10.shape + (1,))
                RK_10 = RK_10.reshape(RK_10.shape + (1,))

                MK_20 = model_kernel.C20([p.location],
                                         [q.location],
                                         basis_l=p.tangent_basis)[0,0]
                MK_20 = MK_20.reshape(MK_20.shape + (1,))
                RK_20 = randomizer_kernel.C20([p.location],
                                              [q.location],
                                              basis_l=p.tangent_basis)[0,0]
                RK_20 = RK_20.reshape(RK_20.shape + (1,))

                c_g.append(MK_10 + RK_10)
                c_h.append(-(MK_20 + RK_20) * p.sign)
            else:
                c_g.append(np.zeros((0,1)))
                c_h.append(np.zeros((0,0,1)))

        C_p.append(c_p)
        C_g.append(c_g)
        C_h.append(c_h)
        
    C00 = np.array(C_p)
    C00i = np.linalg.inv(C00)

    for i in range(len(C_g)):
        C_g[i] = np.concatenate(C_g[i], axis=-1).reshape(-1)

    # M is how the field enters the Kac-Rice counter condition...
    # for local maxima of the U (subgradient) process
    
    M = C00i @ np.array(C_g)

    # G is the regression mapping for regressing S_E \nabla^2 (f+omega)_E onto (f+omega)_E

    G_blocks = [np.einsum('ijk,kl->ijl',
                          np.concatenate(c_h, axis=-1),
                          C00i) for c_h in C_h]

    return C00i, M, G_blocks


def mle_summary(mle,
                SD,
                param=None,
                signs=None,
                level=None):
    """
    Wald-like summary
    """

    if param is None:
        param = np.zeros_like(mle)
    Z = (mle - param) / SD

    if signs is not None:
        one_sided = False
        P = normal_dbn.sf(Z * signs)
    else:
        one_sided = False
        P = normal_dbn.cdf(Z)
        P = 2 * np.minimum(P, 1-P)
    df = pd.DataFrame({'Estimate':mle,
                       'SD':SD,
                       'Param':param})
    if one_sided:
        df['P-value (1-sided)'] = P
    else:
        df['P-value (2-sided)'] = 2 * np.minimum(P, 1 - P)

    if level is not None:
        q = normal_dbn.ppf(1 - (1 - level) / 2)
        df['L ({:.0%})'.format(level)] = mle - q * SD
        df['U ({:.0%})'.format(level)] = mle + q * SD

    return df

def regression_decomposition(cov, T, N):
    
    nT, nN = T.shape[0], N.shape[0]
    
    TN = np.concatenate([T, N], axis=0)
    cov_TN = TN @ cov @ TN.T
    prec_TN = np.linalg.inv(cov_TN)
    
    # Cov(TZ|NZ)
    cov_TgN = np.linalg.inv(prec_TN[:nT,:nT])
    
    M = np.zeros_like(prec_TN)
    M[:nT,:nT] = cov_TgN
    M[:nT,nT:] = cov_TN[:nT,nT:] @ np.linalg.inv(cov_TN[nT:,nT:])
    M[nT:,nT:] = np.identity(nN)
    
    L = cov @ TN.T @ prec_TN @ M
    
    # compute the difference in covariance matrices of the
    # two estimators

    cov_beta_TN = prec_TN[:nT,:nT]
    
    cov_T = T @ cov @ T.T
    cov_beta_T = np.linalg.inv(cov_T)

    L_beta = L[:,:nT]
    L_NZ = L[:,nT:]

    # compute the estimation matrix, i.e.
    # the matrix that computes the T coords
    # of beta_{N cup T}

    est_matrix = prec_TN[:nT] @ TN

    # find a square root of the
    # covariance of the residual matrix

    cov_R = cov - cov @ TN.T @ prec_TN @ TN @ cov
    U, D, _ = np.linalg.svd(cov_R)

    p = cov.shape[0]
    rank_R = p - nN - nT # assumes prec_TN is full rank
                         # we would have had an exception
                         # earlier when computing prec_TN if exactly singular 
                         # it's possible rank(cov_R) is smaller if cov wasn't full rank
                         # BUT, we have definitely assumed TN @ cov @ TN.T is full rank
    U = U[:,:rank_R]
    D = D[:rank_R]
    sqrt_cov_R = U * np.sqrt(D)[None,:]

    return RegressionInfo(T,
                          N,
                          L_beta,
                          L_NZ,
                          est_matrix,
                          sqrt_cov_R,
                          cov_beta_T,
                          cov_beta_TN)
