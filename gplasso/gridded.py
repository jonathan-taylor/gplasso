from copy import deepcopy
from functools import partial
from itertools import product
from dataclasses import dataclass, asdict, astuple

import numpy as np
import pandas as pd
import jax.numpy as jnp
from scipy.stats import norm as normal_dbn
from scipy.stats import chi2

import jax
from jax import jacfwd

from .base import (LASSOInference,
                   PointWithSlices as PointWithSlicesBase)

from .utils import RegressionInfo

from .peaks import (get_gradient,
                    get_tangent_gradient,
                    get_normal_gradient,
                    get_hessian,
                    InteriorPeak,
                    InteriorPoint,
                    extract_peaks,
                    extract_points)

from .utils import (mle_summary,
                    regression_decomposition,
                    _compute_mle,
                    _obj_maker)

DEBUG = False

@dataclass
class PointWithSlices(PointWithSlicesBase):

    gradient_slice: slice # index into col of cov for gradient coords
    hessian_slice: slice # index into row/col of Hessian for each peak
    
    def get_gradient(self, arr):
        return arr[self.gradient_slice]

    def set_gradient(self, arr, grad):
        arr[self.gradient_slice] =  grad

    def get_hessian_block(self, arr):
        return arr[self.hessian_slice]


@dataclass
class DisplacementEstimate(object):

    location: np.ndarray
    segment: np.ndarray
    cov: np.ndarray
    factor: float
    quantile: float

class GridLASSOInference(LASSOInference):

    def __init__(self,
                 gridvals,
                 penalty,
                 model_kernel,
                 randomizer_kernel,
                 inference_kernel=None):

        self.gridvals = gridvals
        LASSOInference.__init__(self,
                                penalty,
                                model_kernel,
                                randomizer_kernel,
                                inference_kernel=inference_kernel)
        
    def get_location(self, idx):
        return tuple([g[i] for g, i in zip(self.gridvals, idx)])
    
    def extract_peaks(self,
                      selected_idx,     # cluster representatives --
                      model_spec=None): # model default does not use location intervals for now

        selected_df = pd.DataFrame({}, index=selected_idx)
        selected_df['Value'] = False
        selected_df['Displacement'] = False
        selected_df['Location'] = [self.get_location(idx) for idx in selected_df.index]
        selected_df = selected_df.reset_index()
        selected_df = selected_df.set_index(['Location', 'Index'])
        
        if model_spec is None:
            model_spec = pd.DataFrame({'Value':[True] * len(selected_idx),
                                       'Displacement':[False] * len(selected_idx)},
                                      index=selected_idx)
        model_spec['Location'] = [self.get_location(idx) for idx in model_spec.index]
        model_spec = model_spec.reset_index()
        model_spec = model_spec.set_index(['Location', 'Index'])
        
        # could make some NaN's
        selected_df = (selected_df +
                       model_spec[lambda df: df.index.isin(selected_df.index)])
        # set the NaN's to False
        selected_df.loc[selected_df['Displacement'].isnull(), ['Displacement', 'Value']] = False
        
        # effects in model_spec not captured in selected_df -- by default will be empty
        extra_df = model_spec[lambda df: ~ df.index.isin(selected_df.index)]
        
        sufficient_df = pd.concat([selected_df, extra_df]).reset_index().set_index('Location')
        selected_df = selected_df.reset_index().set_index('Location')
        extra_df = extra_df.reset_index().set_index('Location')

        Z, perturbation = self.Z, self.perturbation_
        selected_df['Sign'] = np.sign([self.subgrad[idx] for idx in selected_df['Index']])
        selected_df['Penalty'] = [self.penalty[idx] for idx in selected_df['Index']]

        ndim = len(self.gridvals)
        npt = sufficient_df.shape[0]

        # second order at each of the selected pts

        idx = 0
        hess_idx = 0
        self._data_peaks = []
        self._random_peaks = []

        for i, (data_pt, random_pt) in enumerate(zip(second_order_expansion(self.gridvals,
                                                                            Z,
                                                                            selected_df['Index']),
                                                     second_order_expansion(self.gridvals,
                                                                            perturbation,
                                                                            selected_df['Index']))):
            data_pt = InteriorPeak(location=data_pt.location,
                                   index=data_pt.index,
                                   sign=selected_df['Sign'][i],
                                   penalty=selected_df['Penalty'][i],
                                   value=data_pt.value,
                                   gradient=data_pt.gradient,
                                   hessian=data_pt.hessian,
                                   tangent_basis=np.identity(data_pt.gradient.shape[0]),
                                   n_obs=1,
                                   n_ambient=ndim,
                                   n_tangent=ndim)

            self._data_peaks.append(PointWithSlices(point=data_pt,
                                                    value_idx=idx,
                                                    gradient_slice=slice(idx + 1,
                                                                         idx + 1 + data_pt.n_ambient),
                                                    hessian_slice=slice(hess_idx,
                                                                        hess_idx + data_pt.n_ambient)))
            idx += 1 + data_pt.n_ambient
            hess_idx += data_pt.n_ambient

            random_pt = InteriorPoint(location=random_pt.location,
                                      index=random_pt.index,
                                      value=random_pt.value,
                                      gradient=random_pt.gradient,
                                      hessian=random_pt.hessian,
                                      tangent_basis=np.identity(random_pt.gradient.shape[0]),
                                      n_obs=1,
                                      n_ambient=ndim,
                                      n_tangent=ndim)

            self._random_peaks.append(PointWithSlices(point=random_pt,
                                                      value_idx=idx,
                                                      gradient_slice=slice(idx + 1,
                                                                           idx + 1 + random_pt.n_ambient),
                                                      hessian_slice=self._data_peaks[-1].hessian_slice))
            idx += 1 + random_pt.n_ambient


        self._extra_points = []
        for extra_pt in second_order_expansion(self.gridvals,
                                               Z,
                                               extra_df['Index']):

            extra_pt = InteriorPoint(location=extra_pt.location,
                                     index=extra_pt.index,
                                     value=extra_pt.value,
                                     gradient=extra_pt.gradient,
                                     hessian=None,
                                     tangent_basis=np.identity(extra_pt.gradient.shape[0]),
                                     n_obs=1,
                                     n_ambient=ndim,
                                     n_tangent=ndim)

            self._extra_points.append(PointWithSlices(point=extra_pt,
                                                      value_idx=idx,
                                                      gradient_slice=slice(idx + 1,
                                                                           idx + 1 + extra_pt.n_ambient),
                                                      hessian_slice=None))

            idx += 1 + extra_pt.n_ambient
            
        cov_size = idx
        cov = self.cov = self._compute_cov(cov_size)
        self.prec = np.linalg.inv(self.cov)

        self._spec_df = sufficient_df
        return model_spec

    def setup_inference(self,
                        inactive,
                        model_spec=[]):

        self.inactive = inactive

        cov = self.cov 
        
        # we use our regression decomposition to rewrite this in terms of
        # our "best case" unbiased estimator (ie when no selection effect present)
        # the affine term in the Kac Rice formula and the corresponding residual of
        # (f_E, \omega_E, \nabla f_E, \nabla \omega_E) 

        C00i, M, G_blocks = _compute_random_model_cov(self._data_peaks,
                                                      self.model_kernel, 
                                                      self.randomizer_kernel)

        # form the T / N matrices for the decomposition

        # T is a selector matrix for all sufficient stats in the model
        # N is represents the random vector in the "0" of the Kac-Rice formula

        cols = []

        mle_index = []
        for p in self._data_peaks + self._extra_points:
            if self._spec_df['Value'][p.point.location]:
                value_col = np.zeros((cov.shape[0], 1))
                p.set_value(value_col, 1)
                cols.append(value_col)
                mle_index.append((p.point.location, 'Value'))
                
            if self._spec_df['Displacement'][p.point.location]:
                gradient_cols = np.zeros((cov.shape[0], p.point.n_ambient))
                p.set_gradient(gradient_cols, np.identity(p.point.n_ambient))
                cols.append(gradient_cols)
                mle_index.append([(p.point.location, 'Displacement[%d]' % i)
                                  for i in range(gradient_cols.shape[1])])
                
        T = np.concatenate(cols, axis=1).T

        self._mle_index = pd.MultiIndex.from_tuples(mle_index, names=['Location', 'Type'])
        
        param_df = pd.DataFrame({}, index=self._mle_index)
        param_df['Param'] = 0

        param_df = param_df.reset_index()
        param_df = param_df[lambda df: df['Type'] == 'Value']
        
        param_df = param_df.set_index('Location')
        
        # now specify the value the gradient is pinned at
        
        N = []

        idx = 0 # restart counter
        for i, (p_d, p_r) in enumerate(zip(self._data_peaks, self._random_peaks)):
            ngrad = get_tangent_gradient(p_d.point).shape[0]
            N_cols = np.zeros((cov.shape[0], ngrad))
            p_d.set_gradient(N_cols, np.identity(ngrad))
            p_r.set_gradient(N_cols, np.identity(ngrad))
            p_d.set_value(N_cols, -p_d.get_hessian_block(M[i]))
            p_r.set_value(N_cols, -p_r.get_hessian_block(M[i]))

            N.append(N_cols)

        N = np.hstack(N).T

        self.regress_decomp = regression_decomposition(cov, T, N) 

        # compute the first order data

        self.first_order = np.zeros(cov.shape[0])
        for p in self._data_peaks + self._random_peaks + self._extra_points:
            p.set_value(self.first_order, p.point.value)
            p.set_gradient(self.first_order, get_gradient(p.point))

        self.logdet_info = self._form_logdet(G_blocks,
                                             C00i)
        self.barrier_info = self._form_barrier(C00i)

        # setup a default parameter df
        
        self._default_param = param_df.copy()
        return param_df

    def _solve_MLE(self):

        barrier, G_barrier, N_barrier = self.barrier_info
        logdet, G_logdet, N_logdet = self.logdet_info

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

        if initial_W.shape[0] != (0,): # there is some noise to integrate over
            N_barrier_ = N_barrier + G_barrier @ (offset + L_beta @ beta_nosel)
            G_barrier_ = G_barrier @ sqrt_cov_R

            N_logdet_ = N_logdet + G_logdet @ (offset + L_beta @ beta_nosel)
            G_logdet_ = G_logdet @ sqrt_cov_R

            B_ = _obj_maker(barrier,
                            offset,
                            L_beta,
                            sqrt_cov_R)

            LD_ = _obj_maker(logdet,
                             offset,
                             L_beta,
                             sqrt_cov_R)


            obj_jax = lambda beta, W: B_(beta, W) + LD_(beta, W)
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
                             (H_full[0][0] - H_full[0][1] @
                              np.linalg.inv(I + H_full[1][1]) @
                              H_full[1][0]))

            mle_cov = cov_beta_TN @ observed_info @ cov_beta_TN
        else:
            mle = beta_nosel
            mle_cov = cov_beta_TN

        return mle, mle_cov

    def summary(self,
                level=0.90,
                displacement_level=0.90,
                one_sided=True,
                location=False,
                param=None):

        if param is None:
            param = self._default_param
            if one_sided:
                raise ValueError('must provide a "param" argument with a "Signs" column for one-sided tests')
            
        mle, mle_cov = self._solve_MLE()

        mle_df = pd.DataFrame({'Estimate':mle,
                               'SD':np.sqrt(np.diag(mle_cov))},
                              index=self._mle_index)

        value_df = mle_df.copy()
        value_df = value_df.reset_index()
        value_df = value_df[lambda df: df['Type'] == 'Value']
        value_df = value_df.set_index('Location')
        
        # inference for peak heights / values
        joined_df = pd.merge(param,
                             value_df,
                             left_index=True,
                             right_index=True)
                                 
        if one_sided:
            value_df = mle_summary(joined_df['Estimate'],
                                   joined_df['SD'],
                                   param=joined_df['Param'],
                                   signs=joined_df['Signs'],
                                   level=level)
        else:
            value_df = mle_summary(joined_df['Estimate'],
                                   joined_df['SD'],
                                   param=joined_df['Param'],
                                   level=level)

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
        return value_df, loc_results

    def _compute_cov(self, cov_size):
        cov = np.zeros((cov_size, cov_size))

        IK = self.inference_kernel
        RK = self.randomizer_kernel

        # first fill in values for the covariance of the data

        for peaks_l, peaks_r, K in [(self._data_peaks + self._extra_points,
                                     self._data_peaks + self._extra_points, IK),
                                    (self._random_peaks, self._random_peaks, RK)]:
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
        for peak in self._data_peaks:
            block = np.zeros(get_gradient(peak.point).shape*2 + self.cov.shape[:1])

            for point in self._data_peaks + self._extra_points:
                C20_q = IK.C20([peak.point.location],
                               [point.point.location],
                               basis_l=peak.point.tangent_basis)[0,0] 
                C21_q = IK.C21([peak.point.location],
                               [point.point.location],
                              basis_l=peak.point.tangent_basis,
                              basis_r=point.point.tangent_basis)[0,0] 
                block[:,:,point.value_idx] = C20_q
                block[:,:,point.gradient_slice] = C21_q

            for point in self._random_peaks:
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
        for block, p_d, p_r in zip(blocks, self._data_peaks, self._random_peaks):
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
        for q, block in zip(self._data_peaks, G_blocks):
            for i, (p_d, p_r) in enumerate(zip(self._data_peaks, self._random_peaks)):
                G_regressMR[q.hessian_slice,
                            q.hessian_slice,
                            p_d.value_idx] = block[:,:,i]
                G_regressMR[q.hessian_slice,  # p_d.hessian_slice should match p_r.hessian_slice
                            q.hessian_slice,
                            p_r.value_idx] = block[:,:,i]

        # G_regressMR is now the regression of -p.sign * p.hessian on to f_E (MK+RK form)
        # i.e. it is the corresponding irrepresentable matrix

        arg = np.zeros_like(self.first_order)
        for p_d in self._data_peaks:
            p_d.set_value(arg, -p_d.point.sign * p_d.point.penalty)

        N_regressMR = np.einsum('ijk,k->ij', G_regressMR, arg)

        self.G_regressMR, self.N_regressMR = G_regressMR, N_regressMR

    def _compute_shape_prods(self,
                           C00i):

        MK = self.model_kernel
        RK = self.randomizer_kernel

        D_ = np.zeros_like(self.N_hess)
        E_ = np.zeros(self.N_hess.shape + (len(self._data_peaks),))
        G_shape = np.zeros_like(self.G_hess)

        # matrix of inner products
        # these get scaled on right by diagonal
        # with blocks like s_j beta_j

        for i, peak in enumerate(self._data_peaks):
            if peak.point.n_ambient > 0:
                ngrad = get_tangent_gradient(peak.point).shape[0]
                E_[peak.hessian_slice,
                   peak.hessian_slice,
                   i] = np.identity(ngrad)

        locations = [q.point.location for q in self._data_peaks]

        for i, qs_l in enumerate(self._data_peaks):
            q_l = qs_l.point
            if q_l.n_ambient > 0:
                c10_l = (MK.C10([q_l.location],
                                locations,
                                basis_l=q_l.tangent_basis)[0].T +
                         RK.C10([q_l.location],
                                locations,
                                basis_l=q_l.tangent_basis)[0].T) 
                for j, qs_r in enumerate(self._data_peaks):
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

        for q_d in self._data_peaks:
            D_[q_d.hessian_slice] *= q_d.point.sign
            q_d.set_value(arg, q_d.get_value(arg) - q_d.point.sign * q_d.point.penalty)

        G_ = np.einsum('ij,jkl,lm->ijm',
                       D_,
                       E_,
                       C00i)

        for i, (q_d, q_r) in enumerate(zip(self._data_peaks,
                                           self._random_peaks)):

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
                        [p.point.location for p in self._data_peaks]) +
                 RK.C00(None,
                        [p.point.location for p in self._data_peaks]))[self.inactive] @ C00i

        pre_proj = np.zeros(self.cov.shape[:1] + self.subgrad[self.inactive].shape[:1])

        # fill in covariance with data and extra points

        for peaks, K in [(self._data_peaks + self._extra_points, IK),
                         (self._random_peaks, RK)]:
            for p in peaks:
                p.set_value(pre_proj, K.C00([p.point.location],
                                            None)[0, self.inactive])
                p.set_gradient(pre_proj, K.C10([p.point.location],
                                               None,
                                               basis_l=p.point.tangent_basis)[0, self.inactive].T)

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


@dataclass
class Expansion(object):

    value: np.ndarray
    gradient: np.ndarray
    hessian: np.ndarray
    index: np.ndarray
    location: np.ndarray
    
def second_order_expansion(G,
                           Z,  
                           I):

    value = []

    Z = np.asarray(Z)
    Z = Z
    grad = np.gradient(Z, *G)
    grad = np.array(grad)
    hess = [np.gradient(g, *G) for g in grad]

    hess = np.array(hess)
    value = []
    for idx in I:
        item_ = idx # (slice(None,None,None),) + idx
        val_idx = Z[item_]
        item_ = (slice(None,None,None),) + item_
        grad_idx = grad[item_]
        item_ = (slice(None,None,None),) + item_
        hess_idx = hess[item_]
        hess_idx = np.array([(h + h.T)/2 for h in hess_idx])
        value.append(Expansion(value=val_idx,
                               gradient=grad_idx,
                               hessian=hess_idx,
                               index=idx,
                               location=tuple([g[i] for i, g in zip(idx, G)])))
                
    return value
