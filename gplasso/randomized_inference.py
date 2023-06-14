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

from .peaks import (get_gradient,
                    get_tangent_gradient,
                    get_normal_gradient,
                    get_hessian,
                    Peak)

DEBUG = False

class DisplacementEstimate(NamedTuple):

    location: np.ndarray
    segment: np.ndarray
    cov: np.ndarray
    factor: float
    quantile: float

class SliceInfo(NamedTuple):

    sizes: list       # dimension of gradient and hessian at each peak
    slices: list      # slices for filling in complete gradient vector
    cov_slices: tuple # slices for the data and randomized value and gradient
    nvalue: int      # len(sizes)
    ngrad: int       # dimension of jacobian

class RegressionInfo(NamedTuple):

    T: np.ndarray
    N: np.ndarray
    L_beta: np.ndarray
    L_NZ: np.ndarray
    est_matrix: np.ndarray
    sqrt_cov_R: np.ndarray
    cov_beta_T: np.ndarray
    cov_beta_TN: np.ndarray


class InferenceInfo(NamedTuple):

    peaks: list
    inactive: np.ndarray # inactive set selector 
    subgrad: np.ndarray # 
    penalty: np.ndarray # penalty values at x_inactive
    model_kernel: covariance_structure  # covariance used for fitting
    randomizer_kernel: covariance_structure # covariance used for randomization (and fitting)
    inference_kernel: covariance_structure
    slice_info: SliceInfo
    first_order: np.ndarray
    cov: np.ndarray
    contrast_info: tuple
    logdet_info: tuple
    barrier_info: tuple
    
class PeakWithSlice(NamedTuple):

    peak: Peak
    value_slice: slice
    gradient_slice: slice

    def get_value(self, arr):
        return arr[self.value_slice]

    def set_value(self, arr, val):
        arr[self.value_slice] = val

    def get_gradient(self, arr):
        return arr[self.gradient_slice]

    def set_gradient(self, arr, grad):
        arr[self.gradient_slice] =  grad

def setup_inference(peaks,
                    inactive,
                    subgrad,
                    penalty,
                    model_kernel,
                    randomizer_kernel,
                    displacement=False,
                    inference_kernel=None,
                    extra_points=None):

    if inference_kernel is None:
        inference_kernel = model_kernel
        
    data_peaks = [deepcopy(peak) for peak in peaks]
    random_peaks = [deepcopy(peak) for peak in peaks]

    data_peaks = []
    random_peaks = []
    idx = 0
    for i, peak in enumerate(peaks):
        ngrad = peak.gradient.shape[1]
        data_peak = peak._replace(value=peak.value[0],
                                  gradient=peak.gradient[0],
                                  hessian=peak.hessian[0])
        data_peak_slice = PeakWithSlice(peak=data_peak,
                                        value_slice=slice(idx, idx + 1),
                                        gradient_slice=slice(idx + 1,
                                                             idx + 1 + ngrad))
        idx += (1 + ngrad)
                                        
        random_peak = peak._replace(value=peak.value[1],
                                    gradient=peak.gradient[1],
                                    hessian=peak.hessian[1])

        random_peak_slice = PeakWithSlice(peak=random_peak,
                                          value_slice=slice(idx, idx + 1),
                                          gradient_slice=slice(idx + 1,
                                                               idx + 1 + ngrad))
        data_peaks.append(data_peak_slice)
        random_peaks.append(random_peak_slice)
    # self.peaks = peaks

    # (self.model_kernel,
    #  self.randomizer_kernel,
    #  self.inference_kernel) = (model_kernel,
    #                            randomizer_kernel,
    #                            inference_kernel)

    sizes = [get_hessian(q).shape[-1] for q in peaks]
    slices = [slice(s, e) for s, e in zip(np.cumsum([0] + sizes),
                                          np.cumsum(sizes))]

    field_dim = len(peaks)
    grad_dim = np.sum(sizes)
    data_value_slice = slice(0, field_dim)
    random_value_slice = slice(field_dim, 2 * field_dim)
    data_grad_slice = slice(2 * field_dim, 2 * field_dim + grad_dim)
    random_grad_slice = slice(2 * field_dim + grad_dim, 2 * field_dim + 2 * grad_dim)

    cov_slices = (data_value_slice,
                  random_value_slice,
                  data_grad_slice,
                  random_grad_slice)

    slice_info = SliceInfo(sizes, slices, cov_slices, len(peaks), sum(sizes))
    
    cov = _compute_cov(peaks,
                       inference_kernel,
                       randomizer_kernel,
                       slice_info)

    # cov2 = _compute_cov2(data_peaks,
    #                      random_peaks,
    #                      inference_kernel,
    #                      randomizer_kernel)

    # we use our regression decomposition to rewrite this in terms of
    # our "best case" unbiased estimator (ie when no selection effect present)
    # the affine term in the Kac Rice formula and the corresponding residual of
    # (f_E, \omega_E, \nabla f_E, \nabla \omega_E) 

    C00, C00i, C_g, C_h = _compute_C0012(peaks,
                                         model_kernel,
                                         randomizer_kernel)
    
    # form the T / N matrices for the decomposition

    T = np.zeros((cov.shape[0]//2, cov.shape[0]))
    T[:len(peaks),data_value_slice] = np.identity(len(peaks))
    T[len(peaks):,data_grad_slice] = np.identity(sum(sizes))
    if not displacement:
        T = T[:len(peaks)]

    N = np.zeros((sum(sizes), cov.shape[0]))
    N[:,random_grad_slice] = np.identity(N.shape[0])
    N[:,data_grad_slice] = np.identity(N.shape[0])
    for i in range(len(C_g)):
        C_g[i] = np.concatenate(C_g[i], axis=-1).reshape(-1)
    M = C00i @ np.array(C_g)
    N[:,random_value_slice] = N[:,data_value_slice] = -M.T
    
    # the affine condition in the Kac-Rice formula

    # why isn't this used? hmm...
    NZ_obs = -M.T @ np.array([p.sign * p.penalty for p in peaks])

    regress_decomp = regression_decomposition(cov, T, N) 

    # compute data for regression

    if extra_points is not None:
        sizes = [get_gradient(peak).shape[0] for peak in extra_points]
        extra_slices = [slice(s, e) for s, e in zip(np.cumsum([0] + sizes),
                                                    np.cumsum(sizes))]
        nextra = len(extra_points) + sum(sizes)
        extra_data_slice = slice(2*ntotal, 2*ntotal+len(extra_points))
        extra_grad_slice = slice(2*ntotal+len(extra_points), nextra)

    first_order = np.zeros(2 * (slice_info.ngrad + slice_info.nvalue))
    first_order[data_value_slice] = [p.value[0] for p in peaks]
    first_order[random_value_slice] = [p.value[1] for p in peaks]

    grads = []
    for p in peaks:
        G = get_gradient(p)[0]
        if hasattr(p, 'tangent_basis') and p.tangent_basis is not None:
            T = p.tangent_basis @ G
            grads.append(T)
            if hasattr(p, 'normal_basis') and p.normal_basis is not None:
                N = p.normal_basis @ G
                grads.append(N)
        else:
            grads.append(G)
        
    first_order[data_grad_slice] = np.hstack(grads)

    grads = []
    for p in peaks:
        G = get_gradient(p)[1]
        if hasattr(p, 'tangent_basis') and p.tangent_basis is not None:
            T = p.tangent_basis @ G
            grads.append(T)
            if hasattr(p, 'normal_basis') and p.normal_basis is not None:
                N = p.normal_basis @ G
                grads.append(N)
        else:
            grads.append(G)
    first_order[random_grad_slice] = np.hstack(grads)

    if extra_points is not None:
        grads = []
        for p in extra_points:
            G = get_gradient(p)[0]
            if hasattr(p, 'tangent_basis') and p.tangent_basis is not None:
                T = p.tangent_basis @ G
                grads.append(T)
                if hasattr(p, 'normal_basis') and p.normal_basis is not None:
                    N = p.normal_basis @ G
                grads.append(N)
            else:
                grads.append(G)
        first_order[extra_grad_slice] = np.hstack(grads)
        first_order[extra_data_slice] = [p.value for p in extra_points]
        
    info = InferenceInfo(peaks,
                         inactive,
                         subgrad,
                         penalty,
                         model_kernel,
                         randomizer_kernel,
                         inference_kernel,
                         slice_info,
                         first_order,
                         cov,
                         regress_decomp,
                         logdet_info=None,
                         barrier_info=None)

    logdet_info = _form_logdet(info,
                               C_h,
                               C00i)

    barrier_info = _form_barrier(info)

    info = info._replace(barrier_info=barrier_info,
                         logdet_info=logdet_info)
    return info

# we will now compute terms needed to restrict G_hess to each (f_j,
# \nabla f_j, \omega_j, \nabla \omega_j) and
# fix the R(\nabla(f+\omega)_E;(f+\omega)_E)= - \nabla irrep_path

def _form_barrier(info):

    # decompose the KKT conditions
    
    penalty, inactive = info.penalty, info.inactive

    ((L_inactive, offset_inactive),
     (L_active, offset_active)) = _decompose_KKT(info)

    # active KKT condtion is L_active @ obs + offset_active >= 0
    
    L_inactive /= (penalty[inactive,None] / 2)
    offset_inactive /= (penalty[inactive] / 2)
    
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
    
    V = L_active @ info.first_order + offset_active

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

def _form_logdet(info,
                 C_h,
                 C00i,
                 extra_points=None):
    
    # compute terms necessary for Jacobian

    G_hess, N_hess = _compute_G_hess(info, extra_points=extra_points)
    G_proj, N_proj = _compute_G_proj(info, C_h, C00i) # G_proj will be of different shape if
                                                      # extra_points is not None
    G_dot, N_dot = _dot_prods(info, G_hess.shape)

    G_logdet = G_hess + G_proj + G_dot
    N_logdet = N_hess + N_proj + N_dot

    def logdet(G_logdet,
               N_logdet,
               first_order):
        hessian_mat = jnp.einsum('ijk,k->ij', G_logdet, first_order) + N_logdet
        if DEBUG:
            for (G, N, n) in zip([G_logdet, G_hess, G_proj, G_dot],
                                 [N_logdet, N_hess, N_proj, N_dot],
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

def _compute_C0012(peaks,
                   model_kernel,
                   randomizer_kernel):
    """
    Compute C00, C01, C02 for randomized process
    
    """
    C_p = []
    C_g = []
    C_h = []

    for i, p in enumerate(peaks):

        c_p = []
        c_g = []
        c_h = []
        
        for j, q in enumerate(peaks):

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

    return C00, C00i, C_g, C_h

def _compute_G_hess(info,
                    extra_points=None):  

    # decompose each part of the Hessian into
    # terms dependent on gradient and field at E
    
    (data_value_slice,
     random_value_slice,
     data_grad_slice,
     random_grad_slice) = info.slice_info.cov_slices

    GI_p, GI_g = _decompose_hessian(info.peaks,
                                    info.peaks,
                                    info.inference_kernel,
                                    info.slice_info)
    GR_p, GR_g = _decompose_hessian(info.peaks,
                                    info.peaks,
                                    info.randomizer_kernel,
                                    info.slice_info)
    if extra_points is not None:
        GE_p, GE_g = _decompose_hessian(info.peaks,
                                        extra_points,
                                        info.inference_kernel,
                                        info.slice_info)
    sizes = info.slice_info.sizes
    slices = info.slice_info.slices

    N_hess = np.zeros(GI_p.shape[:2])
    for p, size_p, slice_p in zip(info.peaks,
                                  sizes,
                                  slices):
        if p.n_ambient > 0:
            N_hess[slice_p, slice_p] += -p.sign * (get_hessian(p).sum(0))

    ntotal = info.slice_info.ngrad + info.slice_info.nvalue
    if extra_points is not None:
        sizes = [get_gradient(peak).shape[0] for peak in extra_points]
        nextra = len(extra_points) + sum(sizes)
        extra_data_slice = slice(2*ntotal, 2*ntotal+len(extra_points))
        extra_grad_slice = slice(2*ntotal+len(extra_points), nextra)
    else:
        nextra = 0
    G_hess = np.zeros((info.slice_info.ngrad, info.slice_info.ngrad, 2*ntotal + nextra,))

    G_hess[:,:,data_value_slice] = GI_p
    G_hess[:,:,random_value_slice] = GR_p
    G_hess[:,:,data_grad_slice] = GI_g
    G_hess[:,:,random_grad_slice] = GR_g

    if extra_points is not None:
        G_hess[:,:,extra_data_slice] = GE_p
        G_hess[:,:,extra_grad_slice] = GE_g

    N_hess -= np.einsum('ijk,k->ij', G_hess, info.first_order)
    return G_hess, N_hess

def _compute_G_proj(info,
                    C_h,
                    C00i):
    
    # let's compute the P(\nabla^2 f_E; f_E) term
    
    (data_value_slice,
     random_value_slice,
     data_grad_slice,
     random_grad_slice) = info.slice_info.cov_slices

    ngrad = info.slice_info.ngrad
    ntotal = ngrad + info.slice_info.nvalue

    N_proj = np.zeros((ngrad, ngrad))
    G_proj = np.zeros((ngrad, ngrad, 2*ntotal))

    for i, (p, s_p) in enumerate(zip(info.peaks, info.slice_info.slices)):
        preG_ = np.concatenate(C_h[i], axis=-1)
        G_proj[s_p,s_p,data_value_slice] = np.einsum('ijk,kl->ijl', preG_, C00i)
        G_proj[s_p,s_p,random_value_slice] = G_proj[s_p,s_p,data_value_slice]

    # G_proj is now the regression of -p.sign * p.hessian on to f_E
    # this will be subtracted in the Hessian 
    G_proj = -G_proj
    arg = np.zeros_like(info.first_order)
    delta = np.array([p.sign * p.penalty for p in info.peaks])
    arg[data_value_slice] = -delta

    N_proj = np.einsum('ijk,k->ij', G_proj, arg)

    return G_proj, N_proj

def inference(info,
              level=0.90,
              displacement_level=0.95,
              one_sided=True,
              location=False,
              param=None):

    (peaks,
     inactive,
     subgrad,
     penalty,
     model_kernel,
     randomizer_kernel,
     inference_kernel,
     slice_info,
     first_order,
     cov,
     contrast_info,
     _,
     _) = info

    barrier, G_barrier, N_barrier = info.barrier_info
    logdet, G_logdet, N_logdet = info.logdet_info

    if param is None:
        param = np.zeros(len(peaks))
    
    (T,
     N,
     L_beta,
     L_NZ,
     est_matrix,
     sqrt_cov_R,
     cov_beta_T,
     cov_beta_TN) = contrast_info

    first_order = info.first_order
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
        
    height_mle, loc_mle = mle[:len(peaks)], mle[len(peaks):]
    peaks = info.peaks
    height_param = param[:len(peaks)]
    height_SD = np.sqrt(np.diag(mle_cov)[:len(peaks)])
    height_Z = (height_mle - height_param) / height_SD

    if DEBUG:
        print(mle, 'mle')
        print(param, 'param')
        print(beta_nosel, 'no selection')

    signs = np.array([p.sign for p in peaks])
    P = normal_dbn.sf(height_Z * signs)
    df = pd.DataFrame({'Location':[tuple(p.location) for p in peaks],
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
    df['Location'] = [tuple(p.location) for p in peaks]

    # now confidence regions for the peaks

    loc_cov = mle_cov[len(peaks):,len(peaks):]
    loc_mle = mle[len(peaks):]
    loc_results = []

    # should only output this for peaks where requested...
    for i, (s, p) in enumerate(zip(info.slice_info.slices, info.peaks)):
        if hasattr(p, 'n_tangent') and len(loc_mle) > 0:
            df_p = df.iloc[i]
            up_lab = 'U ({:.0%})'.format(level)
            low_lab = 'L ({:.0%})'.format(level)
            if df_p[up_lab] * df_p[low_lab] > 0:
                factor = 1 / np.fabs([df_p[up_lab], df_p[low_lab]]).min()
            else:
                factor = np.inf
            n_spatial = p.n_tangent
            if hasattr(p, 'n_normal'):
                n_spatial += p.n_normal
            q = chi2.ppf(displacement_level, n_spatial)
            loc_results.append(DisplacementEstimate(location=p.location,
                                                    segment=np.array([loc_mle[s] / df_p[low_lab],
                                                                      loc_mle[s] / df_p[up_lab]]),
                                                    cov=loc_cov[s,s],
                                                    quantile=q,
                                                    factor=factor))
        else:
            loc_results.append(DisplacementEstimate(p.location, None, None, None, None))
        
    return df.set_index('Location'), loc_results

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

def _compute_cov(peaks,
                 inference_kernel,
                 randomizer_kernel,
                 slice_info):

    sizes, slices, cov_slices, nvalue, ngrad = slice_info

    (data_value_slice,
     random_value_slice,
     data_grad_slice,
     random_grad_slice) = cov_slices
    
    IK = inference_kernel
    RK = randomizer_kernel

    ntotal = nvalue + ngrad
    cov = np.zeros((ntotal*2,)*2)

    # first fill in values for the data
    
    cov[data_value_slice, data_value_slice] = IK.C00([p.location for p in peaks],
                                                     [p.location for p in peaks])
    cov_01 = np.zeros([len(peaks), sum(sizes)])
    cov_11 = np.zeros([sum(sizes), sum(sizes)])
    for i, (p_l, s_l) in enumerate(zip(peaks, slices)):
        for p_r, s_r in zip(peaks, slices):
            cov_01[i,s_r] = IK.C01([p_l.location],
                                   [p_r.location],
                                   basis_r=p_r.tangent_basis)[0,0]
            cov_11[s_l,s_r] = IK.C11([p_l.location],
                                     [p_r.location],
                                     basis_l=p_l.tangent_basis,
                                     basis_r=p_r.tangent_basis)[0,0]
            
    cov[data_value_slice, data_grad_slice] = cov_01
    cov[data_grad_slice, data_value_slice] = cov_01.T
    cov[data_grad_slice, data_grad_slice] = cov_11
    
    # now values for the randomization
    
    cov[random_value_slice, random_value_slice] = RK.C00([p.location for p in peaks],
                                                         [p.location for p in peaks])
    cov_01 = np.zeros([len(peaks), sum(sizes)])
    cov_11 = np.zeros([sum(sizes), sum(sizes)])
    for i, (p_l, s_l) in enumerate(zip(peaks, slices)):
        for p_r, s_r in zip(peaks, slices):
            cov_01[i,s_r] = RK.C01([p_l.location],
                                   [p_r.location],
                                   basis_r=p_r.tangent_basis)[0,0]
            cov_11[s_l,s_r] = RK.C11([p_l.location],
                                     [p_r.location],
                                     basis_l=p_l.tangent_basis,
                                     basis_r=p_r.tangent_basis)[0,0]
            
    cov[random_value_slice, random_grad_slice] = cov_01
    cov[random_grad_slice, random_value_slice] = cov_01.T
    cov[random_grad_slice, random_grad_slice] = cov_11

    return cov

def _decompose_hessian(peaks_obs,
                       peaks_regress,
                       kernel,
                       slice_info):
    """
    Compute decomposition of Hessian at `peaks`,
    projecting onto gradient and the field at `peaks`.
    """

    sizes = slice_info.sizes
    slices = slice_info.slices

    K = kernel

    _shape = (slice_info.ngrad,
              slice_info.ngrad)
    
    G_pts = []
    G_grads = []
    for p, size_p, slice_p in zip(peaks_obs,
                                            sizes,
                                            slices):
        if hasattr(p, 'hessian'):
            G_p = np.zeros(_shape + (1,))
            G_g = np.zeros(_shape + (size_p,))

            for q, s in zip(peaks_regress, slices):
                C20_q = K.C20([q.location],
                               [p.location],
                              basis_l=q.tangent_basis)[0,0] 
                C21_q = K.C21([q.location],
                               [p.location],
                              basis_l=q.tangent_basis,
                              basis_r=p.tangent_basis)[0,0] 
                G_p[s,s,0] = C20_q
                G_g[s,s,:] = C21_q
            G_pts.append(G_p)
            G_grads.append(G_g)

    return np.concatenate(G_pts, axis=-1), np.concatenate(G_grads, axis=-1)


def _dot_prods(info, full_shape):

    (peaks,
     model_kernel,
     randomizer_kernel,
     sizes,
     slices,
     cov_slices) = (info.peaks,
                    info.model_kernel,
                    info.randomizer_kernel,
                    info.slice_info.sizes,
                    info.slice_info.slices,
                    info.slice_info.cov_slices)
     
    MK = model_kernel
    RK = randomizer_kernel
    
    N_ = np.zeros((sum(sizes), sum(sizes)))
    G_ = np.zeros((sum(sizes), sum(sizes), len(peaks)))
    D_ = np.zeros((sum(sizes), sum(sizes)))
    G_full = np.zeros(full_shape)
    
    # matrix of inner products
    # these get scaled on right by diagonal
    # with blocks like s_j beta_j

    E_ = np.zeros_like(G_)
    for i, (p, s, n) in enumerate(zip(info.peaks, slices, sizes)):
        if hasattr(p, 'hessian'):
            E_[s,s,i] = np.identity(n)

    locations = [q.location for q in peaks]

    C00 = np.zeros((len(peaks),
                    len(peaks)))
    
    for i, q_l in enumerate(peaks):
        for j, q_r in enumerate(peaks):
            C00[i,j] = (MK.C00([q_l.location],
                               [q_r.location])[0,0] +
                        RK.C00([q_l.location],
                               [q_r.location])[0,0])
    C00i = np.linalg.inv(C00)

    for i, (q_l, s_l) in enumerate(zip(peaks, slices)):
        if hasattr(q_l, 'hessian'):
            c10_l = (MK.C10([q_l.location],
                            locations,
                            basis_l=q_l.tangent_basis)[0].T +
                     RK.C10([q_l.location],
                            locations,
                            basis_l=q_l.tangent_basis)[0].T) 
            for j, (q_r, s_r) in enumerate(zip(peaks, slices)):
                if hasattr(q_r, 'hessian'):
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

                    D_[s_l,s_r] = c11 - c10_l @ C00i @ c10_r.T
                    D_[s_r,s_l] = D_[s_l,s_r].T

    for q_l, s_l in zip(peaks, slices):
        D_[s_l] *= q_l.sign

    G_ = np.einsum('ij,jkl,lm->ijm',
                   D_,
                   E_,
                   C00i)

    data_value_slice, random_value_slice, _, _ = cov_slices

    arg = np.zeros_like(info.first_order)
    delta = np.array([p.sign * p.penalty for p in info.peaks])
    arg[data_value_slice] -= delta

    G_full[:,:,data_value_slice] = G_
    G_full[:,:,random_value_slice] = G_
    N_full = np.einsum('ijk,k->ij', G_full, arg)
    return G_full, N_full

def _decompose_KKT(info,
                   extra_points=None):

    (peaks,
     inactive,
     subgrad,
     penalty,
     model_kernel,
     randomizer_kernel,
     inference_kernel,
     slice_info,
     first_order,
     cov,
     contrast_info,
     _,
     _) = info

    sizes, slices, cov_slices, nvalue, ngrad = slice_info
    
    (data_value_slice,
     random_value_slice,
     data_grad_slice,
     random_grad_slice) = cov_slices

    if extra_points is not None:
        sizes = [get_gradient(peak).shape[0] for peak in extra_points]
        extra_slices = [slice(s, e) for s, e in zip(np.cumsum([0] + sizes),
                                                    np.cumsum(sizes))]
        nextra = len(extra_points) + sum(sizes)
        extra_data_slice = slice(2*ntotal, 2*ntotal+len(extra_points))
        extra_grad_slice = slice(2*ntotal+len(extra_points), nextra)
    else:
        nextra = 0
        
    if inference_kernel is None:
        inference_kernel = model_kernel

    MK = model_kernel
    RK = randomizer_kernel
    IK = inference_kernel

    C00 = (MK.C00([p.location for p in peaks],
                  [p.location for p in peaks]) +
           RK.C00([p.location for p in peaks],
                  [p.location for p in peaks]))
    C00i = np.linalg.inv(C00)
    irrep = (MK.C00(None,
                    [p.location for p in peaks]) +
             RK.C00(None,
                    [p.location for p in peaks]))[inactive] @ C00i
    
    pre_proj = np.zeros(subgrad[inactive].shape[:1] + cov.shape[:1])

    # fill in covariance with data and random values
    
    pre_proj[:,data_value_slice] = IK.C00(None,
                                          [p.location for p in peaks])[inactive]
    pre_proj[:,random_value_slice] = RK.C00(None,
                                            [p.location for p in peaks])[inactive]
    # now covariance with data gradient
    
    C01 = np.zeros_like(pre_proj[:,data_grad_slice])
    for i, (p, s) in enumerate(zip(peaks, slices)):
        C01[:,s] = IK.C01(None,
                          [p.location],
                          basis_r=p.tangent_basis)[inactive,0]
    pre_proj[:,data_grad_slice] = C01

    # finally with randomizer gradient

    C01 = np.zeros_like(pre_proj[:,random_grad_slice])
    for i, (p, s) in enumerate(zip(peaks, slices)):
        C01[:,s] = RK.C01(None,
                          [p.location],
                          basis_r=p.tangent_basis)[inactive,0]
    pre_proj[:,random_grad_slice] = C01

    # if any extra points:
    
    if extra_points is not None:
        pre_proj[:,extra_data_slice] = IK.C00(None,
                                              [p.location for p in extra_points])[inactive]
        C01 = np.zeros_like(pre_proj[:,extra_grad_slice])
        for i, (p, s) in enumerate(zip(extra_points, extra_slices)):
            C01[:,s] = IK.C01(None,
                              [p.location],
                              basis_r=p.tangent_basis)[inactive,0]
        pre_proj[:,extra_grad_slice] = C01
        
    proj = pre_proj @ np.linalg.inv(cov)

    L_inactive = proj

    L_inactive[:,data_value_slice] -= irrep
    L_inactive[:,random_value_slice] -= irrep
    offset_inactive = subgrad[inactive] - L_inactive @ info.first_order

    signs = np.array([p.sign for p in peaks])
    L_active = np.zeros((len(peaks), cov.shape[0]))
    L_active[:,data_value_slice] = C00i
    L_active[:,random_value_slice] = C00i

    offset_active = -C00i @ np.array([p.penalty * p.sign for p in peaks])
    L_active = (np.diag(signs) @ L_active) / np.sqrt(np.diag(C00i))[:,None]
    offset_active = signs * offset_active / np.sqrt(np.diag(C00i))

    return ((L_inactive, offset_inactive),
            (L_active, offset_active))

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
