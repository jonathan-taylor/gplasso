from itertools import product
from functools import partial

import numpy as np
import jax.numpy as jnp

from joblib import hash as hash_


def decompose(peaks,
              model_kernel,
              decomp_kernel):

    # shorthand
    DK = decomp_kernel 

    sizes = [q.hessian.shape[0] for q in peaks]
    slices = [slice(s, e) for s, e in zip(np.cumsum([0] + sizes),
                                          np.cumsum(sizes))]

    locations = [q.location for q in peaks]
    obs_ = [q.value for q in peaks]
    C00_model = np.zeros((len(peaks), len(peaks)))

    MK = model_kernel
    C00_model = MK.C00(locations,
                       locations)
    C00i_model = np.linalg.inv(C00_model)

    # decompose the values with respect to
    # value at reference point and possibly
    # gradient at reference point

    common_calcs = (sizes,
                    slices,
                    C00i_model)

    preG_hess, preN_hess, C00i_DK = _hessian(peaks,
                                             decomp_kernel,
                                             sizes,
                                             slices)

    preG_proj, preN_proj = _proj_hessian(peaks,
                                         model_kernel,
                                         sizes,
                                         slices,
                                         C00i_model)

    preG_dot, preN_dot = _dot_prods(peaks,
                                    model_kernel,
                                    sizes,
                                    slices,
                                    C00i_model)

    preG_total = preG_hess + preG_proj + preG_dot
    preN_total = preN_hess + preN_proj + preN_dot

    result = {}
    for g, n, label in zip([preG_total, preG_hess, preG_proj, preG_dot],
                           [preN_total, preN_hess, preN_proj, preN_dot],
                           ['total', 'hessian', 'proj', 'dot_prod']):
        result[label] = (g, n, partial(_logdet, g, n))                          

    return result, obs_, C00i_DK

def _logdet(G, N, val):
    V = -jnp.log(jnp.linalg.det(jnp.einsum('ijk,k->ij', G, val) + N))
    if not np.isnan(V):
        return V
    else:
        return np.inf

def _hessian(peaks,
             decomp_kernel,
             sizes,
             slices):

    DK = decomp_kernel
    locations = [p.location for p in peaks]
    C00_DK = DK.C00(locations,
                    locations)
    C00i_DK = np.linalg.inv(C00_DK)
    
    N_ = np.zeros((sum(sizes), sum(sizes)))
    G_ = np.zeros((sum(sizes), sum(sizes), len(peaks)))

    obs_ = np.array([p.value for p in peaks]).reshape(-1)
    
    for q, s_q in zip(peaks, slices):
        N_[s_q,s_q] = -q.sign * q.hessian

        for i, (p, s_p) in enumerate(zip(peaks, slices)):

            C20_DK = DK.C20([q.location],
                            [p.location])[0,0]

            G_[s_q,s_q,i] = -q.sign * C20_DK

    G_ = np.einsum('ijk,kl->ijl',
                   G_,
                   C00i_DK)

    N_ -= np.einsum('ijk,k->ij',
                    G_,
                    obs_)
    return G_, N_, C00i_DK

def _proj_hessian(peaks,
                  model_kernel,
                  sizes,
                  slices,
                  C00i_model):

    MK = model_kernel
    N_ = np.zeros((sum(sizes), sum(sizes)))
    A_ = np.zeros((sum(sizes), sum(sizes), len(peaks)))

    locations = [p.location for p in peaks]

    loc_obs_ = np.array([p.value - p.sign * p.penalty for p in peaks]).reshape(-1)
    for q, s_q in zip(peaks, slices):
        for i, (p, s_p) in enumerate(zip(peaks, slices)):
            A_[s_q,s_q,i] = -q.sign * MK.C20([q.location],
                                             [p.location])[0,0]  

    G_ = np.einsum('kij,kl->ijl',
                   A_,
                   C00i_model)

    N_ = np.einsum('ijk,k->ij',
                   G_,
                   loc_obs_)

    return G_, N_

def _dot_prods(peaks,
               model_kernel,
               sizes,
               slices,
               C00i_model):

    MK = model_kernel

    N_ = np.zeros((sum(sizes), sum(sizes)))
    G_ = np.zeros((sum(sizes), sum(sizes), len(peaks)))
    D_ = np.zeros((sum(sizes), sum(sizes)))

    # matrix of inner products
    # these get scaled on right by diagonal
    # with blocks like s_j beta_j

    E_ = np.zeros_like(G_)
    for i, (q, s, n) in enumerate(zip(peaks, slices, sizes)):
        E_[s,s,i] = np.identity(n)

    loc_obs_ = np.array([p.value - p.sign * p.penalty for p in peaks]).reshape(-1)

    locations = [q.location for q in peaks]

    for q_l, s_l in zip(peaks, slices):
        c10_l = MK.C10([q_l.location],
                       locations)[0].T
        for q_r, s_r in zip(peaks, slices):
            c11 = MK.C11([q_l.location],
                         [q_r.location])[0,0]
            c10_r = MK.C10([q_r.location],
                           locations)[0].T

            D_[s_l,s_r] = c11 - c10_l @ C00i_model @ c10_r.T
            D_[s_r,s_l] = D_[s_l,s_r].T

    for q_l, s_l in zip(peaks, slices):
        D_[s_l] *= q_l.sign

    G_ = np.einsum('ij,jkl,lm->ijm',
                   D_,
                   E_,
                   C00i_model)
    N_ = np.einsum('ijk,k->ij',
                   G_,
                   loc_obs_)
    return G_, N_

def _compute_resid_cov(covariance_struct,
                       peaks):
    """
    compute Cov(resid(\nabla f_E;f_E)) and Cov(\nabla f_E, f_E) Cov(f_E)^{-1}
    """

    peaks = peaks
    locations = [p.location for p in peaks]
    sizes = [p.gradient.shape[0] for p in peaks]
    slices = [slice(s, e) for s, e in zip(np.cumsum([0] + sizes),
                                          np.cumsum(sizes))]        
    C11_uncond = np.zeros((np.sum(sizes),
                           np.sum(sizes)))
    C10_uncond = np.zeros((np.sum(sizes),
                           len(peaks)))
    K = covariance_struct

    C00_uncond = K.C00(locations,
                       locations)
    for s_l, loc_l in zip(slices, locations):
        C10_uncond[s_l] = K.C10([loc_l],
                                locations)[0].T
        for s_r, loc_r in zip(slices, locations):
            C11_uncond[s_l,s_r] = K.C11([loc_l],
                                        [loc_r])[0,0]

    return C11_uncond, C10_uncond, C00_uncond

