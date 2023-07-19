from functools import partial
from dataclasses import dataclass
import numpy as np

from .jax.optimization_problem import jax_spec

@dataclass
class barrier(object):

    shift : float = 1
    scale : float = 0.5

    def value(self, arg):
        if np.any(arg < 0):
            return np.inf
        val = self.scale * (-np.log(arg / (arg + self.shift)).sum())
        return val
    
    def gradient(self, arg, d):
        # d of shape arg.shape + ...
        d_shape = d.shape
        d = d.reshape(arg.shape + (-1,))
        G =  self.scale * (-self.shift / (arg * (arg + self.shift)))
        val = (G @ d).reshape(d_shape[1:])
        if np.any(arg < 0):
            val = val * np.nan
        return val

    def hessian(self, arg, l, r):
        # l of shape (arg.shape,-1)
        # r of shape (arg.shape,-1)
        l_shape = l.shape
        l = l.reshape(arg.shape + (-1,))
        r_shape = r.shape
        r = r.reshape(arg.shape + (-1,))
        H = self.scale * ((self.shift**2 + 2 * self.shift * arg) / ((arg + self.shift)**2 * arg**2))
        val = l.T @ (H[:,None] * r)
        val = val.reshape(l_shape[1:] + r_shape[1:])
        if np.any(arg < 0):
            val = val * np.nan
        return val

    @staticmethod
    def compose(G,
                N,
                offset,
                A1,
                A2,
                shift=1,
                scale=0.5):
        B = barrier(shift=shift,
                    scale=scale)

        return _compose(B,
                        G,
                        N,
                        offset,
                        A1,
                        A2)

class logdet(object):

    def value(self, arg):
        arg = 0.5 * (arg + arg.T)
        eigvals = np.linalg.eigvalsh(arg)
        if np.any(eigvals < 0):
            return np.inf
        return -np.log(eigvals).sum()

    def gradient(self, arg, d):
        # d of shape (arg.shape,-1)
        arg = 0.5 * (arg + arg.T)
        d_shape = d.shape
        d = d.reshape(arg.shape + (-1,))
        arg_i = np.linalg.inv(arg)
        eigvals = np.linalg.eigvalsh(arg)
        if np.any(eigvals < 0):
            factor = np.nan
        else:
            factor = 1
        return -factor * np.einsum('ij,ijk->k', arg_i, d).reshape(d_shape[2:])

    def hessian(self, arg, l, r):
        # l of shape (arg.shape,-1)
        # r of shape (arg.shape,-1)
        arg = 0.5 * (arg + arg.T)
        eigvals = np.linalg.eigvalsh(arg)
        if np.any(eigvals < 0):
            factor = np.nan
        else:
            factor = 1
        l_shape = l.shape
        l = l.reshape(arg.shape + (-1,))
        r_shape = r.shape
        r = r.reshape(arg.shape + (-1,))
        arg_i = np.linalg.inv(arg)
        Vl = np.einsum('ijk,jm->imk', l, arg_i)
        Vr = np.einsum('ijk,jm->imk', r, arg_i)
        value = np.einsum('ijk,jim->km', Vl, Vr)
        return factor * value.reshape(l_shape[2:] + r_shape[2:])

    @staticmethod
    def compose(G,
                N,
                offset,
                A1,
                A2,
                shift=1,
                scale=0.5):
        L = logdet()

        return _compose(L,
                        G,
                        N,
                        offset,
                        A1,
                        A2)

def _compose(O,
             G,
             N,
             offset,
             A1,
             A2):

    N_ = N + G @ offset
    G1 = G @ A1
    G2 = G @ A2

    def obj(v1, v2):
        arg = G1 @ v1 + G2 @ v2 + N_
        return O.value(arg)

    def grad(v1, v2):
        arg = G1 @ v1 + G2 @ v2 + N_
        return O.gradient(arg, G1), O.gradient(arg, G2)

    def hess(v1, v2):
        arg = G1 @ v1 + G2 @ v2 + N_
        H11 = O.hessian(arg, G1, G1)
        H12 = O.hessian(arg, G1, G2)
        H21 = H12.T
        H22 = O.hessian(arg, G2, G2)
        return [[H11, H12], [H21, H22]]

    return obj, grad, hess

def optimization_spec(offset,
                      L_beta,
                      sqrt_cov_R,
                      logdet_info,
                      barrierI_info,
                      barrierA_info,
                      use_jax=False):

    G_logdet, N_logdet = logdet_info
    G_barrierI, N_barrierI = barrierI_info
    G_barrierA, N_barrierA = barrierA_info

    if not use_jax:
        (O_L,
         G_L,
         H_L) = logdet.compose(G_logdet,
                               N_logdet,
                               offset,
                               L_beta,
                               sqrt_cov_R)

        (O_BI,
         G_BI,
         H_BI) = barrier.compose(G_barrierI,
                                 N_barrierI,
                                 offset,
                                 L_beta,
                                 sqrt_cov_R,
                                 shift=0.5,
                                 scale=1/G_barrierI.shape[0])

        (O_BA,
         G_BA,
         H_BA) = barrier.compose(G_barrierA,
                                 N_barrierA,
                                 offset,
                                 L_beta,
                                 sqrt_cov_R,
                                 shift=1,
                                 scale=1)

        def O_np(beta, W):
            return O_L(beta, W) + O_BI(beta, W) + O_BA(beta, W)

        def G_np(beta, W):
            g_L = G_L(beta, W)
            g_BI = G_BI(beta, W)
            g_BA = G_BA(beta, W)
            
            return [g_L[0]+g_BI[0]+g_BA[0],
                    g_L[1]+g_BI[1]+g_BA[1]]

        def H_np(beta, W):
            h_L = H_L(beta, W)
            h_BI = H_BI(beta, W)
            h_BA = H_BA(beta, W)
            
            H_00 = (h_L[0][0] +
                    h_BI[0][0] +
                    h_BA[0][0])
            H_01 = (h_L[0][1] +
                    h_BI[0][1] +
                    h_BA[0][1])
            H_10 = H_01.T
            H_11 = (h_L[1][1] +
                    h_BI[1][1] +
                    h_BA[1][1])
            return [[H_00,H_01],[H_10,H_11]]

        return O_np, G_np, H_np
    else:
        return jax_spec(offset,
                        L_beta,
                        sqrt_cov_R,
                        logdet_info,
                        barrierI_info,
                        barrierA_info)
