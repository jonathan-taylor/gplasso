from functools import partial
from dataclasses import dataclass
import numpy as np

import jax.numpy as jnp
from jax import jacfwd

def logdet_jax(G,
               N,
               arg):
    A = jnp.einsum('ijk,k->ij', G, arg) + N
    A = 0.5 * (A + A.T)
    eigvals = jnp.linalg.eigvalsh(A)
    if jnp.any(eigvals) < 0:
        return jnp.inf
    return -jnp.sum(jnp.log(eigvals))

def barrier_jax(G,
                N,
                scale,
                shift,
                arg):
    arg = G @ arg + N
    val = jnp.log(arg / (arg + shift))
    if jnp.any(arg <= 0):
        return np.inf
    val = -jnp.sum(val)
    return scale * val

def _obj_maker(objs,
               offset,
               L_beta,
               L_W):

    def _new(offset,
             L_beta,
             L_W,
             beta,
             W):
        arg = offset + L_W @ W + L_beta @ beta
        val = 0
        for obj in objs:
            val = val + obj(arg)
        return val
    
    final_obj = partial(_new,
                        offset,
                        L_beta,
                        L_W)
    final_grad = jacfwd(final_obj, argnums=(0,1))
    final_hess = jacfwd(final_grad, argnums=(0,1))

    return final_obj, final_grad, final_hess

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
