from functools import partial

import numpy as np
import jax.numpy as jnp
from jax import jacfwd

from gplasso.optimization_problem import logdet, barrier

def test_logdet(seed=0, q=5, p=6):

    # make a few PSD matrices

    rng = np.random.default_rng(seed)
    
    A = []

    def _simW(q):
        X = rng.standard_normal((2*q,q))
        S = X.T @ X / (2*q)
        return S

    G = np.array([_simW(q) for _ in range(p)]).transpose([1,2,0])
    N = _simW(q)
    
    def obj(G, N, offset, A1, A2, v1, v2):

        arg = offset + A1 @ v1 + A2 @ v2
        M = jnp.einsum('ijk,k->ij', G, arg) + N

        evals = jnp.linalg.eigvalsh(M)
        if jnp.any(evals < 0):
            return jnp.inf
        return - jnp.log(evals).sum()

    s1, s2 = 4, 7
    A1 = 0.5 * rng.uniform(1, 5, size=(p, s1))
    A2 = 0.5 * rng.uniform(1, 5, size=(p, s2))
    offset = rng.uniform(3, 4, size=(p,))

    O_jax = partial(obj,
                    G,
                    N, 
                    offset,
                    A1,
                    A2)

    G_jax = jacfwd(O_jax, argnums=(0,1))
    H_jax = jacfwd(G_jax, argnums=(0,1))

    v1 = np.fabs(rng.standard_normal(s1))
    v2 = np.fabs(rng.standard_normal(s2))

    o_jax = O_jax(v1, v2)
    g_jax = G_jax(v1, v2)
    h_jax = H_jax(v1, v2)

    O_, G_, H_ = logdet.compose(G,
                                N,
                                offset,
                                A1,
                                A2)
    o_ = O_(v1, v2)
    g_ = G_(v1, v2)
    h_ = H_(v1, v2)

    assert np.allclose(o_, o_jax)
    assert np.allclose(g_[0], g_jax[0])
    assert np.allclose(g_[1], g_jax[1])
    assert np.allclose(h_[0][0], h_jax[0][0])
    assert np.allclose(h_[0][1], h_jax[0][1])
    assert np.allclose(h_[1][0], h_jax[1][0])        
    assert np.allclose(h_[1][1], h_jax[1][1])        


def test_barrier(seed=0, ncon=10, p=5,
                 scale=2, shift=4):

    # make a few PSD matrices

    s1, s2 = 3, 4

    rng = np.random.default_rng(seed)
    
    G = rng.uniform(2, 5, size=(ncon, p))
    N = rng.uniform(1, 10, size=(ncon,))
    offset = rng.uniform(3, 5, size=(p,))
    A1 = rng.uniform(1, 2, size=(p, s1))
    A2 = rng.uniform(1, 2, size=(p, s2))
    
    def obj(G, N, offset, A1, A2, v1, v2):

        arg = offset + A1 @ v1 + A2 @ v2
        M = G @ arg + N
        return -scale*jnp.log(M/(M+shift)).sum()

    O_jax = partial(obj,
                    G,
                    N, 
                    offset,
                    A1,
                    A2)

    G_jax = jacfwd(O_jax, argnums=(0,1))
    H_jax = jacfwd(G_jax, argnums=(0,1))

    v1 = np.fabs(rng.standard_normal(s1))
    v2 = np.fabs(rng.standard_normal(s2))

    o_jax = O_jax(v1, v2)
    g_jax = G_jax(v1, v2)
    h_jax = H_jax(v1, v2)

    O_, G_, H_ = barrier.compose(G,
                                 N,
                                 offset,
                                 A1,
                                 A2,
                                 shift=shift,
                                 scale=scale)
    o_ = O_(v1, v2)
    g_ = G_(v1, v2)
    h_ = H_(v1, v2)

    assert np.allclose(o_, o_jax)
    assert np.allclose(g_[0], g_jax[0])
    assert np.allclose(g_[1], g_jax[1])
    assert np.allclose(h_[0][0], h_jax[0][0])
    assert np.allclose(h_[0][1], h_jax[0][1])
    assert np.allclose(h_[1][0], h_jax[1][0])        
    assert np.allclose(h_[1][1], h_jax[1][1])        
    
