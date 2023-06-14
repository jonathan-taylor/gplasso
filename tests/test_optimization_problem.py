import numpy as np
import jax.numpy as jnp
from jax import jacfwd

from optimization_problem import logdet, barrier

rng = np.random.default_rng(0)

def test_logdet_jax():
    L = logdet()

    def _logdet(arg):
        return -jnp.log(jnp.linalg.det(arg))

    _logdet_G = jacfwd(_logdet)
    _logdet_H = jacfwd(_logdet_G)

    X = rng.standard_normal((10, 5))
    W = X.T @ X / 10

    l = rng.standard_normal((5, 5))
    l = l + l.T
    r = rng.standard_normal((5, 5))
    r = r + r.T

    assert np.fabs(L.value(W) - _logdet(W)) / np.fabs(L.value(W)) < 1e-4
    assert np.fabs(L.gradient(W, r) - np.diag(_logdet_G(W) @ r).sum()) / np.fabs(L.gradient(W, r))< 1e-4
    assert np.fabs(L.hessian(W, l, r) - np.einsum('ijkl,ij,kl', _logdet_H(W), l, r)) / np.fabs(L.hessian(W, l, r)) < 1e-4

def test_shape_logdet():
    L = logdet()

    X = rng.standard_normal((10, 5))
    W = X.T @ X / 10

    l = rng.standard_normal((5, 5, 4))
    l = l + l.transpose([1,0,2])
    r = rng.standard_normal((5, 5, 6))
    r = r + r.transpose([1,0,2])

    assert L.hessian(W, l, r).shape == (4, 6)
    assert L.gradient(W, l).shape == (4,)

def test_barrier():

    B = barrier()

    arg = np.fabs(rng.standard_normal(50))
    l = rng.standard_normal((50, 4))
    r = rng.standard_normal((50, 6))
    assert B.gradient(arg, l).shape == (4,)
    assert B.hessian(arg, l, r).shape == (4, 6)

