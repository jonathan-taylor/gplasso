import numpy as np
import jax.numpy as jnp

def test_instance(seed=0, q=5, p=6):

    # make a few PSD matrices

    rng = np.random.default_rng(seed)
    
    A = []

    for _ in range(p+1):
        X = rng.standard_normal((2*q,q))
        S = X.T @ X / (2*q)
        A.append(S)

    B = A[-1]
    A = np.array(A[:-1])

    X_v = rng.standard_normal((2*p,p))
    C = X_v.T @ X_v / (2*p)

    W = rng.standard_normal(p)
    
    def obj(A, B, C, W, v):

        D = jnp.einsum('ijk,k->ij', A, v)
        if jnp.any(jnp.eigvalsh(D + B) < 0):
            return jnp.inf
        return - jnp.log(jnp.linalg.det(D)) + (W * v).sum() + 0.5 * (v * (C @ v)).sum()

    

    
