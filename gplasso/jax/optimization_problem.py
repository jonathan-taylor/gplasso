from functools import partial

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

def jax_spec(offset,
             L_beta,
             sqrt_cov_R,
             logdet_info,
             barrierI_info,
             barrierA_info,
             use_logdet=True):

    G_logdet, N_logdet = logdet_info
    G_barrierI, N_barrierI = barrierI_info
    G_barrierA, N_barrierA = barrierA_info


    logdet_ = partial(logdet_jax,
                      G_logdet,
                      N_logdet)

    barrierA_ = partial(barrier_jax,
                        G_barrierA,
                        N_barrierA,
                        1,
                        1)

    barrierI_ = partial(barrier_jax,
                        G_barrierI,
                        N_barrierI,
                        0.5 / G_barrierI.shape[0],
                        1)
    if use_logdet:
        objs = [logdet_, barrierA_, barrierI_]
    else:
        objs = [barrierA_, barrierI_]

    (O_jax,
     G_jax,
     H_jax) = _obj_maker([logdet_, barrierA_, barrierI_],
                         offset,
                         L_beta,
                         sqrt_cov_R)

    return O_jax, G_jax, H_jax

