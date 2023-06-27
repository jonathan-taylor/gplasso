import numpy as np
from scipy.special import gamma
from string import ascii_lowercase

def taylor_expansion(X_pred,
                     X_obs,
                     Y_obs,
                     precision,
                     order=2):

    X_obs = np.asarray(X_obs)
    X_pred = np.asarray(X_pred)
    n_obs, n_dim = X_obs.shape
    Y_shape = Y_obs.shape[1:]
    
    dX = np.array([np.subtract.outer(X_obs[:,i], X_pred[i]) for i in range(n_dim)])
    W = np.exp(-0.5 * np.einsum('ix,jx,ij->x', dX, dX, precision))

    letters = ascii_lowercase[:order]
    dXs = [np.einsum(','.join(['{}i'.format(l) for l in letters[:i]]) + '->i' + letters[:i],
                     *([dX]*i)) / gamma(i+1) for i in range(1, order+1)]
    dXs = [d.reshape((n_obs,-1)) for d in dXs]
    features = np.hstack([np.ones((n_obs,1))] + dXs)
                          
    fW = features * W[:,None]
    XtX = fW.T @ features
    XtY = fW.T @ Y_obs 

    coef = np.linalg.pinv(XtX) @ XtY
    coef = coef.T
    sizes = [n_dim**d for d in range(order+1)]
    shapes = [(n_dim,)*d for d in range(order+1)]
    slices = [slice(s, e) for s, e in zip(np.cumsum([0] + sizes),
                                          np.cumsum(sizes))]
    if Y_shape:
        expansion = [coef[:,sl].reshape((-1,) + s) for sl, s in zip(slices, shapes)]
    else:
        expansion = [coef[sl].reshape(s) for sl, s in zip(slices, shapes)]

    return expansion

def taylor_expansion_window(G,
                            Z,  
                            I,
                            window_size=5,
                            precision=None,
                            order=2,
                            use_numpy=True):

    s = window_size
    value = []

    Z = np.asarray(Z)
    
    if Z.ndim == 1:
        Z_ = Z.reshape((1, -1))
    else:
        Z_ = Z
    if not use_numpy:

        for idx in I:
            W = tuple([slice(max(v-s,0), min(v+s,G.shape[i]))
                       for i, v in enumerate(idx)])
            Y_obs = Z_[(slice(None, None, None),) + W].reshape(Z_.shape[:1] + (-1,)).T
            X_obs = G[W].reshape((-1, G.shape[-1]))
            X_pred = G[tuple(idx)]
            value.append(taylor_expansion(X_pred,
                                          X_obs,
                                          Y_obs,
                                          precision=precision,
                                          order=order))
    else:
        grad = np.gradient(Z_, *G, axis=range(1,len(Z_.shape)))
        if Z_.ndim == 2:
            grad = [grad]
        grad = np.array(grad)
        hess = [np.gradient(g, *G, axis=range(1,len(Z_.shape))) for g in grad]
        if Z_.ndim == 2:
            hess = [hess]

        grad = np.swapaxes(grad, 0, 1)

        hess = np.array(hess)
        hess = np.swapaxes(hess, 0, 2)
        value = []
        for idx in zip(*I):
            item_ = (slice(None,None,None),) + idx
            val_idx = Z_[item_]
            item_ = (slice(None,None,None),) + item_
            grad_idx = grad[item_]
            item_ = (slice(None,None,None),) + item_
            hess_idx = hess[item_]
            hess_idx = np.array([(h + h.T)/2 for h in hess_idx])
            value.append((val_idx, grad_idx, hess_idx))
                
    return value


