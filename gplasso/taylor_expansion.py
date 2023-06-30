from dataclasses import dataclass

import numpy as np
from scipy.special import gamma
from string import ascii_lowercase

from .peaks import InteriorPoint

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


