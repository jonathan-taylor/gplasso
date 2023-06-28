from itertools import product
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from gplasso.api import (discrete_structure,
                         DiscreteLASSOInference)

def instance(seed=10,
             svd_info=None,
             nextra=0):

    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
        
    p = 50

    W = rng.standard_normal((100,p))
    S = W.T @ W / 100
    S += 0.6
    
    K = discrete_structure(S)
    D = discrete_structure(np.diag(np.linspace(1, 1.5, p)))
    Z = K.sample(rng=rng)

    proportion = 0.8
    var_random = (1 - proportion) / proportion
    K_omega = discrete_structure(var_random * np.diag(np.linspace(1, 1.5, p)))
    
    penalty_weights = 2 * np.sqrt(1 + var_random) * np.ones_like(Z)

    lasso = DiscreteLASSOInference(penalty_weights,
                                   D,
                                   K_omega,
                                   inference_kernel=K)

    E, soln, subgrad = lasso.fit(Z,
                                 rng=rng)

    omega = lasso.perturbation_

    if E.sum() > 0:

        signs = np.sign(subgrad[E])

        extra_points = rng.choice(p, nextra)
        model_points = np.hstack([np.nonzero(E)[0], tuple(extra_points)])
        lasso.extract_peaks(model_points)
        
        inactive = np.ones(soln.shape, bool)
        inactive[E] = 0

        lasso.setup_inference(inactive)

        pvalue_carve = lasso.summary(one_sided=False,
                                     param=None,
                                     level=0.9)

        return pvalue_carve, svd_info
    else:
        return None, None

def test_lasso():

    instance(seed=10)

if __name__ == '__main__':

    dfs = []
    svd_info = None
    
    df, svd_info = instance(seed=None, svd_info=svd_info)

    for _ in range(2000):
        try:
            df, svd_info = instance(seed=None, svd_info=svd_info, nextra=2)
            if df is not None:
                dfs.append(df)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(e)
            pass
        if len(dfs) > 0:
            pval = pd.concat(dfs)['P-value (2-sided)']
            print(np.nanmean(pval), np.nanstd(pval), np.nanmean(pval < 0.05))
    
