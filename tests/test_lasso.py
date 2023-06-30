from itertools import product
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from gplasso.api import (discrete_structure,
                         DiscreteLASSOInference)

def instance(seed=10,
             svd_info=None,
             nextra=0,
             ndrop=0,
             s=3):
    """
    fit a misspecified lasso, select a few extra coordinates
    and make some confidence intervals
    """

    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
        
    p = 50

    W = rng.standard_normal((100,p))
    S = W.T @ W / 100
    S[:5,:5] += 0.2
    beta = np.zeros(p)
    supp = rng.choice(p, s)
    beta[supp] = rng.normal(s) + 2 * rng.choice([-1,1], s)
    
    K = discrete_structure(S)
    D = discrete_structure(np.diag(np.linspace(1, 1.5, p)))
    Z_mean = S @ beta 
    Z = K.sample(rng=rng) + Z_mean

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

        extra_points = rng.choice(p, nextra, replace=False)
        selected_points = np.nonzero(E)[0]
        if ndrop > 0:
            ndrop = min(ndrop, selected_points.shape[0])
            selected_points = rng.choice(selected_points, selected_points.shape[0] - ndrop, replace=False)
        model_spec = np.unique(np.hstack([selected_points, tuple(extra_points)]).astype(int))
        print(model_spec)
        
        inactive = np.ones(soln.shape, bool)
        inactive[E] = 0

        lasso.setup_inference(inactive,
                              model_spec)

        param = np.linalg.inv(S[model_spec][:,model_spec]) @ Z_mean[model_spec]
        param = pd.DataFrame({'Param': param,
                              'Location': lasso.model_spec}).set_index('Location')
        pivot_carve = lasso.summary(one_sided=False,
                                    param=param,
                                    level=0.9)

        return pivot_carve, svd_info
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
            df, svd_info = instance(seed=None, svd_info=svd_info, nextra=2, ndrop=2)
            if df is not None:
                dfs.append(df)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(e)
            pass
        if len(dfs) > 0:
            df_ = pd.concat(dfs)
            pval = df_['P-value (2-sided)']
            print(np.nanmean(pval),
                  np.nanstd(pval),
                  np.nanmean(pval < 0.05),
                  np.nanmean((df_['Param'] < df_['U (90%)']) *
                             (df_['Param'] > df_['L (90%)'])))

    
