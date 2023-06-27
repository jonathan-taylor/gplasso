from itertools import product
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from gplasso.api import (covariance_structure,
                         default_clusters,
                         taylor_expansion_window,
                         LASSOInference)

def instance(seed=10,
             svd_info=None,
             plot=False):

    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
        
    nx, ny = 41, 51
    
    xval = np.linspace(-5,5,nx)
    yval = np.linspace(-3,8,ny)

    grid = np.meshgrid(xval, yval, indexing='ij')

    precision = np.diag([1.4, 2.1])
    K = covariance_structure.gaussian(precision=precision,
                                      grid=grid)

    Z = K.sample()

    proportion = 0.8
    var_random = (1 - proportion) / proportion
    K_omega = covariance_structure.gaussian(precision=precision,
                                            grid=grid, var=var_random)
    omega = K_omega.sample()
    
    penalty_weights = 2 * np.sqrt(1 + var_random) * np.ones_like(Z)

    lasso = LASSOInference(Z,
                           penalty_weights,
                           K,
                           K_omega,
                           inference_kernel=None)

    E, soln, subgrad = lasso.fit()
    signs = np.sign(subgrad[E])

    # this is 2d grid specific
    
    second_order = taylor_expansion_window((xval, yval),
                                           [Z, lasso.perturbation_],
                                           np.nonzero(E))

    tangent_bases = [np.identity(2) for _ in range(len(E))]
    normal_info = [(np.zeros((0, 2)), np.zeros((0, 0))) for _ in range(len(E))]

    E_nz = np.nonzero(E)
    signs = np.sign(subgrad[E])

    clusters = default_clusters(E,
                                K,
                                cor_threshold=0.9)

    peaks, idx = lasso.extract_peaks(E_nz,
                                     signs,
                                     second_order,
                                     tangent_bases,
                                     normal_info,
                                     clusters=clusters)

    if plot:
        fig, ax = plt.subplots(figsize=(8, 10))
        ax = plt.gca()
        im = ax.imshow(Z.T, cmap='coolwarm')
        fig.colorbar(im, ax=ax, alpha=0.5)
        ax.scatter(E_nz[0][signs==1], E_nz[1][signs==1], c='r', s=70)
        ax.scatter(E_nz[0][signs==-1], E_nz[1][signs==-1], c='b', s=70)
        ax.scatter(idx[:,0], idx[:,1], c='k', marker='x', s=100)

    inactive = np.ones(soln.shape, bool)
    for i, j in zip(*E_nz):
        inactive[max(i-2, 0):(i+2),
                 max(j-2, 0):(j+2)] = 0

    lasso.setup_inference(peaks,
                          inactive,
                          subgrad)

    pivot_carve, disp_carve = lasso.summary(one_sided=False,
                                            param=None,
                                            level=0.9)

    return pivot_carve, svd_info

def test_2d():

    instance()
    

if __name__ == '__main__':
    dfs = []
    svd_info = None
    
    for _ in range(200):
        try:
            df, svd_info = instance(seed=None, svd_info=svd_info)
            dfs.append(df)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(e)
            pass
        if len(dfs) > 0:
            pval = pd.concat(dfs)['P-value (2-sided)']
            print(np.nanmean(pval), np.nanstd(pval), np.nanmean(pval < 0.05))
    
