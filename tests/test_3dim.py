from itertools import product
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from kernel_calcs import covariance_structure
from peaks import extract_peaks, extract_points
from taylor_expansion import taylor_expansion_window
from randomized_inference import setup_inference, inference

from gplasso import fit_gp_lasso

def test_3dim(seed=10, svd_info=None):

    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
        
    nx, ny, nz = 31, 21, 11
    
    xval = np.linspace(-5,5,nx)
    yval = np.linspace(-3,8,ny)
    zval = np.linspace(-2,0,nz)
    grid = np.meshgrid(xval, yval, zval, indexing='ij')

    precision = np.diag([1.4, 1.3, 1.1])
    K = covariance_structure.gaussian(precision=precision,
                                      grid=grid)

    if svd_info is None:
        S_ = K.C00(None, None).reshape(nx*ny*nz, nx*ny*nz)
        A, D = np.linalg.svd(S_)[:2]
        svd_info = A, D
    else:
        A, D = svd_info

    Z = A @ (np.sqrt(D) * rng.standard_normal(nx*ny*nz))
    Z = Z.reshape((nx, ny, nz))

    proportion = 0.8
    var_random = (1 - proportion) / proportion
    K_omega = covariance_structure.gaussian(precision=precision,
                                            grid=grid, var=var_random)
    omega = A @ (np.sqrt(D) * rng.standard_normal(nx*ny*nz) * np.sqrt(var_random))
    omega = omega.reshape((nx, ny, nz))
    
    penalty_weights = 2 * np.sqrt(1 + var_random) * np.ones_like(Z)
    E, soln, subgrad = fit_gp_lasso(Z + omega,
                                    [K, K_omega],
                                    penalty_weights)
    
    signs = np.sign(subgrad[E])

    second_order = taylor_expansion_window((xval, yval, zval),
                                           [Z, omega],
                                           np.nonzero(E))

    tangent_bases = [np.identity(3) for _ in range(len(E))]
    normal_info = [(np.zeros((0, 3)), np.zeros((0, 0))) for _ in range(len(E))]

    E_nz = np.nonzero(E)

    peaks, clusters, _ = extract_peaks(E_nz,
                                       second_order,
                                       tangent_bases,
                                       normal_info,
                                       K,
                                       signs,
                                       penalty_weights,
                                       seed=1)

    inactive = np.ones(soln.shape, bool)
    for i, j, k in zip(*E_nz):
        inactive[max(i-2, 0):(i+2),
                 max(j-2, 0):(j+2),
                 max(k-2, 0):(k+2)] = 0

    info = setup_inference(peaks,
                           inactive,
                           subgrad,
                           penalty_weights,
                           K,
                           K_omega,
                           inference_kernel=None,
                           displacement=True)

    pivot_carve, disp_carve = inference(info,
                                        one_sided=False,
                                        param=None,
                                        level=0.9)

    return pivot_carve, svd_info

if __name__ == '__main__':
    dfs = []
    svd_info = None
    for _ in range(200):
        try:
            df, svd_info = test_3dim(seed=None, svd_info=svd_info)
            dfs.append(df)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(e)
            pass
        if len(dfs) > 0:
            pval = pd.concat(dfs)['P-value (2-sided)']
            print(np.nanmean(pval), np.nanstd(pval), np.nanmean(pval < 0.05))
    
