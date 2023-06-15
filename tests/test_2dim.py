from itertools import product
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from gplasso.kernel_calcs import covariance_structure
from gplasso.peaks import extract_peaks, extract_points
from gplasso.taylor_expansion import taylor_expansion_window
from gplasso.general_inference import LASSOInference, inference
from gplasso.gplasso import fit_gp_lasso

def instance(seed=10,
             svd_info=None):

    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
        
    nx, ny = 31, 51
    
    xval = np.linspace(-5,5,nx)
    yval = np.linspace(-3,8,ny)

    grid = np.meshgrid(xval, yval, indexing='ij')

    precision = np.diag([1.4, 2.1])
    K = covariance_structure.gaussian(precision=precision,
                                      grid=grid)

    if svd_info is None:
        S_ = K.C00(None, None).reshape(nx*ny, nx*ny)
        A, D = np.linalg.svd(S_)[:2]
        svd_info = A, D
    else:
        A, D = svd_info
        
    Z = A @ (np.sqrt(D) * rng.standard_normal(nx*ny))
    Z = Z.reshape((nx, ny))

    proportion = 0.8
    var_random = (1 - proportion) / proportion
    K_omega = covariance_structure.gaussian(precision=precision,
                                            grid=grid, var=var_random)
    omega = A @ (np.sqrt(D) * rng.standard_normal(nx*ny) * np.sqrt(var_random))
    omega = omega.reshape((nx, ny))
    
    penalty_weights = 2 * np.sqrt(1 + var_random) * np.ones_like(Z)
    E, soln, subgrad = fit_gp_lasso(Z + omega,
                                    [K, K_omega],
                                    penalty_weights)

    signs = np.sign(subgrad[E])

    second_order = taylor_expansion_window((xval, yval),
                                           [Z, omega],
                                           np.nonzero(E))

    tangent_bases = [np.identity(2) for _ in range(len(E))]
    normal_info = [(np.zeros((0, 2)), np.zeros((0, 0))) for _ in range(len(E))]

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
    for i, j in zip(*E_nz):
        inactive[max(i-2, 0):(i+2),
                 max(j-2, 0):(j+2)] = 0

    info = LASSOInference(peaks,
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
    
