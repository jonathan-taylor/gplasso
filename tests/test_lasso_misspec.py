from itertools import product
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from kernel_calcs import covariance_structure, discrete_structure
from peaks import extract_peaks, extract_points
from taylor_expansion import taylor_expansion_window
from randomized_inference import setup_inference, inference

import regreg.api as rr


def test_lasso(seed=10,
               svd_info=None):

    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
        
    p = 50

    W = rng.standard_normal((1000,p))
    S = W.T @ W / p**2
    
    K = discrete_structure(S)

    if svd_info is None:
        S_ = K.C00(None, None).reshape(p, p)
        A, D = np.linalg.svd(S_)[:2]
        svd_info = A, D
    else:
        A, D = svd_info
        S_ = (A * D[None,:]) @ A.T
    Z = A @ (np.sqrt(D) * rng.standard_normal(p))

    proportion = 0.8
    var_random = (1 - proportion) / proportion
    K_omega = discrete_structure(var_random * S)
    K_infer = discrete_structure(S)

    omega = A @ (np.sqrt(D) * rng.standard_normal(p) * np.sqrt(var_random))
    
    penalty_weights = 2 * np.sqrt(1 + var_random) * np.ones_like(Z)

    S_r = S_ * (1 + var_random)
    loss = rr.quadratic_loss(Z.shape[0], Q=S_r)
    linear_term = rr.identity_quadratic(0, 0, -Z-omega,0)
    penalty = rr.weighted_l1norm(penalty_weights, lagrange=1)
    problem = rr.simple_problem(loss,
                                penalty)
    soln = problem.solve(linear_term, min_its=200, tol=1e-12)
    E = soln != 0
    subgrad = Z+omega - S_r @ soln
    
    if E.sum() > 0:
        signs = np.sign(subgrad[E])

        second_order = []
        for i in np.nonzero(E)[0]:
            second_order.append((np.array([Z[i], omega[i]]),
                                 np.zeros((2,0)),
                                 np.zeros((2,0,0))))

        tangent_bases = [np.identity(0) for _ in range(len(E))]
        normal_info = [(np.zeros((0, 0)), np.zeros((0, 0))) for _ in range(len(E))]

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
        inactive[E_nz] = 0

        info = setup_inference(peaks,
                               inactive,
                               subgrad,
                               penalty_weights,
                               K,
                               K_omega,
                               inference_kernel=K_infer,
                               displacement=True)

        pivot_carve, disp_carve = inference(info,
                                            one_sided=False,
                                            param=None,
                                            level=0.9)

        return pivot_carve, svd_info
    else:
        return None, None

if __name__ == '__main__':

    dfs = []
    svd_info = None
    
    df, svd_info = test_lasso(seed=None, svd_info=svd_info)

    for _ in range(2000):
        try:
            df, svd_info = test_lasso(seed=None, svd_info=svd_info)
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
    
