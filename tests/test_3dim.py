from itertools import product
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from gplasso.api import (gaussian_kernel,
                         default_clusters,
                         GridLASSOInference,
                         GSToolsSampler)

def instance(seed=9,
             svd_info=None,
             plot=False):

    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
        
    nx, ny, nz = 40, 25, 30
    
    xval = np.linspace(-5,5,nx)
    yval = np.linspace(-3,8,ny)
    zval = np.linspace(0,2,nz)

    grid = np.meshgrid(xval, yval, zval, indexing='ij')

    precision = np.diag([1.4, 2.1, 1.1])

    K_sampler = GSToolsSampler.gaussian(grid, precision, var=1)
    K = gaussian_kernel(Q=precision,
                        grid=grid,
                        sampler=K_sampler,
                        var=1)

    proportion = 0.8
    var_random = (1 - proportion) / proportion
    omega_sampler = GSToolsSampler.gaussian(grid, precision, var=var_random)
    
    K_omega = gaussian_kernel(Q=precision,
                              grid=grid,
                              var=var_random,
                              sampler=omega_sampler)

    while True:
        Z = K.sample(seed=rng.integers(0, 1e6))

        penalty_weights = 3.2 * np.sqrt(1 + var_random) * np.ones_like(Z)

        lasso = GridLASSOInference(penalty_weights,
                                   K,
                                   K_omega,
                                   inference_kernel=None)

        E, soln, subgrad = lasso.fit(Z, seed=rng.integers(0, 1e6))


        # this is 3d grid specific

        if E.sum():
            cluster_df = default_clusters(E,
                                          K,
                                          cor_threshold=0.9)
            selection = []

            for label in np.unique(cluster_df['Cluster']):
                cur_df = cluster_df[lambda df: cluster_df['Cluster'] == label]
                selection.append(cur_df.iloc[rng.choice(cur_df.shape[0], 1)])
            selection = pd.concat(selection)

            tests = []
            for ix, iy, iz in selection['Index']:
                test_x = (ix > 1) * (ix < nx-2)
                test_y = (iy > 1) * (iy < ny-2)
                test_z = (iz > 1) * (iz < nz-2)
                tests.append(test_x * test_y * test_z)
            if np.all(tests):
                break
            else:
                print('try again')

    model_spec = pd.DataFrame({'Value':[True] * len(selection['Index']),
                               'Displacement':[False] * len(selection['Index'])},
                              index=selection['Index'])
    model_spec.loc[[selection['Index'].iloc[0]],'Value'] = False
    mid = (xval.shape[0]//2, yval.shape[0]//2, zval.shape[0]//2)
    extra_pt = pd.DataFrame({'Value':[True],
                             'Displacement':[False],
                             'Index':[mid]}).set_index('Index')

    model_spec = pd.concat([model_spec, extra_pt])

    model_spec = lasso.extract_peaks(selection['Index'],
                                     model_spec=model_spec)

    E_nz = np.nonzero(E)

    inactive = np.ones(soln.shape, bool)
    for i, j, k in zip(*E_nz):
        inactive[max(i-2, 0):(i+2),
                 max(j-2, 0):(j+2),
                 max(k-2, 0):(k+2)] = 0

    param_default = lasso.setup_inference(inactive)

    pivot_carve, disp_carve = lasso.summary(one_sided=False,
                                            param=param_default,
                                            level=0.9)

    return pivot_carve

def test_3d():

    instance(seed=7)
    

if __name__ == '__main__':
    dfs = []
    
    for _ in range(200):
        try:
            df = instance(seed=None)
            if df is not None:
                dfs.append(df)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print('except:', type(e), e)
            raise(e)
            pass
        if len(dfs) > 0:
            pval = pd.concat(dfs)['P-value (2-sided)']
            print(np.nanmean(pval), np.nanstd(pval), np.nanmean(pval < 0.05))
    
