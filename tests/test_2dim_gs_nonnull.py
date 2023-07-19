from itertools import product
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from gplasso.api import (gaussian_kernel,
                         default_clusters,
                         GridLASSOInference,
                         GSToolsSampler)
from gplasso.utils import MLEInfo

from joblib import Memory
location = './cachedir'
memory = Memory(location, verbose=0)

def get_cov(grid,
            precision):
    K = gaussian_kernel(Q=precision,
                        grid=grid)
    return K.C00(None, None)

get_cov = memory.cache(get_cov)

def instance(seed=10,
             svd_info=None,
             plot=False,
             use_jax=False,
             use_logdet=True,
             null=False):

    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
        
    nx, ny = 41, 51
    
    xval = np.linspace(-5,5,nx)
    yval = np.linspace(-3,8,ny)

    grid = np.meshgrid(xval, yval, indexing='ij')

    precision = np.diag([1.4, 2.1])

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

    # find a solution not on the boundary
    
    s = 5
    
    while True:

        S = get_cov(grid, precision)
        beta = np.zeros((nx, ny))
        supp_x = rng.choice(np.arange(5, nx-5), s)
        supp_y = rng.choice(np.arange(5, ny-5), s)        
        beta[supp_x, supp_y] = (3 + rng.uniform(-1, 1, size=s)) * rng.choice([-1,1], size=s)
        if null:
            beta *= 0
        Z_mean = np.einsum('ijkl,kl->ij', S, beta)

        Z = K.sample(seed=rng.integers(0, 1e6)) + Z_mean

        penalty_weights = 2.5 * np.sqrt(1 + var_random) * np.ones_like(Z)

        lasso = GridLASSOInference(penalty_weights,
                                   K,
                                   K_omega,
                                   inference_kernel=None)

        E, soln, subgrad = lasso.fit(Z, seed=rng.integers(0, 1e6))

        E_nz = np.nonzero(E)
        print(E.sum())
        
        if ((E_nz[0].min() > 1 and E_nz[0].max() < nx-2) and
            (E_nz[1].min() > 1 and E_nz[1].max() < ny-2)):
            break
        else:
            print('try again')

    # this is 2d grid specific
    
    cluster_df = default_clusters(E,
                                  K,
                                  cor_threshold=0.9)
    selection = []

    for label in np.unique(cluster_df['Cluster']):
        cur_df = cluster_df[lambda df: cluster_df['Cluster'] == label]
        selection.append(cur_df.iloc[rng.choice(cur_df.shape[0], 1)])
    selection = pd.concat(selection)

    model_spec = pd.DataFrame({'Value':[True] * len(selection['Index']),
                               'Displacement':[False] * len(selection['Index'])},
                              index=selection['Index'])

    mid = (xval.shape[0]//2, yval.shape[0]//2)
    extra_pt = pd.DataFrame({'Value':[True],
                             'Displacement':[False],
                             'Index':[mid]}).set_index('Index')

    #model_spec = pd.concat([model_spec, extra_pt])
    
    model_spec = lasso.extract_peaks(selection['Index'],
                                     model_spec=model_spec)
    
    idx = model_spec.index.droplevel('Location')
    S_idx = np.array([[S[p_l[0], p_l[1], p_r[0], p_r[1]]
                       for p_l in idx] for p_r in idx])
    mu_idx = np.array([Z_mean[p[0], p[1]] for p in idx])
    
    param_truth = pd.Series(np.linalg.inv(S_idx) @ mu_idx,
                            name='Param',
                            index=model_spec.index.droplevel('Index'))
    E_nz = np.nonzero(E)
    if plot:
        signs = np.sign(subgrad[E_nz])
        fig, ax = plt.subplots(figsize=(8, 10))
        ax = plt.gca()
        im = ax.imshow(Z.T, cmap='coolwarm')
        fig.colorbar(im, ax=ax, alpha=0.5)
        ax.scatter(E_nz[0][signs==1], E_nz[1][signs==1], c='r', s=70)
        ax.scatter(E_nz[0][signs==-1], E_nz[1][signs==-1], c='b', s=70)

    inactive = np.ones(soln.shape, bool)
    for i, j in zip(*E_nz):
        inactive[max(i-2, 0):(i+2),
                 max(j-2, 0):(j+2)] = 0

    param = lasso.setup_inference(inactive)

    pvalue_carve, _ = lasso.summary(one_sided=False,
                                    param=param,
                                    level=0.9,
                                    use_jax=use_jax,
                                    use_logdet=use_logdet)

    param['Param'].values[:] = param_truth

    pivot_carve, _ = lasso.summary(one_sided=False,
                                   param=param,
                                   level=0.9,
                                   use_jax=use_jax,
                                   use_logdet=use_logdet)

    Z_ds = Z - lasso.perturbation_ / var_random
    Z_ds_idx = np.array([Z_ds[p[0], p[1]] for p in idx])
    Q_idx = np.linalg.inv(S_idx)
    est_ds = Q_idx @ Z_ds_idx
    mle_ds = MLEInfo(estimate=est_ds,
                     cov=Q_idx * (1 + var_random) / var_random)

    pvalue_ds = mle_ds.summary(param=param_truth*0,
                               level=0.9)
    pivot_ds = mle_ds.summary(param=param_truth,
                               level=0.9)
    
    pivot_global = lasso.mle_info.linear_hypothesis(np.identity(pivot_carve.shape[0]),
                                                    truth=param_truth).pvalue
    pvalue_global = lasso.mle_info.linear_hypothesis(np.identity(pivot_carve.shape[0]),
                                                     truth=param_truth*0).pvalue

    pivot_global_ds = mle_ds.linear_hypothesis(np.identity(pivot_carve.shape[0]),
                                               truth=param_truth).pvalue
    pvalue_global_ds = mle_ds.linear_hypothesis(np.identity(pivot_carve.shape[0]),
                                                truth=param_truth*0).pvalue
    
    return ((pivot_carve, pvalue_carve, pivot_global, pvalue_global),
            (pivot_ds, pvalue_ds, pivot_global_ds, pvalue_global_ds))

def test_2d():

    instance(seed=7)
    

if __name__ == '__main__':
    dfs = []
    P_global = []
    Pivot_global = []
    for _ in range(2000):
        try:
            carve, ds = instance(seed=None)
            pivot, pvalue, pivot_global, pvalue_global = carve
            pivot_ds, pvalue_ds, pivot_global_ds, pvalue_global_ds = ds
            print('num peaks: ', pivot.shape[0])
            pvalue['Hypothesis'] = pvalue_ds['Hypothesis'] = 'Null'
            pivot['Hypothesis'] = pivot_ds['Hypothesis'] = 'Alternative'
            pvalue['Method'] = pivot['Method'] = 'Carve'
            pvalue_ds['Method'] = pivot_ds['Method'] = 'Split'
            df = pd.concat([pvalue, pivot, pvalue_ds, pivot_ds])
            dfs.append(df)
            P_global.extend([(pvalue_global, 'Null', 'Carve'),
                             (pvalue_global_ds, 'Null', 'Split')])
            Pivot_global.extend([(pivot_global, 'Alternative', 'Carve'),
                                 (pivot_global_ds, 'Alternative', 'Split')])

        except KeyboardInterrupt:
            break
        except Exception as e:
            print('except:', type(e), e)
            pass
        if len(dfs) > 0:
            df_ = pd.concat(dfs)
            for (l, m), d in df_.groupby(['Hypothesis', 'Method']):
                pval = d['P-value (2-sided)']
                L = d['U (90%)'] - d['L (90%)']
                if l == 'Alternative':
                    coverage = np.nanmean((d['Param'] < d['U (90%)']) *
                                          (d['Param'] > d['L (90%)']))
                else:
                    coverage = np.nan
                    L *= np.nan
                print('Coord,{0},{1}:'.format(l,m),
                      np.nanmean(pval),
                      np.nanstd(pval),
                      np.nanmean(pval < 0.05),
                      np.nanmedian(L),
                      coverage)

            df_pvalue = pd.DataFrame(P_global, columns=['P-value', 'Hypothesis', 'Method'])
            df_pivot = pd.DataFrame(Pivot_global, columns=['P-value', 'Hypothesis', 'Method'])
            df_ = pd.concat([df_pvalue, df_pivot])
            for (l, m), d in df_.groupby(['Hypothesis', 'Method']):
                pval = d['P-value']
                print('Omnibus,{0},{1}:'.format(l,m),
                      np.nanmean(pval),
                      np.nanstd(pval),
                      np.nanmean(pval < 0.05))
                
