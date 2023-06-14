# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: py:light,ipynb
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: gplasso
#     language: python
#     name: gplasso
# ---

from itertools import product
import matplotlib.pyplot as plt
from joblib import hash as _hash
import os

import numpy as np
import pandas as pd
from kernel_calcs import covariance_structure, gaussian_kernel
from peaks import InteriorPeak, extract_peaks, extract_points
from randomized_inference import (inference,
                                  setup_inference,
                                  regression_decomposition,
                                  mle_summary)
from gplasso import fit_gp_lasso
from taylor_expansion import taylor_expansion_window

seed = None
rng = np.random.default_rng(seed)


def test_inference(nx=500,
                   plot_dendrogram=True,
                   penalty=2,
                   marginal=False,
                   two_sided=True,
                   prop_train=0.9,
                   one_sided=True,
                   seed=None,
                   null=False,
                   svd_info=None,
                   level=0.95,
                   displacement=False,
                   spacing=1.5):

    xval = np.linspace(-10,10,nx)

    sigma = 1
    covK = covariance_structure(gaussian_kernel,
                                kernel_args={'precision': sigma**(-2) * np.identity(1)},
                                grid=[xval])
    random_var = (1 - prop_train) / prop_train
    cov_omega = covariance_structure(gaussian_kernel,
                                     kernel_args={'precision': sigma**(-2) * np.identity(1),
                                                  'var':random_var},
                                     grid=[xval])

    if svd_info is None:
        S = np.asarray(covK.C00(None,
                                None))
        U, D = np.linalg.svd(S)[:2]
        A = U * np.sqrt(D[None,:])

        S_omega = np.asarray(cov_omega.C00(None,
                                           None))
        U, D = np.linalg.svd(S_omega)[:2]
        A_omega = U * np.sqrt(D[None,:])
    else:
        S, S_omega, A, A_omega = svd_info
        
    Z = A.dot(rng.standard_normal(A.shape[0]))
    penalty_weights = np.ones_like(Z) * penalty * np.sqrt(1 + random_var)
    
    omega = A_omega.dot(rng.standard_normal(A_omega.shape[0]))

    beta = np.zeros_like(Z)
    beta[int(0.4 * nx)] = 4.
    delta = int(nx / (xval[-1]-xval[0]) * sigma)
    beta[int(0.4 * nx + spacing * delta)] = 3.5
    beta[int(0.4 * nx + (spacing + 0.1) * delta)] = 2.5
    if null:
        beta *= 0
    truth = S.dot(beta)
    Z += truth 
    
    soln, subgrad = fit_gp_lasso(Z + omega, S + S_omega, penalty_weights)
    E = np.nonzero((soln != 0))[0]
    print(E)

    if E.shape[0] > 0 and 0 not in E and nx-1 not in E:

        E_ = [E[0]]

        signs = np.sign(subgrad[E])
        
        second_order = taylor_expansion_window(covK.grid,
                                               [Z, omega],
                                               E,
                                               window_size=10,
                                               precision=sigma**(-2)*np.identity(1))
        
        true_second_order = taylor_expansion_window(covK.grid,
                                                    truth,
                                                    E,
                                                    window_size=10,
                                                    precision=sigma**(-2)*np.identity(1))

        indep = Z - 1 / random_var * omega
        indep_second_order = taylor_expansion_window(covK.grid,
                                                     indep,
                                                     E,
                                                     window_size=10,
                                                     precision=sigma**(-2)*np.identity(1))
        
        true_second_order = taylor_expansion_window(covK.grid,
                                                    truth,
                                                    E,
                                                    window_size=10,
                                                    precision=sigma**(-2)*np.identity(1))
        
        
        
        tangent_bases = [np.identity(1) for _ in range(len(E))]
        normal_info = [(np.zeros((0, 1)), np.zeros((0, 0))) for _ in range(len(E))]

        peaks, clusters = extract_peaks(E,
                                        second_order,
                                        tangent_bases,
                                        normal_info,
                                        covK,
                                        signs,
                                        penalty_weights,
                                        seed=seed)

        true_points, _ = extract_points(E,
                                        true_second_order,
                                        tangent_bases,
                                        normal_info,
                                        covK,
                                        seed=seed,
                                        clusters=clusters)
        
        E_beta = np.nonzero(beta)[0]
        C_ = covK.C00(np.array([p.location for p in true_points]).reshape((-1,1)),
                               xval[E_beta].reshape((-1,1))) 
        true_values = C_ @ beta[E_beta]
        
        true_grads = covK.C10(np.array([p.location for p in true_points]).reshape((-1,1)),
                                        xval[E_beta].reshape((-1,1)))
        true_grads = np.einsum('ikj,k->ij', true_grads, beta[E_beta])
        
        DEBUG = False
        if DEBUG:
            print(true_values, 'exact values')
            print([p.value for p in true_points], 'taylor values')
            print(true_grads, 'exact grads')
            print([p.gradient for p in true_points], 'taylor grads')
            print(np.gradient(truth, xval)[E], 'huh')
        
        indep_points, _ = extract_points(E,
                                         indep_second_order,
                                         tangent_bases,
                                         normal_info,
                                         covK,
                                         seed=seed,
                                         clusters=clusters)

        inactive = np.ones(soln.shape, bool)
        for i in E:
            inactive[(i-3):(i+3)] = 0

        info = setup_inference(peaks,
                               inactive,
                               subgrad,
                               penalty_weights,
                               covK,
                               cov_omega,
                               inference_kernel=None,
                               displacement=displacement)
        (data_value_slice,
         random_value_slice,
         data_grad_slice,
         random_grad_slice) = info.slice_info.cov_slices

        regress_decomp = info.contrast_info
        est_matrix = regress_decomp.est_matrix

        true_first_order = np.zeros(regress_decomp.cov_beta_T.shape[0])
        true_first_order[:len(peaks)] = np.hstack([p.value for p in true_points])
        if displacement:
            true_first_order[len(peaks):] = np.hstack([p.gradient for p in true_points])

        indep_first_order = np.zeros(regress_decomp.cov_beta_T.shape[0])
        indep_first_order[:len(peaks)] = np.hstack([p.value for p in indep_points])
        if displacement:
            indep_first_order[len(peaks):] = np.hstack([p.gradient for p in indep_points])

        param = regress_decomp.cov_beta_T @ true_first_order
        
        data_first_order = np.zeros(regress_decomp.cov_beta_T.shape[0])
        data_first_order[:len(peaks)] = np.hstack([p.value[0] for p in peaks])
        if displacement:
            data_first_order[len(peaks):] = np.hstack([p.gradient[0] for p in peaks])
        naive_mle = regress_decomp.cov_beta_T @ data_first_order

        indep_mle = regress_decomp.cov_beta_T @ indep_first_order

        if one_sided:
            pivot_indep = mle_summary(indep_mle[:len(peaks)],
                                       np.sqrt(np.diag((1 + 1 / random_var) * regress_decomp.cov_beta_T))[:len(peaks)],
                                       signs=[p.sign for p in peaks],
                                       param=param[:len(peaks)],
                                       level=level)
            pvalue_indep = mle_summary(indep_mle[:len(peaks)],
                                       np.sqrt(np.diag((1 + 1 / random_var) * regress_decomp.cov_beta_T))[:len(peaks)],
                                       signs=[p.sign for p in peaks],
                                       param=param[:len(peaks)]*0,
                                       level=level)
        else:
            pivot_indep = mle_summary(indep_mle[:len(peaks)],
                                      np.sqrt(np.diag((1 + 1 / random_var) * regress_decomp.cov_beta_T))[:len(peaks)],
                                      param=param[:len(peaks)],
                                      level=level)
            pvalue_indep = mle_summary(indep_mle[:len(peaks)],
                                       np.sqrt(np.diag((1 + 1 / random_var) * regress_decomp.cov_beta_T))[:len(peaks)],
                                       param=param[:len(peaks)]*0,
                                       level=level)
                 
        # now do data splitting -- watch out for extra points --shape will change
        
        
        T = info.contrast_info.T
        p = T.shape[1] // 2
        N = np.zeros((p,2*p))
        npeak = len(info.peaks)
        N[:npeak,random_value_slice] = N[:npeak,data_value_slice] = np.identity(npeak)
        N[npeak:,random_grad_slice] = N[npeak:,data_grad_slice] = np.identity(sum(info.slice_info.sizes))
        regress_decomp = regression_decomposition(info.cov, T, N) 
        info_ds = info._replace(contrast_info=regress_decomp)
        
        pivot_ds = inference(info_ds,
                             one_sided=one_sided,
                             param=param,
                             level=level)
        
        pvalue_ds = inference(info_ds,
                              one_sided=one_sided,
                              param=param*0,
                              level=level)
        
        pivot_ds['Pivot'] = [True] * pivot_ds.shape[0]
        pivot_indep['Pivot'] = [True] * pivot_ds.shape[0]
        pvalue_ds['Pivot'] = [False] * pvalue_ds.shape[0]
        pvalue_indep['Pivot'] = [False] * pvalue_ds.shape[0]
        
        pivot_ds['Model'] = ['Splitting'] * pivot_ds.shape[0]
        pivot_indep['Model'] = ['Indep'] * pivot_ds.shape[0]
        pvalue_ds['Model'] = ['Splitting'] * pvalue_ds.shape[0]
        pvalue_indep['Model'] = ['Indep'] * pvalue_ds.shape[0]
        
        peak_hash = _hash(peaks)

        df = pd.concat([pivot_ds,
                        pvalue_ds,
                        pivot_indep,
                        pvalue_indep])
        df['spacing'] = spacing
        return df, (S, S_omega, A, A_omega)
    else:
        return None, None


df, svd_info = test_inference(plot_dendrogram=True, seed=0, displacement=True)
df

dfs = []

# +
nfail = 0
null, displacement, spacing = False, True, 1.5
one_sided = False
if one_sided:
    pval_label = 'P-value (1-sided)'
else:
    pval_label = 'P-value (2-sided)'
up_label = [col for col in df.columns if col[0] == 'U'][0]    
low_label = up_label.replace('U', 'L')

if os.path.exists('data_split.csv'):
    prev = [pd.read_csv('data_split.csv')]
else:
    prev = []
    
for i in range(2000):
    for null, displacement, spacing in [(False, True, 1.5),
                                        (False, True, 2),
                                        (False, False, 2),
                                        (False, False, 1.5),
                                        (True, False, 1.5)][:2]:
        try:
            val = test_inference(plot_dendrogram=False, 
                                 prop_train=0.8,
                                 one_sided=one_sided,
                                 svd_info=svd_info,
                                 displacement=displacement,
                                 null=null,
                                 spacing=spacing)
            if val[0] is not None:
               
                dfs.append(val[0])
        except KeyboardInterrupt:
            break
        except:
            nfail += 1
            pass
        if len(dfs) > 0:
            df = pd.concat(dfs + prev)
            def summary(df):
                vals = [df[pval_label].mean(), df[pval_label].std(),
                        (df[pval_label] < 0.05).mean(),
                        (df[up_label] - df[low_label]).median(), 
                        ((df[up_label] > df['Param']) * (df['Param'] > df[low_label])).mean()]
                return pd.Series(vals, index=['mean(P-value)', 'std(P-value)', 'P-value < 5%', 'median(CI length)', 'CI coverage'])
    if i % 10 == 0:
        display(df.groupby(by=['Model', 'Pivot', 'spacing']).apply(summary))
        df.to_csv('data_split.csv')

        print('{} of {} have failed'.format(nfail, i+1))
# -
df.columns



