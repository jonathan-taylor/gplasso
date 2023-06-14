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

import numpy as np
from kernel_calcs import covariance_structure, gaussian_kernel
from jacobian import SpatialPoint
from inference import inference
from taylor_expansion import taylor_expansion_window
import regreg.api as rr
from sklearn.linear_model import lars_path_gram

from sklearn.cluster import \
     (KMeans,
      AgglomerativeClustering)
from scipy.cluster.hierarchy import \
     (dendrogram,
      cut_tree)
from ISLP.cluster import compute_linkage

rng = np.random.default_rng(0)

def fit_gp_lasso(observed_process,
                 covariance_kernel,
                 penalty_weights,
                 tol=1e-12,
                 min_its=200,
                 solve_args={}):

    Z, S = observed_process, covariance_kernel # shorthand

    loss = rr.quadratic_loss(S.shape, Q=S)
    linear_term = rr.identity_quadratic(0,0,-Z,0)
    penalty = rr.weighted_l1norm(penalty_weights, lagrange=1)
    problem = rr.simple_problem(loss, penalty)
    
    soln = problem.solve(linear_term,
                         tol=tol,
                         min_its=min_its,
                         **solve_args)

    fit = S.dot(soln)
    subgrad = -Z + fit

    return soln, subgrad


def test_inference(nx=2000,
                   plot_dendrogram=True,
                  penalty=2,
                  marginal=False,
                  two_sided=False,
                  null=False):

    sigma = 1
    covK = covariance_structure(gaussian_kernel,
                                kernel_args={'precision': sigma**(-2) * np.identity(1)})
    xval = np.linspace(-10,10,nx)
    S = np.asarray(covK.C00(xval.reshape((-1,1)),
                            xval.reshape((-1,1))))
    U, D = np.linalg.svd(S)[:2]
    A = U * np.sqrt(D[None,:])

    Z = A.dot(rng.standard_normal(A.shape[0]))
    penalty_weights = np.ones_like(Z) * penalty
    beta = np.zeros_like(Z)
    beta[int(0.4 * nx)] = 3.
    delta = int(nx / (xval[-1]-xval[0]) * sigma)
    beta[int(0.4 * nx + 2 * delta)] = 4.
    beta[int(0.4 * nx + 3 * delta)] = 2.
    if null:
        beta *= 0
    print(np.nonzero(beta != 0)[0], 'beta')
    truth = S.dot(beta)
    Z += truth 
    
    # L = lars_path_gram(Z, S, n_samples=1, alpha_min=lagrange)
    # _, E, path = L
    # E = np.sort(E)
    # soln = path[:,-1]
    soln, subgrad = fit_gp_lasso(Z, S, penalty_weights)
    E = np.nonzero((soln != 0))[0]
    print(E)
    fit = S.dot(soln)
    subgrad = -(Z - fit)

    
    if E.shape[0] > 0 and 0 not in E and nx-1 not in E:

        E_ = [E[0]]

        T = taylor_expansion_window(xval.reshape((-1,1)),
                                    Z,
                                    E.reshape((-1,1)),
                                    window_size=10,
                                    precision=sigma**(-2)*np.identity(1))

        

        points = []
        
        if len(E) > 1:
            HClust = AgglomerativeClustering
            locations = xval[E].reshape((-1,1))
            hc = HClust(distance_threshold=0.1,
                        n_clusters=None,
                        metric='precomputed',
                        linkage='single')
            C = S[E][:,E]
            diagC = np.diag(C)
            C /= np.multiply.outer(np.sqrt(diagC), np.sqrt(diagC))
            hc.fit(1 - C)
            print(hc.labels_)
            if plot_dendrogram:
                
                cargs = {'color_threshold':-np.inf,
                         'above_threshold_color':'black'}
                linkage_comp = compute_linkage(hc)
                fig, ax = plt.subplots(1, 2, figsize=(4, 4))
                print(locations)
                dendrogram(linkage_comp, 
                           ax=ax[0], 
                           **cargs)
                ax[1].plot(xval, -subgrad, 'r--')
                ax[1].plot(xval, Z)
                ax[1].scatter(locations[:,0], [0]*locations.shape[0])
            clusters = hc.labels_
        else:
            clusters = np.array([0])
            
        truth_ = []
            
        gradients = np.array([l for _, l, _ in T])
        hessians = np.array([h for _, _, h in T])
        for lab in np.unique(clusters):
            cur_cluster = np.nonzero(clusters == lab)[0]
            if cur_cluster.shape[0] > 1:
                gradient = gradients[cur_cluster].mean(0)
                hessian = hessians[cur_cluster].mean(0)
                value = Z[E[cur_cluster]].mean(0)
                sign = np.sign(-subgrad[E[cur_cluster]][0])
                penalty = penalty_weights[cur_cluster].mean()
            else:
                gradient = gradients[cur_cluster].mean(0)
                hessian = hessians[cur_cluster].mean(0)
                value = Z[E[cur_cluster]][0]
                sign = np.sign(-subgrad[E[cur_cluster]][0])
                penalty = penalty_weights[cur_cluster].mean()
                
            truth_.append(truth[E[cur_cluster]].mean())
                
            point = SpatialPoint(location=rng.choice(xval[E[cur_cluster]], 1).reshape(-1),
                                 value=value,
                                 penalty=penalty,
                                 sign=sign, # presumes all signs are the same, reasonable?
                                 gradient=gradient,
                                 hessian=hessian,
                                 tangent_basis=None,
                                 normal_basis=None
                                 )
            points.append(point)

        S_E = S[E][:,E]
        Q = np.zeros((len(points), len(points)))
        for i, l in enumerate(np.unique(clusters)):
            l_cluster = np.nonzero(clusters == l)[0]
            for j, r in enumerate(np.unique(clusters)):
                r_cluster = np.nonzero(clusters == r)[0]
                Q[i,j] = S_E[l_cluster][:,r_cluster].mean()
        
        proj = np.linalg.inv(Q) @ truth_

        I_J = inference(points,
                        covK,
                        param=proj,
                        C00=Q,
                        marginal=marginal,
                        two_sided=two_sided,
                        use_jacobian=True)
        I_noJ = inference(points,
                        covK,
                        param=proj,
                        C00=Q,
                        marginal=marginal,
                        two_sided=two_sided,
                        use_jacobian=False)
        
        return I_J, I_noJ
    else:
        return None

with_jac, without_jac = test_inference(null=False)
with_jac

without_jac

import pandas as pd
dfs_with, dfs_without = [], []

for _ in range(1000):
    result = test_inference(plot_dendrogram=False, null=False)
    if result is not None:
        w, wo = result
        dfs_with.append(w)
        dfs_without.append(wo)
    if len(dfs_with) > 0:
        df_with = pd.concat(dfs_with)
        df_without = pd.concat(dfs_without)
        print('pivot with', df_with['P-value (1-sided)'].mean(), df_with['P-value (1-sided)'].std())
        print('pivot without', df_without['P-value (1-sided)'].mean(), df_without['P-value (1-sided)'].std())

fig, ax = plt.subplots(1, 1, figsize=(8,8))
ax = df_with['P-value (1-sided)'].hist(cumulative=True, density=1, bins=100, histtype='step', color='b', label='w Jacobian', ax=ax)
df_without['P-value (1-sided)'].hist(cumulative=True, density=1, bins=100, histtype='step', color='r', label='w/o Jacobian', ax=ax)
ax.axline((0,0), slope=1, ls='--', c='k')
ax.legend();


