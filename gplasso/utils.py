from dataclasses import dataclass
from functools import partial

import numpy as np

import pandas as pd

from scipy.stats import norm as normal_dbn
from scipy.stats import chi2


@dataclass
class RegressionInfo(object):

    T: np.ndarray
    N: np.ndarray
    L_beta: np.ndarray
    L_NZ: np.ndarray
    est_matrix: np.ndarray
    sqrt_cov_R: np.ndarray
    cov_beta_T: np.ndarray
    cov_beta_TN: np.ndarray
    
def regression_decomposition(cov, T, N):
    
    nT, nN = T.shape[0], N.shape[0]
    
    TN = np.concatenate([T, N], axis=0)
    cov_TN = TN @ cov @ TN.T
    prec_TN = np.linalg.inv(cov_TN)
    
    # Cov(TZ|NZ)
    cov_TgN = np.linalg.inv(prec_TN[:nT,:nT])
    
    M = np.zeros_like(prec_TN)
    M[:nT,:nT] = cov_TgN
    M[:nT,nT:] = cov_TN[:nT,nT:] @ np.linalg.inv(cov_TN[nT:,nT:])
    M[nT:,nT:] = np.identity(nN)
    
    L = cov @ TN.T @ prec_TN @ M
    
    # compute the difference in covariance matrices of the
    # two estimators

    cov_beta_TN = prec_TN[:nT,:nT]
    
    cov_T = T @ cov @ T.T
    cov_beta_T = np.linalg.inv(cov_T)

    L_beta = L[:,:nT]
    L_NZ = L[:,nT:]

    # compute the estimation matrix, i.e.
    # the matrix that computes the T coords
    # of beta_{N cup T}

    est_matrix = prec_TN[:nT] @ TN

    # find a square root of the
    # covariance of the residual matrix

    cov_R = cov - cov @ TN.T @ prec_TN @ TN @ cov
    U, D, _ = np.linalg.svd(cov_R)

    p = cov.shape[0]
    rank_R = p - nN - nT # assumes prec_TN is full rank
                         # we would have had an exception
                         # earlier when computing prec_TN if exactly singular 
                         # it's possible rank(cov_R) is smaller if cov wasn't full rank
                         # BUT, we have definitely assumed TN @ cov @ TN.T is full rank
    U = U[:,:rank_R]
    D = D[:rank_R]
    sqrt_cov_R = U * np.sqrt(D)[None,:]

    return RegressionInfo(T,
                          N,
                          L_beta,
                          L_NZ,
                          est_matrix,
                          sqrt_cov_R,
                          cov_beta_T,
                          cov_beta_TN)

@dataclass
class MLEInfo(object):

    estimate: np.ndarray
    cov: np.ndarray

    def linear_hypothesis(self,
                          contrast,
                          truth=None):
        '''
        Test $H_0:C\hat{\beta}-h=0$
        '''
        df = contrast.shape[0] # presumes full-rank
        if contrast.ndim > 1:
            if truth is None:
                truth = np.zeros(df)
            est = contrast @ self.estimate - truth
            cov_est = contrast @ self.cov @ contrast.T
            prec_est = np.linalg.inv(cov_est)
            stat = np.einsum('i,j,ij',
                             est,
                             est,
                             prec_est)
            return Chi2Result(est,
                              truth,
                              cov_est,
                              stat,
                              df,
                              chi2.sf(stat, df=df))
        else:
            if truth is None:
                truth = 0
            est = contrast @ self.estimate - truth
            var_est = np.einsum('i,j,ij->',
                                contrast,
                                contrast,
                                self.cov)
            sd_est = np.sqrt(var_est)
            stat = np.fabs(est) / sd_est
            pvalue = chi2.sf(stat**2, df=1)
            return ZResult(est,
                           truth,
                           sd_est,
                           stat,
                           pvalue)

    def summary(self, 
                param=None,
                signs=None,
                level=None):
        return mle_summary(self.estimate,
                           np.sqrt(np.diag(self.cov)),
                           param=param,
                           signs=signs,
                           level=level)

@dataclass
class ZResult(object):

    estimate: float
    truth: float
    std: float
    stat: float
    pvalue: float

@dataclass
class Chi2Result(object):

    estimate: np.ndarray
    truth: np.ndarray
    cov: np.ndarray
    stat: float
    df: float
    pvalue: float


def mle_summary(mle,
                SD,
                param=None,
                signs=None,
                level=None):
    """
    Wald-like summary
    """

    if param is None:
        param = np.zeros_like(mle)
    Z = (mle - param) / SD

    if signs is not None:
        one_sided = True
        P = normal_dbn.sf(Z * signs)
    else:
        one_sided = False
        P = normal_dbn.cdf(Z)
        P = 2 * np.minimum(P, 1-P)

    df = pd.DataFrame({'Estimate':mle,
                       'SD':SD,
                       'Param':param})
    if one_sided:
        df['P-value (1-sided)'] = P
    else:
        df['P-value (2-sided)'] = P

    if level is not None:
        q = normal_dbn.ppf(1 - (1 - level) / 2)
        df['L ({:.0%})'.format(level)] = mle - q * SD
        df['U ({:.0%})'.format(level)] = mle + q * SD

    return df

def _compute_mle(initial_W,
                 val_,
                 grad_,
                 hess_,
                 DEBUG=False,
                 num_newton=20):

    W = initial_W.copy()
    I = np.identity(W.shape[0])

    num_newton = 20

    if W.shape != (0,): # for data splitting W has shape (0,)
        for i in range(num_newton):
            if DEBUG:
                print('newton iterate {}'.format(i))

            H = I + hess_(W)
            G = W + grad_(W)

            # do a line search

            factor = 1
            niter = 0
            cur_val = np.inf
            step = np.linalg.inv(H) @ G

            while True:
                W_new = W - factor * step
                new_val = val_(W_new) + (W_new**2).sum() * 0.5                
                if new_val < cur_val:
                    break

                factor *= 0.5
                niter += 1
                if niter >= 30:
                    raise ValueError('no descent')

            if np.linalg.norm(W - W_new) < 1e-9 * np.linalg.norm(W):
                break

            W = W_new
            cur_val = new_val

        if DEBUG:
            print('W', W)

        return W

