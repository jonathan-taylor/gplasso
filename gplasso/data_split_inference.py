import numpy as np
import pandas as pd
from scipy.stats import norm as normal_dbn

from .taylor_expansion import taylor_expansion_window

def inference(covK,
              xval,
              E,
              Z,
              truth=None,
              factor=1,
              level=0.95):

    Q = cov_peak = covK.C00(xval[E].reshape((-1,1)), 
                            xval[E].reshape((-1,1)))
    Qi = np.linalg.inv(Q)
    estimate = Qi @ Z[E] 
    SE = np.sqrt(factor * np.diag(Qi))
    
    q = normal_dbn.ppf(1 - (1-level)/2)
    U = estimate + q * SE
    L = estimate - q * SE
    
    df = pd.DataFrame({'Location':xval[E],
                       'Estimate':estimate,
                       'SE':SE,
                       'L':L,
                       'U':U})
    if truth is not None:
        proj = Qi @ truth[E]
        df['Truth'] = proj
        df['Covered'] = (proj < U) * (proj > L)

    df = df.set_index('Location')
    return df


def location_interval(Z_height,
                      U_height,
                      L_height,
                      Z_delta,
                      SE_delta,
                      level=0.95):

    pt_estimate = Z_delta / Z_height
    if L_height * U_height < 0:
        return pt_estimate, -np.inf, np.inf
    else:
        q = normal_dbn.ppf(1 - (1-level)/2)
        recip = np.min(np.fabs([L_height, U_height]))
        a, b = np.sort([Z_delta / U_height, Z_delta / L_height])
        L = a - q * SE_delta / recip
        U = b + q * SE_delta / recip
        
        return pt_estimate, L, U


def peak_inference(covK,
                   xval,
                   E,
                   Z,
                   truth=None,
                   factor=1,
                   level=0.95,
                   sigma=1.5):
    
    npt = len(E)
    Q = np.zeros((2*npt, 2*npt))
    Q[:npt][:,:npt] = covK.C00(xval[E].reshape((-1,1)), 
                               xval[E].reshape((-1,1)))
    Q[:npt][:,npt:] = covK.C01(xval[E].reshape((-1,1)), 
                               xval[E].reshape((-1,1))).reshape((npt, npt))
    Q[npt:][:,:npt] = Q[:npt][:,npt:]
    Q[npt:][:,npt:] = covK.C11(xval[E].reshape((-1,1)), 
                               xval[E].reshape((-1,1))).reshape((npt, npt))

    T = taylor_expansion_window(covK.grid,
                                Z,
                                E.reshape((-1,1)),
                                window_size=10,
                                precision=sigma**(-2)*np.identity(1))
    Qi = np.linalg.inv(Q)
    obs = np.hstack([Z[E], np.squeeze([l[0] for _, l, _ in T])])
    
    Qi = np.linalg.inv(Q)
    estimate = Qi @ obs
    SE = np.sqrt(factor * np.diag(Qi))
    
    q = normal_dbn.ppf(1 - (1-level)/2)
    U = estimate + q * SE
    L = estimate - q * SE
    
    df = pd.DataFrame({'Location':list(xval[E]) + [f'peak(x)' for x in xval[E]],
                       'Estimate':estimate,
                       'SE':SE,
                       'P-value (2-sided)': 2 * normal_dbn.sf(np.fabs(estimate / SE)),
                       'L':L,
                       'U':U})

    if truth is not None:
        T0 = taylor_expansion_window(covK.grid,
                                     truth,
                                     E.reshape((-1,1)),
                                     window_size=10,
                                     precision=sigma**(-2)*np.identity(1))

        true_val = np.hstack([truth[E], np.squeeze([l[0] for _, l, _ in T0])])

        proj = Qi @ true_val
        df['Truth'] = proj
        df['Covered'] = (proj < U) * (proj > L)

    estimates_peak, U_peak, L_peak = [], [], []
    
    estimate = np.asarray(df['Estimate'].iloc[:npt])
    U = np.asarray(df['U'].iloc[:npt])
    L = np.asarray(df['L'].iloc[:npt])
    del_estimate = np.asarray(df['Estimate'].iloc[npt:])
    del_SE = np.asarray(df['SE'].iloc[npt:])

    for i in range(npt):
        pt_est, l, u = location_interval(estimate[i],
                                         U[i],
                                         L[i],
                                         del_estimate[i],
                                         del_SE[i],
                                         level=level)
        estimates_peak.append(pt_est)
        U_peak.append(u)
        L_peak.append(l)

    true_peak = []
    for i in range(len(E)):
        if np.fabs(true_val[npt+i]) > 0:
            true_peak.append(true_val[i] / true_val[npt+i])
        else:
            true_peak.append(np.nan)
    true_peak = np.array(true_peak)

    peak_df = pd.DataFrame({'Location':xval[E],
                           'Estimate':xval[E] + estimates_peak,
                           'L':L_peak+xval[E],
                           'U':U_peak+xval[E],
                           'Truth':true_peak+xval[E],
                           'Covered':(true_peak < U_peak) * (true_peak > L_peak)})
    df = df.set_index('Location')
    peak_df = peak_df.set_index('Location')
    peak_df['Covered'] = peak_df['Covered'] * np.isfinite(peak_df['U']) + True * (~np.isfinite(peak_df['U']))
    return df[:npt], peak_df

