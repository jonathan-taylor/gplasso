import numpy as np

from randomized_inference import regression_decomposition

def test_data_splitting():

    rng = np.random.default_rng(0)
    n, p = 20, 7
    prop = 0.4
    
    X = rng.standard_normal((n, p))
    S = X.T @ X / n
    Si = np.linalg.inv(S)
    
    A = np.linalg.cholesky(S)
    Z = A @ rng.standard_normal(p)

    I = np.identity(p)

    N = np.zeros((p, 2*p))
    N[:,:p] = N[:,p:] = I

    for (S1, S2) in [(I, I),
                     (S, I),
                     (S, S)]:
        S2 = S2 * prop / (1 - prop)
        A1, A2 = [np.linalg.cholesky(s) for s in [S1, S2]]
        S1i, S2i = [np.linalg.inv(s) for s in [S1, S2]]

        Z = A1 @ rng.standard_normal(p)
        W = A2 @ rng.standard_normal(p)
        stat = np.hstack([Z, W])

        assert np.allclose(Z + W, N @ stat)

        data_splitting = Z - S1 @ S2i @ W

        cov = np.zeros((2*p, 2*p))
        cov[:p,:p] = S1
        cov[p:,p:] = S2

        # the usual estimator of beta would be S1i @ Z
        T = np.zeros((p, 2*p))
        T[:,:p] = S1i

        regress_decomp = regression_decomposition(cov, T, N) 
        assert np.allclose(regress_decomp.est_matrix @ stat, data_splitting)

        
