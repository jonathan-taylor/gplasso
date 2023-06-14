from itertools import product

import numpy as np
from kernel_calcs import covariance_structure, gaussian_kernel
from jacobian import Point, decompose, decompose_random
from taylor_expansion import taylor_expansion_window
import regreg.api as rr
from sklearn.linear_model import lars_path_gram

sigma = 2
covK = covariance_structure(gaussian_kernel,
                            kernel_args={'precision': sigma**(-2) * np.identity(2)})
rng = np.random.default_rng(0)

def test_decompose(nx=200):

    sigma = 1.5
    covK = covariance_structure(gaussian_kernel,
                                kernel_args={'precision': sigma**(-2) * np.identity(2)})
    xval = np.linspace(-10,10,nx)
    S = np.asarray(covK.C00(xval.reshape((-1,1)), xval.reshape((-1,1))))
    U, D = np.linalg.svd(S)[:2]
    A = U * np.sqrt(D[None,:])

    Z = A.dot(rng.standard_normal(A.shape[0]))
    beta = np.zeros_like(Z)
    beta[60] = 0.
    beta[120] = 0.

    truth = S.dot(beta)
    Z += truth 
    lagrange = 2
    L = lars_path_gram(Z, S, n_samples=1, alpha_min=lagrange)
    _, E, path = L
    E = np.sort(E)
    soln = path[:,-1]
    fit = S.dot(soln)
    subgrad = -(Z - fit)

    if E.shape[0] > 1 and 0 not in E and nx-1 not in E:

        E_ = [E[0]]

        for e1, e2 in zip(E, E[1:]):
            if e2 - e1 > 1:
                E_.append(e2)
        E = np.array(E_)
        T = taylor_expansion_window(xval.reshape((-1,1)),
                                    Z,
                                    E.reshape((-1,1)),
                                    window_size=10,
                                    precision=1**(-2)*np.identity(1))

        points = []

        for e, (c, l, q) in zip(E, T):
            point = Point(location=[xval[e]],
                          value=Z[e],
                          penalty=lagrange,
                          sign=np.sign(-subgrad[e]),
                          gradient=l,
                          hessian=q,
                          normal_gradient=0)
            points.append(point)

        logdet_info = decompose([(points[0], False)],
                                points,
                                [covK, covK],
                                covK)
        logdet_info = decompose([(points[0], True)],
                                points,
                                [covK, covK],
                                covK)
        if len(points) > 1:
            logdet_info = decompose([(points[0], True),
                                     (points[-1], True)],
                                    points,
                                    [covK, covK],
                                    covK)
            logdet_info = decompose_random([(points[0], False),
                                            (points[-1], True)],
                                           [(points[0], True)],
                                           points,
                                           covK,
                                           covK)

                                    
    
    
