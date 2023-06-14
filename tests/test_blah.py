from itertools import product

import numpy as np
from kernel_calcs import covariance_structure, gaussian_kernel
from jacobian import Point, decomposition, jacobian

# %%
s = np.array([[0,0]], float)
t = np.linspace(-3,3,101).reshape((-1,1))
G = np.meshgrid(t, t)
G = np.array(G).transpose([1,2,0])
print(G.shape, s.shape)
K = gaussian_kernel(s, G)
K.shape

# %%
sigma = 2
covK = covariance_structure(gaussian_kernel,
                            kernel_args={'precision': sigma**(-2) * np.identity(2)})

# %%
rng = np.random.default_rng(0)
E_x = rng.choice(G.shape[0], 5, replace=False)
E_y = rng.choice(G.shape[1], 5, replace=False)
E = G[E_x,E_y]
E.shape

E_lx = rng.choice(G.shape[0], 4, replace=False)
E_ly = rng.choice(G.shape[1], 4, replace=False)
E_l = G[E_lx,E_ly]

E_rx = rng.choice(G.shape[0], 6, replace=False)
E_ry = rng.choice(G.shape[1], 6, replace=False)
E_r = G[E_rx,E_ry]


def test_decomposition(E, covK):

    points = [Point(location=e,
                    value=rng.normal(),
                    penalty=1,
                    sign=rng.choice([-1,1],1),
                    gradient=rng.normal(size=2),
                    hessian=rng.normal(size=(2,2)),
                    normal_gradient=0) for e in E]

    decomposition(points, covK, gradient=False)

    J = jacobian(covK,
                 covK)
    
    D = decomposition(points, covK, gradient=True)
    J = jacobian(covK,
                 covK,
                 use_gradient=True)
    J.decompose(points[0],
                points)
    return D

logd = test_decomposition(E, covK)[0]
