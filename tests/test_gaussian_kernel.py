from itertools import product

import numpy as np
from kernel_calcs import covariance_structure, gaussian_kernel
from jacobian import Point, CrossCov, full_hessian, cov_matrices_from_dict

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


# %% [markdown]
# # Test
#
# ## $\text{Cov}(f_L, f_R)$

# %%
dE = np.array([np.subtract.outer(E_l[:,i], E_r[:,i]) for i in range(2)]).transpose([1,2,0])
E_l.shape, E_r.shape, dE.shape

# %%
Q = (dE**2).sum(-1)
K = np.exp(-0.5*Q/(sigma**2))
assert np.linalg.norm(covK.C00(E_l,E_r) - K) / np.linalg.norm(K) < 1e-5

# %% [markdown]
# ## $\text{Cov}(f_L, \nabla f_R)$

# %%
M10 = np.einsum('ij,ijk->ijk', K, -dE / sigma**2)
assert np.linalg.norm(covK.C10(E_l,E_r) - M10) / np.linalg.norm(M10) < 1e-5
assert np.linalg.norm(covK.C01(E_l,E_r) + M10) / np.linalg.norm(M10) < 1e-5

# %% [markdown]
# ## $\text{Cov}(f_L, \nabla^2 f_R)$

# %%
M20 = (np.einsum('ij,ijk,ijl->ijkl', K, -dE / sigma**2, -dE / sigma**2)
       - np.einsum('ij,kl->ijkl', K, np.identity(2) / sigma**2))
assert np.linalg.norm(covK.C20(E_l,E_r) - M20) / np.linalg.norm(M20) < 1e-5

# %% [markdown]
# ## $\text{Cov}(\nabla f_L, \nabla f_R)$

# %%
M11 = - M20 
assert np.linalg.norm(covK.C11(E_l,E_r) - M11) / np.linalg.norm(M11) < 1e-5
assert np.linalg.norm(covK.C02(E_l,E_r) - M20) / np.linalg.norm(M20) < 1e-5

# %% [markdown]
# ## $\text{Cov}(\nabla^2 f_L, \nabla f_R)$

# %%
M21 = (np.einsum('ij,ijk,ijl,ijm->ijklm', K, dE, dE, dE) / sigma**6
       - (np.einsum('ij,kl,ijm->ijklm', K, np.identity(2), dE) +
          np.einsum('ij,kl,ijm->ijmkl', K, np.identity(2), dE) +
          np.einsum('ij,kl,ijm->ijlmk', K, np.identity(2), dE)) / sigma**4)
assert np.linalg.norm(covK.C21(E_l, E_r) - M21) / np.linalg.norm(M21) < 1e-5
assert np.linalg.norm(covK.C12(E_l, E_r) + M21) / np.linalg.norm(M21) < 1e-5

# %% smoke test

covK.C22(E_l, E_r)

# make the dict for covariance matrices

def test_cov_matrices_from_dict(E, covK):

    points = [Point(location=e,
                    value=rng.normal(),
                    penalty=1,
                    sign=rng.choice([-1,1],1),
                    gradient=rng.normal(size=2),
                    hessian=rng.normal(size=(2,2)),
                    normal_gradient=0) for e in E]

    C00 = covK.C00(E, E)
    C01 = covK.C01(E, E)
    C11 = covK.C11(E, E)

    cov_info = {}

    for i, j in product(range(len(points)),
                        range(len(points))):
        cov_info[(i,j)] = CrossCov(point_L=points[i],
                                   point_R=points[j],
                                   C00=C00[i,j],
                                   C01=C01[i,j],
                                   C11=C11[i,j])
    return points, cov_matrices_from_dict(points, cov_info)

if __name__ == '__main__':
    print([v.shape for v in test_cov_matrices_from_dict(E, covK)[1]])
