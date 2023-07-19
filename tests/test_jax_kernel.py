#!/usr/bin/env python
# coding: utf-8

# We want to differentiate functions of the form
# $$
# R(s,t)=f(s-t)
# $$
# with
# $$
# f(v) = g\left(\frac{1}{2}v'Qv\right).
# $$
# 
# The canonical example being the radial basis function (Gaussian covariance) with
# $g(r)=e^{-r}$ and $Q=\Lambda^{-1}$ being the precision of the covariance (the inverse covariance matrix of the derivatives of  such a field).

# $$
# \begin{aligned}
# \frac{\partial f}{\partial v_{i_1}}
# &= g'\left(\frac{1}{2} v'Qv\right) \cdot \sum_{j_1} Q_{i_1j_1}v_{j_1} \\
# &= g'\left(\frac{1}{2}v'Qv\right) \cdot \left(e_{i_1}'Qv\right)
# \end{aligned}
# $$

# $$
# \begin{aligned}
# \frac{\partial^2 f}{\partial v_{i_1} \partial v_{i_2}}
# &= g''\left(\frac{1}{2}v'Qv\right) \cdot \sum_{j_1,j_2} Q_{i_1j_1}v_{j_1}
#        Q_{i_2j_2}v_{j_2} + \\
# & \qquad g'\left(\frac{1}{2}v'Qv\right) \cdot Q_{i_1i_2} \\
# &= g''\left(\frac{1}{2}v'Qv\right) \cdot \left(e_{i_1}'Qv \right) \left(e_{i_2}'Qv \right) + \\
# & \qquad g'\left(\frac{1}{2}v'Qv\right) \cdot Q_{i_1i_2}
# \end{aligned}
# $$

# $$
# \begin{aligned}
# \frac{\partial^3 f}{\partial v_{i_1} \partial v_{i_2}
#                    \partial v_{i_3}}
# &= g'''\left(\frac{1}{2}v'Qv\right) \cdot  \left(e_{i_1}'Qv \right) \left(e_{i_2}'Qv \right)
# \left(e_{i_3}'Qv \right) + \\
# & \qquad g''\left(\frac{1}{2}v'Qv\right) \left[Q_{i_1i_3} \left(e_{i_2}'Qv \right) + Q_{i_2i_3} \left(e_{i_1}'Qv \right)\right] \\
# & \qquad g''\left(\frac{1}{2}v'Qv\right) \cdot Q_{i_1i_2} \left(e_{i_3}'Qv \right) \\
# &= g'''\left(\frac{1}{2}v'Qv\right) \cdot  \left(e_{i_1}'Qv \right) \left(e_{i_2}'Qv \right)
# \left(e_{i_3}'Qv \right) + \\
# & \qquad g''\left(\frac{1}{2}v'Qv\right) \left[Q_{i_1i_3} \left(e_{i_2}'Qv \right) + Q_{i_2i_3} \left(e_{i_1}'Qv \right) + Q_{i_1i_2} \left(e_{i_3}'Qv \right) \right] \\
# \end{aligned}
# $$

# $$
# \begin{aligned}
# \frac{\partial^4 f}{\partial v_{i_1} \partial v_{i_2}
#                    \partial v_{i_3} \partial v_{i_4}}
# &= g''''\left(\frac{1}{2}v'Qv\right) \left(e_{i_1}'Qv \right) \left(e_{i_2}'Qv \right)
# \left(e_{i_3}'Qv \right) \left(e_{i_4}'Qv \right) + \\
# & \qquad g'''\left(\frac{1}{2}v'Qv\right) \cdot \biggl[Q_{i_1i_4} \left(e_{i_2}'Qv \right) \left(e_{i_3}'Qv \right) + 
# Q_{i_2i_4} \left(e_{i_1}'Qv \right) \left(e_{i_3}'Qv \right) +
# Q_{i_3i_4} \left(e_{i_1}'Qv \right) \left(e_{i_2}'Qv \right) +
# Q_{i_1i_3} \left(e_{i_2}'Qv \right) \left(e_{i_4}'Qv \right)
# + Q_{i_2i_3} \left(e_{i_1}'Qv \right) \left(e_{i_4}'Qv\right) + Q_{i_1i_2} \left(e_{i_3}'Qv \right) \left(e_{i_4}'Qv\right) \biggr] + \\
# & \qquad g''\left(\frac{1}{2}v'Qv\right) \left[Q_{i_1i_3}Q_{i_2i_4} + Q_{i_2i_3}Q_{i_1i_4} + Q_{i_1i_2}Q_{i_3i_4}\right]
# \end{aligned}
# $$

# In[ ]:


import numpy as np
from gplasso.kernels import gaussian_kernel
from gplasso.jax_kernels import jax_covariance_kernel

def test_jax_kernel():

    xval = np.linspace(0, 10, 21)
    yval = np.linspace(0, 10, 21)
    zval = np.linspace(0, 10, 21)

    grid = np.meshgrid(xval, yval, zval, indexing='ij')

    rng = np.random.default_rng(0)
    X = rng.standard_normal((5,3))
    Q = X.T @ X / 5
    G = gaussian_kernel(Q,
                        grid=grid)

    V = rng.standard_normal((10,3))
    W = rng.standard_normal((5,3))


    J = jax_covariance_kernel.gaussian(precision=Q, var=1, grid=grid)

    for m in ['C00', 'C01', 'C10',
              'C11', 'C02', 'C20',
              'C12', 'C21', 'C22']:
        J_ = getattr(J, m)(V, W)
        G_ = getattr(G, m)(V, W)
        tol = np.linalg.norm(J_-G_) / np.linalg.norm(G_)
        print('m: ', tol, np.linalg.norm(G_))
        assert tol < 1e-5
