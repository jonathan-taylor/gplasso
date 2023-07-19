import numpy as np

import jax.numpy as jnp
from jax import jacfwd

from ..kernels import covariance_kernel

def _jax_outer_subtract(s, t):
    tmp = jnp.outer(jnp.exp(s), jnp.exp(-t))
    tmp = jnp.reshape(tmp, s.shape + t.shape)
    return jnp.log(tmp)

def gaussian_kernel_(s,
                     t,
                     precision=None,
                     var=1):
    
    s, t = jnp.asarray(s), jnp.asarray(t)
    dim_s, dim_t = s.shape[-1], t.shape[-1]
    diff = jnp.array([_jax_outer_subtract(s[...,i], t[...,i]) for i in range(dim_s)])
    if precision is None:
        precision = jnp.identity(dim_s)
    quadratic_form = jnp.einsum('i...,k...,ik->...',
                                diff, 
                                diff, 
                                precision,
                                optimize=True)
    return var * jnp.exp(-0.5 * quadratic_form)

class jax_covariance_kernel(covariance_kernel):

    def __init__(self,
                 kernel,
                 kernel_args={},
                 grid=None,
                 sampler=None): # default grid of x values
        """
        Compute covariance structure of field
        and its first two derivatives at points
        loc_l and loc_r.
        """

        self.kernel_ = lambda loc_l, loc_r: kernel(loc_l,
                                                   loc_r,
                                                   **kernel_args)

        self.grid = grid
        self._grid = np.asarray(grid)
        self.sampler = sampler
        
    @staticmethod
    def gaussian(precision=None,
                 var=1,
                 grid=None,
                 sampler=None):
        return jax_covariance_kernel(gaussian_kernel_,
                                     kernel_args={'precision':precision,
                                                  'var':var},
                                     grid=grid,
                                     sampler=sampler)

    # location based computations

    def C00(self,
            loc_l,
            loc_r,
            reshape=True):
        loc_l, loc_r, grid_l, grid_r = _get_LR(self._grid, loc_l, loc_r)
        return self._reshape(self.kernel_(loc_l, loc_r),
                             reshape,
                             grid_l,
                             grid_r)
    
    def C10(self,
            loc_l,
            loc_r,
            basis_l=None,
            reshape=True):
        loc_l, loc_r, grid_l, grid_r = _get_LR(self._grid, loc_l, loc_r)
        if not hasattr(self, 'C10_kernel_'):
            self.C10_kernel_ = jacfwd(self.kernel_,
                                      argnums=(0,))
        D = self.C10_kernel_(loc_l, loc_r)
        value = jnp.array([D[0][i,:,i] for i in range(loc_l.shape[0])])
        if basis_l is not None:
            value = jnp.einsum('ijk,lk->ijl',
                               value,
                               basis_l,
                               optimize=True)
        return self._reshape(value,
                             reshape,
                             grid_l,
                             grid_r)
    
    def C01(self,
            loc_l,
            loc_r,
            basis_r=None,
            reshape=True):
        loc_l, loc_r, grid_l, grid_r = _get_LR(self._grid, loc_l, loc_r)
        value = jnp.transpose(self.C10(loc_r,
                                       loc_l,
                                       basis_l=basis_r,
                                       reshape=False),
                             [1,0,2])
        return self._reshape(value,
                             reshape,
                             grid_l,
                             grid_r)

    def C20(self,
            loc_l,
            loc_r,
            basis_l=None,
            reshape=True):
        loc_l, loc_r, grid_l, grid_r = _get_LR(self._grid, loc_l, loc_r)
        if not hasattr(self, 'D2_kernel_'):
            self.D2_kernel_ = jacfwd(self.C10,
                                     argnums=(0,1))
        D = self.D2_kernel_(loc_l, loc_r)
        value = jnp.array([D[0][i,:,:,i] for i in range(loc_l.shape[0])])
        if basis_l is not None:
            value = jnp.einsum('ijkl,ak,bl->ijab',
                               value,
                               basis_l,
                               basis_l,
                               optimize=True)
        return self._reshape(value,
                             reshape,
                             grid_l,
                             grid_r)

    
    def C02(self,
            loc_l,
            loc_r,
            basis_r=None,
            reshape=True):
        loc_l, loc_r, grid_l, grid_r = _get_LR(self._grid, loc_l, loc_r)
        value =  jnp.transpose(self.C20(loc_r,
                                        loc_l,
                                        basis_l=basis_r,
                                        reshape=False),
                               [1,0,2,3])
        return self._reshape(value,
                             reshape,
                             grid_l,
                             grid_r)


    def C11(self,
            loc_l,
            loc_r,
            basis_l=None,
            basis_r=None,
            reshape=True):
        loc_l, loc_r, grid_l, grid_r = _get_LR(self._grid, loc_l, loc_r)
        if not hasattr(self, 'D2_kernel_'):
            self.D2_kernel_ = jacfwd(self.C10,
                                     argnums=(0,1))
        D = self.D2_kernel_(loc_l, loc_r)
        C11 = jnp.array([D[1][:,i,:,i] for i in range(loc_r.shape[0])])
        value = jnp.transpose(C11, [1,0,2,3])
        if basis_l is not None:
            value = jnp.einsum('ijkl,ak->ijal',
                               value,
                               basis_l,
                               optimize=True)
        if basis_r is not None:
            value = jnp.einsum('ijkl,bl->ijkb',
                               value,
                               basis_r,
                               optimize=True)
        return self._reshape(value,
                             reshape,
                             grid_l,
                             grid_r)
    
    def C21(self,
            loc_l,
            loc_r,
            basis_l=None,
            basis_r=None,
            reshape=True):
        loc_l, loc_r, grid_l, grid_r = _get_LR(self._grid, loc_l, loc_r)
        if not hasattr(self, 'C21_kernel_'):
            self.C21_kernel_ = jacfwd(self.C20,
                                      argnums=(1,))
        D = self.C21_kernel_(loc_l, loc_r)
        C21 = jnp.array([D[0][:,i,:,:,i] for i in range(loc_r.shape[0])])
        value = jnp.transpose(C21, [1,0,2,3,4])
        if basis_l is not None:
            value = jnp.einsum('ijklm,ak,bl->ijabm',
                               value,
                               basis_l,
                               basis_l,
                               optimize=True)
        if basis_r is not None:
            value = jnp.einsum('ijklm,cm->ijklc',
                               value,
                               basis_r,
                               optimize=True)
        return self._reshape(value,
                             reshape,
                             grid_l,
                             grid_r)
    
    def C12(self,
            loc_l,
            loc_r,
            basis_l=None,
            basis_r=None,
            reshape=True):
        loc_l, loc_r, grid_l, grid_r = _get_LR(self._grid, loc_l, loc_r)
        value = self.C21(loc_r,
                         loc_l,
                         basis_l=basis_r,
                         basis_r=basis_l,
                         reshape=False).transpose([1,0,4,2,3])
        return self._reshape(value,
                             reshape,
                             grid_l,
                             grid_r)

    def C22(self,
            loc_l,
            loc_r,
            basis_l=None,
            basis_r=None,
            reshape=True):
        loc_l, loc_r, grid_l, grid_r = _get_LR(self._grid, loc_l, loc_r)
        if not hasattr(self, 'C22_kernel_'):
            self.C22_kernel_ = jacfwd(self.C21,
                                      argnums=(1,))
        D = self.C22_kernel_(loc_l, loc_r)
        C22 = jnp.array([D[0][:,i,:,:,:,i] for i in range(loc_r.shape[0])])
        C22 = jnp.transpose(C22, [1,0,2,3,4,5])
        if basis_l is not None:
            C22 = jnp.einsum('ijklmn,ak,bl->ijabmn',
                             C22,
                             basis_l,
                             basis_l,
                             optimize=True)
        if basis_r is not None:
            C22 = jnp.einsum('ijklmn,am,bn->ijklab',
                             C22,
                             basis_r,
                             basis_r,
                             optimize=True)
        return self._reshape(C22,
                             reshape,
                             grid_l,
                             grid_r)
    
def _get_LR(grid, loc_l, loc_r):
    G = grid.transpose(list(range(1, grid[0].ndim+1)) + [0])
    G = G.reshape((-1, G.shape[-1]))
    grid_l, grid_r = False, False
    if loc_l is None:
        loc_l = G
        grid_l = True
    if loc_r is None:
        loc_r = G 
        grid_r = True
    loc_l, loc_r = jnp.asarray(loc_l), jnp.asarray(loc_r)
    return loc_l, loc_r, grid_l, grid_r

