import numpy as np
import jax.numpy as jnp
from jax import jacfwd

def _jax_outer_subtract(s, t):
    tmp = jnp.outer(jnp.exp(s), jnp.exp(-t))
    tmp = jnp.reshape(tmp, s.shape + t.shape)
    return jnp.log(tmp)

def gaussian_kernel(s,
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

def np_gaussian_kernel(s,
                       t,
                       precision=None,
                       var=1,
                       use_jax=False):

    s, t = np.asarray(s), np.asarray(t)
    dim_s, dim_t = s.shape[-1], t.shape[-1]
    diff = np.array([np.subtract.outer(s[...,i], t[...,i]) for i in range(dim_s)])
    if precision is None:
        precision = np.identity(dim_s)
    quadratic_form = np.einsum('i...,k...,ik->...',
                               diff, 
                               diff, 
                               precision,
                               optimize=True)
    return var * np.exp(-0.5 * quadratic_form)


class covariance_structure(object):

    def __init__(self,
                 kernel,
                 kernel_args={},
                 grid=None): # default grid of x values
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
        
    @staticmethod
    def gaussian(precision=None,
                 var=1,
                 grid=None):
        return covariance_structure(gaussian_kernel,
                                    kernel_args={'precision':precision,
                                                 'var':var},
                                    grid=grid)

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
            value = np.einsum('ijk,lk->ijl',
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
        value = np.transpose(self.C10(loc_r,
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
            value = np.einsum('ijkl,ak,bl->ijab',
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
        value =  np.transpose(self.C20(loc_r,
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
        value = np.transpose(C11, [1,0,2,3])
        if basis_l is not None:
            value = np.einsum('ijkl,ak->ijal',
                              value,
                              basis_l,
                              optimize=True)
        if basis_r is not None:
            value = np.einsum('ijkl,bl->ijkb',
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
        value = np.transpose(C21, [1,0,2,3,4])
        if basis_l is not None:
            value = np.einsum('ijklm,ak,bl->ijabm',
                              value,
                              basis_l,
                              basis_l,
                              optimize=True)
        if basis_r is not None:
            value = np.einsum('ijklm,cm->ijklc',
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
        C22 = np.transpose(C22, [1,0,2,3,4,5])
        if basis_l is not None:
            C22 = np.einsum('ijklmn,ak,bl->ijabmn',
                            C22,
                            basis_l,
                            basis_l,
                            optimize=True)
        if basis_r is not None:
            C22 = np.einsum('ijklmn,am,bn->ijklab',
                            C22,
                            basis_r,
                            basis_r,
                            optimize=True)
        return self._reshape(C22,
                             reshape,
                             grid_l,
                             grid_r)
    
    # from indices into the grid

    def _reshape(self,
                 value,
                 do_reshape,
                 grid_l,
                 grid_r):
        if do_reshape:
            # first take care of the left most indices
            if grid_l:
                shape_l = self.grid[0].shape
            else:
                shape_l = (value.shape[0],)
            if grid_r:
                shape_r = self.grid[0].shape
            else:
                shape_r = (value.shape[1],)
                
            value = value.reshape(shape_l + shape_r + value.shape[2:])
        return value
    
    def get_locations(self, idx):
        loc = np.array([g[idx] for g in self._grid])
        return np.transpose(loc, list(range(1, loc.ndim)) + [0])
    
    def C00_idx(self,
                idx_l,
                idx_r,
                reshape=True):
        loc_l, loc_r = self.get_locations(idx_l), self.get_locations(idx_r)
        return self.C00(loc_l,
                        loc_r,
                        reshape=reshape)
    
    def C10_idx(self,
                loc_l,
                loc_r,
                basis_l=None,
                reshape=True):
        loc_l, loc_r = self.get_locations(idx_l), self.get_locations(idx_r)
        return self.C10(loc_l,
                        loc_r,
                        basis_l=basis_l,
                        reshape=reshape)
    
    def C01_idx(self,
                loc_l,
                loc_r,
                basis_r=None,
                reshape=True):
        loc_l, loc_r = self.get_locations(idx_l), self.get_locations(idx_r)
        return self.C01(loc_l,
                        loc_r,
                        basis_r=basis_r,
                        reshape=reshape)

    def C20_idx(self,
                loc_l,
                loc_r,
                basis_l=None,
                reshape=True):
        loc_l, loc_r = self.get_locations(idx_l), self.get_locations(idx_r)
        return self.C20(loc_l,
                        loc_r,
                        basis_l=basis_l,
                        reshape=reshape)
    
    def C02_idx(self,
                loc_l,
                loc_r,
                basis_r=None,
                reshape=True):
        loc_l, loc_r = self.get_locations(idx_l), self.get_locations(idx_r)
        return self.C02(loc_l,
                        loc_r,
                        basis_r=basis_r,
                        reshape=reshape)

    def C11_idx(self,
                loc_l,
                loc_r,
                basis_l=None,
                basis_r=None,
                reshape=True):
        loc_l, loc_r = self.get_locations(idx_l), self.get_locations(idx_r)
        return self.C20(loc_l,
                        loc_r,
                        basis_l=basis_l,
                        basis_r=basis_r,
                        reshape=reshape)
    
    def C21_idx(self,
                loc_l,
                loc_r,
                basis_l=None,
                basis_r=None,
                reshape=True):
        loc_l, loc_r = self.get_locations(idx_l), self.get_locations(idx_r)
        return self.C21(loc_l,
                        loc_r,
                        basis_l=basis_l,
                        basis_r=basis_r,
                        reshape=reshape)
    
    def C12_idx(self,
                loc_l,
                loc_r,
                basis_l=None,
                basis_r=None,
                reshape=True):
        loc_l, loc_r = self.get_locations(idx_l), self.get_locations(idx_r)
        return self.C12(loc_l,
                        loc_r,
                        basis_l=basis_l,
                        basis_r=basis_r,
                        reshape=reshape)

    def C22_idx(self,
                loc_l,
                loc_r,
                basis_l=None,
                basis_r=None,
                reshape=True):
        loc_l, loc_r = self.get_locations(idx_l), self.get_locations(idx_r)
        return self.C22(loc_l,
                        loc_r,
                        basis_l=basis_l,
                        basis_r=basis_r,
                        reshape=reshape)

class discrete_structure(covariance_structure):

    def __init__(self,
                 S):
        self.grid = (np.arange(S.shape[0]),)
        self._grid = np.asarray(self.grid)
        self.S_ = S
        
    def C00(self,
            loc_l,
            loc_r,
            reshape=True):
        
        loc_l, loc_r, grid_l, grid_r = _get_LR(self._grid, loc_l, loc_r)
        loc_l = loc_l[:,0].astype(int)
        loc_r = loc_r[:,0].astype(int)
        return self._reshape(self.S_[loc_l][:,loc_r],
                             reshape,
                             grid_l,
                             grid_r)

    def C01(self,
            loc_l,
            loc_r,
            basis_r=None,
            reshape=True):
        loc_l, loc_r, grid_l, grid_r = _get_LR(self._grid, loc_l, loc_r)            
        loc_l = loc_l[:,0].astype(int)
        loc_r = loc_r[:,0].astype(int)
        nl = self.S_[loc_l,0].shape[0]
        nr = self.S_[loc_r,0].shape[0]
        return self._reshape(np.zeros((nl, nr, 0)),
                             reshape,
                             grid_l,
                             grid_r)

    def C10(self,
            loc_l,
            loc_r,
            basis_l=None,
            reshape=True):
        loc_l, loc_r, grid_l, grid_r = _get_LR(self._grid, loc_l, loc_r)            
        loc_l = loc_l[:,0].astype(int)
        loc_r = loc_r[:,0].astype(int)
        nl = self.S_[loc_l,0].shape[0]
        nr = self.S_[loc_r,0].shape[0]
        return self._reshape(np.zeros((nl, nr, 0)),
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
        loc_l = loc_l[:,0].astype(int)
        loc_r = loc_r[:,0].astype(int)
        nl = self.S_[loc_l,0].shape[0]
        nr = self.S_[loc_r,0].shape[0]
        return self._reshape(np.zeros((nl, nr, 0, 0)),
                             reshape,
                             grid_l,
                             grid_r)

    def C20(self,
            loc_l,
            loc_r,
            basis_l=None,
            reshape=True):
        loc_l, loc_r, grid_l, grid_r = _get_LR(self._grid, loc_l, loc_r)            
        loc_l = loc_l[:,0].astype(int)
        loc_r = loc_r[:,0].astype(int)
        nl = self.S_[loc_l,0].shape[0]
        nr = self.S_[loc_r,0].shape[0]
        return self._reshape(np.zeros((nl, nr, 0, 0)),
                             reshape,
                             grid_l,
                             grid_r)

    def C02(self,
            loc_l,
            loc_r,
            basis_r=None,
            reshape=True):
        loc_l, loc_r, grid_l, grid_r = _get_LR(self._grid, loc_l, loc_r)            
        loc_l = loc_l[:,0].astype(int)
        loc_r = loc_r[:,0].astype(int)
        nl = self.S_[loc_l,0].shape[0]
        nr = self.S_[loc_r,0].shape[0]
        return self._reshape(np.zeros((nl, nr, 0, 0)),
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
        loc_l = loc_l[:,0].astype(int)
        loc_r = loc_r[:,0].astype(int)
        nl = self.S_[loc_l,0].shape[0]
        nr = self.S_[loc_r,0].shape[0]
        return self._reshape(np.zeros((nl, nr, 0, 0, 0)),
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
        loc_l = loc_l[:,0].astype(int)
        loc_r = loc_r[:,0].astype(int)
        nl = self.S_[loc_l,0].shape[0]
        nr = self.S_[loc_r,0].shape[0]
        return self._reshape(np.zeros((nl, nr, 0, 0, 0)),
                             reshape,
                             grid_l,
                             grid_r)

    def C22(self,
            loc_l,
            loc_r,
            basis_l=None,
            reshape=True):
        loc_l, loc_r, grid_l, grid_r = _get_LR(self._grid, loc_l, loc_r)            
        loc_l = loc_l[:,0].astype(int)
        loc_r = loc_r[:,0].astype(int)
        nl = self.S_[loc_l,0].shape[0]
        nr = self.S_[loc_r,0].shape[0]
        return self._reshape(np.zeros((nl, nr, 0, 0, 0, 0)),
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

