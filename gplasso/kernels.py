from dataclasses import dataclass

import numpy as np
import jax.numpy as jnp
from jax import jacfwd
import gstools as gs

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

class covariance_structure(object):

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
        return covariance_structure(gaussian_kernel_,
                                    kernel_args={'precision':precision,
                                                 'var':var},
                                    grid=grid,
                                    sampler=sampler)

    # default simulation method -- better subclasses will overwrite

    def sample(self, **sample_args):

        if self.sampler is None:
            raise ValueError('must provide "sampler" at init to draw a sample')
        return self.sampler(**sample_args)

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
                 S,
                 sampler=None):
        self.grid = (np.arange(S.shape[0]),)
        self._grid = np.asarray(self.grid)
        self.S_ = S
        if sampler is None:
            npt = S.shape[0]
            shape = S.shape[:1]
            U, D = np.linalg.svd(S)[:2]
            sampler = SVDSampler(U, D, npt, shape)

        self.sampler = sampler
        
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

######################

class isotropic(covariance_structure):
    # covariances of the form R(s,t) = g((s-t)'Q(s-t)/2)
    # canonical example: Gaussian, g(r)=exp(-r)

    def __init__(self,
                 Q,
                 grid=None,
                 sampler=None,
                 var=1): # default grid of x values
        self.kernel_ = lambda loc_l, loc_r: kernel(loc_l,
                                                   loc_r,
                                                   **kernel_args)

        self.grid = grid
        self._grid = np.asarray(grid)
        self.sampler = sampler
        self.Q = Q
        self.var = var / self._func(0) 

    # location based computations

    def C00(self,
            loc_l,
            loc_r,
            reshape=True):
        D, grid_l, grid_r = _get_LR_np(self._grid, loc_l, loc_r)
        value = self._reshape(self._deriv0(D),
                              reshape,
                              grid_l,
                              grid_r)
        return value
    
    def C10(self,
            loc_l,
            loc_r,
            basis_l=None,
            reshape=True):
        D, grid_l, grid_r = _get_LR_np(self._grid, loc_l, loc_r)
        value = self._deriv1(D)
        if basis_l is not None:
            value = np.einsum('ijk,lk->ijl', value, basis_l)
        value = self._reshape(value,
                              reshape,
                              grid_l,
                              grid_r)
        return value
    
    def C01(self,
            loc_l,
            loc_r,
            basis_r=None,
            reshape=True):
        D, grid_l, grid_r = _get_LR_np(self._grid, loc_l, loc_r)
        value = -self._deriv1(D)
        if basis_r is not None:
            value = np.einsum('ijk,lk->ijl', value, basis_r)
        value = self._reshape(value,
                              reshape,
                              grid_l,
                              grid_r)
        return value

    def C20(self,
            loc_l,
            loc_r,
            basis_l=None,
            reshape=True):
        D, grid_l, grid_r = _get_LR_np(self._grid, loc_l, loc_r)
        value = self._deriv2(D)
        if basis_l is not None:
            value = np.einsum('ijkl,mk,nl->ijmn',
                              value,
                              basis_l,
                              basis_l)
        value = self._reshape(value,
                              reshape,
                              grid_l,
                              grid_r)
        return value

    def C02(self,
            loc_l,
            loc_r,
            basis_r=None,
            reshape=True):
        D, grid_l, grid_r = _get_LR_np(self._grid, loc_l, loc_r)
        value = self._deriv2(D)
        if basis_r is not None:
            value = np.einsum('ijkl,mk,nl->ijmn',
                              value,
                              basis_r,
                              basis_r)
        value = self._reshape(value,
                              reshape,
                              grid_l,
                              grid_r)
        return value
   
    def C11(self,
            loc_l,
            loc_r,
            basis_l=None,
            basis_r=None,
            reshape=True):
        D, grid_l, grid_r = _get_LR_np(self._grid, loc_l, loc_r)
        value = -self._deriv2(D)
        if basis_l is not None:
            value = np.einsum('ijkl,mk->ijml',
                              value,
                              basis_l)
        if basis_r is not None:
            value = np.einsum('ijkl,ml->ijkm',
                              value,
                              basis_r)
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
        D, grid_l, grid_r = _get_LR_np(self._grid, loc_l, loc_r)
        value = -self._deriv3(D)
        if basis_r is not None:
            value = np.einsum('ijklm,nm->ijkln',
                              value,
                              basis_r)
        if basis_l is not None:
            value = np.einsum('ijklm,kn,lo->ijnom',
                              value,
                              basis_l,
                              basis_l)

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
        D, grid_l, grid_r = _get_LR_np(self._grid, loc_l, loc_r)
        value = self._deriv3(D)
        if basis_l is not None:
            value = np.einsum('ijklm,nk->ijnml',
                              value,
                              basis_l)
        if basis_r is not None:
            value = np.einsum('ijklm,nl,mo->ijkno',
                              value,
                              basis_r,
                              basis_r)

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
        D, grid_l, grid_r = _get_LR_np(self._grid, loc_l, loc_r)
        value = self._deriv3(D)
        if basis_r is not None:
            value = np.einsum('ijklmn,om,pn->ijklop',
                              value,
                              basis_r,
                              basis_r)
        if basis_l is not None:
            value = np.einsum('ijklmn,ok,pl->ijopmn',
                              value,
                              basis_l,
                              basis_l)

        return self._reshape(self._deriv4(D),
                             reshape,
                             grid_l,
                             grid_r)
        
    #####################

    def _func(self, arg, order=0):
        raise NotImplementedError('must implement up to 4th order derivative')
        
    def _deriv0(self, v):
        # v_flat of shape (-1,Q.shape[0])
        d = self.Q.shape[0]
        v_flat = v.reshape((-1,d))
        Qv = np.einsum('ij,jk->ik', v_flat, self.Q)
        arg = (Qv * v_flat).sum(-1)
        return self.var * self._func(arg/2, order=0).reshape(v.shape[:-1])

    def _deriv1(self, v):
        # v_flat of shape (-1,Q.shape[0])
        d = self.Q.shape[0]
        v_flat = v.reshape((-1,d))
        Qv = np.einsum('ij,jk->ik', v_flat, self.Q)
        arg = (Qv * v_flat).sum(-1)
        g_1 = self._func(arg/2, order=1)
        return self.var * (g_1[:,None] * Qv).reshape(v.shape[:-1] + (d,))

    def _deriv2(self, v):
        # v_flat of shape (-1,Q.shape[0])
        d = self.Q.shape[0]
        v_flat = v.reshape((-1,d))
        Qv = np.einsum('ij,jk->ik', v_flat, self.Q)
        arg = (Qv * v_flat).sum(-1)
        g_1 = self._func(arg/2, order=1)
        g_2 = self._func(arg/2, order=2)
        
        V_2 = np.einsum('i,ij,ik->ijk', g_2, Qv, Qv)
        V_1 = np.einsum('i,jk->ijk', g_1, self.Q)
        
        return self.var * (V_1 + V_2).reshape(v.shape[:-1] + (d, d))
    
    def _deriv3(self, v):
        # v_flat of shape (-1,Q.shape[0])
        d = self.Q.shape[0]
        v_flat = v.reshape((-1,d))
        Qv = np.einsum('ij,jk->ik', v_flat, self.Q)
        arg = (Qv * v_flat).sum(-1)
        g_2 = self._func(arg/2, order=2)
        g_3 = self._func(arg/2, order=3)
        
        V_3 = np.einsum('i,ij,ik,il->ijkl', g_3, Qv, Qv, Qv)
        V_2 = (np.einsum('i,ij,kl->ijkl', g_2, Qv, self.Q) +
               np.einsum('i,ik,jl->ijkl', g_2, Qv, self.Q) +
               np.einsum('i,il,jk->ijkl', g_2, Qv, self.Q))
        return self.var * (V_3 + V_2).reshape(v.shape[:-1] + (d,)*3)

    def _deriv4(self, v):
        # v_flat of shape (-1,Q.shape[0])
        d = self.Q.shape[0]
        v_flat = v.reshape((-1,d))
        Qv = np.einsum('ij,jk->ik', v_flat, self.Q)
        arg = (Qv * v_flat).sum(-1)
        
        g_2 = self._func(arg/2, order=2)
        g_3 = self._func(arg/2, order=3)
        g_4 = self._func(arg/2, order=4)
        
        V_4 = np.einsum('i,ij,ik,il,im->ijklm', g_4, Qv, Qv, Qv, Qv)
        
        V_3 = (np.einsum('i,ij,ik,lm->ijklm', g_3, Qv, Qv, self.Q) +
               np.einsum('i,ij,il,km->ijklm', g_3, Qv, Qv, self.Q) +
               np.einsum('i,ij,im,kl->ijklm', g_3, Qv, Qv, self.Q) +
               np.einsum('i,ik,il,jm->ijklm', g_3, Qv, Qv, self.Q) +
               np.einsum('i,ik,im,jl->ijklm', g_3, Qv, Qv, self.Q) +
               np.einsum('i,il,im,jk->ijklm', g_3, Qv, Qv, self.Q))
        
        V_2 = (np.einsum('i,jk,lm->ijklm', g_2, self.Q, self.Q) +
               np.einsum('i,jl,km->ijklm', g_2, self.Q, self.Q) +
               np.einsum('i,jm,kl->ijklm', g_2, self.Q, self.Q))
        
        return self.var * (V_4 + V_3 + V_2).reshape(v.shape[:-1] + (d,)*4)

class gaussian_kernel(isotropic):
    
    def _func(self, arg, order=0):
        return (-1)**order * np.exp(-arg)
    

######################

@dataclass
class SVDSampler(object):

    U : np.ndarray # left singular vectors of S
    D : np.ndarray # square root of singular vectors of S
    npt : int      # when flattened, how many pts?
    shape : tuple  # how to reshape final vector

    def __call__(self, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        elif type(rng) == int:
            rng = np.random.default_rng(rng)

        Z = self.U @ (np.sqrt(self.D) * rng.standard_normal(self.npt))
        Z = Z.reshape(self.shape)

        return Z

class GSToolsSampler(object):

    def __init__(self,
                 model,
                 gridvals,
                 linear_map,
                 generator='RandMeth'):

        self.model = model
        self.generator = generator

        flattened = np.array([g.reshape(-1) for g in gridvals])
        self._shape = gridvals[0].shape
        
        self._modelpts = np.einsum('ij,ki->kj',
                                   flattened,
                                   linear_map) 

    def __call__(self, seed=None):
        
        _srf = gs.SRF(self.model, generator=self.generator, seed=seed)
        Z = _srf(self._modelpts)
        return Z.reshape(self._shape)
    
    @staticmethod
    def gaussian(gridvals,
                 precision,
                 generator='RandMeth',
                 var=1):
        model = gs.Gaussian(rescale=1/np.sqrt(2),
                            var=var,
                            nugget=0,
                            dim=precision.shape[0],
                            len_scale=[1]*precision.shape[0])
        A = np.linalg.cholesky(precision)
        return GSToolsSampler(model,
                              gridvals,
                              A,
                              generator=generator)

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


def _get_LR_np(grid, loc_l, loc_r):
    G = grid.transpose(list(range(1, grid[0].ndim+1)) + [0])
    G = G.reshape((-1, G.shape[-1]))
    grid_l, grid_r = False, False
    if loc_l is None:
        loc_l = G
        grid_l = True
    if loc_r is None:
        loc_r = G 
        grid_r = True
    loc_l, loc_r = np.asarray(loc_l), np.asarray(loc_r)
    D = np.array([np.subtract.outer(loc_l[:,i], loc_r[:,i])
                  for i in range(loc_l.shape[-1])])
    D = np.transpose(D, [1, 2, 0])
    return D, grid_l, grid_r

