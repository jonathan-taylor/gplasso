import numpy as np

class barrier(object):

    def __init__(self,
                 shift=0.5,
                 scale=1):
        self.shift=shift
        self.scale = scale

    def value(self, arg):
        if np.any(arg < 0):
            return np.inf
        val = self.scale * (-np.log(arg) + np.log(arg + self.shift)).sum()
        return val
    
    def gradient(self, arg, d):
        # d of shape arg.shape + ...
        d_shape = d.shape
        d = d.reshape(arg.shape + (-1,))
        G =  self.scale * (-1 / arg + 1 / (arg + self.shift))
        val = (G @ d).reshape(d_shape[1:])
        if np.any(arg < 0):
            val = val * np.nan
        return val

    def hessian(self, arg, l, r):
        # l of shape (arg.shape,-1)
        # r of shape (arg.shape,-1)
        l_shape = l.shape
        l = l.reshape(arg.shape + (-1,))
        r_shape = r.shape
        r = r.reshape(arg.shape + (-1,))
        H = self.scale * ((1 / arg**2 - 1 / (arg + self.shift)**2))
        val = l.T @ (H[:,None] * r)
        val = val.reshape(l_shape[1:] + r_shape[1:])
        if np.any(arg < 0):
            val = val * np.nan
        return val

class logdet(object):

    def value(self, arg):
        eigvals = np.linalg.eigvalsh(arg)
        if np.any(eigvals < 0):
            return np.inf
        return -np.log(eigvals).sum()

    def gradient(self, arg, d):
        # d of shape (arg.shape,-1)
        d_shape = d.shape
        d = d.reshape(arg.shape + (-1,))
        arg_i = np.linalg.inv(arg)
        eigvals = np.linalg.eigvalsh(arg)
        if np.any(eigvals < 0):
            factor = np.nan
        else:
            factor = 1
        return -factor * np.einsum('ij,ijk->k', arg_i, d).reshape(d_shape[2:])

    def hessian(self, arg, l, r):
        # l of shape (arg.shape,-1)
        # r of shape (arg.shape,-1)
        eigvals = np.linalg.eigvalsh(arg)
        if np.any(eigvals < 0):
            factor = np.nan
        else:
            factor = 1
        l_shape = l.shape
        l = l.reshape(arg.shape + (-1,))
        r_shape = r.shape
        r = r.reshape(arg.shape + (-1,))
        arg_i = np.linalg.inv(arg)
        Vl = np.einsum('ijk,jm->imk', l, arg_i)
        Vr = np.einsum('ijk,jm->imk', r, arg_i)
        value = np.einsum('ijk,jim->km', Vl, Vr)
        return factor * value.reshape(l_shape[2:] + r_shape[2:])

