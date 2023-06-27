import numpy as np, pandas as pd
import jax.numpy as jnp
from selectinf.distributions.discrete_family import discrete_family
from selectinf.constraints.affine import constraints as AC
from scipy.interpolate import interp1d

from .jacobian import decompose

def inference(peaks,
              model_kernel,
              inference_kernel=None,
              param=None,
              marginal=False,
              C00=None,
              two_sided=True,
              use_jacobian=True,
              level=None):

    if param is None:
        param = np.zeros(len(peaks))

    if inference_kernel is None:
        inference_kernel = model_kernel
        
    # we don't need to do conditional covariances given the R(\nabla f_E;f_E)
    # because we decompose with respect to f_E, ie for any A
    # Cov(A,f_E|R(\nabla f_E;f_E)) = Cov(A,f_E) becuase Cov(f_E,R(\nable f_E;f_E))=0

    IK = inference_kernel
    
    if C00 is None:
        locations = [p.location for p in peaks]
        C00 = IK.C00(locations,
                     locations)
    C00i = np.linalg.inv(C00)
    s = np.array([p.sign for p in peaks]).reshape(-1)
    con = AC(-np.diag(s) @ C00i,
             -s * C00i @ np.array([p.penalty * p.sign for p in peaks]).reshape(-1),
             covariance=C00)

    Z = np.array([p.value for p in peaks]).reshape(-1)
    P = []
    L_ci = []
    U_ci = []
    
    for i, ref_point in enumerate(peaks):
        target_dir = C00i[i]
        if marginal:
            pts = [ref_point]
        else:
            pts = peaks
        logdet_info = decompose(pts,
                                model_kernel,
                                inference_kernel)[0]
        L, target, U, sd = con.bounds(target_dir, Z)
        G, N, logdet_map = logdet_info['total']

        G_target = C00 @ target_dir / sd**2
        N_target = Z - G_target * target
        if use_jacobian:
            def _jacobian(x):
                return logdet_map(N_target + G_target * x)
        else:
            def _jacobian(x):
                return 0

        xval = np.linspace(max(-10*sd, L), min(10*sd, U), 501)
        f_xval = np.array([_jacobian(x) for x in xval])
        f = interp1d(xval, f_xval)
        xval = np.linspace(max(-10*sd, L), min(10*sd, U), 2001)
        W = np.exp(-np.array([f(x) for x in xval]) - 0.5 * xval**2 / sd**2)
        F = discrete_family(xval, W)
        cdf = F.cdf(param[i], x=C00i[i].dot(Z))
        if level is not None:
            l, u = F.equal_tailed_interval(C00i[i].dot(Z),alpha=1-level)
            L_ci.append(l)
            U_ci.append(u)

        if not two_sided:
            if ref_point.sign == 1:
                P.append(1 - cdf)
            else:
                P.append(cdf)
        else:
            P.append(2 * min(cdf, 1-cdf))

    df = pd.DataFrame({'Location':np.squeeze([p.location for p in peaks]),
                       'Estimate':C00i @ Z,
                       'P-value (1-sided)':P,
                       'Param':param})
    if level is not None:
        df['L ({:.0%})'.format(level)] = L_ci
        df['U ({:.0%})'.format(level)] = U_ci

    return df.set_index('Location')
