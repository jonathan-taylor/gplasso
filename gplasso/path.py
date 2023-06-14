import numpy as np

def path(Z,
         T,
         cov_kernel,
         lagrange,
         penalty=None):

    if penalty is None:
        penalty = np.ones_like(Z)

    active = []
    signs = []
    resid = Z
    b_U = penalty
    b_L = -penalty
    lam_cur = np.inf
    while True:
        inactive_U = np.argmax(resid / b_U)
        inactive_L = np.argmax(resid / b_L)
        if (resid[inactive_U] / b_U[inactive_U] >
            resid[inactive_L] / b_L[inactive_L]):
            inactive_var, inactive_sign = (inactive_U, 1)
            hit_time = resid[inactive_U] / b_U[inactive_U]
        else:
            inactive_var, inactive_sign = (inactive_L, -1)
            hit_time = -resid[inactive_L] / b_L[inactive_L]

        if active:
            S = np.array(signs)
            Q = cov_kernel.C00(T[active], T[active])
            Qi = np.linalg.solve(Q)
            beta = S * Qi @ Z[active]
            offset = S * Qi @ (penalty[active] * S)
            check = offset < 0
            leave_time = max(np.max(-beta[check] / offset[check]), 0)
            

    lam_max = np.fabs(Z) / penalty
