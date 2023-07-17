from copy import copy

import numpy as np
import regreg.api as rr

def fit_gp_lasso(observed_process,
                 covariance_kernels,
                 penalty_weights,
                 tol=1e-12,
                 solve_args={'min_its':200},
                 num_lam=10):
    solve_args.update(tol=tol)
    
    Z = observed_process # shorthand

    lam_max = np.fabs(Z / penalty_weights).max()

    lam_vals = np.exp(np.linspace(np.log(lam_max), 0, num_lam))

    E = ever_active = np.fabs(Z) / penalty_weights >= 0.999 * lam_max
    soln = np.zeros(E.sum())
    E_idx = np.nonzero(E)
    loc_E, S_E, S_IE = _update_S(covariance_kernels,
                                 E_idx,
                                 None)

    Z_flat = Z.reshape(-1)
    
    for i, lam in enumerate(lam_vals):
        
        (cur_E,
         cur_S_E,
         cur_S_IE,
         cur_loc_E) = (E.copy(),
                       S_E.copy(),
                       S_IE.copy(),
                       copy(loc_E))
        
        num_tries = 0
        soln_E = soln.copy()
        
        if i < len(lam_vals):
            solve_args['tol'] = 1e-8
            solve_args['min_its'] = 50
            solve_args['max_its'] = 200
        else:
            solve_args['tol'] = 1e-14
            solve_args['max_its'] = 2000

        while True:
            Z_E = Z[E_idx]
            loss_E = rr.quadratic_loss(cur_S_E.shape, Q=cur_S_E)
            linear_term_E = rr.identity_quadratic(0,0,-Z_E,1e-8)
            penalty_E = rr.weighted_l1norm(penalty_weights[E_idx], lagrange=lam)
            problem_E = rr.simple_problem(loss_E, penalty_E)
            problem_E.coefs[:soln_E.shape[0]] = soln_E
            soln = problem_E.solve(linear_term_E,
                                   **solve_args)

            # which coordinates of ever_active are non-zero

            fit = cur_S_IE @ soln
            subgrad = (Z_flat - fit).reshape(Z.shape)
            
            failing = (np.fabs(subgrad) / penalty_weights) >= 0.999 * lam
            failing[E_idx] = False # current E was already included

            if failing.sum() > 0: # add variables to ever_active
                failing_idx = np.nonzero(failing)
                E_idx = tuple([np.hstack([cur, cand]) for cur, cand in zip(E_idx,
                                                                           failing_idx)])
                # append appropriate columns to S_IE
                
                cur_loc_E, cur_S_E, cur_S_IE = _update_S(covariance_kernels,
                                                         failing_idx,
                                                         (cur_loc_E,
                                                          cur_S_E,
                                                          cur_S_IE))
                num_tries += 1
            else:
                (S_E,
                 S_IE,
                 loc_E) = (cur_S_E,
                           cur_S_IE,
                           cur_loc_E)
                break

            if num_tries >= 10:
                raise ValueError("active set hasn't converged")
            
    full_soln = np.zeros_like(Z)
    full_soln[E_idx] = soln
    return full_soln != 0, full_soln, subgrad

def _update_S(covariance_kernels,
              idx,
              cur_val):
                             
    K = covariance_kernels[0]
    num_grid = np.product(K.grid[0].shape)

    if cur_val is not None:
        loc_E, S_E, S_IE = cur_val
    else:
        loc_E, S_E, S_IE = (np.zeros((0, len(K.grid[0].shape))),
                            np.zeros((0,0)),
                            np.zeros((num_grid, 0)))
    num_cur = S_E.shape[0]
    new_loc = K.get_locations(idx)
    num_new = new_loc.shape[0]

    new_S_E = np.zeros((num_cur+num_new,)*2)
    new_IE = np.zeros((num_grid, num_new))

    # update S[E,E]
    new_S_E[:num_cur,:num_cur] = S_E
    for K_ in covariance_kernels:
        new_S_E[num_cur:,num_cur:] += K_.C00(new_loc,
                                             new_loc)
        if num_cur > 0:
            incr = K_.C00(loc_E,
                          new_loc)
            new_S_E[:num_cur,num_cur:] += incr
            new_S_E[num_cur:,:num_cur] += incr.T

        new_IE += K_.C00(None,
                         new_loc).reshape((num_grid, -1))
    
    new_S_IE = np.concatenate([S_IE, new_IE], axis=1)
    return np.concatenate([loc_E, new_loc], axis=0), new_S_E, new_S_IE 
