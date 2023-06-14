# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: gplasso
#     language: python
#     name: gplasso
# ---

import numpy as np


def solve_barrier(y, rho, c=0.05, niter=20, x=None):
  """
  minimize_x y*x - log(x/(x+c)) + 0.5 * rho * x**2
  """
  y = np.asarray(y)
  x = np.maximum(-y / rho, 0.1)

  value = lambda x: (y * x).sum() - np.log(x/(x+c)).sum() + 0.5 * rho * (x**2).sum()
  cur_value = value(x)
  for i in range(niter):
    G = y - 1/x + 1/(x+c) + rho * x
    H = 1 / x**2 - 1 / (x + c)**2 + rho
    step = - G / H
    f = np.ones_like(G)
    if np.fabs(step / np.maximum(np.fabs(y), 1)).max() < 1e-4:
      break
    cont = True
    while cont:
      W = x + f * step < 0
      if not np.any(W):
        cont = False
      f[W] *= 0.5

    tries = 0
    cont = True
    while cont:
      new_value = value(x + f * step)
      if new_value < cur_value:
        x = x + f * step
        cur_value = new_value
        cont = False
      if tries == 50:
        raise ValueError('no descent')
      f *= 0.5
      tries += 1
  return x

def solve_logdet(M, rho, niter=20):
  """
  minimize_A tr(MA) - log(det(A)) + 0.5 * rho * tr(A**2)
  
  for symmetric M, over non-negdefinite A

  if M is not positive definite, we 
  """
  D, U = np.linalg.eigh(M)
  soln = (-D + np.sqrt(D**2 + 4 * rho)) / (2 * rho)
  V = (U * soln[None,:]) @ U.T
  return V

def solve_problem(linear, 
                  quadratic, 
                  barrier_linear,
                  barrier_offset,
                  logdet_linear,
                  logdet_offset,
                  x_warm=None,
                  barrier_rho=1,
                  logdet_rho=1,
                  niter=500):
    """
    solve problem
    
    minimize_x x'l + 0.5 * x'Qx + B(A_1x+b_1) - logdet(A_2x+b_2)
    
    with l=linear, Q=quadratic, A_1=barrier_linear,
    b_1=barrier_offset, A_2=logdet_linear, b_2=logdet_offset
    
    uses ADMM making duplications
    
    w_1=A_1x+b_1
    w_2=A_2x+b_2
    """
    
    if x_warm is None:
        x_warm = -np.linalg.inv(quadratic) @ linear
        
    barrier_dual = np.zeros(barrier_offset.shape)
    logdet_dual = np.zeros(logdet_offset.shape)
    
    """
    augmented lagrangian adds
    
    y_1'(A_1x+b_1-w_1) + 0.5 * rho_1 \|A_1x+b_1-w_1\|^2_2
    y_2'(A_2x+b_2-w_2) + 0.5 * rho_2 \|A_2x+b_2-w_2\|^2_2
    """
    
    Q_full = (quadratic + 
              barrier_rho * barrier_linear.T @ barrier_linear + 
              logdet_rho * np.einsum('ijk,ijl->kl',
                                     logdet_linear, logdet_linear))
    Qi_full = np.linalg.inv(Q_full)

    x = x_warm.copy()
    for _ in range(niter):
        
        # update w_barrier
        
        barrier_w = solve_barrier(-barrier_dual
                                  -barrier_rho * (barrier_linear @ x 
                                                  + barrier_offset),
                                  barrier_rho)
        
        # update w_logdet
        L = (-logdet_dual 
             -logdet_rho * (np.einsum('ijk,k->ij',
                            logdet_linear, x)
                             + logdet_offset))
        logdet_w = solve_logdet(L, 
                                logdet_rho)
             
        # update x
             
        L = (linear + 
             barrier_linear.T @ (barrier_dual + 
                                 barrier_rho * (barrier_offset - barrier_w)) +
             np.einsum('ijk,ij->k',
                       logdet_linear,
                       logdet_dual + 
                       logdet_rho * (logdet_offset - logdet_w)))
        x = -Qi_full @ L
        
        # update dual variables
        
        DEBUG = False
        if DEBUG:
          print(np.linalg.norm(barrier_linear @ x + barrier_offset - barrier_w) / np.linalg.norm(barrier_w), 'bar')
          print(np.linalg.norm(np.einsum('ijk,k->ij',
                                         logdet_linear, x) + logdet_offset - logdet_w) / np.linalg.norm(logdet_w), 'logdet')
          print(np.linalg.eigvalsh(np.einsum('ijk,k->ij',
                                             logdet_linear, x) + logdet_offset), 'eigvals x')
          print(np.linalg.eigvalsh(logdet_w), 'eigvals w')
        barrier_dual += barrier_rho * (barrier_linear @ x + barrier_offset - barrier_w)
        logdet_dual += logdet_rho * (np.einsum('ijk,k->ij',
                                                logdet_linear, x) + logdet_offset - logdet_w)
             
    return x
  
if __name__ == "__main__":
  y, rho, c, niter = 5, 0.4, 0.5, 20


  soln = solve_barrier(np.random.standard_normal(500), rho, c, niter)
  xval = np.linspace(0.01,2 * soln,501)
  #plt.plot(xval, value(xval, 0)[0])
  #plt.scatter([soln[0]], [value(soln, 0)[0]])

  A = np.random.standard_normal((50, 50))
  A += A.T

  V = solve_logdet(A, 0.4)

  def sym(k):
    A = np.random.standard_normal((k, k))
    return 0.5 * (A + A.T)

  # Commented out IPython magic to ensure Python compatibility.
  # %timeit solve_logdet(sym(10), 1)

  # +
  p, kB, kL = 3, 10, 5

  A_B = np.random.standard_normal((kB,p))
  A_L = np.random.standard_normal((kL,kL,p))
  o_B = np.random.standard_normal(kB)
  o_L = np.random.standard_normal((kL,kL))

  X = np.random.standard_normal((50,p))
  quadratic = X.T @ X / 50
  linear = np.random.standard_normal(p)

  solve_problem(linear,
                quadratic,
                A_B,
                o_B,
                A_L,
                o_L)



  # -



