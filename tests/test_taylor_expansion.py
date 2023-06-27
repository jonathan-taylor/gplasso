import numpy as np

from gplasso.taylor_expansion import taylor_expansion

def test_taylor_expansion(dim=3, order=2):

    rng = np.random.default_rng(4)
    X_pred = rng.uniform(size=(dim,))
    X_obs = rng.uniform(size=(400,dim))
    Y_obs = rng.normal(size=(400,))

    constant_0 = rng.normal()
    linear_0 = rng.normal(size=(dim,))
    quadratic_0 = rng.normal(size=(dim, dim))
    quadratic_0 = 0.5 * (quadratic_0 + quadratic_0.T)
    
    def myquad(x):
        return constant_0 + (linear_0 * (x - X_pred)).sum() + 0.5 * ((x - X_pred) @ quadratic_0 @ (x - X_pred))

    Y_obs = np.array([myquad(x) for x in X_obs])

    Z = rng.normal(size=(500,dim))
    precision = (0.2)**(-2) * (Z.T @ Z / 500)

    expansion = taylor_expansion(X_pred,
                                 X_obs,
                                 Y_obs,
                                 precision,
                                 order=order)

    for t, e in zip([constant_0, linear_0, quadratic_0],
                    expansion[:3]):
        assert np.linalg.norm(t-e) / np.linalg.norm(e) < 1e-5

