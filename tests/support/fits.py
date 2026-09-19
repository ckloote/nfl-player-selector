"""A trusted reference for the Poisson fits: the same estimand reached by a route that
shares no code with `evaluate.poisson_glm`."""

import numpy as np
from scipy import optimize


def reference_mle(y, x):
    """An independent Poisson MLE, sharing no code with `evaluate.poisson_glm`.

    Minimising the negative log-likelihood with a general-purpose optimiser is a
    different route to the same estimand; agreeing with it is evidence the IRLS
    implementation is correct rather than merely self-consistent.
    """
    design = np.column_stack([np.ones_like(x), x])

    def nll(beta):
        eta = design @ beta
        return float(np.sum(np.exp(eta) - y * eta))

    return optimize.minimize(nll, np.zeros(2), method="BFGS", tol=1e-14).x


def reference_model_se(y, x, beta):
    """Model-based SEs from a numerically differentiated Hessian of the same NLL."""
    design = np.column_stack([np.ones_like(x), x])

    def nll(b):
        eta = design @ b
        return float(np.sum(np.exp(eta) - y * eta))

    step, hess = 1e-5, np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            a, b = np.zeros(2), np.zeros(2)
            a[i] = b[i] = step
            a[j] += step
            b[j] -= step
            hess[i, j] = (nll(beta + a) - nll(beta + b) - nll(beta - b) + nll(beta - a)) / (
                4 * step * step
            )
    return np.sqrt(np.diag(np.linalg.inv(hess)))
