import jax
import jax.numpy as jnp
import jax.random as random
import jax.nn

import numpy as np
import numpy.random as nprandom
from scipy.spatial.distance import cdist

import pandas as pd

# ======= Exact Gaussian Process Regression =======

def sq_exp_kernel(U, V, tau, l, nugget = 1e-5):
    dist = jnp.array(cdist(U, V, 'sqeuclidean'))
    exp_term = jnp.exp(-dist / (2 * l ** 2))
    out = tau ** 2 * exp_term
    out = jnp.where(out == 0, nugget, out)
    return out


def eb_likelihood(X, Y, params):
    tau, l, sigma2 = jax.nn.softplus(params)
    C = sq_exp_kernel(X, X, tau, l) + sigma2 * jnp.eye(X.shape[0])
    L = jnp.linalg.cholesky(C)
    log_C_det = 2 * jnp.sum(jnp.log(jnp.diag(L))) 
    C_inv = jnp.linalg.inv(L).T @ jnp.linalg.inv(L)
    out = 0.5 * log_C_det + 0.5 * Y.T @ C_inv @ Y + X.shape[0] / 2 * jnp.log(2 * jnp.pi)
    if(out.shape != (1, 1)):
        return(out)
    else:
        return(out[0,0])
    
class GP_trainer():
    def __init__(self, X, Y, params, tol = 1e-3, maxiter = 1000):
        self.X = jnp.array(X)
        self.Y = jnp.array(Y)
        self.params = params # [tau l sigma^2]
        self.tol = tol
        self.maxiter = maxiter
        self.param_estimates = []
        self.final_loss = 0
    
    def train(self):
        loss = eb_likelihood(self.X, self.Y, self.params)
        loss_grad = jax.grad(eb_likelihood, argnums = 2)

        for i in range(self.maxiter):
            grad = loss_grad(self.X, self.Y, self.params)
            self.params = self.params - 1e-4 * grad
            next_loss = eb_likelihood(self.X, self.Y, self.params)
            if jnp.abs(next_loss - loss) < self.tol: break
            loss = next_loss
            print('Current loss: ', loss, 'Current params: ', jax.nn.softplus(self.params))
        
        return jax.nn.softplus(self.params), loss

class GP_regression():
    def __init__(self, X_train, Y_train, X_test, standardize, params = None, tol = 1e-3, maxiter = 1000, nugget = 1e-5):
        self.X_train = jnp.array(X_train)
        self.Y_train = jnp.array(Y_train)
        self.X_test = jnp.array(X_test)
        self.standardize = standardize
        self.params = params
        self.tol = tol
        self.maxiter = maxiter
        self.nugget = nugget
    def fit(self):
        self.Y_train_mean = jnp.mean(self.Y_train)
        self.Y_train = self.Y_train - self.Y_train_mean

        if(self.standardize):
            self.X_test = (self.X_test - jnp.mean(self.X_train)) / jnp.std(self.X_train)
            self.X_train = (self.X_train - jnp.mean(self.X_train)) / jnp.std(self.X_train)

        if(self.params is None):
            init_params = jnp.array([6.0, 0.0, 22.0])
            trainer = GP_trainer(X = self.X_train, Y = self.Y_train, params = init_params, tol = self.tol, maxiter = self.maxiter)
            self.params, self.loss = trainer.train()
            print('Got parameters: ', self.params)
        else:
            transformed_params = jnp.log(jnp.exp(self.params) - 1)
            self.loss = eb_likelihood(self.X_train, self.Y_train, transformed_params)

        KXX = sq_exp_kernel(self.X_train, self.X_train, self.params[0], self.params[1], self.nugget)
        KXX_ = sq_exp_kernel(self.X_train, self.X_test, self.params[0], self.params[1], self.nugget)
        KX_X_ = sq_exp_kernel(self.X_test, self.X_test, self.params[0], self.params[1], self.nugget)
        KX_X = KXX_.T
        L = jnp.linalg.cholesky(KXX + self.params[2] * jnp.eye(self.X_train.shape[0]))
        K_inv = jnp.linalg.inv(L).T @ jnp.linalg.inv(L)
        # K_inv = np.linalg.inv(KXX + self.params[2] * jnp.eye(self.X_train.shape[0]))
        self.gp_mean = KX_X @ K_inv @ self.Y_train
        self.gp_mean = self.gp_mean + self.Y_train_mean
        self.gp_cov = KX_X_ - KX_X @ K_inv @ KXX_
        return self.gp_mean, self.gp_cov, self.params, self.loss


# ======= Approximate Gaussian Process Regression =======


class SparseGP():
    def __init__(self, train_X, train_Y, test_X, inducing_X, params, method = 'SoR', standardize = False, nugget = 1e-5 ):
        self.train_X = train_X
        self.train_Y = train_Y
        self.test_X = test_X
        self.inducing_X = inducing_X
        self.params = params # [tau, l, sigma^2]
        self.method = method
        self.standardize = standardize
        self.nugget = nugget
    def fit(self):
        self.train_Y_mean = jnp.mean(self.train_Y)
        self.train_Y = self.train_Y - self.train_Y_mean
        if(self.standardize):
            self.test_X = (self.test_X - jnp.mean(self.test_X)) / jnp.std(self.test_X)
            self.train_X = (self.train_X - jnp.mean(self.train_X)) / jnp.std(self.train_X)
            self.inducing_X = (self.inducing_X - jnp.mean(self.inducing_X)) / jnp.std(self.inducing_X)

        Kuu = sq_exp_kernel(self.inducing_X, self.inducing_X, self.params[0], self.params[1], self.nugget)
        KuX = sq_exp_kernel(self.inducing_X, self.train_X, self.params[0], self.params[1], self.nugget)
        KX_u = sq_exp_kernel(self.test_X, self.inducing_X, self.params[0], self.params[1], self.nugget)
        KXu = KuX.T
        KuX_ = KX_u.T
        
        if(self.method == 'SoR'):
            L = jnp.linalg.cholesky(1 / self.params[2] * KuX @ KXu + Kuu)
            # Sigma = jnp.linalg.inv(L).T @ jnp.linalg.inv(L)
            Sigma = np.linalg.inv(1 / self.params[2] * KuX @ KXu + Kuu)


            gp_mean = 1 / self.params[2] * KX_u @ Sigma @ KuX @ self.train_Y
            gp_mean = gp_mean + self.train_Y_mean
            gp_cov = KX_u @ Sigma @ KuX_
            return gp_mean, gp_cov
        
        if(self.method == 'SPGP'):
            Kuu_inv = np.linalg.inv(Kuu)
            QXX_diag = np.sum(np.multiply(KuX.T , np.matmul(KuX.T, Kuu_inv)), axis = 1)
            KXX_diag = self.params[0] ** 2 * jnp.ones(self.train_X.shape[0]) + self.nugget
            QX_X_ = KX_u @ Kuu_inv @ KX_u.T
            KX_X_ = sq_exp_kernel(self.test_X, self.test_X, self.params[0], self.params[1], self.nugget)

            Lambda_diag = KXX_diag - QXX_diag + self.params[2]
            temp_mat2 = np.divide(KXu, np.outer(Lambda_diag, np.ones(KXu.shape[1])))
            Sigma = np.linalg.inv(np.matmul(KuX, temp_mat2) + Kuu)

            # Convert to (n, 1) shape from (n,) if needed
            if(len(Lambda_diag.shape) == 1 and self.train_X.shape[1] == 1): Lambda_diag = Lambda_diag[:, jnp.newaxis]
            if(len(self.train_Y.shape) == 1 and self.train_X.shape[1] == 1): self.train_Y = self.train_Y[:, jnp.newaxis]
            Lambda_inv_Y = jnp.divide(self.train_Y, Lambda_diag)

            gp_mean = KX_u @ Sigma @ KuX @ Lambda_inv_Y
            gp_mean = gp_mean + self.train_Y_mean
            gp_cov = KX_X_ - QX_X_ + KX_u @ Sigma @ KX_u.T
            return gp_mean, gp_cov

