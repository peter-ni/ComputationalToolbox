import numpy as np
import scipy.stats
import scipy.spatial
from scipy.stats import norm, invgamma, multivariate_normal
import pandas as pd
import sys
import cProfile

# ======== Gibbs Sampler for Bayesian Probit Regression =======

class ProbitGibbsSampler():
    def __init__(self, X, y, beta_init = None, tau_sq = 10, burn = 5000, nmc = 5000):
        self.X = np.hstack((np.ones((X.shape[0], 1)), X))
        self.y = y
        if beta_init is None: self.beta_init = np.zeros(self.X.shape[1])
        if beta_init is not None: self.beta_init = beta_init
        self.tau_sq = tau_sq
        self.burn = burn
        self.nmc = nmc
    def train(self, silence = False):
        n = self.X.shape[0]
        p = self.X.shape[1]
        Sigma = np.linalg.inv(self.X.T @ self.X + np.diag(np.full(p, p / self.tau_sq)))
        beta_list = np.zeros((self.nmc, p))
        beta = self.beta_init

        for i in range(self.burn + self.nmc):
            if not silence: print('Doing iteration: ', i)
            z = np.zeros(n)
            for y_ind in range(len(self.y)):
                mean = self.X[y_ind] @ beta
                if self.y[y_ind] == 1: clip_a = 0; clip_b = np.inf
                if self.y[y_ind] == 0: clip_a = -np.inf; clip_b = 0
                z[y_ind] = scipy.stats.truncnorm.rvs(a = clip_a - mean, b = clip_b - mean, loc = mean, scale = 1)
            beta = np.random.multivariate_normal(mean = Sigma @ self.X.T @ z, cov = Sigma)
            if i >= self.burn: beta_list[i - self.burn] = beta

        beta_hat = np.mean(beta_list, axis = 0)
        beta_lower = np.percentile(beta_list, 2.5, axis = 0)
        beta_upper = np.percentile(beta_list, 97.5, axis = 0)
        fitted_values = scipy.stats.norm.cdf(self.X @ beta_hat)
        return beta_hat, beta_lower, beta_upper, beta_list, fitted_values

# ======= MH Algorithm for Poisson Regression =======

def log_posterior(beta, X, y, tau_sq):
    eta = X @ beta
    log_likelihood = np.sum(scipy.stats.poisson.logpmf(y, mu = np.exp(eta)))
    log_prior = np.sum(scipy.stats.norm.logpdf(beta, loc = 0, scale = np.sqrt(tau_sq)))
    return log_likelihood + log_prior


class PoissonMH():
    def __init__(self, x, y, beta_init = None, tau_sq = 10, adapt_freq = 25, burn = 5000, nmc = 5000):
        self.x = np.hstack((np.ones((x.shape[0], 1)), x))
        self.y = y
        if beta_init is None: self.beta_init = np.zeros(self.x.shape[1])
        if beta_init is not None: self.beta_init = beta_init
        self.tau_sq = tau_sq
        self.adapt_freq = adapt_freq
        self.burn = burn
        self.nmc = nmc
    
    def train(self, silence = False):
        beta = self.beta_init
        n = self.x.shape[0]
        p = self.x.shape[1]
        U = np.random.uniform(0, 1, size = self.burn + self.nmc)
        beta_list = np.zeros((self.nmc, p))
        proposal_var = 1
        num_accept = 0


        for i in range(self.burn + self.nmc):
            Sigma = proposal_var * np.eye(p)
            beta_proposal = np.random.multivariate_normal(mean = beta, cov = Sigma)
            v = np.exp(log_posterior(beta_proposal, self.x, self.y, self.tau_sq) - log_posterior(beta, self.x, self.y, self.tau_sq))

            if U[i] <= v:
                beta = beta_proposal
                num_accept += 1
            
            # acceptance_rate = num_accept / self.adapt_freq
            acceptance_rate = num_accept / (i + 1)

            if i % self.adapt_freq == 0:
                if acceptance_rate < 0.2: proposal_var = proposal_var * 0.9
                if acceptance_rate > 0.4: proposal_var = proposal_var * 1.1

            if i >= self.burn: beta_list[i - self.burn] = beta
            if not silence: print('Iteration: ', i, ' Acceptance rate: ', acceptance_rate, ' Proposal variance: ', proposal_var)
        
        beta_hat = np.mean(beta_list, axis = 0)
        beta_lower = np.percentile(beta_list, 2.5, axis = 0)
        beta_upper = np.percentile(beta_list, 97.5, axis = 0)
        return beta_hat, beta_lower, beta_upper, beta_list


# ======= Adaptive MH for Bayesian Spatial Regression =======

def get_H(coordinates, phi2):
    dist_mat = scipy.spatial.distance_matrix(coordinates, coordinates)
    H = np.exp(- dist_mat / (2 * phi2))
    return H

class SpatialRegression():
    def __init__(self, X, y, coordinates, inits, hyperparameters, burn = 5000, nmc = 5000, adapt_freq = 25):
        self.X = np.hstack((np.ones((X.shape[0], 1)), X))
        self.y = y
        self.coordinates = coordinates
        self.inits = inits
        self.hyperparameters = hyperparameters
        self.burn = burn
        self.nmc = nmc
        self.adapt_freq = adapt_freq

    def train(self, silence = True):
        beta, W, sigma2, tau2, phi2 = self.inits
        rho = np.log(phi2)
        xi2, a_sigma, b_sigma, a_tau, b_tau, a_phi, b_phi = self.hyperparameters
        n = self.X.shape[0]
        p = self.X.shape[1]
        num_accepted = 0
        accept_rate = 0

        beta_list = np.zeros((self.nmc, p))
        W_list = np.zeros((self.nmc, n))
        sigma2_list = np.zeros(self.nmc)
        tau2_list = np.zeros(self.nmc)
        phi2_list = np.zeros(self.nmc)


        U = np.random.uniform(0, 1, self.burn + self.nmc)
        XtX = self.X.T @ self.X

        for i in range(self.burn + self.nmc):
            H = get_H(self.coordinates, np.exp(rho))
            H_inv = np.linalg.inv(H)
            Omega_beta = np.linalg.inv(XtX / sigma2 + np.eye(p) / xi2)
            Omega = np.linalg.inv(np.eye(n) / sigma2 + H_inv / tau2)

            beta_new = np.random.multivariate_normal(1/sigma2 * Omega_beta @ self.X.T @ (self.y - W), Omega_beta)
            W_new = np.random.multivariate_normal(1/sigma2 * Omega @ (self.y - self.X @ beta), Omega)

            sigma2_new = 1 / np.random.gamma(a_sigma + n / 2, 1 / (b_sigma + 0.5 * (self.y - self.X @ beta_new - W_new).T @ (self.y - self.X @ beta_new - W_new)))
            tau2_new = 1 / np.random.gamma(a_tau + n / 2, 1 / (b_tau + 0.5 * W_new.T @ H_inv @ W_new))

            # Workaround for bad Sigma2 initialization
            headstart = 25
            if headstart >= 0:
                rho_proposal = np.random.normal(rho, 1)
                headstart -= 1
            else:
                rho_proposal = np.random.normal(rho, sigma2_new)


            # Hardcoded PDFs are much faster because scipy.stats initializes a new object each time
            H_proposal = get_H(self.coordinates, np.exp(rho_proposal))
            log_pdf_mv_normal_proposal = -0.5 * n * np.log(2 * np.pi) - 0.5 * np.linalg.slogdet(tau2_new * H_proposal)[1] - 0.5 * (W_new @ np.linalg.inv(tau2_new * H_proposal) @ W_new)
            log_pdf_mv_normal_current = -0.5 * n * np.log(2 * np.pi) - 0.5 * np.linalg.slogdet(tau2_new * H)[1] - 0.5 * (W_new @ np.linalg.inv(tau2_new * H) @ W_new)

            log_pdf_inv_gamma_proposal = a_phi * np.log(b_phi) - scipy.special.gammaln(a_phi) - (a_phi + 1) * np.log(np.exp(rho_proposal)) - b_phi / np.exp(rho_proposal)
            log_pdf_inv_gamma_current = a_phi * np.log(b_phi) - scipy.special.gammaln(a_phi) - (a_phi + 1) * np.log(np.exp(rho)) - b_phi / np.exp(rho)

            top = log_pdf_mv_normal_proposal + log_pdf_inv_gamma_proposal + rho_proposal
            bottom = log_pdf_mv_normal_current + log_pdf_inv_gamma_current + rho

            # At least 1+ hours to run
            # top = multivariate_normal.logpdf(W_new, mean = np.zeros(n), cov = tau2_new * H_proposal) + invgamma.logpdf(np.exp(rho_proposal), a_phi, scale = b_phi) + rho_proposal
            # bottom = multivariate_normal.logpdf(W_new, mean = np.zeros(n), cov = tau2_new * H) + invgamma.logpdf(np.exp(rho), a_phi, scale = b_phi) + rho

            v = top - bottom
            alpha = min(1, np.exp(v))
            if U[i] < alpha:
                rho = rho_proposal
                num_accepted += 1
            
            accept_rate = num_accepted / (i + 1)
            if i % self.adapt_freq == 0 and i > 0:
                if accept_rate < 0.2: sigma2_new = sigma2_new * 0.1
                if accept_rate > 0.4: sigma2_new = sigma2_new * 1.2
            
            
            if i >= self.burn:
                beta_list[i - self.burn] = beta
                W_list[i - self.burn] = W
                sigma2_list[i - self.burn] = sigma2
                tau2_list[i - self.burn] = tau2
                phi2_list[i - self.burn] = phi2


            phi2 = np.exp(rho)
            beta = beta_new
            W = W_new
            sigma2 = sigma2_new
            tau2 = tau2_new
            if not silence: print('Iteration:', i, 'Accept Rate:', round(accept_rate, 4), 'Alpha:', round(alpha, 4), 'Sigma2:', round(sigma2_new, 4))
            
        return beta_list, W_list, sigma2_list, tau2_list, phi2_list