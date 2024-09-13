import jax
from jax import numpy as jnp, random, vmap

import numpy as np
from numpy import random as nprandom

import pandas as pd
import time
import math


class randomWalkPaths():
    def __init__(self, p, u, d, n, S0):
        self.p = p
        self.u = u
        self.d = d
        self.n = n
        self.S0 = S0
    

    def gen_layer(self, layer_n):
        price = vmap(lambda i: self.S0 * (self.u ** (layer_n - i)) * (self.d ** i))
        density = vmap(lambda i: self.p ** (layer_n - i) * (1 - self.p) ** i)
        adj = jnp.array([math.comb(layer_n, k) for k in range(layer_n + 1)])

        layer_price = price(jnp.arange(layer_n + 1))
        layer_prob = jnp.multiply(density(jnp.arange(layer_n + 1)), adj)
        return layer_price, layer_prob
    
    def gen_tree(self):
        max_pad = self.n + 1
        probs = -1 * jnp.ones((max_pad, max_pad))
        prices = -1 * jnp.ones((max_pad, max_pad))
        for layer in range(1, self.n + 1):
            layer_price, layer_prob = self.gen_layer(layer)
            layer_price = jnp.pad(layer_price, (0, max_pad - layer), constant_values = -1)
            layer_prob = jnp.pad(layer_prob, (0, max_pad - layer), constant_values = -1)
            prices = prices.at[layer, :layer + 1].set(layer_price[:layer + 1])
            probs = probs.at[layer, :layer + 1].set(layer_prob[:layer + 1])
        
        prices = prices.at[0, 0].set(self.S0)
        probs = probs.at[0, 0].set(1)
        return prices, probs
    
    
# ======= Payoff Approximation for European Options =======


class EuroBinomApprox():
    def __init__(self, type, K, r, sigma, S0, T, N):
        self.type = type
        self.K = K
        self.r = r
        self.sigma = sigma
        self.S0 = S0
        self.T = T
        self.N = N
    
    def price(self):
        dt = self.T / self.N
        beta = 0.5 * (jnp.exp(-self.r * dt) + jnp.exp((self.r + self.sigma ** 2) * dt))
        u = beta + jnp.sqrt(beta ** 2 - 1)
        d = beta - jnp.sqrt(beta ** 2 - 1)
        p = (jnp.exp(self.r * dt) - d) / (u - d)

        stock_tree = randomWalkPaths(p = p, u = u, d = d, n = self.N, S0 = self.S0)
        stock_prices, stock_probs = stock_tree.gen_tree()

        
        if self.type == 'call':
            end_payoff = jnp.maximum(stock_prices[-1] - self.K, 0)
        else:
            end_payoff = jnp.maximum(self.K - stock_prices[-1], 0)
        
        payoff_tree = -1 * jnp.ones((self.N + 1, self.N + 1))
        payoff_tree = payoff_tree.at[self.N, :].set(end_payoff)

        for layer in range(self.N - 1, -1, -1):
            discount = jnp.exp(-self.r * dt)
            layer_payoff = discount * (p * payoff_tree[layer + 1, :layer + 1] + (1 - p) * payoff_tree[layer + 1, 1:layer + 2])
            payoff_tree = payoff_tree.at[layer, :layer + 1].set(jnp.maximum(layer_payoff, 0))

        return payoff_tree    
    
    
# ======= Black-Scholes Pricing for European Options =======
    

class BlackScholes():
    def __init__ (self, type, S, t, K, T, r, sigma):
        self.type = type
        self.S = S
        self.t = t
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
    def price_options(self):
        d1 = (jnp.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * (self.T - self.t)) / (self.sigma * jnp.sqrt(self.T - self.t))
        d2 = (jnp.log(self.S / self.K) + (self.r - 0.5 * self.sigma ** 2) * (self.T - self.t)) / (self.sigma * jnp.sqrt(self.T - self.t))
        if self.type == 'call':
            return self.S * jax.scipy.stats.norm.cdf(d1) - self.K * jnp.exp(-self.r * (self.T - self.t)) * jax.scipy.stats.norm.cdf(d2)
        else:
            return self.K * jnp.exp(-self.r * (self.T - self.t)) * jax.scipy.stats.norm.cdf(-d2) - self.S * jax.scipy.stats.norm.cdf(-d1)
        
        
# ======= Greeks based on Black-Scholes Model for European Options =======

class Greeks():
    def __init__ (self, S, type, t, K, T, r, sigma):
        self.S = S
        self.type = type
        self.t = t
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
    def calculate(self):
        d1 = (jnp.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * (self.T - self.t)) / (self.sigma * jnp.sqrt(self.T - self.t))
        d2 = (jnp.log(self.S / self.K) + (self.r - 0.5 * self.sigma ** 2) * (self.T - self.t)) / (self.sigma * jnp.sqrt(self.T - self.t))

        if self.type == 'call':
            delta = jax.scipy.stats.norm.cdf(d1)
            gamma = jax.scipy.stats.norm.pdf(d1) / (self.S * self.sigma * jnp.sqrt(self.T - self.t))
            vega = self.S * jax.scipy.stats.norm.pdf(d1) * jnp.sqrt(self.T - self.t)
            theta = (-self.S * jax.scipy.stats.norm.pdf(d1) * self.sigma / (2 * jnp.sqrt(self.T - self.t)) - self.r * self.K * jnp.exp(-self.r * (self.T - self.t)) * jax.scipy.stats.norm.cdf(d2)) / 252
            rho = self.K * (self.T - self.t) * jnp.exp(-self.r * (self.T - self.t)) * jax.scipy.stats.norm.cdf(d2)
        else:
            delta = -jax.scipy.stats.norm.cdf(-d1)
            gamma = jax.scipy.stats.norm.pdf(d1) / (self.S * self.sigma * jnp.sqrt(self.T - self.t))
            vega = self.S * jax.scipy.stats.norm.pdf(d1) * jnp.sqrt(self.T - self.t)
            theta = (-self.S * jax.scipy.stats.norm.pdf(d1) * self.sigma / (2 * jnp.sqrt(self.T - self.t)) + self.r * self.K * jnp.exp(-self.r * (self.T - self.t)) * jax.scipy.stats.norm.cdf(-d2)) / 252
            rho = -self.K * (self.T - self.t) * jnp.exp(-self.r * (self.T - self.t)) * jax.scipy.stats.norm.cdf(-d2)
        
        return delta, gamma, vega, theta, rho
    
# ======= Geometric Brownian Motion using MLEs ========

class GeoBM:
    def __init__(self, t, St, t_new):
        self.t = t
        self.St = St
        self.Xt = jnp.log(St)
        self.t_new = t_new
    
    def getMLE(self):
        N = self.t.shape[0]
        delta_X = self.Xt[1:] - self.Xt[:-1]
        delta_t = self.t[1:] - self.t[:-1]
        summation = jnp.sum(delta_X ** 2 / delta_t)
        sigma = jnp.sqrt(- 1 / N * (self.Xt[-1] - self.Xt[0]) ** 2 / (self.t[-1] - self.t[0]) + 1 / N * summation)
        mu = (self.Xt[-1] - self.Xt[0]) / (self.t[-1] - self.t[0]) + 0.5 * sigma ** 2
        self.mu = mu
        self.sigma = sigma
        return mu, sigma
    
    def forecast(self):
        return self.St[0] * jnp.exp((self.mu * self.t_new))

def simulate_gbm(S0, mu, sigma, t_new, key):
    dt = jnp.diff(t_new, prepend = 0)
    Z = random.normal(key, shape=(len(t_new) - 1,))
    exponent = (mu - 0.5 * sigma ** 2) * dt[1:] + sigma * jnp.sqrt(dt[1:]) * Z
    St = S0 * jnp.exp(jnp.concatenate([jnp.array([0.0]), jnp.cumsum(exponent)]))
    return St

