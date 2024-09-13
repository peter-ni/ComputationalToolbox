import jax
from jax import numpy as jnp, random

import numpy as np
from numpy import random as nprandom

import pandas as pd
import time
import math


# ======= Homogenous Poisson Process =======

class HPP:
    def __init__(self, lam, T, times_first, num_realizations, key, no_save = False, silence = True):
        if lam <= 0:
            raise ValueError("Lambda must be positive.")
        if (jnp.array(T) <= 0).any():
            raise ValueError("All values of T must be positive.")
        
        self.lam = lam
        self.T = T
        self.times_first = times_first
        self.num_realizations = num_realizations
        self.no_save = no_save
        self.key = key
        self.silence = silence

    def simulate(self):
        results = []
        timings = []

        for t in np.atleast_1d(self.T):
            start_time = time.time()
            realizations = []
            for _ in range(self.num_realizations):
                if not self.silence:
                    print(f'T = {t}, Realization = {_ + 1}')
                
                if self.times_first:
                    realization = self.hpp_sub_times_first(self.lam, t, self.key)
                else:
                    realization = self.hpp_sub_num_first(self.lam, t, self.key)
                
                if not self.no_save:
                    realizations.append(realization)
                self.key, _ = random.split(self.key)
            
            end_time = time.time()
            timings.append(end_time - start_time)

            if not self.no_save:
                results.append(realizations)
        return results, timings

    def hpp_sub_times_first(self, lam, T, key):
        key, subkey = random.split(key)
        prealloc_size = math.ceil(lam * T * 3)
        u = random.uniform(subkey, shape = (prealloc_size, ))
        interarrivals = -jnp.log(u) / lam
        arrival_times = np.cumsum(np.array(interarrivals))
        cutoff = np.searchsorted(arrival_times, T, side = 'left')
        return jnp.array(arrival_times[:cutoff])
    

    def hpp_sub_num_first(self, lam, T, key):
        key, subkey = random.split(key)
        num_arrivals = nprandom.poisson(lam * T)
        return random.uniform(subkey, minval = 0, maxval = T, shape = (num_arrivals,)).sort()

# ======= Non-homogenous Poisson Process =======

class NHPP:
    def __init__(self, lamfn, T, num_realizations, key):
        self.lamfn = lamfn
        self.T = T
        self.num_realizations = num_realizations
        self.key = key

    def simulate(self):
        realizations = []
        for _ in range(self.num_realizations):
            realization = self.nhpp_sub()
            realizations.append(realization)
        return(realizations)

    def nhpp_sub(self):
        self.key, subkey = random.split(self.key)
        max_lam = self.max_finder()
        hpp_points = HPP(lam = max_lam, T = self.T, times_first = True, num_realizations = 1, key = subkey, silence = True).simulate()[0][0][0]
        unifs = random.uniform(shape = hpp_points.shape, key = subkey)
        mask = unifs <= self.lamfn(hpp_points) / max_lam
        nhpp_points = hpp_points[mask]
        return(nhpp_points)
    
    def max_finder(self):
        grid = jnp.arange(0, self.T, 1e-3)
        grid_intensities = self.lamfn(grid)
        return jnp.max(grid_intensities)


# ======= Intensity Function Estimation (piecewise/linear) =======

class intensity_estimator:
    def __init__(self, arrival_times, T, int_size = 100, init_params = jnp.array([0.0, 0.0]), lr = 1e-3, max_iter = 1e4, tol = 1e-3, loss = None, silence = False):
        self.arrival_times = arrival_times
        self.T = T
        self.int_size = int(int_size)
        self.loss = loss
        self.init_params = init_params
        self.lr = lr
        self.max_iter = int(max_iter)
        self.tol = tol
        self.silence = silence
  
    def estimate(self):
        if self.loss is None:
            intensities = []

            for start in range(0, int(self.T), int(self.int_size)):
                end = start + self.int_size
                count = jnp.sum((self.arrival_times >= start) & (self.arrival_times < end))
                intensity = count / (end - start)
                intensities.append(intensity)

            return intensities

        else:            
            hidden_params = self.init_params
            loss_grad = jax.jit(jax.grad(self.loss, argnums = 2))
            prev_loss = self.loss(self.arrival_times, self.T, hidden_params)

            for i in range(self.max_iter):
                hidden_grads = loss_grad(self.arrival_times, self.T, hidden_params)
                hidden_params -= self.lr * hidden_grads
                current_loss = self.loss(self.arrival_times, self.T, hidden_params)

                if jnp.abs(current_loss - prev_loss) < self.tol:
                    break

                prev_loss = current_loss
                if not self.silence:
                    print(f'\rIteration: {i}, Loss: {current_loss:.3f}, Hidden Params: {[f"{param:.3f}" for param in hidden_params]}', end = '', flush = True)
            final_betas = jax.nn.softplus(hidden_params)
            if not self.silence:
                print('\nFinal Betas:', [f"{beta:.3f}" for beta in final_betas])
            return(final_betas)

@jax.jit
def OLS_loss(arrival_times, T, params):
    betas = jax.nn.softplus(params)
    times_mat = jnp.column_stack((jnp.ones(arrival_times.shape), arrival_times))
    loss = jnp.sum(jnp.log(jnp.dot(times_mat, betas))) + jnp.dot(betas, jnp.array([-T, -T ** 2 / 2]))
    return(-loss)


# ======= Ripley's K Function with Monte Carlo Estimate ======= 

class RipleysK:
    def __init__(self, points, rect, n_grid, key):
        self.points = points
        self.rect = rect
        self.n_grid = n_grid
        self.key = key

    def pairwise_distance(self):
        dist_mat = jnp.sqrt(jnp.sum((self.points[: , None , :] - self.points[None, : , :]) ** 2, axis = -1))
        return dist_mat # --> 149 x 149

    
    def calculate(self):
        dist_mat = self.pairwise_distance()
        r_grid = jnp.linspace(1e-1, jnp.max(dist_mat), self.n_grid)
        mle = (self.rect[2] - self.rect[0]) * (self.rect[3] - self.rect[1]) / len(self.points) ** 2
        ripks = []
        progress_ct = 0

        for radius in r_grid:
            self.key, subkey = random.split(self.key)
            mask_mat = jnp.fill_diagonal(dist_mat <= radius, inplace = False, val = False)
            center_inds, edge_inds = np.nonzero(np.array(mask_mat))
            weights = getWeight(centers = self.points[center_inds], radii = dist_mat[center_inds, edge_inds], rect = self.rect, key = subkey)
            ripks.append(mle * jnp.sum(1 / weights))
            progress_ct += 1
            print(f'Progress: {progress_ct}/{len(r_grid)}', end = '\r')

        return r_grid, jnp.stack(ripks, axis = 0)


@jax.jit
def getWeight(centers, radii, rect, key, num_points = int(1e3)):
    r_random = jnp.sqrt(random.uniform(key, shape = (len(centers), num_points)))
    theta_random = 2 * jnp.pi * random.uniform(key, shape = (len(centers), num_points))

    x = centers[:, 0, None] + radii[:, None] * r_random * jnp.cos(theta_random)
    y = centers[:, 1, None] + radii[:, None] * r_random * jnp.sin(theta_random)

    x_min, y_min, x_max, y_max = rect
    in_rect = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)
    return jnp.mean(in_rect, axis = 1)
