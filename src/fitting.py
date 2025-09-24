import numpy as np
import math
# Import the model functions from the models.py file in the same directory
from .models import linear, logistic_fun, ricker_model, exponential, gen_vb, hill

# --- AIC Functions ---

def AIC_linear(q, tdata, data_pop):
    N = len(data_pop)
    penalty = 2 * (2 + 1)  # 2 parameters (m0, k)
    model_pop = linear(tdata, q)
    error = np.sum((data_pop - model_pop)**2)
    if error == 0: return -np.inf # Avoid log(0)
    AIC_OLS = N * np.log((error / N)) + penalty
    AIC_OLS_c = AIC_OLS + ((penalty * 4) / (N - 2))
    return AIC_OLS_c

def AIC_logistic(q, tdata, data_pop):
    N = len(data_pop)
    penalty = 2 * (3 + 1)  # 3 parameters (P0, r, K)
    model_pop = logistic_fun(tdata, q)
    error = np.sum((data_pop - model_pop)**2)
    if error == 0: return -np.inf
    AIC_OLS = N * np.log((error / N)) + penalty
    AIC_OLS_c = AIC_OLS + ((penalty * 5) / (N - 3))
    return AIC_OLS_c

def AIC_ricker(q, tdata, data_pop):
    N = len(data_pop)
    penalty = 2 * (3 + 1)  # 3 parameters (W0, kg, m)
    model_pop = ricker_model(tdata, q)
    error = np.sum((data_pop - model_pop)**2)
    if error == 0: return -np.inf
    AIC_OLS = N * np.log((error / N)) + penalty
    AIC_OLS_c = AIC_OLS + ((penalty * 5) / (N - 3))
    return AIC_OLS_c

def AIC_exp(q, tdata, data_pop):
    N = len(data_pop)
    penalty = 2 * (2 + 1)  # 2 parameters (m0, k)
    model_pop = exponential(tdata, q)
    error = np.sum((data_pop - model_pop)**2)
    if error == 0: return -np.inf
    AIC_OLS = N * np.log((error / N)) + penalty
    AIC_OLS_c = AIC_OLS + ((penalty * 4) / (N - 2))
    return AIC_OLS_c

def AIC_gen_vb(q, tdata, data_pop):
    N = len(data_pop)
    penalty = 2 * (4 + 1)  # 4 parameters (m0, k, f, A)
    model_pop = gen_vb(tdata, q)
    error = np.sum((data_pop - model_pop)**2)
    if error == 0: return -np.inf
    AIC_OLS = N * np.log((error / N)) + penalty
    AIC_OLS_c = AIC_OLS + ((penalty * 6) / (N - 4))
    return AIC_OLS_c

def AIC_hill(q, tdata, data_pop):
    N = len(data_pop)
    penalty = 2 * (4 + 1)  # 4 parameters (a, b, c, d)
    model_pop = hill(tdata, q)
    error = np.sum((data_pop - model_pop)**2)
    if error == 0: return -np.inf
    AIC_OLS = N * np.log((error / N)) + penalty
    AIC_OLS_c = AIC_OLS + ((penalty * 6) / (N - 4))
    return AIC_OLS_c

# --- Cost Functions (Mean Squared Error) ---

def cost_linear(q, tdata, data_pop):
    N = len(data_pop)
    model_pop = linear(tdata, q)
    error = np.sum((data_pop - model_pop)**2)
    cost = error / N
    return cost

def cost_logistic(q, tdata, data_pop):
    N = len(data_pop)
    model_pop = logistic_fun(tdata, q)
    error = np.sum((data_pop - model_pop)**2)
    cost = error / N
    return cost

def cost_ricker(q, tdata, data_pop):
    N = len(data_pop)
    model_pop = ricker_model(tdata, q)
    error = np.sum((data_pop - model_pop)**2)
    cost = error / N
    return cost

def cost_exponential(q, tdata, data_pop):
    N = len(data_pop)
    model_pop = exponential(tdata, q)
    error = np.sum((data_pop - model_pop)**2)
    cost = error / N
    return cost

def cost_gvb(q, tdata, data_pop):
    N = len(data_pop)
    model_pop = gen_vb(tdata, q)
    error = np.sum((data_pop - model_pop)**2)
    cost = error / N
    return cost

def cost_hill(q, tdata, data_pop):
    N = len(data_pop)
    model_pop = hill(tdata, q)
    error = np.sum((data_pop - model_pop)**2)
    cost = error / N
    return cost