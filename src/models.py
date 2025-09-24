import numpy as np 

#define linear model
def linear(t, q):
    # Define parameters of the model
    m0 = q[0]  # initial size (intercept)
    k = q[1]   # growth rate (slope)
    
    # Linear growth model equation
    m = m0 + k * t  
    
    return m

# define the model
def logistic_fun(t,q):
    
    # define the parameters of the model
    P0 = q[0] # initial value
    r = q[1] # rate of growth
    K = q[2] # limiting population

    P = K/(((K - P0)/P0)*np.exp(-r*t) + 1)
    
    return P

# define the model
def ricker_model(t,q):
    
    # define the parameters of the model
    W0 = q[0] # initial value (also changes upper asymptote)
    kg = q[1] # growth coefficient
    m = q[2] # affects time of inflection

    W = W0 * np.exp(m * (1 - np.exp(-kg * t)))
    
    return W

# exponential model
def exponential(t,q):

    # define parameters of the model
    m0 = q[0] # initial size
    k = q[1] # growth rate

    m = m0 * np.exp(k * t)
    
    return m

# generalized von bertalanffy growth function
def gen_vb(t,q):

    # define parameters of the model
    m0 = q[0] # initial size
    k = q[1] # growth rate (> 0)
    f = q[2] # 0 < f < 1
    A = q[3] # 0 < A < 1

    m = m0 * (((1 - f * np.exp(-k * t)) / (1 - f)) ** (-1 / (A - 1)))
    
    return m

def hill(t,q):

    a = q[0]
    b = q[1]
    c = q[2]
    d = q[3]

    y = a + ((b - a) / (1 + ((c / t) ** d)))

    return y