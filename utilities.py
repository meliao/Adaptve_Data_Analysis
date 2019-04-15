import numpy as np
from scipy.stats import truncnorm

def trunc_gauss(tau, seed = None):

    if tau == 0:
        return 0
    else:
        if seed:
            np.random.seed(seed)
        return truncnorm.rvs(-1 * tau, tau)

def random_unit_vector(d, p, seed = None):
    '''
    Inputs:
        d (integer) the dimension or length of the desired vector
        p (float or string) the 'ord' option in numpy's linalg.norm function
            if a float, the L_p norm will be used. Other options are inf, -inf.
        (optional) seed (integer) random seed if desired
    '''
    if seed:
        np.random.seed(seed)
    x = np.random.normal(size = d)
    a = np.linalg.norm(x, ord = p)
    return 1 / a * x
