from src.root_finding import brent_root_finder
from numba import njit
from math import erf, sqrt

def limiting_q_m_Huber(delta_small, delta_large, percentage, beta, a):
    # find m solution
    m = 0.0
    # find q solution
    q = 0.0
    return m, q


def limit_observables_huber(delta_small, delta_large, percentage, beta, a):
    m_limit, q_limit = limiting_q_m_Huber(delta_small, delta_large, percentage, beta, a)
    g = 0.0

    return m_limit, q_limit
