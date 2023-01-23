import numpy as np
from numba import njit


@njit(error_model="numpy", fastmath=True)
def find_first_greather_than_zero(vec, reversed):
    if reversed:
        vec = np.flip(vec)
    for i, elem in enumerate(vec):
        if elem > 0:
            if reversed:
                return len(vec) - i - 1
            return i
    return -1


@njit(error_model="numpy", fastmath=True)  #  'float64(float64, float64)',
def brent_root_finder(fun, xa, xb, xtol, rtol, max_iter, args):
    xpre, xcur = xa, xb
    xblk = 0.0
    fblk = 0.0
    spre = 0.0
    scur = 0.0

    # /* the tolerance is 2*delta */

    fpre = fun(xpre, *args)
    fcur = fun(xcur, *args)

    if fpre * fcur > 0:
        raise ValueError("The endpoints should have different signs.")

    if fpre == 0:
        return xpre
    if fcur == 0:
        return xcur

    for i in range(max_iter):
        if fpre != 0 and fcur != 0 and (np.sign(fpre) != np.sign(fcur)):
            xblk = xpre
            fblk = fpre
            spre = scur = xcur - xpre

        if np.abs(fblk) < np.abs(fcur):
            xpre = xcur
            xcur = xblk
            xblk = xpre

            fpre = fcur
            fcur = fblk
            fblk = fpre

        delta = (xtol + rtol * np.abs(xcur)) / 2
        sbis = (xblk - xcur) / 2

        if fcur == 0 or np.abs(sbis) < delta:
            return xcur

        if np.abs(spre) > delta and np.abs(fcur) < np.abs(fpre):
            if xpre == xblk:
                # /* interpolate */
                stry = -fcur * (xcur - xpre) / (fcur - fpre)
            else:
                # /* extrapolate */
                dpre = (fpre - fcur) / (xpre - xcur)
                dblk = (fblk - fcur) / (xblk - xcur)
                stry = -fcur * (fblk * dblk - fpre * dpre) / (dblk * dpre * (fblk - fpre))

            if 2 * np.abs(stry) < np.minimum(np.abs(spre), 3 * np.abs(sbis) - delta):
                # /* good short step */
                spre = scur
                scur = stry
            else:
                # /* bisect */
                spre = sbis
                scur = sbis
        else:
            # /* bisect */
            spre = sbis
            scur = sbis

        xpre = xcur
        fpre = fcur
        if np.abs(scur) > delta:
            xcur += scur
        else:
            xcur += delta if sbis > 0 else -delta

        fcur = fun(xcur, *args)

    return xcur
