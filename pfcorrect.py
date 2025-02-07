from cmath import rect as __cmath_rect
from typing import Any
import numpy as np
import numpy.typing as npt
from math import acos

def round_step(a: npt.NDArray[np.float64] | float, step: float) -> npt.NDArray[np.float64] | float:
    '''
    Round to nearest value in discrete increments

    Parameters
    ----------
    a : ndarray or float
        the array to round
    step : float
        the step to round to

    Returns
    -------
    b : ndarray or float
        a, rounded to step

    '''
    return np.round(a / step) * step


def correct(s_load: npt.ArrayLike) -> npt.NDArray[np.float64] | float:
    '''
    Compute power factor correction, Q_c for a minumum power factor of 0.85 and maximum of 1.0
    
    Parameters
    ----------
    s_load : complex array_like
        the real power, P + jQ

    Returns
    -------
    Q_c : ndarray or scalar
            the corrective Q value
    '''
    pf_min = 0.85
    pf_max = 1.0

    def v_func(s: complex) -> float:
        q_new = (s.real / pf_min) * np.sin(np.acos(pf_min))
        q_c = q_new - s.imag
        q_c = round_step(q_c, 0.5)
        q_new = q_c + s.imag

        # check if rounding created PF < 0.85 or > 1.0
        s_new = np.sqrt(s.real**2 + q_new**2)
        pf_new = s.real / s_new

        if pf_new < pf_min:
            return q_c - 0.5
        elif pf_new > pf_max:
            # no compensation was req'd
            return s.imag
        return q_c

    return np.vectorize(v_func)(s_load)

if __name__ == "__main__":
    print(correct(1.5+6j))
