from cmath import rect as __cmath_rect
import numpy as np
import numpy.typing as npt
from math import acos

def rect(s_load: npt.ArrayLike, pf: float) -> npt.NDArray[np.complex64] | complex:
    '''
    Compute the rectangular form given magnitude(s) and an angle.
    Similar to cmath.rect, but can operate on array types.
    
    Parameters
    ----------
    s_load : array_like
        the real power
    pf : float
        the target power factor
    
    Returns
    -------
    rect : ndarray or scalar
        the rectangular form, a+bj

    See Also
    --------
    cmath.rect :
        Python cmath module version of this function for scalar inputs
    '''
    def v_func(a: complex, pf: float) -> float:
        # virtual function to vectorize with numpy for applying to every element in the array
        return __cmath_rect(a.real / pf, acos(pf)).imag
    return np.vectorize(v_func)(s_load, pf)


def round_step(a: npt.NDArray[np.float64] | float, step: float) -> npt.NDArray[np.float64] | float:
    '''
    Round to nearest value in discrete increments

    Parameters
    ----------

    '''
    return np.round(a / step) * step


def correct(s_load: npt.ArrayLike, s_min: complex = 1+0j, s_max: complex = 10+10j) -> npt.NDArray[np.float64] | float:
    '''
    Compute power factor correction, Q_c
    
    Parameters
    ----------
    s_load : array_like
        the real power
    s_min : complex
        the minimum allowed s_load

    Returns
    ------
        Q_c : ndarray or scalar
            the corrected Q value
    '''
    pf_target = 0.85
    if (np.max(s_load).real > s_max.real or np.min(s_load).real < s_min.real 
        or np.max(s_load).imag > s_max.imag or np.min(s_load).imag < s_min.imag):
        print("invalid s_load")
        return np.empty_like(s_load)
    return np_rect(s_load, pf_target)
