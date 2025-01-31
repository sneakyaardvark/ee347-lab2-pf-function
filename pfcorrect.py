from cmath import rect as __cmath_rect
import numpy as np
import numpy.typing as npt
from math import acos

def np_rect(s_load: npt.ArrayLike, pf: float) -> npt.ArrayLike:
    '''
    
    '''
    def v_func(a: complex, pf: float) -> float:
        return __cmath_rect(a.real / pf, acos(pf)).imag
    return np.vectorize(v_func)(s_load, pf)


def correct_np(s_load: npt.ArrayLike, s_min: complex = 1+0j, s_max: complex = 10+10j) -> npt.ArrayLike:
    pf_target = 0.85
    if (np.max(s_load).real > s_max.real or np.min(s_load).real < s_min.real 
        or np.max(s_load).imag > s_max.imag or np.min(s_load).imag < s_min.imag):
        print("invalid s_load")
        return []
    return np_rect(s_load, pf_target)


def correct(s_load: complex, s_min: complex = 1+0j, s_max: complex = 10+10j):
    pf_target = 0.85
    if (s_load.real > s_max.real or s_load.real < s_min.real or s_load.imag > s_max.imag or s_load.imag < s_min.imag):
        print("invalid s_load")
        return -1

    return __cmath_rect(s_load.real / pf_target, acos(pf_target)).imag


def main():
    print(correct_np(np.array([1+0j, 2+0j, 3+0j, 4+0j, 5+0j, 6+0j, 7+0j, 50+10j])))

if __name__ == "__main__":
    main()
