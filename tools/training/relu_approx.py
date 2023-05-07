import numpy as np


def relu(x):
    return np.maximum(0, x)


def coeffs1(alpha):
    coeffs = []
    if alpha == 7:
        coeffs = [
            7.30445164958251,
            -3.46825871108659 * 10,
            5.98596518298826 * 10,
            -3.18755225906466 * 10,
        ]
    elif alpha == 8:
        coeffs = [
            8.83133072022416,
            -4.64575039895512 * 10,
            8.30282234720408 * 10,
            -4.49928477828070 * 10,
        ]
    elif alpha == 9:
        coeffs = [
            1.80966285718807 * 10,
            -4.34038703274886 * 100,
            4.15497103545696 * 1000,
            -1.86846943613149 * 10000,
            4.41657177889329 * 10000,
            -5.65527928983401 * 10000,
            3.71156122725781 * 10000,
            -9.78241933892781 * 1000,
        ]
    elif alpha == 10:
        coeffs = [
            1.08541842577442 * 10,
            -6.22833925211098 * 10,
            1.14369227820443 * 100,
            -6.28023496973074 * 10,
        ]
    else:
        raise ValueError(
            f"alpha must be 7, 8 or 9. However, the given alpha is {alpha}"
        )

    coeffs = [elem for item in coeffs for elem in (0, item)]
    return coeffs


def coeffs2(alpha):
    coeffs = []
    if alpha == 7:
        coeffs = [
            2.40085652217597,
            -2.63125454261783,
            1.54912674773593,
            -3.31172956504304 / 10,
        ]
    elif alpha == 8:
        coeffs = [
            3.94881885083263,
            -1.29103010992282 * 10,
            2.80865362174658 * 10,
            -3.55969148965137 * 10,
            2.65159370881337 * 10,
            -1.14184889368449 * 10,
            2.62558443881334,
            -2.49172299998642 / 10,
        ]
    elif alpha == 9:
        coeffs = [
            3.79753323360856,
            -1.17718157771192 * 10,
            2.49771086678346 * 10,
            -3.15238841603993 * 10,
            2.37294863126722 * 10,
            -1.04331800195923 * 10,
            2.46743976260838,
            -2.42130100247617 / 10,
        ]
    elif alpha == 10:
        coeffs = [
            4.13976170985111,
            -5.84997640211679,
            2.94376255659280,
            -4.54530437460152 / 10,
        ]
    else:
        raise ValueError(
            f"alpha must be 7, 8 or 9. However, the given alpha is {alpha}"
        )
    coeffs = [elem for item in coeffs for elem in (0, item)]
    return coeffs


def coeffs3(alpha):
    coeffs = []
    if alpha == 10:
        coeffs = [
            3.29956739043733,
            -7.84227260291355,
            1.28907764115564 * 10,
            -1.24917112584486 * 10,
            6.94167991428074,
            -2.04298067399942,
            2.46407138926031 / 10,
        ]
        coeffs = [elem for item in coeffs for elem in (0, item)]
        return coeffs


def approx1(x, alpha):
    return np.polynomial.polynomial.polyval(x, coeffs1(alpha))


def approx2(x, alpha):
    return np.polynomial.polynomial.polyval(x, coeffs2(alpha))


def approx3(x, alpha):
    if alpha >= 10:
        return np.polynomial.polynomial.polyval(x, coeffs3(alpha))
    else:
        return np.polynomial.polynomial.polyval(x, [0, 1])


def step_approx(x, alpha):
    return approx3(approx2(approx1(x, alpha), alpha), alpha)


def relu_approx(x, alpha, B):
    x = x / B
    return B * (x + x * step_approx(x, alpha)) / 2
