"""
Computational Cancer Analysis Library

Authors:
    Huwate (Kwat) Yeerna (Medetgul-Ernar)
        kwat.medetgul.ernar@gmail.com
        Computational Cancer Analysis Laboratory, UCSD Cancer Center

    Pablo Tamayo
        ptamayo@ucsd.edu
        Computational Cancer Analysis Laboratory, UCSD Cancer Center
Modified by:
    Edwin F. Juárez
    ejuarez@ucsd.edu
    Mesirov Lab -- UCSD Medicine Department.
"""

# import rpy2.robjects as ro
from numpy import asarray, exp, finfo, isnan, log, sign, sqrt, sum, sort
from numpy.random import random_sample, seed
# from rpy2.robjects.numpy2ri import numpy2ri
# from rpy2.robjects.packages import importr
from scipy.stats import pearsonr
from scipy.stats import gaussian_kde
import numpy as np

from .. import RANDOM_SEED
from ..support.d2 import drop_nan_columns

EPS = finfo(float).eps

## Commented-out by EFJ on 2017-07-20 to remove the need to use R.
# ro.conversion.py2ri = numpy2ri
# mass = importr('MASS')
# bcv = mass.bcv
# kde2d = mass.kde2d

## Commented-out by EFJ on 2017-07-20 to remove the need to use R.
# def information_coefficient(x, y, n_grids=25,
#                             jitter=1E-10, random_seed=RANDOM_SEED):
#     """
#     Compute the information coefficient between x and y, which are
#         continuous, categorical, or binary vectors.
#     :param x: numpy array;
#     :param y: numpy array;
#     :param n_grids: int; number of grids for computing bandwidths
#     :param jitter: number;
#     :param random_seed: int or array-like;
#     :return: float; Information coefficient
#     """
#
#     # Can't work with missing any value
#     # not_nan_filter = ~isnan(x)
#     # not_nan_filter &= ~isnan(y)
#     # x = x[not_nan_filter]
#     # y = y[not_nan_filter]
#
#     # Assume that we are not working with NaNs
#     x, y = drop_nan_columns([x, y])  # Commented out by EFJ on 2017-07-12
#
#     # x = drop_nan_columns(x)  # Added by EFJ on 2017-07-12
#     # y = drop_nan_columns(y)  # Added by EFJ on 2017-07-12
#
#     # Need at least 3 values to compute bandwidth
#     if len(x) < 3 or len(y) < 3:
#         return 0
#
#     x = asarray(x, dtype=float)
#     y = asarray(y, dtype=float)
#
#     # Add jitter
#     seed(random_seed)
#     x += random_sample(x.size) * jitter
#     y += random_sample(y.size) * jitter
#
#     # Compute bandwidths
#     cor, p = pearsonr(x, y)
#     bandwidth_x = asarray(bcv(x)[0]) * (1 + (-0.75) * abs(cor))
#     bandwidth_y = asarray(bcv(y)[0]) * (1 + (-0.75) * abs(cor))
#
#     # Compute P(x, y), P(x), P(y)
#     fxy = asarray(
#         kde2d(x, y, asarray([bandwidth_x, bandwidth_y]), n=asarray([n_grids]))[
#             2]) + EPS
#
#     dx = (x.max() - x.min()) / (n_grids - 1)
#     dy = (y.max() - y.min()) / (n_grids - 1)
#     pxy = fxy / (fxy.sum() * dx * dy)
#     px = pxy.sum(axis=1) * dy
#     py = pxy.sum(axis=0) * dx
#
#     # Compute mutual information;
#     mi = (pxy * log(pxy / (asarray([px] * n_grids).T *
#                            asarray([py] * n_grids)))).sum() * dx * dy
#
#     # # Get H(x, y), H(x), and H(y)
#     # hxy = - (pxy * log(pxy)).sum() * dx * dy
#     # hx = -(px * log(px)).sum() * dx
#     # hy = -(py * log(py)).sum() * dy
#     # mi = hx + hy - hxy
#
#     # Compute information coefficient
#     ic = sign(cor) * sqrt(1 - exp(-2 * mi))
#
#     # TODO: debug when MI < 0 and |MI|  ~ 0 resulting in IC = nan
#     if isnan(ic):
#         ic = 0
#
#     return ic

def information_coefficient(x, y, n_grids=25,
                            jitter=1E-10, random_seed=RANDOM_SEED):
    """
    Compute the information coefficient between x and y, which are
        continuous, categorical, or binary vectors. This function uses only python libraries -- No R is needed.
    :param x: numpy array;
    :param y: numpy array;
    :param n_grids: int; number of grids for computing bandwidths
    :param jitter: number;
    :param random_seed: int or array-like;
    :return: float; Information coefficient
    """

    # Can't work with missing any value
    # not_nan_filter = ~isnan(x)
    # not_nan_filter &= ~isnan(y)
    # x = x[not_nan_filter]
    # y = y[not_nan_filter]

    x, y = drop_nan_columns([x, y])

    # Need at least 3 values to compute bandwidth
    if len(x) < 3 or len(y) < 3:
        return 0

    x = asarray(x, dtype=float)
    y = asarray(y, dtype=float)

    # Add jitter
    seed(random_seed)
    x += random_sample(x.size) * jitter
    y += random_sample(y.size) * jitter

    # Compute bandwidths
    cor, p = pearsonr(x, y)

    # bandwidth_x = asarray(bcv(x)[0]) * (1 + (-0.75) * abs(cor))
    # bandwidth_y = asarray(bcv(y)[0]) * (1 + (-0.75) * abs(cor))

    # Compute P(x, y), P(x), P(y)
    # fxy = asarray(
    #     kde2d(x, y, asarray([bandwidth_x, bandwidth_y]), n=asarray([n_grids]))[
    #         2]) + EPS

    # Estimate fxy using scipy.stats.gaussian_kde
    xmin = x.min()
    xmax = x.max()
    ymin = y.min()
    ymax = y.max()
    X, Y = np.mgrid[xmin:xmax:complex(0, n_grids), ymin:ymax:complex(0, n_grids)]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([x, y])
    kernel = gaussian_kde(values)
    fxy = np.reshape(kernel(positions).T, X.shape) + EPS

    dx = (x.max() - x.min()) / (n_grids - 1)
    dy = (y.max() - y.min()) / (n_grids - 1)
    pxy = fxy / (fxy.sum() * dx * dy)
    px = pxy.sum(axis=1) * dy
    py = pxy.sum(axis=0) * dx

    # Compute mutual information;
    mi = (pxy * log(pxy / (asarray([px] * n_grids).T *
                           asarray([py] * n_grids)))).sum() * dx * dy

    # # Get H(x, y), H(x), and H(y)
    # hxy = - (pxy * log(pxy)).sum() * dx * dy
    # hx = -(px * log(px)).sum() * dx
    # hy = -(py * log(py)).sum() * dy
    # mi = hx + hy - hxy

    # Compute information coefficient
    ic = sign(cor) * sqrt(1 - exp(-2 * mi))

    # TODO: debug when MI < 0 and |MI|  ~ 0 resulting in IC = nan
    if isnan(ic):
        ic = 0

    return ic


def compute_entropy(a):
    """
    Compute entropy of a.
    :param a: array; (1, n_values)
    :return float; 0 <= float
    """

    p = a / a.sum()
    return -(p * log(p)).sum()


def compute_brier_entropy(a, n=1):
    """
    Compute brier entropy of a.
    :param a: array; (1, n_values)
    :param n: int;
    :return float; 0 <= float
    """

    p = a / a.sum()
    p = sort(p)
    p = p[::-1]

    brier_error = 0
    for i in range(n):
        brier_error += (1 - p[i]) ** 2 + sum(
            [p[not_i] ** 2 for not_i in range(len(p)) if not_i != i])
    return brier_error


def normalize_information_coefficients(a, method, clip_min=None, clip_max=None):
    """

    :param a: array; (n_rows, n_columns)
    :param method:
    :param clip_min:
    :param clip_max:
    :return array; (n_rows, n_columns); 0 <= array <= 1
    """

    if method == '0-1':
        return (a - a.min()) / (a.max() - a.min())

    elif method == 'p1d2':
        return (a + 1) / 2

    elif method == 'clip':
        return a.clip(clip_min, clip_max)

    else:
        raise ValueError('Unknown method {}.'.format(method))
