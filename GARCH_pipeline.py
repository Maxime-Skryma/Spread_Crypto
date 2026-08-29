import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng()
from numpy.polynomial import Polynomial
from scipy.linalg import toeplitz
from statsmodels.tsa.arima_process import ArmaProcess
from scipy.optimize import minimize
from scipy.special import gammaln

from ARCH_GARCH import log_vraisemblance_GJR_GARCH_student,log_vraisemblance_GARCH

