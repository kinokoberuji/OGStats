import numpy as np
import scipy.stats as stats

from typing import Tuple

### Jacknife
def np_jackknife(data: np.array, func=np.mean):

    k = data.size
    output = np.zeros(k)

    for i in range(k):
        reduced_arr = np.delete(data, i)
        output[i] = func(reduced_arr)

    return output

### Acceleration coefficient
def np_acc_coef(data: np.array, func=np.mean):

    jknf_vals = np_jackknife(data, func)
    mean_jknf = np.mean(jknf_vals)

    nominator, denominator = 0.0, 0.0

    for element in jknf_vals:
        nominator += (mean_jknf - element) ** 3.0
        denominator += (mean_jknf - element) ** 2.0

    return nominator / (6.0 * (denominator ** 1.5))

def Est_bca_CI(
    boot_thetas: np.array, emp_val: float, alpha: float
) -> Tuple[float, float]:

    """Function to calculate BCa confidence intervals

    Parameters
    ----------
    boot_thetas : np.array
        np.array of bootstrap statistics for intercept or slope
    emp_val : float
        Empirical value for intercept or slope
    alpha : float
        Confidence level

    Returns
    -------
    Tuple[float, float]
        Lower and upper confidence interval
    """

    lower_ratio = boot_thetas[boot_thetas <= emp_val].size / boot_thetas.size

    z0 = stats.norm.ppf(lower_ratio)
    z_alpha = stats.norm.ppf(alpha / 2)

    # Acceleration coefficient
    a = np_acc_coef(boot_thetas, np.mean)
    perc_lower = stats.norm.cdf(z0 + (z0 + z_alpha) / (1.0 - a * (z0 + z_alpha)))
    perc_upper = stats.norm.cdf(z0 + (z0 - z_alpha) / (1.0 - a * (z0 - z_alpha)))

    BCa_CI = np.quantile(boot_thetas, [perc_lower, perc_upper])

    return BCa_CI