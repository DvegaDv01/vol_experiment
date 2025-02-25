import numpy as np
from scipy.stats import norm


def gaussian_kernel(x, x0, bandwidth):
    """Gaussian kernel function for weighting

    Parameters:
    -----------
    x : float or array - Point(s) to evaluate
    x0 : float - Center point
    bandwidth : float - Kernel bandwidth

    Returns:
    --------
    float or array - Kernel weight(s)
    """
    return np.exp(-0.5 * ((x - x0) / bandwidth) ** 2)


def calculate_log_moneyness(K, F, sigma, tau):
    """Calculate different log moneyness measures

    Parameters:
    -----------
    K : float or array - Strike price(s)
    F : float - Forward price
    sigma : float - Implied volatility
    tau : float - Time to maturity in years

    Returns:
    --------
    tuple - (l_plus, l_minus)
        l_plus = ln(K/F) + sigma²τ/2
        l_minus = ln(K/F) - sigma²τ/2
    """
    log_strike_ratio = np.log(K / F)
    half_var_time = 0.5 * sigma ** 2 * tau
    l_plus = log_strike_ratio + half_var_time
    l_minus = log_strike_ratio - half_var_time
    return l_plus, l_minus


def compute_standardized_moneyness(l_plus, l_minus, sigma, tau):
    """Compute standardized moneyness measures

    Parameters:
    -----------
    l_plus, l_minus : float or array - Log moneyness measures
    sigma : float - Implied volatility
    tau : float - Time to maturity in years

    Returns:
    --------
    tuple - (z_plus, z_minus)
        z_plus = l_plus/(sigma*√τ)
        z_minus = l_minus/(sigma*√τ)
    """
    denom = sigma * np.sqrt(tau)
    z_plus = l_plus / denom
    z_minus = l_minus / denom
    return z_plus, z_minus


def compute_mean_moneyness_measures(z_plus_p, z_minus_p, z_plus_c, z_minus_c):
    """Compute geometric and arithmetic mean moneyness measures for smile trades

    Parameters:
    -----------
    z_plus_p, z_minus_p : float - Put moneyness measures
    z_plus_c, z_minus_c : float - Call moneyness measures

    Returns:
    --------
    tuple - (l_bar_gt, l_bar_agt)
    """
    # Geometric mean log moneyness (equation 56)
    l_bar_gt = np.sqrt(np.abs(z_minus_p * z_plus_p) * z_minus_c * z_plus_c)

    # Arithmetic-geometric mean (equation 68)
    l_bar_agt = np.sqrt((z_plus_c + np.abs(z_plus_p)) / 2 * (z_minus_c + np.abs(z_minus_p)) / 2)

    return l_bar_gt, l_bar_agt