import numpy as np
from scipy.stats import norm
from ..utils.math_utils import calculate_log_moneyness, compute_standardized_moneyness


class BMSModel:
    def __init__(self):
        """Black-Merton-Scholes option pricing model"""
        pass

    def option_price(self, S, K, tau, r, q, sigma, option_type='call'):
        """Calculate BMS option price

        Parameters:
        -----------
        S : float - Spot price
        K : float - Strike price
        tau : float - Time to maturity in years
        r : float - Risk-free rate
        q : float - Dividend yield
        sigma : float - Implied volatility
        option_type : str - 'call' or 'put'

        Returns:
        --------
        float - Option price
        """
        if tau <= 0:
            # Handle expired options
            if option_type == 'call':
                return max(0, S - K)
            else:
                return max(0, K - S)

        F = S * np.exp((r - q) * tau)  # Forward price
        df = np.exp(-r * tau)  # Discount factor

        l_plus, l_minus = calculate_log_moneyness(K, F, sigma, tau)
        z_plus, z_minus = compute_standardized_moneyness(l_plus, l_minus, sigma, tau)

        if option_type == 'call':
            return df * (F * norm.cdf(-z_minus) - K * norm.cdf(-z_plus))
        else:  # put
            return df * (K * norm.cdf(z_plus) - F * norm.cdf(z_minus))

    def calculate_greeks(self, S, K, tau, r, q, sigma, option_type='call'):
        """Calculate all BMS greeks

        Parameters:
        -----------
        S : float - Spot price
        K : float - Strike price
        tau : float - Time to maturity in years
        r : float - Risk-free rate
        q : float - Dividend yield
        sigma : float - Implied volatility
        option_type : str - 'call' or 'put'

        Returns:
        --------
        dict - Contains all greeks (delta, gamma, vega, theta, volga, vanna)
        """
        if tau <= 0:
            return {
                'delta': 0, 'gamma': 0, 'vega': 0, 'theta': 0,
                'volga': 0, 'vanna': 0, 'cash_gamma': 0, 'cash_vega': 0,
                'cash_vanna': 0, 'cash_volga': 0
            }

        F = S * np.exp((r - q) * tau)  # Forward price
        df = np.exp(-r * tau)  # Discount factor

        l_plus, l_minus = calculate_log_moneyness(K, F, sigma, tau)
        z_plus, z_minus = compute_standardized_moneyness(l_plus, l_minus, sigma, tau)

        # Normal density at the standardized moneyness
        n_zplus = norm.pdf(z_plus)
        n_zminus = norm.pdf(z_minus)

        # Normal CDF at the standardized moneyness
        N_zplus = norm.cdf(z_plus)
        N_zminus = norm.cdf(z_minus)

        # Calculate delta
        if option_type == 'call':
            delta = np.exp(-q * tau) * norm.cdf(-z_minus)
        else:  # put
            delta = np.exp(-q * tau) * (norm.cdf(-z_minus) - 1)

        # Calculate gamma (same for calls and puts)
        gamma = np.exp(-q * tau) * n_zminus / (S * sigma * np.sqrt(tau))

        # Calculate vega (same for calls and puts)
        vega = S * np.exp(-q * tau) * n_zminus * np.sqrt(tau)

        # Calculate theta
        if option_type == 'call':
            theta = -S * np.exp(-q * tau) * n_zminus * sigma / (2 * np.sqrt(tau)) \
                    - r * S * np.exp(-q * tau) * norm.cdf(-z_minus) \
                    + q * S * np.exp(-q * tau) * norm.cdf(-z_minus) \
                    + r * K * np.exp(-r * tau) * norm.cdf(-z_plus)
        else:  # put
            theta = -S * np.exp(-q * tau) * n_zminus * sigma / (2 * np.sqrt(tau)) \
                    + r * K * np.exp(-r * tau) * norm.cdf(z_plus) \
                    - q * S * np.exp(-q * tau) * norm.cdf(-z_minus) \
                    - r * S * np.exp(-q * tau) * norm.cdf(-z_minus)

        # Calculate vanna
        vanna = -np.exp(-q * tau) * n_zminus * z_minus / sigma

        # Calculate volga
        volga = vega * (z_plus * z_minus) / sigma

        # Calculate cash greeks
        cash_gamma = S ** 2 * gamma
        cash_vega = sigma * vega
        cash_vanna = S * sigma * vanna
        cash_volga = sigma ** 2 * volga

        return {
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'theta': theta,
            'vanna': vanna,
            'volga': volga,
            'cash_gamma': cash_gamma,
            'cash_vega': cash_vega,
            'cash_vanna': cash_vanna,
            'cash_volga': cash_volga,
            'z_plus': z_plus,
            'z_minus': z_minus,
            'l_plus': l_plus,
            'l_minus': l_minus
        }

    def find_atm_strike(self, F, tau, sigma):
        """Find strike where z_plus = 0

        Parameters:
        -----------
        F : float - Forward price
        tau : float - Time to maturity in years
        sigma : float - Implied volatility

        Returns:
        --------
        float - ATM strike
        """
        # At z_plus = 0, we have l_plus = 0
        # l_plus = ln(K/F) + sigma²τ/2 = 0
        # ln(K/F) = -sigma²τ/2
        # K = F * exp(-sigma²τ/2)
        return F * np.exp(-0.5 * sigma ** 2 * tau)

    def implied_volatility(self, price, S, K, tau, r, q, option_type='call', tol=1e-8, max_iter=100):
        """Calculate implied volatility using Newton-Raphson method

        Parameters:
        -----------
        price : float - Option market price
        S : float - Spot price
        K : float - Strike price
        tau : float - Time to maturity in years
        r : float - Risk-free rate
        q : float - Dividend yield
        option_type : str - 'call' or 'put'
        tol : float - Convergence tolerance
        max_iter : int - Maximum iterations

        Returns:
        --------
        float - Implied volatility
        """
        if tau <= 0:
            return np.nan

        # Initial guess - use Brenner-Subrahmanyam approximation
        F = S * np.exp((r - q) * tau)
        df = np.exp(-r * tau)

        if option_type == 'call':
            intrinsic = max(0, F - K) * df
        else:
            intrinsic = max(0, K - F) * df

        if np.abs(price - intrinsic) < tol:
            return 0.0

        if F / K > 1.0:
            sigma = np.sqrt(2 * np.pi / tau) * price / S
        else:
            sigma = np.sqrt(2 * np.pi / tau) * price / K

        sigma = max(0.01, min(sigma, 5.0))  # Bound initial guess

        for _ in range(max_iter):
            price_guess = self.option_price(S, K, tau, r, q, sigma, option_type)
            vega = self.calculate_greeks(S, K, tau, r, q, sigma, option_type)['vega']

            if vega == 0:
                # Avoid division by zero
                sigma = sigma * 1.5
                continue

            price_diff = price_guess - price
            if abs(price_diff) < tol:
                return sigma

            sigma_change = price_diff / vega
            sigma = sigma - sigma_change

            # Bound sigma to prevent extreme values
            sigma = max(0.001, min(sigma, 10.0))

        # If no convergence, return NaN
        return np.nan