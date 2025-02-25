import numpy as np
import pandas as pd
from scipy import optimize
from ..utils.math_utils import calculate_log_moneyness, compute_standardized_moneyness


class MomentExtractor:
    def __init__(self, vol_surface):
        """
        Parameters:
        -----------
        vol_surface : VolatilitySurface
        """
        self.vol_surface = vol_surface

    def extract_drift_from_term_structure(self, tau1, tau2, S, r=0.0, q=0.0):
        """Extract risk-neutral drift from ATM term structure

        Parameters:
        -----------
        tau1, tau2 : float - Adjacent maturities
        S : float - Spot price
        r : float - Risk-free rate
        q : float - Dividend yield

        Returns:
        --------
        float - Common expected rate of change μ_t
        """
        # Get ATM volatilities at both maturities
        atm_term_structure = self.vol_surface.get_atm_volatility_term_structure(
            S, r, q, [tau1, tau2]
        )

        if len(atm_term_structure) < 2 or atm_term_structure['A_t'].isna().any():
            return np.nan

        A1 = atm_term_structure.iloc[0]['A_t']
        A2 = atm_term_structure.iloc[1]['A_t']

        # Calculate μ_t using equation (12)
        A1_squared = A1 ** 2
        A2_squared = A2 ** 2

        numerator = A2_squared - A1_squared
        denominator = 2 * (A2_squared * tau2 - A1_squared * tau1)

        if abs(denominator) < 1e-10:
            return np.nan

        mu_t = numerator / denominator

        return mu_t

    def extract_variance_covariance_from_smile(self, tau, S, r=0.0, q=0.0, moneyness_range=(-1, 1)):
        """Extract variance and covariance from local smile

        Parameters:
        -----------
        tau : float - Maturity
        S : float - Spot price
        r : float - Risk-free rate
        q : float - Dividend yield
        moneyness_range : tuple - Range of moneyness for local regression

        Returns:
        --------
        tuple - (γ_t, ω²_t, R², residuals)
        """
        # Get floating series for this maturity
        floating_series = self.vol_surface.calculate_floating_series(
            S, r, q,
            moneyness_levels=np.linspace(moneyness_range[0], moneyness_range[1], 9),
            tau_levels=[tau]
        )

        if len(floating_series) == 0:
            return np.nan, np.nan, np.nan, None

        # Get ATM volatility
        atm_data = floating_series[floating_series['x'] == 0]
        if len(atm_data) == 0:
            return np.nan, np.nan, np.nan, None

        A_t = atm_data.iloc[0]['impl_vol']
        A_t_squared = A_t ** 2

        # Calculate implied variance spreads (I²_t - A²_t)
        floating_series['impl_var'] = floating_series['impl_vol'] ** 2
        floating_series['var_spread'] = floating_series['impl_var'] - A_t_squared

        # Calculate regression variables: 2*z_+ and z_+*z_-
        # Filter for in-range moneyness
        regression_data = floating_series[
            (floating_series['x'] >= moneyness_range[0]) &
            (floating_series['x'] <= moneyness_range[1])
            ]

        if len(regression_data) < 3:
            return np.nan, np.nan, np.nan, None

        # Get moneyness measures for regression
        F = S * np.exp((r - q) * tau)

        z_plus_values = regression_data['z_plus'].values

        # Calculate z_minus using the relationship
        impl_vol_values = regression_data['impl_vol'].values
        log_strike_ratio = np.log(regression_data['K'].values / F)

        l_plus_values = log_strike_ratio + 0.5 * impl_vol_values ** 2 * tau
        l_minus_values = log_strike_ratio - 0.5 * impl_vol_values ** 2 * tau

        z_plus_values = l_plus_values / (impl_vol_values * np.sqrt(tau))
        z_minus_values = l_minus_values / (impl_vol_values * np.sqrt(tau))

        # Prepare variables for regression
        X = np.column_stack([
            2 * z_plus_values,  # For γ_t
            z_plus_values * z_minus_values  # For ω²_t
        ])
        y = regression_data['var_spread'].values

        # Define objective function for constrained regression
        def objective_func(params):
            gamma, omega_sq = params
            # Ensure non-negative variance
            omega_sq = max(0, omega_sq)

            y_pred = 2 * gamma * z_plus_values + omega_sq * z_plus_values * z_minus_values
            return np.sum((y - y_pred) ** 2)

        # Perform constrained optimization
        initial_guess = [0.0, 0.01]  # Initial gamma and omega_sq
        bounds = (None, (0, None))  # No constraint on gamma, non-negative omega_sq

        result = optimize.minimize(
            objective_func,
            initial_guess,
            method='L-BFGS-B',
            bounds=[None, (0, None)]  # No constraint on gamma, non-negative omega_sq
        )

        gamma_t, omega_sq_t = result.x
        omega_sq_t = max(0, omega_sq_t)  # Ensure non-negative variance

        # Calculate R² and residuals
        y_pred = 2 * gamma_t * z_plus_values + omega_sq_t * z_plus_values * z_minus_values
        residuals = y - y_pred

        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_residual = np.sum(residuals ** 2)

        r_squared = 1 - (ss_residual / ss_total) if ss_total > 0 else np.nan

        return gamma_t, omega_sq_t, r_squared, residuals

    def calculate_implied_skew(self, K_p, K_c, tau, S, r=0.0, q=0.0):
        """Calculate implied skew b_t for a given maturity and strikes

        Parameters:
        -----------
        K_p : float - Put strike
        K_c : float - Call strike
        tau : float - Time to maturity
        S : float - Spot price
        r : float - Risk-free rate
        q : float - Dividend yield

        Returns:
        --------
        float - b_t = (I²_ct/2 - I²_pt/2)/(l^c_+t - l^p_+t)
        """
        F = S * np.exp((r - q) * tau)

        # Get implied volatilities
        I_pt = self.vol_surface.get_implied_volatility(K_p, tau, S, r, q)
        I_ct = self.vol_surface.get_implied_volatility(K_c, tau, S, r, q)

        if np.isnan(I_pt) or np.isnan(I_ct):
            return np.nan

        # Calculate log moneyness measures
        l_plus_p, _ = calculate_log_moneyness(K_p, F, I_pt, tau)
        l_plus_c, _ = calculate_log_moneyness(K_c, F, I_ct, tau)

        # Calculate implied skew
        numerator = I_ct ** 2 / 2 - I_pt ** 2 / 2
        denominator = l_plus_c - l_plus_p

        if abs(denominator) < 1e-10:
            return np.nan

        b_t = numerator / denominator

        return b_t

    def calculate_implied_convexity(self, K_p, K_a, K_c, tau, S, r=0.0, q=0.0):
        """Calculate implied convexity c_t for a given maturity and strikes

        Parameters:
        -----------
        K_p : float - Put strike
        K_a : float - ATM strike
        K_c : float - Call strike
        tau : float - Time to maturity
        S : float - Spot price
        r : float - Risk-free rate
        q : float - Dividend yield

        Returns:
        --------
        float - c_t = ((I²_ct + I²_pt)/2 - I²_at)/l̄²_agt
        """
        F = S * np.exp((r - q) * tau)

        # Get implied volatilities
        I_pt = self.vol_surface.get_implied_volatility(K_p, tau, S, r, q)
        I_at = self.vol_surface.get_implied_volatility(K_a, tau, S, r, q)
        I_ct = self.vol_surface.get_implied_volatility(K_c, tau, S, r, q)

        if np.isnan(I_pt) or np.isnan(I_at) or np.isnan(I_ct):
            return np.nan

        # Calculate log moneyness measures
        l_plus_p, l_minus_p = calculate_log_moneyness(K_p, F, I_pt, tau)
        l_plus_a, l_minus_a = calculate_log_moneyness(K_a, F, I_at, tau)
        l_plus_c, l_minus_c = calculate_log_moneyness(K_c, F, I_ct, tau)

        # Calculate standardized moneyness
        z_plus_p, z_minus_p = compute_standardized_moneyness(l_plus_p, l_minus_p, I_pt, tau)
        z_plus_a, z_minus_a = compute_standardized_moneyness(l_plus_a, l_minus_a, I_at, tau)
        z_plus_c, z_minus_c = compute_standardized_moneyness(l_plus_c, l_minus_c, I_ct, tau)

        # Calculate arithmetic-geometric mean (equation 70)
        l_bar_agt = np.sqrt((abs(z_plus_p) + z_plus_c) / 2 * (abs(z_minus_p) + z_minus_c) / 2)

        # Calculate implied convexity
        numerator = (I_ct ** 2 + I_pt ** 2) / 2 - I_at ** 2
        denominator = l_bar_agt ** 2

        if abs(denominator) < 1e-10:
            return np.nan

        c_t = numerator / denominator

        return c_t

    def calculate_breakeven_implied_variance(self, tau, moneyness, gamma_forecast, omega_sq_forecast, atm_var):
        """Calculate breakeven implied variance based on forecasts

        Parameters:
        -----------
        tau : float - Time to maturity
        moneyness : float or array - Standardized moneyness (z_+)
        gamma_forecast : float - Forecasted covariance
        omega_sq_forecast : float - Forecasted variance
        atm_var : float - ATM implied variance

        Returns:
        --------
        float or array - Breakeven implied variance
        """
        # Assuming moneyness is z_+, calculate z_-
        # For simplicity, use an approximate relationship
        z_minus = moneyness - atm_var * np.sqrt(tau)

        # Calculate breakeven implied variance using equation (14)
        breakeven_var = atm_var + 2 * gamma_forecast * moneyness + omega_sq_forecast * moneyness * z_minus

        return breakeven_var