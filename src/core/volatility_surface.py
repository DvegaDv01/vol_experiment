import numpy as np
import pandas as pd
from ..utils.math_utils import gaussian_kernel, calculate_log_moneyness, compute_standardized_moneyness
from .option_pricing import BMSModel


class VolatilitySurface:
    def __init__(self, option_data, kernel_bandwidths=None):
        """
        Parameters:
        -----------
        option_data : DataFrame - Contains columns: [strike, maturity, implied_vol, call_put, etc.]
        kernel_bandwidths : tuple - (moneyness_bandwidth, tau_bandwidth)
        """
        self.option_data = option_data.copy()
        self.kernel_bandwidths = kernel_bandwidths or (0.25, 0.3)  # Default bandwidths
        self.bms_model = BMSModel()
        self.preprocess_data()

    def preprocess_data(self):
        """Preprocess option data for surface construction"""
        # Add standardized moneyness if not present
        if 'z_plus' not in self.option_data.columns:
            # Compute forward price
            self.option_data['forward'] = self.option_data['spot'] * np.exp(
                (self.option_data['r'] - self.option_data['q']) * self.option_data['tau']
            )

            # Compute log moneyness measures
            l_plus, l_minus = calculate_log_moneyness(
                self.option_data['strike'],
                self.option_data['forward'],
                self.option_data['impl_vol'],
                self.option_data['tau']
            )

            # Compute standardized moneyness
            z_plus, z_minus = compute_standardized_moneyness(
                l_plus, l_minus,
                self.option_data['impl_vol'],
                self.option_data['tau']
            )

            self.option_data['l_plus'] = l_plus
            self.option_data['l_minus'] = l_minus
            self.option_data['z_plus'] = z_plus
            self.option_data['z_minus'] = z_minus

    def get_implied_volatility(self, K, tau, S, r=0.0, q=0.0):
        """Get interpolated implied volatility for given strike and maturity

        Parameters:
        -----------
        K : float - Strike price
        tau : float - Time to maturity in years
        S : float - Spot price
        r : float - Risk-free rate
        q : float - Dividend yield

        Returns:
        --------
        float - Implied volatility
        """
        F = S * np.exp((r - q) * tau)

        # Compute reference moneyness for interpolation
        log_strike_ratio = np.log(K / F)

        # Use only appropriate options for interpolation
        valid_data = self.option_data[
            (self.option_data['tau'] > 0) &
            (~self.option_data['impl_vol'].isna())
            ].copy()

        if len(valid_data) == 0:
            return np.nan

        # Calculate weights based on moneyness and maturity
        moneyness_bandwidth, tau_bandwidth = self.kernel_bandwidths

        # Calculate weights for each option
        log_moneyness_weights = gaussian_kernel(
            log_strike_ratio,
            np.log(valid_data['strike'] / valid_data['forward']),
            moneyness_bandwidth
        )

        tau_weights = gaussian_kernel(
            np.log(tau),
            np.log(valid_data['tau']),
            tau_bandwidth
        )

        # Combine weights
        total_weights = log_moneyness_weights * tau_weights

        # Normalize weights
        if np.sum(total_weights) > 0:
            total_weights = total_weights / np.sum(total_weights)
        else:
            return np.nan

        # Calculate weighted average implied volatility
        weighted_iv = np.sum(valid_data['impl_vol'] * total_weights)

        return weighted_iv

    def get_atm_volatility_term_structure(self, S, r=0.0, q=0.0, tau_grid=None):
        """Get term structure of ATM implied volatility

        Parameters:
        -----------
        S : float - Spot price
        r : float - Risk-free rate (can be array matching tau_grid)
        q : float - Dividend yield (can be array matching tau_grid)
        tau_grid : array - Grid of maturities to calculate ATM IV

        Returns:
        --------
        DataFrame - Contains [tau, A_t]
        """
        if tau_grid is None:
            # Default grid: 1M, 2M, 3M, 6M, 1Y
            tau_grid = np.array([1 / 12, 2 / 12, 3 / 12, 6 / 12, 1.0])

        atm_vols = []

        for idx, tau in enumerate(tau_grid):
            r_val = r[idx] if isinstance(r, (list, np.ndarray)) else r
            q_val = q[idx] if isinstance(q, (list, np.ndarray)) else q

            F = S * np.exp((r_val - q_val) * tau)

            # Find ATM strike (z_plus = 0)
            # Use 0.2 as initial volatility guess for ATM strike calculation
            K_atm = self.bms_model.find_atm_strike(F, tau, 0.2)

            # Get implied volatility at ATM strike
            iv_atm = self.get_implied_volatility(K_atm, tau, S, r_val, q_val)

            # Refine ATM strike using the implied volatility
            if not np.isnan(iv_atm):
                K_atm = self.bms_model.find_atm_strike(F, tau, iv_atm)
                iv_atm = self.get_implied_volatility(K_atm, tau, S, r_val, q_val)

            atm_vols.append({
                'tau': tau,
                'A_t': iv_atm,
                'K_atm': K_atm
            })

        return pd.DataFrame(atm_vols)

    def calculate_floating_series(self, S, r=0.0, q=0.0, moneyness_levels=None, tau_levels=None):
        """Calculate interpolated floating IV series

        Parameters:
        -----------
        S : float - Spot price
        r : float - Risk-free rate
        q : float - Dividend yield
        moneyness_levels : array - Standardized moneyness levels
        tau_levels : array - Time to maturity levels

        Returns:
        --------
        DataFrame - Contains [tau, moneyness, implied_vol]
        """
        if moneyness_levels is None:
            moneyness_levels = np.array([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2])

        if tau_levels is None:
            tau_levels = np.array([1 / 12, 2 / 12, 3 / 12, 6 / 12, 1.0])

        floating_series = []

        for tau in tau_levels:
            # Get ATM volatility for this maturity
            atm_data = self.get_atm_volatility_term_structure(S, r, q, [tau])
            atm_vol = atm_data.iloc[0]['A_t']

            if np.isnan(atm_vol):
                continue

            F = S * np.exp((r - q) * tau)

            for x in moneyness_levels:
                # Convert standardized moneyness to strike
                # z_plus = (ln(K/F) + sigma²τ/2) / (sigma√τ)
                # For ATM strike, sigma = atm_vol
                K = F * np.exp(x * atm_vol * np.sqrt(tau) - 0.5 * atm_vol ** 2 * tau)

                # Get implied volatility at this strike
                impl_vol = self.get_implied_volatility(K, tau, S, r, q)

                if not np.isnan(impl_vol):
                    # Calculate actual z_plus using the implied volatility
                    l_plus, _ = calculate_log_moneyness(K, F, impl_vol, tau)
                    z_plus = l_plus / (impl_vol * np.sqrt(tau))

                    floating_series.append({
                        'tau': tau,
                        'x': x,
                        'z_plus': z_plus,
                        'impl_vol': impl_vol,
                        'K': K,
                        'F': F
                    })

        return pd.DataFrame(floating_series)