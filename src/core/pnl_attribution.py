import numpy as np
import pandas as pd
from ..core.option_pricing import BMSModel


class PnLAttributor:
    def __init__(self, bms_model=None):
        """
        Parameters:
        -----------
        bms_model : BMSModel or None
        """
        self.bms_model = bms_model or BMSModel()

    def attribute_pnl(self, initial_state, final_state, realized_moments):
        """Attribute P&L to different exposures

        Parameters:
        -----------
        initial_state, final_state : dict - Contains option parameters and price
        realized_moments : dict - Contains realized σ², γ, ω²

        Returns:
        --------
        dict - P&L attribution to theta, delta, gamma, vega, vanna, volga
        """
        # Extract parameters
        S_initial = initial_state['S']
        S_final = final_state['S']
        I_initial = initial_state['impl_vol']
        I_final = final_state['impl_vol']

        tau_initial = initial_state['tau']
        tau_final = final_state['tau']

        K = initial_state['K']

        # Calculate price change
        price_initial = self.bms_model.option_price(
            S_initial, K, tau_initial,
            initial_state.get('r', 0), initial_state.get('q', 0),
            I_initial, initial_state.get('option_type', 'call')
        )

        price_final = self.bms_model.option_price(
            S_final, K, tau_final,
            final_state.get('r', 0), final_state.get('q', 0),
            I_final, final_state.get('option_type', 'call')
        )

        total_pnl = price_final - price_initial

        # Calculate Greeks at initial state
        greeks_initial = self.bms_model.calculate_greeks(
            S_initial, K, tau_initial,
            initial_state.get('r', 0), initial_state.get('q', 0),
            I_initial, initial_state.get('option_type', 'call')
        )

        # Extract realized moments
        sigma_squared = realized_moments.get('sigma_squared', 0)
        gamma_realized = realized_moments.get('gamma', 0)
        omega_squared = realized_moments.get('omega_squared', 0)

        # Attribute P&L
        # Time decay (theta)
        dt = tau_initial - tau_final  # Time elapsed in years
        theta_pnl = greeks_initial['theta'] * dt

        # Price change (delta)
        dS = S_final - S_initial
        delta_pnl = greeks_initial['delta'] * dS

        # Gamma - Price variance effect
        gamma_pnl = 0.5 * greeks_initial['gamma'] * dS ** 2

        # Implied volatility change (vega)
        dI = I_final - I_initial
        vega_pnl = greeks_initial['vega'] * dI

        # Vanna - Price-volatility covariance effect
        vanna_pnl = greeks_initial['vanna'] * dS * dI

        # Volga - Volatility variance effect
        volga_pnl = 0.5 * greeks_initial['volga'] * dI ** 2

        # Calculate realized contributions based on continuous-time moments
        # These use the formula from equation (3) and the moments over the period
        realized_theta = greeks_initial['theta'] * dt
        realized_gamma = 0.5 * greeks_initial['cash_gamma'] * sigma_squared * dt
        realized_vega = greeks_initial['cash_vega'] * dt * realized_moments.get('mu', 0)
        realized_vanna = greeks_initial['cash_vanna'] * gamma_realized * dt
        realized_volga = 0.5 * greeks_initial['cash_volga'] * omega_squared * dt

        # Calculate unexplained P&L
        explained_pnl = theta_pnl + delta_pnl + gamma_pnl + vega_pnl + vanna_pnl + volga_pnl
        unexplained_pnl = total_pnl - explained_pnl

        # Calculate continuous-time theoretical P&L
        theoretical_pnl = realized_theta + realized_gamma + realized_vega + realized_vanna + realized_volga

        return {
            'total_pnl': total_pnl,
            'theta_pnl': theta_pnl,
            'delta_pnl': delta_pnl,
            'gamma_pnl': gamma_pnl,
            'vega_pnl': vega_pnl,
            'vanna_pnl': vanna_pnl,
            'volga_pnl': volga_pnl,
            'unexplained_pnl': unexplained_pnl,
            'realized_theta': realized_theta,
            'realized_gamma': realized_gamma,
            'realized_vega': realized_vega,
            'realized_vanna': realized_vanna,
            'realized_volga': realized_volga,
            'theoretical_pnl': theoretical_pnl
        }

    def validate_no_arbitrage_condition(self, option_params, moments):
        """Check if no-arbitrage condition is satisfied

        Parameters:
        -----------
        option_params : dict - Option parameters
        moments : dict - Contains μ_t, σ²_t, γ_t, ω²_t moments

        Returns:
        --------
        dict - Validation results
        """
        # Extract parameters
        S = option_params['S']
        K = option_params['K']
        tau = option_params['tau']
        r = option_params.get('r', 0)
        q = option_params.get('q', 0)
        I = option_params['impl_vol']
        option_type = option_params.get('option_type', 'call')

        # Calculate Greeks
        greeks = self.bms_model.calculate_greeks(S, K, tau, r, q, I, option_type)

        # Extract moments
        mu = moments.get('mu', 0)
        sigma_squared = moments.get('sigma_squared', 0)
        gamma = moments.get('gamma', 0)
        omega_squared = moments.get('omega_squared', 0)

        # Calculate LHS and RHS of equation (7)
        lhs = -greeks['theta']

        rhs = (
                greeks['vega'] * I * mu +
                0.5 * greeks['cash_gamma'] * sigma_squared +
                0.5 * greeks['volga'] * I ** 2 * omega_squared +
                greeks['vanna'] * I * S * gamma
        )

        # Calculate deviation
        deviation = lhs - rhs
        relative_deviation = deviation / abs(lhs) if abs(lhs) > 1e-8 else deviation

        is_satisfied = abs(relative_deviation) < 0.01  # 1% threshold

        return {
            'is_satisfied': is_satisfied,
            'deviation': deviation,
            'relative_deviation': relative_deviation,
            'lhs': lhs,
            'rhs': rhs,
            'theta': greeks['theta'],
            'vega_term': greeks['vega'] * I * mu,
            'gamma_term': 0.5 * greeks['cash_gamma'] * sigma_squared,
            'volga_term': 0.5 * greeks['volga'] * I ** 2 * omega_squared,
            'vanna_term': greeks['vanna'] * I * S * gamma
        }