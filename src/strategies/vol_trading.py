import numpy as np
from ..core.option_pricing import BMSModel


class VolTradeStrategy:
    def __init__(self, bms_model=None):
        """
        Parameters:
        -----------
        bms_model : BMSModel or None
        """
        self.bms_model = bms_model or BMSModel()

    def construct_position(self, S, tau, r, q, sigma_atm, notional=1e6):
        """Construct vol trade position (ATM straddles)

        Parameters:
        -----------
        S : float - Spot price
        tau : float - Time to maturity
        r : float - Risk-free rate
        q : float - Dividend yield
        sigma_atm : float - ATM implied volatility
        notional : float - Notional amount to trade

        Returns:
        --------
        dict - Position details including weights, strikes, cash gamma
        """
        # Calculate forward price
        F = S * np.exp((r - q) * tau)

        # Find ATM strike where z_+ = 0
        K_atm = self.bms_model.find_atm_strike(F, tau, sigma_atm)

        # Calculate option prices and Greeks
        call_greeks = self.bms_model.calculate_greeks(S, K_atm, tau, r, q, sigma_atm, 'call')
        put_greeks = self.bms_model.calculate_greeks(S, K_atm, tau, r, q, sigma_atm, 'put')

        call_price = self.bms_model.option_price(S, K_atm, tau, r, q, sigma_atm, 'call')
        put_price = self.bms_model.option_price(S, K_atm, tau, r, q, sigma_atm, 'put')

        # Calculate straddle properties
        straddle_price = call_price + put_price
        straddle_delta = call_greeks['delta'] + put_greeks['delta']
        straddle_gamma = call_greeks['gamma'] + put_greeks['gamma']
        straddle_cash_gamma = call_greeks['cash_gamma'] + put_greeks['cash_gamma']

        # Calculate η^a_t = 2/∆Γ^a_t (equation 52)
        if straddle_cash_gamma > 0:
            eta_a = 2 / straddle_cash_gamma
        else:
            eta_a = 0  # Invalid position if no gamma

        # Scale position based on notional
        num_straddles = notional / straddle_price if straddle_price > 0 else 0

        # Calculate delta hedge ratio
        delta_hedge = -straddle_delta * num_straddles

        return {
            'type': 'vol_trade',
            'strike_atm': K_atm,
            'num_straddles': num_straddles,
            'eta_a': eta_a,
            'straddle_price': straddle_price,
            'straddle_delta': straddle_delta,
            'straddle_gamma': straddle_gamma,
            'straddle_cash_gamma': straddle_cash_gamma,
            'delta_hedge': delta_hedge,
            'S': S,
            'tau': tau,
            'r': r,
            'q': q,
            'sigma_atm': sigma_atm
        }

    def calculate_expected_gain_rate(self, position, instantaneous_variance, implied_variance):
        """Calculate expected gain rate of vol position

        Parameters:
        -----------
        position : dict - Position details from construct_position
        instantaneous_variance : float - Realized variance σ²_t
        implied_variance : float - Implied variance I²_atm

        Returns:
        --------
        float - Expected gain rate (σ²_t - I²_at)
        """
        # Calculate expected gain rate (equation 54)
        expected_gain_rate = instantaneous_variance - implied_variance

        # Scale by position size
        expected_pnl = expected_gain_rate * position['straddle_cash_gamma'] * position['num_straddles'] / 2

        return {
            'expected_gain_rate': expected_gain_rate,
            'expected_pnl': expected_pnl
        }

    def update_position(self, position, S_new, tau_new, r_new=None, q_new=None, sigma_atm_new=None):
        """Update position for new market conditions

        Parameters:
        -----------
        position : dict - Current position details
        S_new : float - New spot price
        tau_new : float - New time to maturity
        r_new : float or None - New risk-free rate (None to keep current)
        q_new : float or None - New dividend yield (None to keep current)
        sigma_atm_new : float or None - New ATM implied volatility (None to keep current)

        Returns:
        --------
        dict - Updated position details
        """
        # Use current values if new ones not provided
        r = r_new if r_new is not None else position['r']
        q = q_new if q_new is not None else position['q']
        sigma_atm = sigma_atm_new if sigma_atm_new is not None else position['sigma_atm']

        # Recalculate Greeks for call and put
        call_greeks = self.bms_model.calculate_greeks(
            S_new, position['strike_atm'], tau_new, r, q, sigma_atm, 'call'
        )
        put_greeks = self.bms_model.calculate_greeks(
            S_new, position['strike_atm'], tau_new, r, q, sigma_atm, 'put'
        )

        # Update straddle properties
        straddle_delta = call_greeks['delta'] + put_greeks['delta']
        straddle_gamma = call_greeks['gamma'] + put_greeks['gamma']
        straddle_cash_gamma = call_greeks['cash_gamma'] + put_greeks['cash_gamma']

        # Calculate new delta hedge
        delta_hedge = -straddle_delta * position['num_straddles']

        # Calculate delta hedge adjustment
        delta_hedge_adjustment = delta_hedge - position['delta_hedge']

        updated_position = position.copy()
        updated_position.update({
            'S': S_new,
            'tau': tau_new,
            'r': r,
            'q': q,
            'sigma_atm': sigma_atm,
            'straddle_delta': straddle_delta,
            'straddle_gamma': straddle_gamma,
            'straddle_cash_gamma': straddle_cash_gamma,
            'delta_hedge': delta_hedge,
            'delta_hedge_adjustment': delta_hedge_adjustment
        })

        return updated_position