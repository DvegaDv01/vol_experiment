import numpy as np
from ..core.option_pricing import BMSModel


class HedgingManager:
    def __init__(self, bms_model=None):
        """
        Parameters:
        -----------
        bms_model : BMSModel or None
        """
        self.bms_model = bms_model or BMSModel()

    def calculate_delta_hedge_adjustment(self, position, S_new):
        """Calculate required delta hedge adjustment when spot price changes

        Parameters:
        -----------
        position : dict - Position details
        S_new : float - New spot price

        Returns:
        --------
        float - Delta hedge adjustment amount
        """
        if position['type'] == 'vol_trade':
            return self._delta_hedge_vol(position, S_new)
        elif position['type'] == 'skew_trade':
            return self._delta_hedge_skew(position, S_new)
        elif position['type'] == 'smile_trade':
            return self._delta_hedge_smile(position, S_new)
        else:
            raise ValueError(f"Unknown position type: {position['type']}")

    def _delta_hedge_vol(self, position, S_new):
        """Delta hedge adjustment for vol trade"""
        # Recalculate straddle delta at new spot
        call_greeks = self.bms_model.calculate_greeks(
            S_new, position['strike_atm'], position['tau'],
            position['r'], position['q'], position['sigma_atm'], 'call'
        )
        put_greeks = self.bms_model.calculate_greeks(
            S_new, position['strike_atm'], position['tau'],
            position['r'], position['q'], position['sigma_atm'], 'put'
        )

        new_straddle_delta = call_greeks['delta'] + put_greeks['delta']
        new_delta_hedge = -new_straddle_delta * position['num_straddles']

        return new_delta_hedge - position['delta_hedge']

    def _delta_hedge_skew(self, position, S_new):
        """Delta hedge adjustment for skew trade"""
        # Recalculate deltas at new spot
        put_greeks = self.bms_model.calculate_greeks(
            S_new, position['K_put'], position['tau'],
            position['r'], position['q'], position['sigma_put'], 'put'
        )
        call_greeks = self.bms_model.calculate_greeks(
            S_new, position['K_call'], position['tau'],
            position['r'], position['q'], position['sigma_call'], 'call'
        )
        straddle_call_greeks = self.bms_model.calculate_greeks(
            S_new, position['K_atm'], position['tau'],
            position['r'], position['q'], position['sigma_atm'], 'call'
        )
        straddle_put_greeks = self.bms_model.calculate_greeks(
            S_new, position['K_atm'], position['tau'],
            position['r'], position['q'], position['sigma_atm'], 'put'
        )

        total_delta = (
                position['num_puts'] * put_greeks['delta'] +
                position['num_calls'] * call_greeks['delta'] +
                position['num_straddles'] * (straddle_call_greeks['delta'] + straddle_put_greeks['delta'])
        )

        new_delta_hedge = -total_delta
        return new_delta_hedge - position['delta_hedge']

    def _delta_hedge_smile(self, position, S_new):
        """Delta hedge adjustment for smile trade"""
        # Same structure as skew trade
        return self._delta_hedge_skew(position, S_new)

    def calculate_cash_vega_hedge(self, position, sigma_atm_new=None):
        """Calculate cash vega hedge adjustment when vol changes

        Parameters:
        -----------
        position : dict - Position details
        sigma_atm_new : float or None - New ATM volatility

        Returns:
        --------
        dict - Cash vega hedge adjustments
        """
        if position['type'] == 'skew_trade':
            return self._vega_hedge_skew(position, sigma_atm_new)
        elif position['type'] == 'smile_trade':
            return self._vega_hedge_smile(position, sigma_atm_new)
        else:
            # Vol trades don't need cash vega hedging
            return {'straddle_adjustment': 0}

    def _vega_hedge_skew(self, position, sigma_atm_new=None):
        """Cash vega hedge for skew trade per equation (64)"""
        sigma_atm = sigma_atm_new if sigma_atm_new is not None else position['sigma_atm']

        # Get straddle vega
        straddle_call_greeks = self.bms_model.calculate_greeks(
            position['S'], position['K_atm'], position['tau'],
            position['r'], position['q'], sigma_atm, 'call'
        )
        straddle_put_greeks = self.bms_model.calculate_greeks(
            position['S'], position['K_atm'], position['tau'],
            position['r'], position['q'], sigma_atm, 'put'
        )

        straddle_cash_gamma = straddle_call_greeks['cash_gamma'] + straddle_put_greeks['cash_gamma']

        # Calculate implied variance difference
        implied_var_diff = position['sigma_call'] ** 2 - position['sigma_put'] ** 2
        l_diff = position['l_plus_c'] - position['l_plus_p']

        # Calculate ideal number of straddles from equation (64)
        ideal_num_straddles = -position['tau'] / (straddle_cash_gamma * l_diff) * (implied_var_diff / sigma_atm ** 2)

        # Return adjustment needed
        return {'straddle_adjustment': ideal_num_straddles - position['num_straddles']}

    def _vega_hedge_smile(self, position, sigma_atm_new=None):
        """Cash vega hedge for smile trade per equation (79)"""
        sigma_atm = sigma_atm_new if sigma_atm_new is not None else position['sigma_atm']

        # Get straddle cash gamma
        straddle_call_greeks = self.bms_model.calculate_greeks(
            position['S'], position['K_atm'], position['tau'],
            position['r'], position['q'], sigma_atm, 'call'
        )
        straddle_put_greeks = self.bms_model.calculate_greeks(
            position['S'], position['K_atm'], position['tau'],
            position['r'], position['q'], sigma_atm, 'put'
        )

        straddle_cash_gamma = straddle_call_greeks['cash_gamma'] + straddle_put_greeks['cash_gamma']

        # Calculate variance ratio term
        var_ratio_term = (position['sigma_call'] ** 2 + position['sigma_put'] ** 2) / (2 * sigma_atm ** 2) - 1

        # Calculate ideal number of straddles from equation (79)
        eta_a_base = -2 / (position['l_bar_agt'] ** 2 * straddle_cash_gamma)
        eta_a_vega_hedge = -1 / (position['l_bar_agt'] ** 2 * straddle_cash_gamma) * position['tau'] * var_ratio_term
        ideal_eta_a = eta_a_base + eta_a_vega_hedge

        ideal_num_straddles = ideal_eta_a * position['position_scale']

        # Return adjustment needed
        return {'straddle_adjustment': ideal_num_straddles - position['num_straddles']}