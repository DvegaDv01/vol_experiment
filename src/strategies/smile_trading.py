import numpy as np
from ..core.option_pricing import BMSModel
from ..utils.math_utils import calculate_log_moneyness, compute_standardized_moneyness


class SmileTradeStrategy:
    def __init__(self, bms_model=None):
        """
        Parameters:
        -----------
        bms_model : BMSModel or None
        """
        self.bms_model = bms_model or BMSModel()

    def construct_position(self, S, tau, r, q, sigma_atm, sigma_put, sigma_call, K_put, K_call, notional=1e6):
        """Construct smile trade position (butterfly spreads)

        Parameters:
        -----------
        S : float - Spot price
        tau : float - Time to maturity
        r : float - Risk-free rate
        q : float - Dividend yield
        sigma_atm : float - ATM implied volatility
        sigma_put : float - OTM put implied volatility
        sigma_call : float - OTM call implied volatility
        K_put : float - OTM put strike
        K_call : float - OTM call strike
        notional : float - Notional amount to trade

        Returns:
        --------
        dict - Position details including weights for puts, calls, and straddles
        """
        # Calculate forward price
        F = S * np.exp((r - q) * tau)

        # Find ATM strike where z_+ = 0
        K_atm = self.bms_model.find_atm_strike(F, tau, sigma_atm)

        # Calculate log moneyness and standardized moneyness for each option
        l_plus_p, l_minus_p = calculate_log_moneyness(K_put, F, sigma_put, tau)
        l_plus_a, l_minus_a = calculate_log_moneyness(K_atm, F, sigma_atm, tau)
        l_plus_c, l_minus_c = calculate_log_moneyness(K_call, F, sigma_call, tau)

        z_plus_p, z_minus_p = compute_standardized_moneyness(l_plus_p, l_minus_p, sigma_put, tau)
        z_plus_a, z_minus_a = compute_standardized_moneyness(l_plus_a, l_minus_a, sigma_atm, tau)
        z_plus_c, z_minus_c = compute_standardized_moneyness(l_plus_c, l_minus_c, sigma_call, tau)

        # Check for equal absolute moneyness (equation 69)
        # l_plus_c = -l_plus_p
        # We'll continue even if this isn't exactly satisfied

        # Calculate arithmetic-geometric mean (equations 68, 70)
        l_bar_agt = np.sqrt((abs(z_plus_p) + z_plus_c) / 2 * (abs(z_minus_p) + z_minus_c) / 2)

        # Calculate Greeks
        put_greeks = self.bms_model.calculate_greeks(S, K_put, tau, r, q, sigma_put, 'put')
        call_greeks = self.bms_model.calculate_greeks(S, K_call, tau, r, q, sigma_call, 'call')
        straddle_call_greeks = self.bms_model.calculate_greeks(S, K_atm, tau, r, q, sigma_atm, 'call')
        straddle_put_greeks = self.bms_model.calculate_greeks(S, K_atm, tau, r, q, sigma_atm, 'put')

        # Get prices
        put_price = self.bms_model.option_price(S, K_put, tau, r, q, sigma_put, 'put')
        call_price = self.bms_model.option_price(S, K_call, tau, r, q, sigma_call, 'call')
        straddle_price = (
                self.bms_model.option_price(S, K_atm, tau, r, q, sigma_atm, 'call') +
                self.bms_model.option_price(S, K_atm, tau, r, q, sigma_atm, 'put')
        )

        # Calculate cash gamma
        cash_gamma_p = put_greeks['cash_gamma']
        cash_gamma_c = call_greeks['cash_gamma']
        straddle_cash_gamma = straddle_call_greeks['cash_gamma'] + straddle_put_greeks['cash_gamma']

        # Calculate weights (equation 72)
        # η^p_t = 1/(l̄²_agt*$Γ^p_t)
        # η^a_t = -2/(l̄²_agt*$Γ^a_t)
        # η^c_t = 1/(l̄²_agt*$Γ^c_t)
        if l_bar_agt ** 2 < 1e-8 or cash_gamma_p < 1e-8 or cash_gamma_c < 1e-8 or straddle_cash_gamma < 1e-8:
            return None  # Invalid position

        eta_p = 1 / (l_bar_agt ** 2 * cash_gamma_p)
        eta_a_base = -2 / (l_bar_agt ** 2 * straddle_cash_gamma)
        eta_c = 1 / (l_bar_agt ** 2 * cash_gamma_c)

        # Add cash vega hedge (equation 79)
        # η^a_t = -1/(l̄²_agt*$Γ^a_t)*{2 + τ[(I²_ct + I²_pt)/(2*I²_at) - 1]}
        var_ratio_term = (sigma_call ** 2 + sigma_put ** 2) / (2 * sigma_atm ** 2) - 1
        eta_a_vega_hedge = -1 / (l_bar_agt ** 2 * straddle_cash_gamma) * tau * var_ratio_term
        eta_a = eta_a_base + eta_a_vega_hedge

        # Scale position for notional
        # First, calculate total position cost
        position_cost = (
                eta_p * put_price +
                eta_a * straddle_price +
                eta_c * call_price
        )

        # Apply scaling
        position_scale = notional / abs(position_cost) if abs(position_cost) > 0 else 0

        num_puts = eta_p * position_scale
        num_straddles = eta_a * position_scale
        num_calls = eta_c * position_scale

        # Calculate delta hedge
        total_delta = (
                num_puts * put_greeks['delta'] +
                num_calls * call_greeks['delta'] +
                num_straddles * (straddle_call_greeks['delta'] + straddle_put_greeks['delta'])
        )

        delta_hedge = -total_delta

        return {
            'type': 'smile_trade',
            'K_put': K_put,
            'K_atm': K_atm,
            'K_call': K_call,
            'num_puts': num_puts,
            'num_calls': num_calls,
            'num_straddles': num_straddles,
            'eta_p': eta_p,
            'eta_c': eta_c,
            'eta_a': eta_a,
            'eta_a_base': eta_a_base,
            'eta_a_vega_hedge': eta_a_vega_hedge,
            'l_bar_agt': l_bar_agt,
            'z_plus_p': z_plus_p,
            'z_minus_p': z_minus_p,
            'z_plus_c': z_plus_c,
            'z_minus_c': z_minus_c,
            'put_price': put_price,
            'call_price': call_price,
            'straddle_price': straddle_price,
            'delta_hedge': delta_hedge,
            'position_cost': position_cost * position_scale,
            'position_scale': position_scale,
            'S': S,
            'tau': tau,
            'r': r,
            'q': q,
            'sigma_put': sigma_put,
            'sigma_call': sigma_call,
            'sigma_atm': sigma_atm
        }

    def calculate_expected_gain_rate(self, position, variance_of_vol, implied_convexity):
        """Calculate expected gain rate of smile position

        Parameters:
        -----------
        position : dict - Position details from construct_position
        variance_of_vol : float - Realized variance of volatility ω²_t
        implied_convexity : float - Implied convexity c_t

        Returns:
        --------
        dict - Expected gain metrics
        """
        # Calculate expected gain rate (equation 73)
        expected_gain_rate = variance_of_vol - implied_convexity

        # Calculate implied convexity based on position
        sigma_c_squared = position['sigma_call'] ** 2
        sigma_p_squared = position['sigma_put'] ** 2
        sigma_a_squared = position['sigma_atm'] ** 2

        position_implied_convexity = ((sigma_c_squared + sigma_p_squared) / 2 - sigma_a_squared) / position[
            'l_bar_agt'] ** 2

        # Expected P&L based on position size
        expected_pnl = expected_gain_rate * position['position_scale']

        return {
            'expected_gain_rate': expected_gain_rate,
            'position_implied_convexity': position_implied_convexity,
            'expected_pnl': expected_pnl
        }

    def update_position(self, position, S_new, tau_new, r_new=None, q_new=None,
                        sigma_put_new=None, sigma_call_new=None, sigma_atm_new=None):
        """Update position for new market conditions

        Parameters:
        -----------
        position : dict - Current position details
        S_new : float - New spot price
        tau_new : float - New time to maturity
        r_new, q_new : float or None - New rates (None to keep current)
        sigma_*_new : float or None - New implied volatilities (None to keep current)

        Returns:
        --------
        dict - Updated position details
        """
        # Use current values if new ones not provided
        r = r_new if r_new is not None else position['r']
        q = q_new if q_new is not None else position['q']
        sigma_put = sigma_put_new if sigma_put_new is not None else position['sigma_put']
        sigma_call = sigma_call_new if sigma_call_new is not None else position['sigma_call']
        sigma_atm = sigma_atm_new if sigma_atm_new is not None else position['sigma_atm']

        # Recalculate Greeks
        put_greeks = self.bms_model.calculate_greeks(
            S_new, position['K_put'], tau_new, r, q, sigma_put, 'put'
        )
        call_greeks = self.bms_model.calculate_greeks(
            S_new, position['K_call'], tau_new, r, q, sigma_call, 'call'
        )
        straddle_call_greeks = self.bms_model.calculate_greeks(
            S_new, position['K_atm'], tau_new, r, q, sigma_atm, 'call'
        )
        straddle_put_greeks = self.bms_model.calculate_greeks(
            S_new, position['K_atm'], tau_new, r, q, sigma_atm, 'put'
        )

        # Calculate new delta hedge
        total_delta = (
                position['num_puts'] * put_greeks['delta'] +
                position['num_calls'] * call_greeks['delta'] +
                position['num_straddles'] * (straddle_call_greeks['delta'] + straddle_put_greeks['delta'])
        )

        delta_hedge = -total_delta
        delta_hedge_adjustment = delta_hedge - position['delta_hedge']

        updated_position = position.copy()
        updated_position.update({
            'S': S_new,
            'tau': tau_new,
            'r': r,
            'q': q,
            'sigma_put': sigma_put,
            'sigma_call': sigma_call,
            'sigma_atm': sigma_atm,
            'delta_hedge': delta_hedge,
            'delta_hedge_adjustment': delta_hedge_adjustment
        })

        return updated_position