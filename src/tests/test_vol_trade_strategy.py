import unittest
import numpy as np
from volatility_trading.core.option_pricing import BMSModel
from volatility_trading.strategies.vol_trading import VolTradeStrategy


class TestVolTradeStrategy(unittest.TestCase):
    def setUp(self):
        self.bms_model = BMSModel()
        self.strategy = VolTradeStrategy(self.bms_model)

        # Test parameters
        self.S = 100.0
        self.tau = 0.25  # 3 months
        self.r = 0.02
        self.q = 0.01
        self.sigma_atm = 0.2
        self.notional = 1e6

    def test_construct_position(self):
        """Test vol trade position construction"""
        position = self.strategy.construct_position(
            self.S, self.tau, self.r, self.q, self.sigma_atm, self.notional
        )

        # Check that the position has the expected keys
        expected_keys = ['type', 'strike_atm', 'num_straddles', 'eta_a',
                         'straddle_price', 'straddle_delta',
                         'straddle_gamma', 'straddle_cash_gamma',
                         'delta_hedge', 'S', 'tau', 'r', 'q', 'sigma_atm']

        for key in expected_keys:
            self.assertIn(key, position)

        # Check that the type is correct
        self.assertEqual(position['type'], 'vol_trade')

        # Check that the ATM strike is reasonable
        # For ATM options with z_+ = 0, K = F * exp(-sigma^2*tau/2)
        F = self.S * np.exp((self.r - self.q) * self.tau)
        expected_K = F * np.exp(-0.5 * self.sigma_atm ** 2 * self.tau)
        self.assertAlmostEqual(position['strike_atm'], expected_K, delta=0.01)

        # Check that delta hedge is opposite sign to straddle delta
        if position['straddle_delta'] != 0:
            self.assertLess(position['straddle_delta'] * position['delta_hedge'], 0)

    def test_expected_gain_rate(self):
        """Test expected gain rate calculation"""
        position = self.strategy.construct_position(
            self.S, self.tau, self.r, self.q, self.sigma_atm, self.notional
        )

        instantaneous_variance = 0.05  # Higher than implied
        implied_variance = self.sigma_atm ** 2

        expected_gain = self.strategy.calculate_expected_gain_rate(
            position, instantaneous_variance, implied_variance
        )

        # Check expected gain rate equals the difference between variances
        self.assertAlmostEqual(
            expected_gain['expected_gain_rate'],
            instantaneous_variance - implied_variance,
            delta=1e-10
        )

        # Check expected P&L scaled by position size
        expected_pnl = (instantaneous_variance - implied_variance) * position['straddle_cash_gamma'] * position[
            'num_straddles'] / 2
        self.assertAlmostEqual(
            expected_gain['expected_pnl'],
            expected_pnl,
            delta=1e-6
        )

    def test_update_position(self):
        """Test position update"""
        position = self.strategy.construct_position(
            self.S, self.tau, self.r, self.q, self.sigma_atm, self.notional
        )

        # Spot moves up, time passes, vol increases
        S_new = 102.0
        tau_new = self.tau - 1 / 252  # One day passed
        sigma_new = 0.21

        updated_position = self.strategy.update_position(
            position, S_new, tau_new, self.r, self.q, sigma_new
        )

        # Check that fields were updated correctly
        self.assertEqual(updated_position['S'], S_new)
        self.assertEqual(updated_position['tau'], tau_new)
        self.assertEqual(updated_position['sigma_atm'], sigma_new)

        # Delta hedge should be adjusted
        self.assertIn('delta_hedge_adjustment', updated_position)

        # Number of straddles should remain the same
        self.assertEqual(updated_position['num_straddles'], position['num_straddles'])


if __name__ == '__main__':
    unittest.main()