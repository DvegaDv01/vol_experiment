import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from .core.option_pricing import BMSModel
from .core.volatility_surface import VolatilitySurface
from .core.moment_extraction import MomentExtractor
from .core.pnl_attribution import PnLAttributor

from .strategies.vol_trading import VolTradeStrategy
from .strategies.skew_trading import SkewTradeStrategy
from .strategies.smile_trading import SmileTradeStrategy
from .strategies.hedging import HedgingManager

from .analysis.performance_metrics import PerformanceAnalyzer
from .analysis.regression import RegressionAnalyzer

from .utils.visualization import VolatilityVisualizer


class VolatilityTradingSystem:
    def __init__(self, option_data=None):
        """
        Initialize the volatility trading system.

        Parameters:
        -----------
        option_data : DataFrame or None - Option data with implied volatilities
        """
        self.bms_model = BMSModel()
        self.vol_surface = None
        self.moment_extractor = None
        self.regression_analyzer = RegressionAnalyzer()
        self.performance_analyzer = PerformanceAnalyzer()
        self.pnl_attributor = PnLAttributor(bms_model=self.bms_model)
        self.visualizer = VolatilityVisualizer()

        # Initialize strategies
        self.vol_strategy = VolTradeStrategy(bms_model=self.bms_model)
        self.skew_strategy = SkewTradeStrategy(bms_model=self.bms_model)
        self.smile_strategy = SmileTradeStrategy(bms_model=self.bms_model)
        self.hedging_manager = HedgingManager(bms_model=self.bms_model)

        # Track active positions
        self.active_positions = {
            'vol': None,
            'skew': None,
            'smile': None
        }

        # Track historical data
        self.extracted_moments = []

        # Initialize with data if provided
        if option_data is not None:
            self.initialize_with_data(option_data)

    def initialize_with_data(self, option_data, kernel_bandwidths=None):
        """
        Initialize the system with option data.

        Parameters:
        -----------
        option_data : DataFrame - Option data with implied volatilities
        kernel_bandwidths : tuple or None - Bandwidths for surface interpolation
        """
        self.vol_surface = VolatilitySurface(option_data, kernel_bandwidths)
        self.moment_extractor = MomentExtractor(self.vol_surface)

    def extract_moments(self, S, r=0.0, q=0.0, moneyness_range=(-1, 1), tau_levels=None):
        """
        Extract moments from the volatility surface.

        Parameters:
        -----------
        S : float - Spot price
        r : float - Risk-free rate
        q : float - Dividend yield
        moneyness_range : tuple - Range of moneyness for extraction
        tau_levels : list or None - Maturities to extract moments for

        Returns:
        --------
        DataFrame - Extracted moments (μ, γ, ω²)
        """
        if self.moment_extractor is None:
            raise ValueError("System not initialized with data. Call initialize_with_data first.")

        if tau_levels is None:
            tau_levels = [1 / 12, 2 / 12, 3 / 12, 6 / 12, 1.0]

        results = []
        date = pd.Timestamp.now().date()

        for tau_idx, tau in enumerate(tau_levels):
            if tau_idx < len(tau_levels) - 1:
                # Extract μ from term structure
                mu = self.moment_extractor.extract_drift_from_term_structure(
                    tau, tau_levels[tau_idx + 1], S, r, q
                )
            else:
                # For the last maturity, use the previous μ
                mu = results[-1]['mu'] if results else np.nan

            # Extract γ and ω² from smile
            gamma, omega_sq, r_squared, _ = self.moment_extractor.extract_variance_covariance_from_smile(
                tau, S, r, q, moneyness_range
            )

            # Get ATM volatility
            atm_data = self.vol_surface.get_atm_volatility_term_structure(S, r, q, [tau])
            if len(atm_data) > 0:
                A_t = atm_data.iloc[0]['A_t']
                implied_var = A_t ** 2
            else:
                A_t = np.nan
                implied_var = np.nan

            results.append({
                'date': date,
                'tau': tau,
                'S': S,
                'r': r,
                'q': q,
                'mu': mu,
                'gamma': gamma,
                'omega_squared': omega_sq,
                'r_squared': r_squared,
                'implied_vol': A_t,
                'implied_var': implied_var
            })

        moments_df = pd.DataFrame(results)
        self.extracted_moments.append(moments_df)

        return moments_df

    def establish_vol_position(self, S, tau, r, q, sigma_atm, notional=1e6):
        """
        Establish a vol trading position.

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
        dict - Position details
        """
        position = self.vol_strategy.construct_position(S, tau, r, q, sigma_atm, notional)
        self.active_positions['vol'] = position
        return position

    def establish_skew_position(self, S, tau, r, q, sigma_atm, sigma_put, sigma_call,
                                K_put, K_call, notional=1e6):
        """
        Establish a skew trading position.

        Parameters:
        -----------
        S : float - Spot price
        tau : float - Time to maturity
        r, q : float - Rates
        sigma_* : float - Implied volatilities
        K_* : float - Strike prices
        notional : float - Notional amount to trade

        Returns:
        --------
        dict - Position details
        """
        position = self.skew_strategy.construct_position(
            S, tau, r, q, sigma_atm, sigma_put, sigma_call, K_put, K_call, notional
        )
        self.active_positions['skew'] = position
        return position

    def establish_smile_position(self, S, tau, r, q, sigma_atm, sigma_put, sigma_call,
                                 K_put, K_call, notional=1e6):
        """
        Establish a smile trading position.

        Parameters:
        -----------
        S : float - Spot price
        tau : float - Time to maturity
        r, q : float - Rates
        sigma_* : float - Implied volatilities
        K_* : float - Strike prices
        notional : float - Notional amount to trade

        Returns:
        --------
        dict - Position details
        """
        position = self.smile_strategy.construct_position(
            S, tau, r, q, sigma_atm, sigma_put, sigma_call, K_put, K_call, notional
        )
        self.active_positions['smile'] = position
        return position

    def update_positions(self, S_new, tau_new, r_new=None, q_new=None,
                         sigma_atm_new=None, sigma_put_new=None, sigma_call_new=None):
        """
        Update all active positions for new market conditions.

        Parameters:
        -----------
        S_new : float - New spot price
        tau_new : float - New time to maturity
        r_new, q_new : float or None - New rates
        sigma_*_new : float or None - New implied volatilities

        Returns:
        --------
        dict - Updated positions
        """
        updated_positions = {}

        # Update vol position if active
        if self.active_positions['vol'] is not None:
            updated_positions['vol'] = self.vol_strategy.update_position(
                self.active_positions['vol'], S_new, tau_new, r_new, q_new, sigma_atm_new
            )
            self.active_positions['vol'] = updated_positions['vol']

        # Update skew position if active
        if self.active_positions['skew'] is not None:
            updated_positions['skew'] = self.skew_strategy.update_position(
                self.active_positions['skew'], S_new, tau_new, r_new, q_new,
                sigma_put_new, sigma_call_new, sigma_atm_new
            )
            self.active_positions['skew'] = updated_positions['skew']

        # Update smile position if active
        if self.active_positions['smile'] is not None:
            updated_positions['smile'] = self.smile_strategy.update_position(
                self.active_positions['smile'], S_new, tau_new, r_new, q_new,
                sigma_put_new, sigma_call_new, sigma_atm_new
            )
            self.active_positions['smile'] = updated_positions['smile']

        return updated_positions

    def calculate_expected_gain_rates(self, realized_moments):
        """
        Calculate expected gain rates for all active positions.

        Parameters:
        -----------
        realized_moments : dict - Contains realized moments (sigma_squared, gamma, omega_squared)

        Returns:
        --------
        dict - Expected gain rates for each active position
        """
        expected_gains = {}

        # Calculate for vol position
        if self.active_positions['vol'] is not None:
            expected_gains['vol'] = self.vol_strategy.calculate_expected_gain_rate(
                self.active_positions['vol'],
                realized_moments.get('sigma_squared', 0),
                self.active_positions['vol']['sigma_atm'] ** 2
            )

        # Calculate for skew position
        if self.active_positions['skew'] is not None:
            # Calculate implied skew from position
            l_diff = self.active_positions['skew']['l_plus_c'] - self.active_positions['skew']['l_plus_p']
            sigma_c_squared = self.active_positions['skew']['sigma_call'] ** 2
            sigma_p_squared = self.active_positions['skew']['sigma_put'] ** 2
            implied_skew = (sigma_c_squared / 2 - sigma_p_squared / 2) / l_diff

            expected_gains['skew'] = self.skew_strategy.calculate_expected_gain_rate(
                self.active_positions['skew'],
                realized_moments.get('gamma', 0),
                implied_skew
            )

        # Calculate for smile position
        if self.active_positions['smile'] is not None:
            # Use moment extractor to calculate implied convexity
            if 'K_put' in self.active_positions['smile'] and 'K_atm' in self.active_positions['smile'] and 'K_call' in \
                    self.active_positions['smile']:
                implied_convexity = self.moment_extractor.calculate_implied_convexity(
                    self.active_positions['smile']['K_put'],
                    self.active_positions['smile']['K_atm'],
                    self.active_positions['smile']['K_call'],
                    self.active_positions['smile']['tau'],
                    self.active_positions['smile']['S'],
                    self.active_positions['smile'].get('r', 0),
                    self.active_positions['smile'].get('q', 0)
                )
            else:
                implied_convexity = 0

            expected_gains['smile'] = self.smile_strategy.calculate_expected_gain_rate(
                self.active_positions['smile'],
                realized_moments.get('omega_squared', 0),
                implied_convexity
            )

        return expected_gains

    def attribute_pnl(self, initial_state, final_state, realized_moments):
        """
        Attribute P&L between two states.

        Parameters:
        -----------
        initial_state, final_state : dict - Contains option parameters and price
        realized_moments : dict - Contains realized σ², γ, ω²

        Returns:
        --------
        dict - P&L attribution to theta, delta, gamma, vega, vanna, volga
        """
        return self.pnl_attributor.attribute_pnl(initial_state, final_state, realized_moments)

    def validate_no_arbitrage(self, option_params, moments):
        """
        Validate the no-arbitrage condition for option pricing.

        Parameters:
        -----------
        option_params : dict - Option parameters
        moments : dict - Contains μ_t, σ²_t, γ_t, ω²_t moments

        Returns:
        --------
        dict - Validation results
        """
        return self.pnl_attributor.validate_no_arbitrage_condition(option_params, moments)

    def analyze_performance(self, returns, benchmark_returns=None, risk_free_rate=0):
        """
        Analyze performance of trading returns.

        Parameters:
        -----------
        returns : array or DataFrame - Trading returns
        benchmark_returns : array or None - Benchmark returns
        risk_free_rate : float - Annualized risk-free rate

        Returns:
        --------
        dict - Performance metrics
        """
        if isinstance(returns, pd.DataFrame):
            metrics = {}
            for col in returns.columns:
                sharpe = self.performance_analyzer.calculate_sharpe_ratio(
                    returns[col], risk_free_rate
                )
                ir = self.performance_analyzer.calculate_information_ratio(
                    returns[col], benchmark_returns
                )
                stats = self.performance_analyzer.calculate_pnl_statistics(returns[col])
                metrics[col] = {
                    'sharpe_ratio': sharpe,
                    'information_ratio': ir,
                    **stats
                }
            return metrics
        else:
            sharpe = self.performance_analyzer.calculate_sharpe_ratio(
                returns, risk_free_rate
            )
            ir = self.performance_analyzer.calculate_information_ratio(
                returns, benchmark_returns
            )
            stats = self.performance_analyzer.calculate_pnl_statistics(returns)
            return {
                'sharpe_ratio': sharpe,
                'information_ratio': ir,
                **stats
            }

    def forecast_moments(self, historical_data, horizon=21):
        """
        Forecast future moments using historical and extracted data.

        Parameters:
        -----------
        historical_data : DataFrame - Historical realized moments
        horizon : int - Forecast horizon in days

        Returns:
        --------
        dict - Forecasted moments
        """
        # Get the latest extracted moments
        if not self.extracted_moments:
            raise ValueError("No extracted moments available. Call extract_moments first.")

        extracted_moments = pd.concat(self.extracted_moments).reset_index(drop=True)
        extracted_moments = extracted_moments.drop_duplicates(['date', 'tau'])

        return self.regression_analyzer.forecast_moment_conditions(
            historical_data, extracted_moments, horizon
        )

    def calculate_breakeven_implied_variance(self, tau, moneyness, gamma_forecast, omega_sq_forecast, atm_var):
        """
        Calculate breakeven implied variance based on forecasts.

        Parameters:
        -----------
        tau : float - Time to maturity
        moneyness : float or array - Standardized moneyness
        gamma_forecast : float - Forecasted covariance
        omega_sq_forecast : float - Forecasted variance
        atm_var : float - ATM implied variance

        Returns:
        --------
        float or array - Breakeven implied variance
        """
        if self.moment_extractor is None:
            raise ValueError("System not initialized with data. Call initialize_with_data first.")

        return self.moment_extractor.calculate_breakeven_implied_variance(
            tau, moneyness, gamma_forecast, omega_sq_forecast, atm_var
        )

    def visualize_volatility_surface(self, S, r=0.0, q=0.0):
        """
        Visualize the volatility surface.

        Parameters:
        -----------
        S : float - Spot price
        r, q : float - Rates

        Returns:
        --------
        fig, ax - Matplotlib figure and axis objects
        """
        if self.vol_surface is None:
            raise ValueError("System not initialized with data. Call initialize_with_data first.")

        return self.visualizer.plot_volatility_surface(self.vol_surface, S, r, q)

    def visualize_volatility_smile(self, tau, S, r=0.0, q=0.0):
        """
        Visualize the volatility smile at a specific maturity.

        Parameters:
        -----------
        tau : float - Time to maturity
        S : float - Spot price
        r, q : float - Rates

        Returns:
        --------
        fig, ax - Matplotlib figure and axis objects
        """
        if self.vol_surface is None:
            raise ValueError("System not initialized with data. Call initialize_with_data first.")

        return self.visualizer.plot_volatility_smile(self.vol_surface, tau, S, r, q)

    def visualize_moment_extraction(self, S, r=0.0, q=0.0, tau=1 / 12):
        """
        Visualize moment extraction from the volatility smile.

        Parameters:
        -----------
        S : float - Spot price
        r, q : float - Rates
        tau : float - Time to maturity

        Returns:
        --------
        fig, ax - Matplotlib figure and axis objects
        """
        if self.moment_extractor is None:
            raise ValueError("System not initialized with data. Call initialize_with_data first.")

        return self.visualizer.plot_moment_extraction(self.moment_extractor, S, r, q, tau)

    def visualize_strategy_comparison(self, returns_df, cumulative=True):
        """
        Visualize comparison of strategy returns.

        Parameters:
        -----------
        returns_df : DataFrame - Daily returns for different strategies
        cumulative : bool - Whether to plot cumulative returns

        Returns:
        --------
        fig, ax - Matplotlib figure and axis objects
        """
        return self.visualizer.plot_strategy_comparison(returns_df, cumulative)