import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from volatility_trading import VolatilityTradingSystem


def load_sample_data():
    """Load sample option data for demonstration"""
    # In a real application, this would load actual market data
    # For demonstration, we'll create synthetic data

    # Create a range of strikes around current spot
    S = 100
    strikes = np.linspace(80, 120, 21)

    # Create a range of maturities
    maturities = [1 / 12, 2 / 12, 3 / 12, 6 / 12, 1.0]  # In years

    # Create a volatility surface with a skew
    data = []
    for tau in maturities:
        for K in strikes:
            # Create a volatility skew
            moneyness = np.log(K / S)
            base_vol = 0.2  # Base ATM volatility of 20%
            skew = -0.1 * moneyness  # Negative skew
            smile = 0.05 * moneyness ** 2  # Positive smile/convexity

            impl_vol = base_vol + skew + smile

            # Add some randomness
            impl_vol = max(0.05, impl_vol + np.random.normal(0, 0.01))

            # Create synthetic option data
            data.append({
                'date': pd.Timestamp.now().date(),
                'spot': S,
                'strike': K,
                'tau': tau,
                'r': 0.02,  # Risk-free rate
                'q': 0.01,  # Dividend yield
                'impl_vol': impl_vol,
                'option_type': 'call' if K >= S else 'put',
                'forward': S * np.exp((0.02 - 0.01) * tau)
            })

    return pd.DataFrame(data)


def main():
    """Main example usage function"""
    # Load sample data
    option_data = load_sample_data()

    # Initialize the volatility trading system
    system = VolatilityTradingSystem(option_data)

    # Visualize the current volatility surface
    fig_surface, _ = system.visualize_volatility_surface(S=100, r=0.02, q=0.01)
    fig_surface.savefig('volatility_surface.png')

    # Visualize the volatility smile at the 3-month maturity
    fig_smile, _ = system.visualize_volatility_smile(tau=3 / 12, S=100, r=0.02, q=0.01)
    fig_smile.savefig('volatility_smile_3month.png')

    # Extract moments from the volatility surface
    moments = system.extract_moments(
        S=100, r=0.02, q=0.01,
        tau_levels=[1 / 12, 2 / 12, 3 / 12, 6 / 12, 1.0]
    )
    print("Extracted Moments:")
    print(moments[['tau', 'mu', 'gamma', 'omega_squared']].round(6))

    # Visualize moment extraction
    fig_moments, _ = system.visualize_moment_extraction(S=100, r=0.02, q=0.01, tau=3 / 12)
    fig_moments.savefig('moment_extraction.png')

    # Establish trading positions
    print("\nEstablishing Trading Positions:")

    # Vol trade (ATM straddle)
    vol_position = system.establish_vol_position(
        S=100, tau=3 / 12, r=0.02, q=0.01, sigma_atm=0.2, notional=1e6
    )
    print("Vol Position: ATM Strike =", vol_position['strike_atm'])
    print("Number of Straddles =", vol_position['num_straddles'])
    print("Delta Hedge =", vol_position['delta_hedge'])

    # Skew trade (risk-reversal with cash vega hedging)
    K_put = 90  # 90% of spot
    K_call = 110  # 110% of spot

    # Get implied volatilities at these strikes
    sigma_put = system.vol_surface.get_implied_volatility(K_put, 3 / 12, 100, 0.02, 0.01)
    sigma_call = system.vol_surface.get_implied_volatility(K_call, 3 / 12, 100, 0.02, 0.01)
    sigma_atm = system.vol_surface.get_implied_volatility(100, 3 / 12, 100, 0.02, 0.01)

    skew_position = system.establish_skew_position(
        S=100, tau=3 / 12, r=0.02, q=0.01,
        sigma_atm=sigma_atm, sigma_put=sigma_put, sigma_call=sigma_call,
        K_put=K_put, K_call=K_call, notional=1e6
    )

    if skew_position:
        print("\nSkew Position:")
        print("Put Strike =", skew_position['K_put'])
        print("Call Strike =", skew_position['K_call'])
        print("Number of Puts =", skew_position['num_puts'])
        print("Number of Calls =", skew_position['num_calls'])
        print("Number of Straddles (for vega hedging) =", skew_position['num_straddles'])
        print("Delta Hedge =", skew_position['delta_hedge'])

    # Smile trade (butterfly with cash vega hedging)
    smile_position = system.establish_smile_position(
        S=100, tau=3 / 12, r=0.02, q=0.01,
        sigma_atm=sigma_atm, sigma_put=sigma_put, sigma_call=sigma_call,
        K_put=K_put, K_call=K_call, notional=1e6
    )

    if smile_position:
        print("\nSmile Position:")
        print("Put Strike =", smile_position['K_put'])
        print("ATM Strike =", smile_position['K_atm'])
        print("Call Strike =", smile_position['K_call'])
        print("Number of Puts =", smile_position['num_puts'])
        print("Number of Calls =", smile_position['num_calls'])
        print("Number of Straddles (for vega hedging) =", smile_position['num_straddles'])
        print("Delta Hedge =", smile_position['delta_hedge'])

    # Simulate some realized moments for P&L attribution
    realized_moments = {
        'sigma_squared': 0.04,  # 20% annualized volatility squared
        'gamma': -0.005,  # Negative correlation between returns and volatility
        'omega_squared': 0.01,  # Volatility of volatility
        'mu': 0.2  # Expected rate of change in implied volatility
    }

    # Calculate expected gain rates
    expected_gains = system.calculate_expected_gain_rates(realized_moments)

    print("\nExpected Gain Rates:")
    if 'vol' in expected_gains:
        print("Vol Strategy: Expected Gain Rate =", expected_gains['vol']['expected_gain_rate'])
    if 'skew' in expected_gains:
        print("Skew Strategy: Expected Gain Rate =", expected_gains['skew']['expected_gain_rate'])
    if 'smile' in expected_gains:
        print("Smile Strategy: Expected Gain Rate =", expected_gains['smile']['expected_gain_rate'])

    # Simulate market move for position updates
    print("\nSimulating Market Move:")
    S_new = 102  # 2% increase in spot
    tau_new = 3 / 12 - 1 / 252  # One day less

    # Update positions
    updated_positions = system.update_positions(
        S_new=S_new, tau_new=tau_new,
        r_new=0.02, q_new=0.01,
        sigma_atm_new=0.21  # Slight increase in volatility
    )

    if 'vol' in updated_positions:
        print("Updated Vol Position: Delta Hedge Adjustment =",
              updated_positions['vol']['delta_hedge_adjustment'])

    # P&L attribution
    initial_state = {
        'S': 100,
        'K': 100,
        'tau': 3 / 12,
        'r': 0.02,
        'q': 0.01,
        'impl_vol': 0.2,
        'option_type': 'call'
    }

    final_state = {
        'S': 102,
        'K': 100,
        'tau': 3 / 12 - 1 / 252,
        'r': 0.02,
        'q': 0.01,
        'impl_vol': 0.21,
        'option_type': 'call'
    }

    pnl_attribution = system.attribute_pnl(initial_state, final_state, realized_moments)

    print("\nP&L Attribution:")
    print("Total P&L =", pnl_attribution['total_pnl'])
    print("Theta P&L =", pnl_attribution['theta_pnl'])
    print("Delta P&L =", pnl_attribution['delta_pnl'])
    print("Gamma P&L =", pnl_attribution['gamma_pnl'])
    print("Vega P&L =", pnl_attribution['vega_pnl'])
    print("Vanna P&L =", pnl_attribution['vanna_pnl'])
    print("Volga P&L =", pnl_attribution['volga_pnl'])

    # Validate no-arbitrage condition
    arbitrage_check = system.validate_no_arbitrage(initial_state, realized_moments)

    print("\nNo-Arbitrage Condition Check:")
    print("Condition Satisfied?", arbitrage_check['is_satisfied'])
    print("LHS (negative theta) =", arbitrage_check['lhs'])
    print("RHS (sum of other terms) =", arbitrage_check['rhs'])

    # Simulate strategy returns for performance analysis
    np.random.seed(42)  # For reproducibility
    dates = pd.date_range(start='2022-01-01', periods=252)  # 1 year of daily data

    # Simulate returns for each strategy
    vol_returns = np.random.normal(-0.0001, 0.005, len(dates))  # Slightly negative mean
    skew_returns = np.random.normal(0.0002, 0.004, len(dates))  # Positive mean
    smile_returns = np.random.normal(-0.0001, 0.006, len(dates))  # Slightly negative mean

    # Create returns DataFrame
    returns_df = pd.DataFrame({
        'Vol': vol_returns,
        'Skew': skew_returns,
        'Smile': smile_returns
    }, index=dates)

    # Analyze performance
    performance = system.analyze_performance(returns_df)

    print("\nPerformance Analysis:")
    for strategy, metrics in performance.items():
        print(f"\n{strategy} Strategy:")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Annualized Mean: {metrics['mean'] * 252:.2%}")
        print(f"Annualized Std: {metrics['std'] * np.sqrt(252):.2%}")
        print(f"Positive Days: {metrics['positive_pct']:.2f}%")

    # Visualize strategy comparison
    fig_perf, _ = system.visualize_strategy_comparison(returns_df)
    fig_perf.savefig('strategy_comparison.png')


if __name__ == "__main__":
    main()