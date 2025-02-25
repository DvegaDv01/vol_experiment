import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
from matplotlib.gridspec import GridSpec


class VolatilityVisualizer:
    def __init__(self, figsize=(12, 8)):
        """
        Parameters:
        -----------
        figsize : tuple - Figure size for plots
        """
        self.figsize = figsize

    def plot_volatility_surface(self, vol_surface, S, r=0.0, q=0.0,
                                moneyness_range=(-2, 2), tau_range=(1 / 12, 1),
                                title="Implied Volatility Surface"):
        """Plot 3D volatility surface

        Parameters:
        -----------
        vol_surface : VolatilitySurface object
        S : float - Spot price
        r, q : float - Risk-free rate and dividend yield
        moneyness_range : tuple - Range of standardized moneyness to plot
        tau_range : tuple - Range of maturities to plot (in years)
        title : str - Title for the plot

        Returns:
        --------
        fig, ax - Matplotlib figure and axis objects
        """
        # Create grid of moneyness and maturities
        moneyness_points = np.linspace(moneyness_range[0], moneyness_range[1], 50)
        tau_points = np.linspace(tau_range[0], tau_range[1], 20)

        X, Y = np.meshgrid(moneyness_points, tau_points)
        Z = np.zeros_like(X)

        # Get ATM volatilities for each maturity
        atm_vols = vol_surface.get_atm_volatility_term_structure(S, r, q, tau_points)

        # Fill Z with implied volatilities
        for i, tau in enumerate(tau_points):
            atm_vol = atm_vols.loc[atm_vols['tau'] == tau, 'A_t'].values[0]
            if np.isnan(atm_vol):
                continue

            F = S * np.exp((r - q) * tau)

            for j, x in enumerate(moneyness_points):
                # Convert standardized moneyness to strike
                K = F * np.exp(x * atm_vol * np.sqrt(tau) - 0.5 * atm_vol ** 2 * tau)

                # Get implied volatility
                impl_vol = vol_surface.get_implied_volatility(K, tau, S, r, q)
                Z[i, j] = impl_vol

        # Create 3D plot
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection='3d')

        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, alpha=0.8)

        # Add labels and title
        ax.set_xlabel('Standardized Moneyness')
        ax.set_ylabel('Time to Maturity (years)')
        ax.set_zlabel('Implied Volatility')
        ax.set_title(title)

        # Add colorbar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

        return fig, ax

    def plot_volatility_smile(self, vol_surface, tau, S, r=0.0, q=0.0,
                              moneyness_range=(-2, 2), title=None):
        """Plot volatility smile at a specific maturity

        Parameters:
        -----------
        vol_surface : VolatilitySurface object
        tau : float - Time to maturity
        S : float - Spot price
        r, q : float - Risk-free rate and dividend yield
        moneyness_range : tuple - Range of standardized moneyness to plot
        title : str or None - Title for the plot

        Returns:
        --------
        fig, ax - Matplotlib figure and axis objects
        """
        # Create grid of moneyness points
        moneyness_points = np.linspace(moneyness_range[0], moneyness_range[1], 100)

        # Get ATM volatility
        atm_data = vol_surface.get_atm_volatility_term_structure(S, r, q, [tau])
        if len(atm_data) == 0 or atm_data.iloc[0]['A_t'] is None:
            raise ValueError(f"No ATM volatility available for tau={tau}")

        atm_vol = atm_data.iloc[0]['A_t']
        F = S * np.exp((r - q) * tau)

        # Calculate implied volatilities across moneyness
        strikes = []
        impl_vols = []

        for x in moneyness_points:
            # Convert standardized moneyness to strike
            K = F * np.exp(x * atm_vol * np.sqrt(tau) - 0.5 * atm_vol ** 2 * tau)
            strikes.append(K)

            # Get implied volatility
            impl_vol = vol_surface.get_implied_volatility(K, tau, S, r, q)
            impl_vols.append(impl_vol)

        # Create plot
        fig, ax = plt.subplots(figsize=self.figsize)

        ax.plot(moneyness_points, impl_vols, 'b-', linewidth=2)

        # Plot ATM volatility for reference
        ax.axhline(y=atm_vol, color='r', linestyle='--', alpha=0.5)
        ax.scatter(0, atm_vol, color='r', s=50, zorder=5)

        # Add labels and title
        ax.set_xlabel('Standardized Moneyness')
        ax.set_ylabel('Implied Volatility')
        if title is None:
            title = f'Volatility Smile at τ = {tau:.2f}'
        ax.set_title(title)

        ax.grid(True, alpha=0.3)

        return fig, ax

    def plot_strategy_comparison(self, returns_df, cumulative=True,
                                 risk_free_rate=0, title="Strategy Comparison"):
        """Plot comparison of strategy returns

        Parameters:
        -----------
        returns_df : DataFrame - Daily returns for different strategies
        cumulative : bool - Whether to plot cumulative returns
        risk_free_rate : float - Annualized risk-free rate
        title : str - Title for the plot

        Returns:
        --------
        fig, ax - Matplotlib figure and axis objects
        """
        from ..analysis.performance_metrics import PerformanceAnalyzer
        analyzer = PerformanceAnalyzer()

        # Calculate performance metrics
        metrics = {}
        for col in returns_df.columns:
            sharpe = analyzer.calculate_sharpe_ratio(returns_df[col], risk_free_rate)
            metrics[col] = {
                'Sharpe': sharpe,
                'Mean': returns_df[col].mean() * 252,  # Annualized
                'Std': returns_df[col].std() * np.sqrt(252)  # Annualized
            }

        # Create cumulative returns if requested
        if cumulative:
            cum_returns = (1 + returns_df).cumprod() - 1

        # Create figure
        fig = plt.figure(figsize=self.figsize)
        gs = GridSpec(2, 2, height_ratios=[3, 1], figure=fig)

        # Plot returns/cumulative returns
        ax1 = fig.add_subplot(gs[0, :])

        if cumulative:
            for col in cum_returns.columns:
                ax1.plot(cum_returns.index, cum_returns[col], label=f"{col}")
            ax1.set_ylabel('Cumulative Return')
        else:
            for col in returns_df.columns:
                ax1.plot(returns_df.index, returns_df[col].rolling(21).mean(), label=f"{col}")
            ax1.set_ylabel('21-Day Rolling Average Return')

        ax1.set_title(title)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot metrics table
        ax2 = fig.add_subplot(gs[1, :])
        ax2.axis('off')

        metric_df = pd.DataFrame(metrics).T
        metric_df = metric_df[['Sharpe', 'Mean', 'Std']]

        # Format metrics table
        cell_text = []
        for idx, row in metric_df.iterrows():
            cell_text.append([f"{idx}", f"{row['Sharpe']:.2f}",
                              f"{row['Mean'] * 100:.2f}%", f"{row['Std'] * 100:.2f}%"])

        table = ax2.table(cellText=cell_text,
                          colLabels=['Strategy', 'Sharpe', 'Ann. Return', 'Ann. Volatility'],
                          loc='center', cellLoc='center')

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)

        plt.tight_layout()

        return fig, ax1

    def plot_moment_extraction(self, moment_extractor, S, r=0.0, q=0.0, tau=1 / 12,
                               moneyness_range=(-1, 1), title=None):
        """Plot moment extraction from the volatility smile

        Parameters:
        -----------
        moment_extractor : MomentExtractor object
        S : float - Spot price
        r, q : float - Risk-free rate and dividend yield
        tau : float - Time to maturity
        moneyness_range : tuple - Range of standardized moneyness for extraction
        title : str or None - Title for the plot

        Returns:
        --------
        fig, ax - Matplotlib figure and axis objects
        """
        # Extract variance and covariance from smile
        gamma, omega_sq, r_squared, residuals = moment_extractor.extract_variance_covariance_from_smile(
            tau, S, r, q, moneyness_range
        )

        # Get volatility surface
        vol_surface = moment_extractor.vol_surface

        # Get floating series for this maturity
        floating_series = vol_surface.calculate_floating_series(
            S, r, q,
            moneyness_levels=np.linspace(moneyness_range[0], moneyness_range[1], 21),
            tau_levels=[tau]
        )

        if len(floating_series) == 0:
            raise ValueError(f"No data available for tau={tau}")

        # Get ATM volatility
        atm_data = floating_series[floating_series['x'] == 0]
        if len(atm_data) == 0:
            raise ValueError(f"No ATM data available for tau={tau}")

        A_t = atm_data.iloc[0]['impl_vol']
        A_t_squared = A_t ** 2

        # Calculate implied variance spreads
        floating_series['impl_var'] = floating_series['impl_vol'] ** 2
        floating_series['var_spread'] = floating_series['impl_var'] - A_t_squared

        # Calculate model prediction
        floating_series['z_plus_times_z_minus'] = floating_series['z_plus'] * floating_series['z_minus']
        floating_series['model_spread'] = 2 * gamma * floating_series['z_plus'] + omega_sq * floating_series[
            'z_plus_times_z_minus']

        # Create plot
        fig, ax = plt.subplots(figsize=self.figsize)

        # Plot data points
        ax.scatter(floating_series['x'], floating_series['var_spread'],
                   label='Data Points', color='blue', alpha=0.7)

        # Plot model fit
        ax.plot(floating_series['x'], floating_series['model_spread'],
                label=f'Model Fit: γ={gamma:.6f}, ω²={omega_sq:.6f}, R²={r_squared:.3f}',
                color='red', linewidth=2)

        # Add labels and title
        ax.set_xlabel('Standardized Moneyness')
        ax.set_ylabel('Implied Variance Spread (I² - A²)')
        if title is None:
            title = f'Moment Extraction from Volatility Smile at τ = {tau:.2f}'
        ax.set_title(title)

        ax.grid(True, alpha=0.3)
        ax.legend()

        return fig, ax