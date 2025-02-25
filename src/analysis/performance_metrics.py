import numpy as np
import pandas as pd
from scipy import stats


class PerformanceAnalyzer:
    def __init__(self):
        pass

    def calculate_sharpe_ratio(self, returns, risk_free_rate=0, annualization_factor=252):
        """Calculate annualized Sharpe ratio

        Parameters:
        -----------
        returns : array - Daily returns
        risk_free_rate : float - Annualized risk-free rate
        annualization_factor : int - Factor to annualize (252 for daily returns)

        Returns:
        --------
        float - Sharpe ratio
        """
        if len(returns) == 0:
            return np.nan

        daily_excess_returns = returns - risk_free_rate / annualization_factor

        mean_excess_return = np.mean(daily_excess_returns)
        std_excess_return = np.std(daily_excess_returns, ddof=1)

        if std_excess_return == 0:
            return np.nan

        daily_sharpe = mean_excess_return / std_excess_return
        annualized_sharpe = daily_sharpe * np.sqrt(annualization_factor)

        return annualized_sharpe

    def calculate_information_ratio(self, returns, benchmark_returns=None, annualization_factor=252):
        """Calculate information ratio

        Parameters:
        -----------
        returns : array - Daily returns
        benchmark_returns : array or None - Benchmark returns
        annualization_factor : int - Factor to annualize (252 for daily returns)

        Returns:
        --------
        float - Information ratio
        """
        if len(returns) == 0:
            return np.nan

        if benchmark_returns is None:
            # Without benchmark, just use standard deviation like Sharpe
            mean_return = np.mean(returns)
            std_return = np.std(returns, ddof=1)

            if std_return == 0:
                return np.nan

            daily_ir = mean_return / std_return
        else:
            # With benchmark, use tracking error
            if len(benchmark_returns) != len(returns):
                raise ValueError("Benchmark returns must have same length as returns")

            excess_returns = returns - benchmark_returns
            mean_excess = np.mean(excess_returns)
            tracking_error = np.std(excess_returns, ddof=1)

            if tracking_error == 0:
                return np.nan

            daily_ir = mean_excess / tracking_error

        annualized_ir = daily_ir * np.sqrt(annualization_factor)

        return annualized_ir

    def calculate_pnl_statistics(self, pnl_series):
        """Calculate various statistics for P&L series

        Parameters:
        -----------
        pnl_series : array - P&L values

        Returns:
        --------
        dict - Contains mean, std, skew, kurtosis, etc.
        """
        if len(pnl_series) == 0:
            return {
                'mean': np.nan,
                'std': np.nan,
                'skew': np.nan,
                'kurtosis': np.nan,
                'min': np.nan,
                'max': np.nan,
                'positive_pct': np.nan,
                'drawdown': np.nan
            }

        # Calculate general statistics
        mean = np.mean(pnl_series)
        std = np.std(pnl_series, ddof=1)
        skewness = stats.skew(pnl_series)
        kurtosis = stats.kurtosis(pnl_series)
        minimum = np.min(pnl_series)
        maximum = np.max(pnl_series)

        # Calculate percentage of positive P&L days
        positive_pct = np.mean(pnl_series > 0) * 100

        # Calculate max drawdown
        cumulative = np.cumsum(pnl_series)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = np.min(cumulative - running_max)

        return {
            'mean': mean,
            'std': std,
            'skew': skewness,
            'kurtosis': kurtosis,
            'min': minimum,
            'max': maximum,
            'positive_pct': positive_pct,
            'drawdown': drawdown
        }

    def calculate_risk_premium(self, strategy_returns, moment_swaps=None):
        """Calculate risk premium embedded in strategy returns

        Parameters:
        -----------
        strategy_returns : DataFrame - Returns for vol/skew/smile strategies
        moment_swaps : DataFrame - Returns on corresponding moment swaps (optional)

        Returns:
        --------
        dict - Risk premium estimates with statistical significance
        """
        # Convert to DataFrame if necessary
        if not isinstance(strategy_returns, pd.DataFrame):
            strategy_returns = pd.DataFrame(strategy_returns)

        result = {}

        # Calculate mean returns and t-stats
        for col in strategy_returns.columns:
            returns = strategy_returns[col].dropna()

            if len(returns) == 0:
                result[col] = {
                    'mean': np.nan,
                    't_stat': np.nan,
                    'p_value': np.nan
                }
                continue

            mean_return = np.mean(returns)
            std_error = np.std(returns, ddof=1) / np.sqrt(len(returns))
            t_stat = mean_return / std_error
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(returns) - 1))

            result[col] = {
                'mean': mean_return,
                't_stat': t_stat,
                'p_value': p_value
            }

        # If moment swaps are provided, calculate hedged returns
        if moment_swaps is not None:
            for col in strategy_returns.columns:
                if col + '_swap' in moment_swaps.columns:
                    swap_returns = moment_swaps[col + '_swap']
                    common_index = strategy_returns.index.intersection(swap_returns.index)

                    if len(common_index) == 0:
                        continue

                    hedged_returns = strategy_returns.loc[common_index, col] - swap_returns.loc[common_index]

                    mean_return = np.mean(hedged_returns)
                    std_error = np.std(hedged_returns, ddof=1) / np.sqrt(len(hedged_returns))
                    t_stat = mean_return / std_error
                    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(hedged_returns) - 1))

                    result[col + '_hedged'] = {
                        'mean': mean_return,
                        't_stat': t_stat,
                        'p_value': p_value
                    }

        return result