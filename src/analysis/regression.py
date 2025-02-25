import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


class RegressionAnalyzer:
    def __init__(self):
        pass

    def cross_sectional_regression(self, implied_var_spreads, moneyness_data, constrain_variance=True):
        """Perform cross-sectional regression for variance/covariance extraction

        Parameters:
        -----------
        implied_var_spreads : array - I²_t - A²_t values
        moneyness_data : DataFrame - Contains z_plus and z_plus*z_minus columns
        constrain_variance : bool - Whether to constrain variance to be non-negative

        Returns:
        --------
        tuple - (γ_t, ω²_t, R², coefficients, residuals)
        """
        # Prepare data
        if isinstance(moneyness_data, pd.DataFrame):
            if 'z_plus' in moneyness_data.columns and 'z_minus' in moneyness_data.columns:
                X = np.column_stack([
                    2 * moneyness_data['z_plus'],
                    moneyness_data['z_plus'] * moneyness_data['z_minus']
                ])
            else:
                raise ValueError("moneyness_data must contain 'z_plus' and 'z_minus' columns")
        else:
            X = moneyness_data

        y = implied_var_spreads

        # Handle case with just one observation
        if len(y) < 2:
            return np.nan, np.nan, np.nan, None, None

        if constrain_variance:
            # Constrained regression (non-negative variance)
            from scipy import optimize

            # Define objective function
            def objective_func(params):
                gamma, omega_sq = params
                omega_sq = max(0, omega_sq)  # Ensure non-negative variance

                y_pred = 2 * gamma * X[:, 0] + omega_sq * X[:, 1]
                return np.sum((y - y_pred) ** 2)

            # Perform constrained optimization
            initial_guess = [0.0, 0.01]  # Initial gamma and omega_sq

            result = optimize.minimize(
                objective_func,
                initial_guess,
                method='L-BFGS-B',
                bounds=[None, (0, None)]  # No constraint on gamma, non-negative omega_sq
            )

            gamma_t, omega_sq_t = result.x
            omega_sq_t = max(0, omega_sq_t)  # Ensure non-negative variance

            # Calculate R² and residuals
            y_pred = 2 * gamma_t * X[:, 0] + omega_sq_t * X[:, 1]
            residuals = y - y_pred

            ss_total = np.sum((y - np.mean(y)) ** 2)
            ss_residual = np.sum(residuals ** 2)

            r_squared = 1 - (ss_residual / ss_total) if ss_total > 0 else np.nan

            coefs = np.array([gamma_t, omega_sq_t])

        else:
            # Unconstrained OLS regression
            X_with_const = sm.add_constant(X) if len(X.shape) > 1 and X.shape[1] > 0 else sm.add_constant(
                X.reshape(-1, 1))

            try:
                model = sm.OLS(y, X_with_const)
                results = model.fit()

                # Extract coefficients (intercept, gamma, omega_sq)
                coefs = results.params

                if len(coefs) == 3:
                    # With intercept
                    gamma_t = coefs[1]
                    omega_sq_t = coefs[2]
                elif len(coefs) == 2:
                    # No intercept or only one X variable
                    if X.shape[1] == 1:
                        gamma_t = coefs[1]
                        omega_sq_t = 0
                    else:
                        gamma_t = coefs[0]
                        omega_sq_t = coefs[1]
                else:
                    gamma_t = 0
                    omega_sq_t = 0

                r_squared = results.rsquared
                residuals = results.resid

            except:
                # Handle regression errors
                gamma_t = np.nan
                omega_sq_t = np.nan
                r_squared = np.nan
                residuals = np.full_like(y, np.nan)
                coefs = np.array([np.nan, np.nan])

        return gamma_t, omega_sq_t, r_squared, coefs, residuals

    def time_series_regression(self, y, X, newey_west_lags=21):
        """Perform time series regression with Newey-West standard errors

        Parameters:
        -----------
        y : array - Dependent variable
        X : array - Independent variables
        newey_west_lags : int - Number of lags for Newey-West standard errors

        Returns:
        --------
        dict - Regression statistics and coefficients
        """
        # Convert to arrays if needed
        y = np.asarray(y)
        X = np.asarray(X)

        # Reshape X if needed
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Add constant
        X_with_const = sm.add_constant(X)

        # Handle missing values
        valid_idx = ~np.isnan(y)
        if X.ndim > 1:
            valid_idx = valid_idx & ~np.any(np.isnan(X), axis=1)
        else:
            valid_idx = valid_idx & ~np.isnan(X)

        if np.sum(valid_idx) < 2:
            return {
                'coefficients': np.full(X.shape[1] + 1, np.nan),
                't_stats': np.full(X.shape[1] + 1, np.nan),
                'p_values': np.full(X.shape[1] + 1, np.nan),
                'r_squared': np.nan,
                'adj_r_squared': np.nan,
                'residuals': np.full_like(y, np.nan)
            }

        y_valid = y[valid_idx]
        X_valid = X_with_const[valid_idx]

        # Run regression
        model = sm.OLS(y_valid, X_valid)

        # Use Newey-West standard errors
        results = model.fit(cov_type='HAC', cov_kwds={'maxlags': newey_west_lags})

        return {
            'coefficients': results.params,
            't_stats': results.tvalues,
            'p_values': results.pvalues,
            'r_squared': results.rsquared,
            'adj_r_squared': results.rsquared_adj,
            'residuals': results.resid,
            'full_results': results
        }

    def forecast_moment_conditions(self, historical_data, extracted_moments, horizon=21):
        """Create forecasts for variance and covariance

        Parameters:
        -----------
        historical_data : DataFrame - Historical variance and covariance
        extracted_moments : DataFrame - Cross-sectionally extracted moments
        horizon : int - Forecast horizon in days

        Returns:
        --------
        dict - Forecasted moment conditions
        """
        # Need both datasets
        if historical_data is None or extracted_moments is None:
            return None

        # Ensure we have common dates
        common_dates = historical_data.index.intersection(extracted_moments.index)

        if len(common_dates) == 0:
            return None

        # Prepare datasets
        hist = historical_data.loc[common_dates]
        ext = extracted_moments.loc[common_dates]

        # Forecasting regression for variance
        if 'variance' in hist.columns and 'omega_squared' in ext.columns:
            # Create target: future realized variance
            future_var = hist['variance'].shift(-horizon)

            # Run regression with both predictors
            X = np.column_stack([hist['variance'], ext['omega_squared']])
            var_reg = self.time_series_regression(future_var.dropna(), X[:-horizon])

            var_coefs = var_reg['coefficients']

            # Create forecast using latest data
            latest_hist_var = hist['variance'].iloc[-1]
            latest_ext_var = ext['omega_squared'].iloc[-1]

            var_forecast = var_coefs[0] + var_coefs[1] * latest_hist_var + var_coefs[2] * latest_ext_var
        else:
            var_forecast = np.nan
            var_reg = None

        # Forecasting regression for covariance
        if 'covariance' in hist.columns and 'gamma' in ext.columns:
            # Create target: future realized covariance
            future_cov = hist['covariance'].shift(-horizon)

            # Run regression with both predictors
            X = np.column_stack([hist['covariance'], ext['gamma']])
            cov_reg = self.time_series_regression(future_cov.dropna(), X[:-horizon])

            cov_coefs = cov_reg['coefficients']

            # Create forecast using latest data
            latest_hist_cov = hist['covariance'].iloc[-1]
            latest_ext_cov = ext['gamma'].iloc[-1]

            cov_forecast = cov_coefs[0] + cov_coefs[1] * latest_hist_cov + cov_coefs[2] * latest_ext_cov
        else:
            cov_forecast = np.nan
            cov_reg = None

        return {
            'variance_forecast': var_forecast,
            'covariance_forecast': cov_forecast,
            'variance_regression': var_reg,
            'covariance_regression': cov_reg
        }