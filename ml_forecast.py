# ml_forecast.py
# Machine Learning Forecasting System

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

class MLForecaster:
    """
    ML forecaster for time series volume prediction.
    Implements Linear Regression, Random Forest, and XGBoost.
    """

    def __init__(self, data):
        """
        Initialize with historical volume data.
        :param data: DataFrame with 'volume' column and datetime index
        """
        self.data = data.copy()
        self.models = {}
        self.prediction_horizon = 365  # forecast 1 year by default

    def prepare_features(self, df):
        """
        Create lag features for ML models
        """
        df_feat = df.copy()
        for lag in range(1, 8):
            df_feat[f'lag_{lag}'] = df_feat['volume'].shift(lag)
        df_feat.dropna(inplace=True)
        X = df_feat[[f'lag_{lag}' for lag in range(1, 8)]]
        y = df_feat['volume']
        return X, y

    def train_models(self):
        """
        Train all ML models on historical data
        """
        X, y = self.prepare_features(self.data)

        # Linear Regression
        lr = LinearRegression()
        lr.fit(X, y)
        self.models['linear'] = lr

        # Random Forest
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        self.models['rf'] = rf

        # XGBoost
        xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42, objective='reg:squarederror')
        xgb_model.fit(X, y)
        self.models['xgb'] = xgb_model

        print("✓ ML models trained: Linear Regression, Random Forest, XGBoost")

    def forecast_future(self):
        """
        Forecast future volumes for the prediction horizon
        Returns a DataFrame with columns: date, linear, rf, xgb, lower_bound, upper_bound
        """
        last_date = self.data.index.max()
        last_values = self.data['volume'][-7:].tolist()
        forecasts = []

        for i in range(self.prediction_horizon):
            X_pred = np.array(last_values[-7:]).reshape(1, -1)
            linear_pred = self.models['linear'].predict(X_pred)[0]
            rf_pred = self.models['rf'].predict(X_pred)[0]
            xgb_pred = self.models['xgb'].predict(X_pred)[0]

            # Confidence bounds (simple heuristic: +/-10% of ensemble mean)
            ensemble_mean = np.mean([linear_pred, rf_pred, xgb_pred])
            lower_bound = ensemble_mean * 0.9
            upper_bound = ensemble_mean * 1.1

            forecasts.append({
                'date': last_date + pd.Timedelta(days=i + 1),
                'linear': linear_pred,
                'rf': rf_pred,
                'xgb': xgb_pred,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            })

            # Update last_values for next prediction
            last_values.append(ensemble_mean)

        forecast_df = pd.DataFrame(forecasts)
        return forecast_df

    def evaluate_models(self):
        """
        Optional: Evaluate models on historical data
        Returns a dictionary with MAE and RMSE
        """
        X, y = self.prepare_features(self.data)
        metrics = {}
        for name, model in self.models.items():
            y_pred = model.predict(X)
            mae = mean_absolute_error(y, y_pred)
            rmse = mean_squared_error(y, y_pred, squared=False)
            metrics[name] = {'MAE': mae, 'RMSE': rmse}
        return metrics
