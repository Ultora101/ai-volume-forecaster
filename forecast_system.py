# forecast_system.py
# Quantitative AI Statistical Forecasting System - Part 1

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class QuantitativeForecaster:
    """
    Quantitative AI forecasting system using statistical mathematics
    Implements numerical methods and probability theory from Chapter 15
    Focuses on mathematical modeling and statistical inference
    """

    def __init__(self, data_path):
        """Initialize with numerical historical data"""

        self.df = pd.read_csv(data_path)
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df.set_index('date', inplace=True)

        # American flag colors for visualizations
        self.colors = {
            'red': '#B22234',
            'white': '#FFFFFF',
            'blue': '#3C3B6E'
        }

        print("\033[91m" + "=" * 60)
        print("\033[97m" + " 📊 QUANTITATIVE AI FORECASTING SYSTEM 📊")
        print("\033[94m" + "=" * 60 + "\033[0m")
        print(f"Loaded {len(self.df)} numerical data points")
        print(f"Data dimensionality: {self.df.shape}")

    def quantitative_analysis(self):
        """Perform comprehensive quantitative analysis"""

        print("\n\033[96m🔍 QUANTITATIVE PATTERN ANALYSIS...\033[0m")

        # Calculate statistical moments
        stats_dict = {
            'mean': self.df['volume'].mean(),
            'variance': self.df['volume'].var(),
            'std_dev': self.df['volume'].std(),
            'skewness': stats.skew(self.df['volume']),
            'kurtosis': stats.kurtosis(self.df['volume']),
            'cv': self.df['volume'].std() / self.df['volume'].mean(),
            'trend_coefficient': np.polyfit(range(len(self.df)), self.df['volume'], 1)[0]
        }

        print(f"Statistical Moments:")
        print(f" Mean (μ): {stats_dict['mean']:.2f}")
        print(f" Variance (σ²): {stats_dict['variance']:.2f}")
        print(f" Std Dev (σ): {stats_dict['std_dev']:.2f}")
        print(f" Skewness: {stats_dict['skewness']:.3f}")
        print(f" Kurtosis: {stats_dict['kurtosis']:.3f}")
        print(f" Coefficient of Variation: {stats_dict['cv']:.3f}")
        print(f" Linear Trend (β): {stats_dict['trend_coefficient']:.3f} units/day")

        # Quantitative day-of-week analysis
        print("\n📊 Numerical Weekly Patterns:")

        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        weekly_stats = self.df.groupby(self.df.index.dayofweek)['volume'].agg(['mean', 'std'])

        for i, day in enumerate(days):
            mean_val = weekly_stats.iloc[i]['mean']
            std_val = weekly_stats.iloc[i]['std']
            bar = '█' * int(mean_val / 50)
            print(f" {day}: {bar} μ={mean_val:.0f}, σ={std_val:.0f}")

        return stats_dict

    def moving_average_forecast(self, window=7):
        """Simple moving average forecast"""

        print(f"\n\033[93m📈 MOVING AVERAGE FORECAST (Window={window})\033[0m")

        # Calculate moving average
        self.df[f'MA_{window}'] = self.df['volume'].rolling(window=window).mean()

        # Use last MA value as forecast
        last_ma = self.df[f'MA_{window}'].iloc[-1]

        # Calculate accuracy
        mae = mean_absolute_error(
            self.df['volume'][window:],
            self.df[f'MA_{window}'][window:].shift(1).dropna()
        )

        print(f"Next period forecast: {last_ma:.0f} units")
        print(f"Historical MAE: {mae:.2f} units")

        return last_ma, mae

    def exponential_smoothing(self, alpha=0.3):
        """Exponential smoothing forecast"""

        print(f"\n\033[95m📊 EXPONENTIAL SMOOTHING (α={alpha})\033[0m")

        # Initialize
        result = [self.df['volume'].iloc[0]]

        # Apply exponential smoothing
        for i in range(1, len(self.df)):
            forecast = alpha * self.df['volume'].iloc[i - 1] + (1 - alpha) * result[-1]
            result.append(forecast)

        self.df['exp_smooth'] = result

        # Forecast next period
        next_forecast = alpha * self.df['volume'].iloc[-1] + (1 - alpha) * result[-1]

        # Accuracy
        mae = mean_absolute_error(
            self.df['volume'][1:],
            self.df['exp_smooth'][:-1]
        )

        print(f"Next period forecast: {next_forecast:.0f} units")
        print(f"Historical MAE: {mae:.2f} units")

        return next_forecast, mae

    def decompose_time_series(self):
        """Decompose time series into trend, seasonal, residual"""

        print("\n\033[92m🔧 DECOMPOSING TIME SERIES...\033[0m")

        decomposition = seasonal_decompose(
            self.df['volume'],
            model='additive',
            period=365
        )

        self.df['trend'] = decomposition.trend
        self.df['seasonal'] = decomposition.seasonal
        self.df['residual'] = decomposition.resid

        print("Components extracted:")
        print(" ✓ Trend (long-term direction)")
        print(" ✓ Seasonal (recurring patterns)")
        print(" ✓ Residual (random variation)")

        return decomposition

    def create_visualizations(self):
        """Create analysis visualizations"""

        print("\n\033[94m📊 CREATING VISUALIZATIONS...\033[0m")

        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        fig.patch.set_facecolor('white')

        # 1 — Historical Data
        axes[0].plot(self.df.index, self.df['volume'],
                     color=self.colors['blue'], alpha=0.5, label='Actual')

        if 'MA_7' in self.df.columns:
            axes[0].plot(self.df.index, self.df['MA_7'],
                         color=self.colors['red'], linewidth=2, label='7-Day MA')

        axes[0].set_title('Historical Volume Data', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Volume')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 2 — Weekly Patterns
        weekly_avg = self.df.groupby(self.df.index.dayofweek)['volume'].mean()
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

        bars = axes[1].bar(days, weekly_avg, color=self.colors['blue'],
                           edgecolor=self.colors['red'], linewidth=2)

        axes[1].set_title('Average Volume by Day of Week', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Average Volume')
        axes[1].grid(True, alpha=0.3, axis='y')

        for bar, val in zip(bars, weekly_avg):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width() / 2., height,
                         f'{val:.0f}', ha='center', va='bottom')

        # 3 — Monthly Patterns
        monthly_avg = self.df.groupby(self.df.index.month)['volume'].mean()
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        axes[2].plot(months, monthly_avg, color=self.colors['red'],
                     marker='o', markersize=8, linewidth=2,
                     markerfacecolor=self.colors['blue'],
                     markeredgecolor=self.colors['red'])

        axes[2].set_title('Average Volume by Month', fontsize=14, fontweight='bold')
        axes[2].set_ylabel('Average Volume')
        axes[2].grid(True, alpha=0.3)
        axes[2].set_xlabel('Month')

        plt.tight_layout()
        plt.savefig('volume_analysis.png', dpi=100, bbox_inches='tight')
        print("✓ Saved visualization to volume_analysis.png")

        return fig

# Main Execution

def main():

    print("\033[91m" + "=" * 60)
    print("\033[97m" + " 🇺🇸 STATISTICAL AI FORECASTING 🇺🇸")
    print("\033[94m" + "=" * 60 + "\033[0m")
    print("Week 7: Building Foundation\n")

    forecaster = QuantitativeForecaster('data/historical_volumes.csv')

    # Statistical and pattern analysis
    stats = forecaster.quantitative_analysis()

    # Forecasting
    ma_forecast, _ = forecaster.moving_average_forecast(window=7)
    exp_forecast, _ = forecaster.exponential_smoothing(alpha=0.3)

    # Decomposition
    forecaster.decompose_time_series()

    # Visualization
    forecaster.create_visualizations()

    print("\n" + "=" * 60)
    print("\033[92m✅ ANALYSIS COMPLETE!\033[0m")
    print("=" * 60)

    print("\nForecast Summary:")
    print(f" Moving Average (7-day): {ma_forecast:.0f} units")
    print(f" Exponential Smoothing: {exp_forecast:.0f} units")
    print(f" Average of methods: {(ma_forecast + exp_forecast) / 2:.0f} units")

    return forecaster


if __name__ == "__main__":
    forecaster = main()
