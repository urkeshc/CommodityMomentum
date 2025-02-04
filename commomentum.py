import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class CommodityMomentum:
    def __init__(self, data):
        self.data = data

    def generate_rolling_returns(self, range=12):  # default yearly
        """
        Inputs:
            - Rolling returns period
        Output:
            - dataframe with returns in the past X periods at each point in time
        """
        return self.data.rolling(window=range).sum()

    def commodity_momentum_strategy(self, K, X, RiskFreeRate):
        """
        Buy the top K best performing commodity(ies) in the past X periods, and sell the bottom K performing commodity(ies).
        Inputs:
            - K: Number of commodities to buy/sell
            - X: Number of periods to look back
            - Risk Free Rate
        Returns:
            - DataFrame with strategy returns
            - Sharpe Ratio (in DataFrame attributes)
            - Max Drawdown
            - Average Yearly Return
            - Yearly Standard Deviation
        """
        rolling_data = self.generate_rolling_returns(range=X)

        def get_top_bottom_commodities(row, top_n=K, bottom_n=K):
            ranked = row.rank(ascending=False, method='min', na_option='keep')
            long_positions = (ranked <= top_n).astype(int)
            short_positions = -1 * (ranked > (len(row) - bottom_n)).astype(int)
            return long_positions + short_positions

        trading_signals = rolling_data.apply(get_top_bottom_commodities, axis=1)
        strategy_returns = (trading_signals.shift(1) * self.data).sum(axis=1) / (2 * K)
        annualized_returns = strategy_returns.mean() * 12
        annualized_std_dev = strategy_returns.std() * np.sqrt(12)
        sharpe = (annualized_returns - RiskFreeRate) / annualized_std_dev if annualized_std_dev != 0 else 0
        cum_returns = strategy_returns.cumsum()

        running_max = cum_returns.cummax()
        drawdowns = cum_returns - running_max
        max_drawdown = drawdowns.min()

        # Create expanded results DataFrame with signals and rolling returns
        results = pd.DataFrame({
            'Strategy_Returns': strategy_returns,
            'Cumulative_Returns': cum_returns
        })
        
        # Add trading signals with prefix 'Signal_'
        for column in trading_signals.columns:
            results[f'Signal_{column}'] = trading_signals[column]
        
        # Add rolling returns with prefix 'Rolling_'
        for column in rolling_data.columns:
            results[f'Rolling_{column}'] = rolling_data[column]
            
        # Add raw returns with prefix 'Return_'
        for column in self.data.columns:
            results[f'Return_{column}'] = self.data[column]
        
        # Add strategy metrics as attributes
        results.attrs['Sharpe_Ratio'] = sharpe
        results.attrs['Max_Drawdown'] = max_drawdown
        results.attrs['Annualized_Return'] = annualized_returns
        results.attrs['Annualized_StdDev'] = annualized_std_dev
        
        # Print metrics
        print(f"Sharpe Ratio: {sharpe:.2f}")
        print(f"Max Drawdown: {max_drawdown:.2f}")
        print(f"Average Yearly Return: {annualized_returns:.2f}")
        print(f"Yearly Standard Deviation: {annualized_std_dev:.2f}")
        
        return results

    def compare_momentum_periods(self, range=[3, 6, 12], K=4, RiskFreeRate=0.02):
        """
        Compare momentum strategies across different lookback periods.
        Inputs:
            - range: List of lookback periods to compare
            - K: Number of commodities to buy/sell
            - Risk Free Rate
        Returns:
            - DataFrame with metrics for each lookback period
        """
        results = {}
        for period in range:
            # Calculate returns for this lookback period
            roll_data = self.data.rolling(window=period).sum()
            
            # Get trading signals
            def get_top_bottom_commodities(row):
                ranked = row.rank(ascending=False, method='min', na_option='keep')
                long_positions = (ranked <= K).astype(int)
                short_positions = -1 * (ranked > (len(row) - K)).astype(int)
                return long_positions + short_positions

            signals = roll_data.apply(get_top_bottom_commodities, axis=1)
            strategy_returns = (signals.shift(1) * self.data).sum(axis=1) / (2 * K)
            
            # Calculate metrics
            ann_returns = strategy_returns.mean() * 12
            ann_std_dev = strategy_returns.std() * np.sqrt(12)
            sharpe = (ann_returns - RiskFreeRate) / ann_std_dev if ann_std_dev != 0 else 0
            cum_returns = (1 + strategy_returns).cumprod().iloc[-1] - 1 if not strategy_returns.empty else 0
            
            results[period] = {
                'Annualized_Return': ann_returns,
                'Annualized_StdDev': ann_std_dev,
                'Sharpe_Ratio': sharpe,
                'Cumulative_Return': cum_returns
            }
            
        return pd.DataFrame(results).T

    def plot_strategy_returns(self, strategy_returns):
        """
        Plot the returns of a single momentum strategy.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        strategy_returns['Cumulative_Returns'].plot(ax=ax)
        ax.set_title('Commodity Momentum Strategy')
        ax.set_xlabel('Time')
        ax.set_ylabel('Cumulative Returns')
        plt.show()

    def histogram_momentum_strategies(self, strategies_returns, metric='Sharpe_Ratio'):
        """
        Plot histogram of momentum strategy metrics.
        Inputs:
            - strategies_returns: DataFrame from compare_momentum_periods
            - metric: Which metric to plot ('Sharpe_Ratio', 'Annualized_Return', etc.)
        """
        if metric not in strategies_returns.columns:
            raise ValueError(f"Metric {metric} not found in data. Available metrics: {strategies_returns.columns.tolist()}")
            
        plt.figure(figsize=(10, 6))
        ax = strategies_returns[metric].plot(kind='bar')
        plt.title(f'{metric} Across Different Lookback Periods')
        plt.xlabel('Lookback Period (Months)')
        plt.ylabel(metric)
        
        # Add value labels on top of each bar
        for i, v in enumerate(strategies_returns[metric]):
            ax.text(i, v, f'{v:.2f}', ha='center', va='bottom')
            
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()