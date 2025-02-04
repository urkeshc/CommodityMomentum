import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class SMAStrategy:
    def __init__(self, df):
        # Assume df is already cleaned and loaded
        self.df = df

    def run(self):
        column_name = input("Enter the column name to run the SMA on: ")

        # Ensure daily returns
        self.df['daily_return'] = self.df[column_name].pct_change()

        choice = input("Use custom SMA range? (y/n): ")
        if choice.lower() == 'y':
            short_ma = int(input("Enter short SMA window: "))
            long_ma = int(input("Enter long SMA window: "))
            self._compute_single_sma_strategy(column_name, short_ma, long_ma)
        else:
            top_n = int(input("How many top strategies to display? "))
            self._compute_best_sma_strategies(column_name, top_n)

    def _compute_single_sma_strategy(self, col, short_ma, long_ma):
        SMA_short = self.df[col].rolling(window=short_ma).mean()
        SMA_long = self.df[col].rolling(window=long_ma).mean()
        signal = (SMA_short > SMA_long).astype(int) - (SMA_short < SMA_long).astype(int)
        strategy_return = self.df['daily_return'] * signal.shift()
        self.df['strategy_cumulative'] = (1 + strategy_return).cumprod() - 1
        self.df['baseline_cumulative'] = (self.df[col] / self.df[col].iloc[0]) - 1

        self._plot_results(col, short_ma, long_ma)

    def _compute_best_sma_strategies(self, col, top_n):
        moving_averages = range(1, 252)
        strategy_results = {}
        strategy_std_dev = {}

        for short_ma in moving_averages:
            for long_ma in moving_averages:
                if short_ma < long_ma:
                    SMA_short = self.df[col].rolling(window=short_ma).mean()
                    SMA_long = self.df[col].rolling(window=long_ma).mean()
                    signal = (SMA_short > SMA_long).astype(int) - (SMA_short < SMA_long).astype(int)
                    strategy_return = self.df['daily_return'] * signal.shift()
                    cumulative_return = (1 + strategy_return).cumprod().iloc[-1] - 1
                    strategy_results[(short_ma, long_ma)] = cumulative_return
                    strategy_std_dev[(short_ma, long_ma)] = strategy_return.dropna().std()

        sorted_strategies = sorted(strategy_results.items(), key=lambda x: x[1], reverse=True)[:top_n]
        best_short, best_long = sorted_strategies[0][0]

        # Compute best strategy for plotting
        SMA_short_best = self.df[col].rolling(window=best_short).mean()
        SMA_long_best = self.df[col].rolling(window=best_long).mean()
        best_signal = (SMA_short_best > SMA_long_best).astype(int) - (SMA_short_best < SMA_long_best).astype(int)
        best_strategy_return = self.df['daily_return'] * best_signal.shift()
        self.df['strategy_cumulative'] = (1 + best_strategy_return).cumprod() - 1
        self.df['baseline_cumulative'] = (self.df[col] / self.df[col].iloc[0]) - 1

        self._plot_results(col, best_short, best_long)
        self._print_top_strategies(sorted_strategies, strategy_std_dev)

    def _plot_results(self, col, short_ma, long_ma):
        plt.figure(figsize=(14, 8))
        ax = plt.gca()
        ax.plot(self.df.index, self.df['baseline_cumulative'], color='blue', label=f'{col} Cumulative Returns')
        ax.plot(self.df.index, self.df['strategy_cumulative'], color='green',
                label=f'Cumulative Returns (SMA {short_ma}-{long_ma})')
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Returns', color='black')
        ax.legend(loc='upper left')
        plt.title(f'Comparison of Cumulative Returns: {col} vs. SMA {short_ma}-{long_ma}')
        plt.grid(True)
        plt.show()

    def _print_top_strategies(self, sorted_strategies, strategy_std_dev):
        days_total = (self.df.index[-1] - self.df.index[0]).days
        years = days_total / 252
        risk_free_rate = 0.0175
        for (short_ma, long_ma), cum_return in sorted_strategies:
            cagr = (1 + cum_return) ** (1 / years) - 1
            annualized_std = np.sqrt(252) * strategy_std_dev[(short_ma, long_ma)]
            sharpe_ratio = (cagr - risk_free_rate) / annualized_std if annualized_std > 0 else 0
            print(f"SMA ({short_ma},{long_ma}): Cumulative={cum_return:.4f}, "
                  f"Annualized={100*cagr:.2f}%, StdDev={100*annualized_std:.2f}%, Sharpe={sharpe_ratio:.3f}")
