import pandas as pd
import os
import yfinance as yf

DEFAULT_COMMODITIES = {
    "Crude Oil": "CL=F",
    "Gold": "GC=F",
    "Natural Gas": "NG=F",
    "Silver": "SI=F",
    "Copper": "HG=F",
    "Corn": "ZC=F",
    "Soybeans": "ZS=F",
    "Wheat": "KE=F",
    "Coffee": "KC=F",
    "Cotton": "CT=F",
    "Sugar": "SB=F",
    "Aluminum": "ALI=F",
    "Platinum": "PL=F",
    "Palladium": "PA=F",
    "Brent Crude": "BZ=F",
    "Frozen Orange Juice": "OJ=F",
    "Micro Gold": "MGC=F",
    "Micro Silver": "SIL=F",
    "Heating Oil": "HO=F",
    "RBOB Gasoline": "RB=F",
    "Oat Futures": "ZO=F",
    "Rough Rice": "ZR=F",
    "Soybean Oil": "ZL=F",
    "Lean Hogs": "HE=F",
    "Live Cattle": "LE=F",
    "Feeder Cattle": "GF=F",
    "Cocoa": "CC=F"
}

class LoadData:
    def __init__(self, path=None):
        self.path = path

    
    def load_data(self): # If you want to use an arlready existing CSV file, specify file path
        if not self.path:
            print("No path specified, skipping local CSV load.")
            return None
        return pd.read_csv(self.path)

    
    def show_available_commodities(self):
        print("Available commodities:")
        for name, ticker in DEFAULT_COMMODITIES.items():
            print(f"{name}: {ticker}")

    
    def remove_outliers(self, data, period='daily'):
        """
        Removes outliers based on period-specific thresholds.
        Replaces outliers with mean of 3 previous and 3 next values.
        """
        threshold_map = {
            'daily': 1.0,      # 100%
            'monthly': 2.0,    # 200%
            'quarterly': 5.0,  # 500%
            'yearly': 5.0      # 500%
        }
        
        if period not in threshold_map:
            raise ValueError("Invalid period. Choose 'daily', 'monthly', 'quarterly', or 'yearly'")
        
        threshold = threshold_map[period]
        outlier_mask = abs(data) > threshold
        
        # Create a copy to avoid modifying the original data
        cleaned_data = data.copy()
        
        # For each outlier, replace with mean of 3 previous and 3 next values
        for idx in data[outlier_mask].index:
            prev_vals = data.loc[:idx].tail(4)[:-1]  # 3 previous values
            next_vals = data.loc[idx:].head(4)[1:]   # 3 next values
            replacement = pd.concat([prev_vals, next_vals]).mean()
            cleaned_data.loc[idx] = replacement
            
        return cleaned_data

    def calculate_period_returns(self, data, period='daily', remove_outliers=True):
        """
        Calculate returns for the specified period and optionally remove outliers.
        """
        if period == 'daily':
            returns = data['Adj Close'].pct_change()
        elif period == 'monthly':
            returns = data['Adj Close'].resample('M').last().pct_change()
        elif period == 'quarterly':
            returns = data['Adj Close'].resample('Q').last().pct_change()
        elif period == 'yearly':
            returns = data['Adj Close'].resample('Y').last().pct_change()
        else:
            raise ValueError("Invalid period. Choose 'daily', 'monthly', 'quarterly', or 'yearly'")
        
        if remove_outliers:
            returns = self.remove_outliers(returns, period)
            
        return returns

    def load_data_from_yf(self, commodities=None, years=15, load_all=False, return_period='daily', remove_outliers=True):
        if load_all:
            commodities_dict = DEFAULT_COMMODITIES
        elif not commodities:
            self.show_available_commodities()
            user_in = input("Enter comma-separated names of commodities (or leave blank for default): ")
            if user_in.strip():
                selected = [c.strip() for c in user_in.split(",")]
                commodities_dict = {k: v for k, v in DEFAULT_COMMODITIES.items() if k in selected}
            else:
                commodities_dict = DEFAULT_COMMODITIES
        else:
            commodities_dict = {k: v for k, v in DEFAULT_COMMODITIES.items() if k in commodities}

        end_date = pd.Timestamp.today()
        start_date = end_date - pd.DateOffset(years=years)

        print(f"Fetching data from {start_date.date()} to {end_date.date()}")
        commodity_data = {}
        for name, ticker in commodities_dict.items():
            data = yf.download(ticker, start=start_date, end=end_date)
            data['Adj Close'] = data['Adj Close'].astype(float)
            returns = self.calculate_period_returns(data, return_period, remove_outliers)
            commodity_data[name] = returns

        combined_df = pd.DataFrame(commodity_data).dropna(how='all') # can fillna(0) if needed - investigate this in the future...

        self._save_combined_data(combined_df) # comment out to avoid systematically saving the data (for example, for exploration purposes)
        return combined_df


    def _save_combined_data(self, combined_df, filename="combined_commodities.csv"):
        os.makedirs("./data", exist_ok=True)
        save_path = os.path.join("./data", filename)
        combined_df.to_csv(save_path)
        print(f"Combined data saved to {save_path}")

