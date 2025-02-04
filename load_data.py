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

    def load_data(self):
        if not self.path:
            print("No path specified, skipping local CSV load.")
            return None
        return pd.read_csv(self.path)

    def show_available_commodities(self):
        print("Available commodities:")
        for name, ticker in DEFAULT_COMMODITIES.items():
            print(f"{name}: {ticker}")

    def load_data_from_yf(self, commodities=None, years=15):
        if not commodities:
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
            data['daily_return'] = data['Adj Close'].pct_change()
            commodity_data[name] = data

        # Combine data into a single DataFrame (optional example)
        combined_df = pd.DataFrame({
            nm: df['daily_return'] for nm, df in commodity_data.items()
        }).dropna(how='all')

        self._save_combined_data(combined_df)
        return combined_df

    def _save_combined_data(self, combined_df, filename="combined_commodities.csv"):
        os.makedirs("./data", exist_ok=True)
        save_path = os.path.join("./data", filename)
        combined_df.to_csv(save_path)
        print(f"Combined data saved to {save_path}")

