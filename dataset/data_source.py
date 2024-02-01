import os

import pandas as pd
import yfinance as yf


class DataSource:
    def __init__(self, indicators, tickers, file_name='indicator_data.csv'):
        self.indicators = indicators
        self.tickers = tickers
        self.save_path = file_name

    def download_dataset(self, period="1Y"):
        data = {}
        for ticker in self.tickers:
            data[ticker] = yf.download(ticker, period=period)

        data_list = []
        for ticker in self.tickers:
            time_series = data[ticker]
            indicator_list = ['Close']
            for k, v in self.indicators.items():
                indicator_list.append(k)
                time_series = v(time_series, col_name=k)
            time_series = time_series.drop(['Open', 'High', 'Low', 'Adj Close', 'Volume'], axis=1)
            data[ticker] = time_series
            data_list.append(time_series)

        states_df = pd.concat(data_list, axis=1)
        states_df.to_csv(self.save_path)

        return states_df.to_numpy()[45:]

    def load_dataset(self):
        if os.path.isfile(self.save_path):
            df = pd.read_csv(self.save_path)
            df = df.drop(columns=df.columns[0])
            return df.to_numpy()[45:]

        return self.download_dataset()
