import os

import numpy as np
import pandas as pd
import yfinance as yf
from matplotlib import pyplot as plt


class DataSource:
    def __init__(self, indicators, tickers, file_name='indicator_data.csv'):
        self.indicators = indicators
        self.tickers = tickers
        self.save_path = file_name

    def download_dataset(self, period="5Y"):
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
            # for column in time_series.columns:
            #     # Plot the column
            #     plt.figure(figsize=(10, 6))
            #     plt.plot(time_series[column])
            #     plt.title(f'{ticker} - {column}')
            #     plt.xlabel('Index')
            #     plt.ylabel(column)
            #
            #     # Save the plot as PNG
            #     plt.savefig('TESTS/reports/dataset/' + ticker + '_' + column + '.png')
            #
            #     # Close the plot to avoid displaying multiple plots in a single figure
            #     plt.close()


        states_df = pd.concat(data_list, axis=1)
        states_df.to_csv(self.save_path)

        states_data = states_df.to_numpy()[45:]

        min_vals = np.min(states_data, axis=0)
        max_vals = np.max(states_data, axis=0)

        normalized_arr = (states_data - min_vals) / (max_vals - min_vals)

        return normalized_arr * 100

    def load_dataset(self):
        # if os.path.isfile(self.save_path):
        #     df = pd.read_csv(self.save_path)
        #     df = df.drop(columns=df.columns[0])
        #     return df.to_numpy()[45:]

        return self.download_dataset()
