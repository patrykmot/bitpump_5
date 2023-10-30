from enum import Enum

import yfinance as yf
import pandas as pd
import os as os
import glob

from pandas import DataFrame


class StockTicker(Enum):
    BITCOIN_USD = 1
    SP_500 = 2
    OIL = 3
    GOLD = 4


class StockInterval(Enum):
    #  Valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
    WEEK = 1
    DAY = 2
    HOUR = 3
    MINUTES_5 = 4


# Yahoo ticker list and symbols https://finance.yahoo.com/lookup/


class StockDataSource:
    def __init__(self):
        # Create data folder if needed
        self.data_directory = "stock_data"
        if not os.path.exists(self.data_directory):
            os.makedirs(self.data_directory)

    def get_data(self, ticker: StockTicker = StockTicker.BITCOIN_USD, interval: StockInterval = StockInterval.HOUR) \
            -> pd.DataFrame:
        file_name = self._get_file_name(ticker, interval)
        data : DataFrame
        if os.path.exists(file_name):
            data = pd.read_csv(file_name, index_col=0)
        else:
            ticker_name = self._get_ticker_name(ticker)
            interval_name = self._get_interval_name(interval)
            print("Connecting to Yahoo to get ticker data for " + ticker_name + " for interval " + interval_name)
            yf_ticker = yf.Ticker(ticker_name)
            data = yf_ticker.history(period="max", interval=interval_name)
            if data.size <= 0:
                print("Downloaded ticker from yahoo is empty! Please check you query parameters.")
            else:
                print("Saving result to file " + file_name)
                data.to_csv(file_name)
        return data

    def _get_file_name(self, ticker: StockTicker, interval: StockInterval):
        file_name: str = self._remove_file_name_disallowed_characters(
            "ticker_" + self._get_ticker_name(ticker) + "_" + self._get_interval_name(interval))
        return os.path.join(self.data_directory, file_name) + ".csv"

    @staticmethod
    def _remove_file_name_disallowed_characters(s: str):
        return s.replace("-", "_").replace("=", "_").replace("^", "_")

    def _get_ticker_name(self, ticker: StockTicker):
        if StockTicker.BITCOIN_USD == ticker:
            return "BTC-USD"
        if StockTicker.OIL == ticker:
            return "CL=F"
        if StockTicker.GOLD == ticker:
            return "GLD"
        if StockTicker.SP_500 == ticker:
            return "^GSPC"

    def _get_interval_name(self, interval: StockInterval):
        if StockInterval.HOUR == interval:
            return "1h"
        if StockInterval.DAY == interval:
            return "1d"
        if StockInterval.WEEK == interval:
            return "1wk"
        if StockInterval.MINUTES_5 == interval:
            return "5m"

    def remove_all_cached_data(self):
        print("Clear cached files.")
        files = glob.glob(os.path.join(self.data_directory,"*.csv"))
        for f in files:
            os.remove(f)
