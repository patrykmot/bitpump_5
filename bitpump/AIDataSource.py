import StockDataSource as sds
import pandas as pd
import bitpump as bit
import numpy as np


class AIDataSource:
    COL_VOLUME = "Volume"
    COL_TIMESTAMP = "Timestamp"
    allowed_columns = ["Open", "High", "Low", "Close", COL_VOLUME, COL_TIMESTAMP]

    def __init__(self, stock_data_source: sds.StockDataSource() = sds.StockDataSource()):
        self.stock_data_source = stock_data_source
        self.max = 0
        self.min = 0

    def get_data(self, stock_ticker: sds.StockTicker, stock_interval: sds.StockInterval,
                 number_of_candles_in_row: int = 5) -> pd.DataFrame:
        print(f"Starting to fetch data for ticker = {str(stock_ticker)} and interval {str(stock_interval)}")
        data: pd.DataFrame = self.stock_data_source.get_data(stock_ticker, stock_interval)

        # Remove not needed columns
        column_names = [str(c) for c in data.columns]
        for allowed_column in column_names:
            if allowed_column not in AIDataSource.allowed_columns:
                data = data.drop(columns=[allowed_column])

        # Create new column with timestamp from index (it contain dates)
        data = bit.create_date_column_from_index(data)

        # Reset index and put number_of_candles_in_row as a column
        data = self.reshape(data, number_of_candles_in_row)

        # Normalize candles values per each row, and volumes globally
        data = self.normalize(data)

        # check if we have duplicates
        # data = data.drop_duplicates(subset=self.allowed_columns[:-1])
        # size_after = data.size
        # print(f"removed {size_before - size_after} number of rows from input data.")
        return data

    # Move join_rows into columns
    def reshape(self, data: pd.DataFrame, join_rows: int) -> pd.DataFrame:
        reshaped_data: pd.DataFrame = pd.DataFrame()
        for i in range(0, len(data), join_rows):
            cumulated_row: pd.DataFrame = pd.DataFrame()
            for g in range(0, join_rows):
                if i + join_rows < len(data.index):
                    row: pd.Series = data.iloc[i + g]
                    row_part = pd.DataFrame(row).T.reset_index(drop=True)  # THIS WORKS!!!!!!!!!!!!
                    cumulated_row = pd.concat([cumulated_row, row_part], axis=1, ignore_index=True)

            if cumulated_row.size > 0:
                # Put index value as last row in reshaped_data
                cumulated_row.index = [max([reshaped_data.size - 1, 0])]
                reshaped_data = pd.concat([reshaped_data, cumulated_row], axis=0, ignore_index=True)

        # setup colum name
        reshaped_data.columns = self._get_column_names(data.columns, join_rows)
        return reshaped_data;

    def normalize(self, data: pd.DataFrame):

        # Normalize candles in each rows
        filter_candle_col = [col for col in data if
                             not (col.startswith(AIDataSource.COL_VOLUME) | col.startswith(AIDataSource.COL_TIMESTAMP))]
        candles: pd.DataFrame = data[filter_candle_col]
        norm_out = candles.apply(normalize_row, axis=1)
        candles.update(norm_out)
        data.update(candles)

        filter_volume_col = [col for col in data if col.startswith(AIDataSource.COL_VOLUME)]
        volumes: pd.DataFrame = data[filter_volume_col]

        self.max = volumes.max().max()
        self.min = volumes.min().min()

        volumes2 = volumes.apply(self._normalize_data_frame, axis=1)
        data.update(volumes2)
        # Should we normalize volumes somehow?
        return data

    def _normalize_data_frame(self, df):
        return (df - self.min) / (self.max - self.min)

    def _get_column_names(self, columns: [], count: int) -> []:
        new_names = []
        for i in range(0, count):
            for name in columns:
                new_names.append(name + "_" + str(i + 1))
        return new_names


def normalize_row(row: pd.Series):
    min_val = row.min()
    max_val = row.max()
    return (row - min_val) / (max_val - min_val)


def create_date_column_from_index(df: pd.DataFrame) -> pd.DataFrame:
    df[bit.AIDataSource.COL_TIMESTAMP] = df.index
    return df


def keep_last_timestamp_column_only(df: pd.DataFrame) -> pd.DataFrame:
    timestamps: pd.DataFrame = df[(col for col in df if col.startswith(bit.AIDataSource.COL_TIMESTAMP))]
    df_with_removed_timestamps = pd.DataFrame(df)
    for col in timestamps.columns[0:-1]:
        df_with_removed_timestamps = df_with_removed_timestamps.drop(col, axis=1)
    return df_with_removed_timestamps


def get_last_candle_as_result_and_modify(df: pd.DataFrame) -> pd.DataFrame:
    # Last candle columns
    last_candle_columns = df.columns[-6:]

    # Return last candle as result
    last = df[last_candle_columns]

    # Remove last candle from source
    for col in last_candle_columns:
        df = df.drop(col, axis=1)
    return last, df

def remove_not_unique_results(df: pd.DataFrame):
    duplicates_removed = df.drop_duplicates(df)
    print(f"Removed { df.size - duplicates_removed.size } duplicated rows.")
    return duplicates_removed

