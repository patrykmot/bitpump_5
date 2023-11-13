import torch

import bitpump as bit
import pandas as pd
import numpy as np
import pytest as pytest
import datetime


def test_stock_data_source():
    data: pd.DataFrame = load_data()
    assert data is not None
    assert data.size > 1


def test_play_with_pandas():
    data: pd.DataFrame = load_data()
    print("Found ticker data with size = " + str(data))
    print("Data before removal: " + str(data.columns))
    data = data.drop(columns=["Dividends", "Stock Splits"])
    print("Data after removal: " + str(data.columns))

    # Data frames
    technologies = [["Spark", 20000, "30days"],
                    ["Pandas", 25000, "40days"], ]
    t1 = pd.DataFrame(technologies)
    technologies = [["Spark2", 40000, "50days"],
                    ["Pandas2", 55000, "60days"], ]
    t2 = pd.DataFrame(technologies)
    print(t1)
    print(t2)
    t3 = pd.concat([t1, t2], axis=1)
    print("After concat:")
    print(t3)

    print("Example of candles moving:")
    c1 = ["Open", "Close", "High", "Low", "Volume"]
    c2 = ["Open2", "Close2", "High2", "Low2", "Volume2"]
    t4 = pd.DataFrame(np.random.randn(8, 5), columns=c1)
    print("t4 = \n" + str(t4))
    t5 = pd.DataFrame(columns=c1 + c2)
    print("t5 = \n" + str(t5))


def test_ai_data_source():
    # Load data and do initial filtering
    data_source: bit.AIDataSource = bit.AIDataSource()
    data: pd.DataFrame = data_source.get_data(bit.StockTicker.BITCOIN_USD, bit.StockInterval.DAY)
    assert data is not None
    assert data.size > 10

    # Get last candle as result for AI
    result, data = bit.get_last_candle_as_result_and_modify(data)
    assert result.columns.size == 6  # open, max, min, close, volume, timestamp

    # Remove all timestamp columns except last one
    data_last_timestamp: pd.DataFrame = bit.keep_last_timestamp_column_only(data)
    assert data_last_timestamp.size > 10
    assert data_last_timestamp.columns.size == 21
    assert data.columns.size == 24

    # Filter out not unique results
    bit.find_not_unique_results(data_last_timestamp)


def test_ai_train():
    model = bit.AIModel(2, 60, 1)
    data_in = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0], "B": [4.0, 3.0, 2.0, 2.0]}).astype(dtype='float32')
    data_target = pd.DataFrame({"OUT": [4.0, 5.0, 5.0, 6.5]}).astype(dtype='float32')
    bit.train(model, data_in, data_target, 0.0002, 0.003, 50000)

    _asert_model_training(data_in, 0, data_target, model)
    _asert_model_training(data_in, 1, data_target, model)
    _asert_model_training(data_in, 2, data_target, model)
    _asert_model_training(data_in, 3, data_target, model)


def test_fetch_bitcoin_minutes_prices():
    prices: pd.DataFrame = load_data(ticker=bit.StockTicker.BITCOIN_USD, interval=bit.StockInterval.MINUTES_5)
    assert prices is not None
    assert prices.size > 10


def test_fetch_bitcoin_hour_prices():
    prices: pd.DataFrame = load_data(ticker=bit.StockTicker.BITCOIN_USD, interval=bit.StockInterval.HOUR)
    assert prices is not None
    assert prices.size > 10


def _asert_model_training(data_in, data_index, data_target, model):
    input_tensor = torch.tensor(data_in.iloc[data_index])
    out = model(input_tensor)
    assert abs((out.item() - data_target.iloc[data_index])[0]) < 0.1


def test_create_date_column_from_index():
    df: pd.DataFrame = load_data()
    df_with_date: pd.DataFrame = bit.create_date_column_from_index(df)
    assert df_with_date[bit.AIDataSource.COL_VOLUME].size > 10

def test_merge_candles_with_timestamp_column():
    # freq -> https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
    # Create a date range from '2023-01-01' to '2023-01-10'
    #date_range = pd.date_range(start='2023-01-01', end='2023-01-10', freq='D')

    # Create a DataFrame with a single column 'Date'
    #df = pd.DataFrame({'Date': pd.date_range(start='2023-01-01', end='2023-01-10', freq='h')})
    range_h: pd.DatetimeIndex =  pd.date_range(start='2023-01-01', end='2023-01-10', freq='h')
    range_d: pd.DatetimeIndex = pd.date_range(start='2022-12-10', end='2023-01-05', freq="D")

    data_h_colum: [] = range(0, len(range_h))
    data_d_column: [] = range(100, len(range_d)+100)

    # Freq -> https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
    data_h = pd.DataFrame({"Data": data_h_colum, bit.AIDataSource.COL_TIMESTAMP: range_h})
    data_d = pd.DataFrame({"Data": data_d_column, bit.AIDataSource.COL_TIMESTAMP: range_d})
    merged = bit.merge_data_with_timestamp(data_h, data_d)
    assert merged.columns.size == 3
    assert merged.size == 651
    # 2023.01.01 00:00:00 is after 2022.12.31
    assert merged.loc[0][2] == 121
    assert merged.loc[97][2] == 126

def _get_datetime(year=2023, month=11, day=4, hour=16, minutes=52):
    return datetime.datetime(year=year, month=month, day=day, hour=hour, minute=minutes)

def load_data(ticker: bit.StockTicker = bit.StockTicker.SP_500,
              interval: bit.StockInterval = bit.StockInterval.WEEK) -> pd.DataFrame:
    stock_data_source: bit.StockDataSource = bit.StockDataSource()
    data: pd.DataFrame = stock_data_source.get_data(ticker, interval)
    return data


# Run before every test!
@pytest.fixture(autouse=True)
def run_around_tests():
    bit.StockDataSource().remove_all_cached_data()
