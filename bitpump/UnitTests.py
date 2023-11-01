import torch

import bitpump as bit
import pandas as pd
import numpy as np
import pytest as pytest


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
    bit.remove_not_unique_results(data_last_timestamp)


def test_ai_train():
    model = bit.AIModel(2, 20, 1)
    data_in = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0], "B": [4.0, 3.0, 2.0, 2.0]}).astype(dtype='float32')
    data_target = pd.DataFrame({"OUT": [4.0, 5.0, 5.0, 6.5]}).astype(dtype='float32')
    bit.train(model, data_in, data_target, 0.00005, 0.05)
    input_tensor = torch.tensor(data_in.iloc[0])
    out = model(input_tensor)
    assert abs(out.item()) - 4.0 < 0.2


def test_create_date_column_from_index():
    df: pd.DataFrame = load_data()
    df_with_date: pd.DataFrame = bit.create_date_column_from_index(df)
    assert df_with_date[bit.AIDataSource.COL_VOLUME].size > 10


def load_data() -> pd.DataFrame:
    stock_data_source: bit.StockDataSource = bit.StockDataSource()
    data: pd.DataFrame = stock_data_source.get_data(bit.StockTicker.SP_500, bit.StockInterval.WEEK)
    return data


# Run before every test!
@pytest.fixture(autouse=True)
def run_around_tests():
    bit.StockDataSource().remove_all_cached_data()
