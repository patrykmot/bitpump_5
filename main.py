import bitpump as bit
import pandas as pd


def example_gold_day_with_super_low_error_001():
    data_source: bit.AIDataSource = bit.AIDataSource(bit.StockDataSource())
    data: pd.dataFrame = data_source.get_data(bit.StockTicker.GOLD, bit.StockInterval.DAY)
    result, data = bit.get_last_candle_as_result_and_modify(data)
    data = bit.keep_last_timestamp_column_only(data)
    # data = bit.remove_not_unique_results(data) - risky to remove!!!! Result will be different!
    data = bit.remove_all_timestamps(data)
    result = bit.keep_only_close_candle(result)
    model = bit.AIModel(data.columns.size, data.columns.size * 31, result.columns.size)
    bit.train(model, data, result, 0.001, 0.01)


def example_merge_two_candles_with_different_interval():
    # Load day and hour candles and transform data
    data_source: bit.AIDataSource = bit.AIDataSource(bit.StockDataSource())
    data_day: pd.dataFrame = data_source.get_data(bit.StockTicker.BITCOIN_USD, bit.StockInterval.DAY)
    data_hour: pd.dataFrame = data_source.get_data(bit.StockTicker.BITCOIN_USD, bit.StockInterval.HOUR)

    # Keep only latest timestamps
    data_day = bit.keep_last_timestamp_column_only(data_day)
    data_hour = bit.keep_last_timestamp_column_only(data_hour)

    # Merge data_day into data_hour so in each row we get days and hours data
    data = bit.merge_data_with_timestamp(data_hour, data_day)
    print(data.head())


def predicting_bitcoin_price_base_on_candles():
    data_source: bit.AIDataSource = bit.AIDataSource(bit.StockDataSource())

    # Fetch hour, day, week candles
    btc_week: pd.dataFrame = data_source.get_data(bit.StockTicker.BITCOIN_USD, bit.StockInterval.WEEK)
    btc_day: pd.dataFrame = data_source.get_data(bit.StockTicker.BITCOIN_USD, bit.StockInterval.DAY)
    btc_hour: pd.dataFrame = data_source.get_data(bit.StockTicker.BITCOIN_USD, bit.StockInterval.HOUR, 6)

    btc_target, btc_hour = bit.get_last_candle_as_result_and_modify(btc_hour)
    btc_hour_not_unique: pd.Series = bit.find_not_unique_results(btc_hour)
    if btc_hour_not_unique.size > 0:
        print("Can't continue not unique records found " + btc_hour_not_unique)
        return

    btc_week = bit.keep_last_timestamp_column_only(btc_week)
    btc_day = bit.keep_last_timestamp_column_only(btc_day)
    btc_hour = bit.keep_last_timestamp_column_only(btc_hour)

    # Merge candles as follows hour <- day <- week
    btc_data = bit.merge_data_with_timestamp(btc_hour, btc_day)
    btc_data = bit.merge_data_with_timestamp(btc_data, btc_week)

    # Since data are joined in columns, no need to keep timestamp column
    btc_data = bit.drop_column_with_name(btc_data, bit.AIDataSource.COL_TIMESTAMP)

    model = bit.AIModel(len(btc_data.columns), 2000, 1)
    bit.train(model,
              btc_data,
              pd.DataFrame(bit.get_columns_value_with_name(btc_target, bit.AIDataSource.COL_CLOSE)),
              0.0002,
              0.003,
              50000)

    # TODO split data for training/validation
    # TODO run some tests/statistics to check how good model is
    print(btc_data.head())


def run_application():
    predicting_bitcoin_price_base_on_candles()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run_application()
