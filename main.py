import bitpump as bit
import pandas as pd


def example_gold_day_with_super_low_error_001():
    freezer: bit.Freezer = bit.Freezer("Example_gold_day", "Example gold day with super low error.")
    data_source: bit.AIDataSource = bit.AIDataSource(bit.StockDataSource(freezer))
    data: pd.dataFrame = data_source.get_data(bit.StockTicker.GOLD, bit.StockInterval.DAY)
    result, data = bit.get_last_candle_as_result_and_modify(data)
    data = bit.keep_last_timestamp_column_only(data)
    # data = bit.remove_not_unique_results(data) - risky to remove!!!! Result will be different!
    data = bit.remove_all_timestamps(data)
    result = bit.keep_only_close_candle(result)
    model = bit.AIModel(data.columns.size, data.columns.size * 31, result.columns.size, freezer)
    model.train_me(data, result, 0.001, 0.01)


def example_merge_two_candles_with_different_interval():
    freezer: bit.Freezer = bit.Freezer("Example_merge", "Example merge two data frames.")
    # Load day and hour candles and transform data
    data_source: bit.AIDataSource = bit.AIDataSource(bit.StockDataSource(freezer))
    data_day: pd.dataFrame = data_source.get_data(bit.StockTicker.BITCOIN_USD, bit.StockInterval.DAY)
    data_hour: pd.dataFrame = data_source.get_data(bit.StockTicker.BITCOIN_USD, bit.StockInterval.HOUR)

    # Keep only latest timestamps
    data_day = bit.keep_last_timestamp_column_only(data_day)
    data_hour = bit.keep_last_timestamp_column_only(data_hour)

    # Merge data_day into data_hour so in each row we get days and hours data
    data = bit.merge_data_with_timestamp(data_hour, data_day)
    print(data.head())


def predicting_bitcoin_price_base_on_other_assets():
    freezer: bit.Freezer = bit.Freezer("predicting_bitcoin_price_base_on_other_assets"
                                       , "Predicting bitcoin price with oil gold etc")
    data_source: bit.AIDataSource = bit.AIDataSource(bit.StockDataSource(freezer))

    # Fetch hour, day, week
    # Bitcoin
    btc_week: pd.dataFrame = data_source.get_data(bit.StockTicker.BITCOIN_USD, bit.StockInterval.WEEK)
    btc_day: pd.dataFrame = data_source.get_data(bit.StockTicker.BITCOIN_USD, bit.StockInterval.DAY)
    btc_hour: pd.dataFrame = data_source.get_data(bit.StockTicker.BITCOIN_USD, bit.StockInterval.HOUR, 6)

    # Gold
    gld_week: pd.dataFrame = data_source.get_data(bit.StockTicker.GOLD, bit.StockInterval.WEEK)
    gld_day: pd.dataFrame = data_source.get_data(bit.StockTicker.GOLD, bit.StockInterval.DAY)
    gld_hour: pd.dataFrame = data_source.get_data(bit.StockTicker.GOLD, bit.StockInterval.HOUR)

    # SP 500
    sp5_week: pd.dataFrame = data_source.get_data(bit.StockTicker.SP_500, bit.StockInterval.WEEK)
    sp5_day: pd.dataFrame = data_source.get_data(bit.StockTicker.SP_500, bit.StockInterval.DAY)
    sp5_hour: pd.dataFrame = data_source.get_data(bit.StockTicker.SP_500, bit.StockInterval.HOUR)

    btc_target, btc_hour = bit.get_last_candle_as_result_and_modify(btc_hour)

    btc_week = bit.keep_last_timestamp_column_only(btc_week)
    btc_day = bit.keep_last_timestamp_column_only(btc_day)
    btc_hour = bit.keep_last_timestamp_column_only(btc_hour)
    gld_week = bit.keep_last_timestamp_column_only(gld_week)
    gld_day = bit.keep_last_timestamp_column_only(gld_day)
    gld_hour = bit.keep_last_timestamp_column_only(gld_hour)
    sp5_week = bit.keep_last_timestamp_column_only(sp5_week)
    sp5_day = bit.keep_last_timestamp_column_only(sp5_day)
    sp5_hour = bit.keep_last_timestamp_column_only(sp5_hour)

    # Merge candles as follows hour <- day <- week (from smallest to largest interval)
    btc_data = bit.merge_data_with_timestamp(btc_hour, gld_hour)
    btc_data = bit.merge_data_with_timestamp(btc_data, sp5_hour)
    btc_data = bit.merge_data_with_timestamp(btc_data, btc_day)
    btc_data = bit.merge_data_with_timestamp(btc_data, gld_day)
    btc_data = bit.merge_data_with_timestamp(btc_data, sp5_day)
    btc_data = bit.merge_data_with_timestamp(btc_data, btc_week)
    btc_data = bit.merge_data_with_timestamp(btc_data, gld_week)
    btc_data = bit.merge_data_with_timestamp(btc_data, sp5_week)

    btc_data = bit.drop_column_with_name(btc_data, bit.AIDataSource.COL_TIMESTAMP)

    btc_data, btc_data_validation = bit.split_data(btc_data, 0.90)
    btc_target, btc_target_validation = bit.split_data(btc_target, 0.90)

    model = bit.AIModel(len(btc_data.columns), 4000, 1, freezer)
    model.train_me(btc_data,
                   pd.DataFrame(bit.get_columns_value_with_name(btc_target, bit.AIDataSource.COL_CLOSE)),
                   0.0002,
                   0.003,
                   50000)

    error: float = model.calculate_error(btc_data_validation,
                                         bit.get_columns_value_with_name(btc_target_validation,
                                                                         bit.AIDataSource.COL_CLOSE))
    print("Validation error = " + str(error))
    print(btc_data.head())


def predicting_bitcoin_price_base_on_candles():
    freezer: bit.Freezer = bit.Freezer("main_predicting_bitcoin_price_base_on_candles",
                                       "Predicting bitcoin price with data merge.")
    data_source: bit.AIDataSource = bit.AIDataSource(bit.StockDataSource(freezer))

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

    btc_data, btc_data_validation = bit.split_data(btc_data, 0.93)
    btc_target, btc_target_validation = bit.split_data(btc_target, 0.93)

    model = bit.AIModel(len(btc_data.columns), 2000, 1, freezer)
    model.train_me(btc_data,
                   pd.DataFrame(bit.get_columns_value_with_name(btc_target, bit.AIDataSource.COL_CLOSE)),
                   0.0001,
                   0.004,
                   50000)

    error: float = model.calculate_error(btc_data_validation,
                                         bit.get_columns_value_with_name(btc_target_validation,
                                                                         bit.AIDataSource.COL_CLOSE))
    print("Validation error = " + str(error))  # 0.23!!!!

    print(btc_data.head())


def run_application():
    predicting_bitcoin_price_base_on_candles()
    # example_gold_day_with_super_low_error_001()
    # predicting_bitcoin_price_base_on_other_assets()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run_application()
