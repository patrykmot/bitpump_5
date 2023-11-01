import bitpump as bit
import pandas as pd


def run_application():
    data_source: bit.AIDataSource = bit.AIDataSource(bit.StockDataSource())
    data: pd.dataFrame = data_source.get_data(bit.StockTicker.GOLD, bit.StockInterval.DAY)
    result, data = bit.get_last_candle_as_result_and_modify(data)
    data = bit.keep_last_timestamp_column_only(data)
    # data = bit.remove_not_unique_results(data) - risky to remove!!!! Result will be different!
    data = bit.remove_all_timestamps(data)
    result = bit.keep_only_close_candle(result)
    model = bit.AIModel(data.columns.size, data.columns.size * 2, result.columns.size)
    bit.train(model, data, result, 0.000001, 0.01)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run_application()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
