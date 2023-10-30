
import bitpump as bit
import pandas as pd

import numpy as np

def run_application():
    data_source: bit.AIDataSource = bit.AIDataSource(bit.StockDataSource())
    data: pd.dataFrame = data_source.get_data(bit.StockTicker.GOLD, bit.StockInterval.DAY)
    data = bit.create_date_column_from_index(data)
    model = bit.AIModel(12, 200, 1)
    bit.train(model,)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run_application()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
