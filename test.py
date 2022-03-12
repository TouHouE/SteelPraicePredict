# import logging
#
# logging.basicConfig(level=logging.INFO)
# logging.debug('Hello debug')
# logging.info('Hello info')
# logging.warning('Hello warning')
# logging.error("Hello Error")
# logging.critical("Hello Critical")
from nkust.steel_price.helper.sql_controller import ToDoTabeManager
import pandas as pd
import make_train_data as md
pd.set_option('display.max_columns', 1000)


def test_1():
    STOCKS_SIGN_MAPS = {'taiwan': '台灣鋼鐵指數',
                        'djusst': '道瓊鋼鐵指數',
                        'steel': '鋼筋價格指數'}

    ALL_TABLE = ['commodity_monthly', 'commodity_daily', 'commodity_weekly']
    DB = ToDoTabeManager()
    DB.connect_to_db()

    for table_name in ALL_TABLE:
        df = DB.load_table(f'{table_name}_prediction')
        print(table_name)
        print(df)


def test_2():
    df = md.get_data()
    print(df)

test_2()
