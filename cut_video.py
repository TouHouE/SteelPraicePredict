# from nkust.steel_price.helper.sql_controller import ToDoTabeManager
import make_data as md
from nkust.steel_price.helper import io
import pandas as pd

ALL_TABLE = ['commodity_monthly', 'commodity_daily', 'commodity_weekly']
TABLE_TO_TYPE_MAPS = {
    ALL_TABLE[0]: 'monthly',
    ALL_TABLE[1]: 'daily',
    ALL_TABLE[2]: 'weekly'
}
# ALL_STOCK = {
#     'djusst': '道瓊鋼鐵指數'
# }

ALL_STOCK = md.STOCKS_SIGN_MAPS

default_range = {
    'commodity_monthly': 12,
    'commodity_daily': 30,
    'commodity_weekly': 12
}

def driff(df, driff_len) -> pd.DataFrame:
    df.loc[:, 'c_date'] = pd.to_datetime(df['c_date'])
    print(f'debug:\n{df}')

    if df['c_date'].iloc[0] < df['c_date'].iloc[1]:
        df = df[::-1]

    df_dont_move = df[['c_date', 'c_price']]
    df_regressor = df[['c_open', 'c_high', 'c_low']]

    df_regressor = df_regressor[driff_len:].reset_index()
    df_dont_move = df_dont_move[:-1 * driff_len].reset_index()
    df_final = pd.concat([df_dont_move, df_regressor], axis=1).drop(['index'], axis=1)
    # print(f'debug:\n{df_final}\n')
    return df_final


if __name__ == '__main__':
    for table_name in ALL_TABLE:
        df = md.DB.load_table(table_name)

        for stock_name_sign in ALL_STOCK.keys():
            print(f'This Table Name: {table_name}')
            df_target = df[df['c_name'] == md.STOCKS_SIGN_MAPS[stock_name_sign]].reset_index()
            df_for_train = driff(df_target, driff_len=default_range[table_name])
            df_for_train.to_csv(f'./sql/{stock_name_sign}/{stock_name_sign}_train_{TABLE_TO_TYPE_MAPS[table_name]}.csv', index=False)
