import pandas as pd
from nkust.steel_price.helper.sql_controller import ToDoTabeManager

STOCKS_SIGN_MAPS = {'taiwan': '台灣鋼鐵指數',
                    'djusst': '道瓊鋼鐵指數',
                    'steel': '鋼筋價格指數'}
TYPE_MAP = {
    'D': 'daily',
    'M': 'monthly',
    'W': 'weekly'
}
TEN_DAY = pd.to_timedelta('10d')
ONE_DAY = pd.to_timedelta('1d')
ONE_WEEK = pd.to_timedelta('7d')
SIX_M = pd.to_timedelta(f'{30 * 6}d')
ONE_M = pd.to_timedelta('30d')
DB = ToDoTabeManager()
DB.connect_to_db()


def M(how_long: int):
    return pd.to_timedelta(f'{30 * how_long}d')


def D(how_long: int):
    return pd.to_timedelta(f'{how_long}d')


def W(how_long: int):
    return pd.to_timedelta(f'{7 * how_long}d')


def monthly(name: str, start_date=None, range=12):
    """

    :param name: the stock name, like djusst, taiwan, steel...
    :param range: how long you want to predict
    :return:
    """
    range = int(range)
    RANGE_M = M(range)
    df = DB.load_table("commodity_monthly")
    df.loc[:, 'c_date'] = pd.to_datetime(df.loc[:, 'c_date'])
    df = df[df['c_name'] == STOCKS_SIGN_MAPS[name]]  # split DataFrame by column value

    if start_date is None or pd.to_datetime(start_date) not in df['c_date'].values:
        start_date = df['c_date'].max()
        end_date = start_date + RANGE_M
        start_date = start_date + ONE_M

    else:
        end_date = start_date + RANGE_M
        start_date = start_date + ONE_M

    future_date = pd.date_range(start_date, end_date, freq='1M')

    df_param = df[['c_open', 'c_high', 'c_low', 'c_price']]
    future_param = df_param.iloc[:range].reset_index()
    date = pd.DataFrame({'c_date': future_date})
    future_ = pd.concat([date, future_param], axis=1)
    future_.drop(['index'], axis=1, inplace=True)
    return {'data': future_, 'keep': df}


def daily(name: str, start_date=None, range=30):
    """
        Using stock sign name
    :param name: the stock sign name
    :param range: how long you want to predict
    :return:
    """
    range = int(range)
    RANGE_D = D(range)

    df = DB.load_table('commodity_daily')
    df = df[df['c_name'] == STOCKS_SIGN_MAPS[name]]

    if start_date is None or pd.to_datetime(start_date) not in df['c_date'].values:
        start_date = df['c_date'].max()
        end_date = start_date + RANGE_D
        start_date = start_date + ONE_DAY

    else:
        end_date = start_date + RANGE_D
        start_date = start_date + ONE_DAY

    # Produce whole date
    future_date = pd.date_range(start_date, end_date, freq='1d')

    df_param = df[['c_open', 'c_high', 'c_low', 'c_price']]
    future_param = df_param.iloc[:range].reset_index()  # predict part

    date = pd.DataFrame({'c_date': future_date})

    future_ = pd.concat([date, future_param], axis=1)
    future_.drop(['index'], axis=1, inplace=True)

    return {'data': future_, 'keep': df}


def weekly(name: str, start_date=None, range=12):
    range = int(range)
    RANGE_W = W(range)

    df = DB.load_table('commodity_weekly')
    # print(f'debug: weekly df\n{df}')
    df.loc[:, 'c_date'] = pd.to_datetime(df.loc[:, 'c_date'])
    df = df[df['c_name'] == STOCKS_SIGN_MAPS[name]]

    if start_date is None or pd.to_datetime(start_date) not in df['c_date'].values:
        start_date = df['c_date'].max()
        end_date = start_date + RANGE_W
        start_date = start_date + ONE_WEEK
    else:
        end_date = start_date + RANGE_W
        start_date = start_date + ONE_WEEK

    # Produce whole date
    future_date = pd.date_range(start_date, end_date, freq='1W')

    df_param = df[['c_open', 'c_high', 'c_low', 'c_price']]
    future_param = df_param.iloc[:range].reset_index()  # predict part

    date = pd.DataFrame({'c_date': future_date})

    future_ = pd.concat([date, future_param], axis=1)
    future_.drop(['index'], axis=1, inplace=True)
    return {'data': future_, 'keep': df}
#
# def start(is_daily: bool, stock_name: str, start_date, range: int):
#     return daily(name=stock_name, start_date=start_date, range=range) if is_daily else monthly(stock_name, range)


def start(type: str, stock_name: str, start_date, range: int):
    """
        Using Stock Sign Name, type Using 'D', 'W', 'M'
    :param type:
    :param stock_name:
    :param start_date:
    :param range:
    :return:
    """

    if type == 'D':
        return daily(name=stock_name, start_date=start_date, range=range)
    if type == 'W':
        return weekly(name=stock_name, start_date=start_date, range=range)
    if type == 'M':
        return monthly(name=stock_name, start_date=start_date, range=range)
