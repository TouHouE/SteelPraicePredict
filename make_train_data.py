import pandas as pd
pd.set_option('display.max_row', 500)
DATE = 'Date'
PRICE = 'Close'

def move(df, move_len) -> pd.DataFrame:
    df.loc[:, DATE] = pd.to_datetime(df[DATE])
    # print(f'debug:\n{df}')

    if df[DATE].iloc[0] < df[DATE].iloc[1]:
        df = df[::-1]

    df_dont_move = df[[DATE, PRICE]]
    df_regressor = df[['Open', 'High', 'Low']]

    df_regressor = df_regressor[move_len:].reset_index()
    df_dont_move = df_dont_move[:-1 * move_len].reset_index()
    df_final = pd.concat([df_dont_move, df_regressor], axis=1).drop(['index'], axis=1)
    # print(f'debug:\n{df_final}\n')
    return df_final



def get_data() -> pd.DataFrame:
    df = pd.read_csv('./for_demo/djusst_yahoo_m.csv')
    df_move = move(df, 12)
    df = df.rename({
        DATE: 'ds',
        PRICE: 'y',
    }, axis=1)

    return {'train': df_move, 'org': df}
