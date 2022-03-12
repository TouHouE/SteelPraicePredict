import argparse
from nkust.steel_price.helper import BetterShower as bs
import pandas as pd
import matplotlib.pyplot as plt

def read_df(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    time_col = 'c_date' if 'c_date' in df.columns else 'ds'

    df[time_col] = pd.to_datetime(df[time_col])

    for key in df.columns:
        if key == time_col:
            continue
        try:
            df[key] = pd.to_numeric(df[key])
        except Exception as e:
            continue
    return df


def cut_history(history: pd.DataFrame, min_date, max_date, range) -> pd.DataFrame:
    left_lim = min_date - range
    right_lim = max_date + range
    history = history[history['c_date'] < right_lim]
    history = history[history['c_date'] > left_lim]
    history = history.rename({
            'c_price': 'y',
            "c_date": "ds"
        }, inplace=False, axis='columns')
    return history

def main(arg):
    predicted_path = arg.predict_csv
    history_path = arg.history_csv
    image_name = arg.image_name
    predict_df = read_df(predicted_path)
    history_df = read_df(history_path)
    min_date = predict_df['ds'].min()
    max_date = predict_df['ds'].max()
    final_history_df = cut_history(history_df, min_date, max_date, range=max_date - min_date)
    print(final_history_df)
    print(predict_df)

    bs.show(final_history_df, predict_df)
    plt.legend()

    if image_name is None:
        plt.show()
    else:
        plt.savefig(f'./image/{image_name}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--predict_csv', type=str, help='The predicted csv file')
    parser.add_argument('--history_csv', type=str, help='The csv file of history csv')
    parser.add_argument('--image_name', type=str, help='The name which plot image ')
    main(parser.parse_args())