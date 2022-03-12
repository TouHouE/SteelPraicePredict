from nkust.steel_price.helper.io import get_files, load_model, load_csv
from nkust.steel_price.helper.preprossor import min_max_scaler
from nkust.steel_price.utilities.save_path import PATH
from fbprophet import Prophet
import pandas as pd
import argparse
import datetime
import make_data as md

pd.set_option('display.max_rows', 500)
TYPE_MAP = {
    'D': 'daily',
    'M': 'monthly',
    'W': 'weekly'
}

MAPS = {'taiwan': '台灣鋼鐵指數',
        'djusst': '道瓊鋼鐵指數',
        'steel': '鋼筋價格指數'}


def load(stock_name) -> {}:
    model_ = load_model(stock_name)
    model_name = stock_name.split('.')[0]

    map = {
        'name': model_name,
        'model': model_
    }

    return map


def isHistory2Future(date_df: pd.DataFrame):
    if len(date_df) < 2:
        return True
    return date_df.iloc[0]['ds'] < date_df.iloc[1]['ds']


def precessing(model: Prophet, date_: pd.DataFrame) -> pd.DataFrame:
    """

    :param model:
    :return: the data after MinMaxScaler
    """
    if len(model.extra_regressors) != 0:
        cols = date_.columns
        rename_dict = {}

        for col in cols:
            date_[col] = date_[col].fillna(date_[col].mean())

        rename_dict['c_date'] = 'ds'
        rename_dict['c_price'] = 'y'
        date_ = date_.rename(rename_dict, axis=1)

        if not isHistory2Future(date_):
            date_ = date_.iloc[::-1]
    else:
        date_ = model.make_future_dataframe(periods=10 * 1, freq='D', include_history=True)

    return min_max_scaler(date_)


def predict(model_stock_name, start_date, predict_range, type) -> list:
    """
    :param model_stock_name:
    :param start_date:
    :param predict_range:
    :param type: only "D", "W", "M"
    :return:
    """
    df_and_save = md.start(type=type,
                           stock_name=model_stock_name,
                           start_date=start_date,
                           range=predict_range)
    # TODO: Fix model name problem
    model_map = load_model(f"{model_stock_name}_train_{TYPE_MAP[type]}2.json")
    model = model_map
    df = df_and_save['data']
    date_range = precessing(model, df)
    predicted = model.predict(date_range)

    # print(predicted)
    return {'predict': predicted, 'history': df_and_save['keep']}


def update(all_df: pd.DataFrame, type: str):
    # print(all_df)
    size = len(all_df)
    md.DB.delete_all(f'commodity_{TYPE_MAP[type]}_prediction')
    for i in range(size):
        # bprint(all_df.iloc[i])
        md.DB.insert(table_name=f'commodity_{TYPE_MAP[type]}_prediction', info=all_df.iloc[i])


def main(parser: argparse.ArgumentParser):
    args = parser.parse_args()

    PATH.MODEL_PATH = args.weight_dir
    # PATH.CSV_PATH = parser.input_csv
    predict_range_and_type = args.predict_range
    predicted_dir = args.predicted_dir
    type = predict_range_and_type[-1]
    true_range = int(predict_range_and_type[:-1])

    model_and_stock_names = ['djusst', 'taiwan', 'steel']
    start_date = pd.to_datetime(args.start_date)
    now = pd.to_datetime(datetime.datetime.now())
    all_results = list()

    for model_and_stock_name in model_and_stock_names:
        map = predict(model_and_stock_name, predict_range=true_range, start_date=start_date, type=type)
        results = map['predict']
        # else:
        #     #TODO: Start Date Not include in DB
        #     start_date = datetime.datetime.now()
        #     sub_range = args.start_daate - start_date
        #     results = predict(model_and_stock_name)

        if predicted_dir is not None:
            results.to_csv(f'{predicted_dir}/{model_and_stock_name}_{type}.csv', index=False)
            map['history'].to_csv(f'{predicted_dir}/{model_and_stock_name}_{type}_DF.csv', index=False)

        results = results[['ds', 'yhat']]
        results['c_name'] = MAPS[model_and_stock_name]
        all_results.append(results)

    # print(df)
    df = pd.concat(all_results, axis=0)
    update(df, type=type)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--weight_dir', help='The dictionary which model weight saved', default='./model')
    parser.add_argument('--predicted_dir', help='The dictionary where predicted data to save', default=None)
    # parser.add_argument('--input_csv', help='The variable model need', required=True)
    parser.add_argument('--predict_range',
                        help='How long you want to predict (days num)D (weeks num)W (months num)M\nD: Day\n W: Week\nM: Month',
                        default=1)
    parser.add_argument('--start_date', help='Predict Start date, format is "yyyy-MM-dd"')
    # parser.add_argument('--target_table_name')
    # parser.add_argument('--stock_name', help='Which Stock(model). just has "djusst", "steel", "taiwan')
    main(parser)
