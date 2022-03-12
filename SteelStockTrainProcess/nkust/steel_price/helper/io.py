import json
from fbprophet import Prophet
from fbprophet.serialize import model_to_json, model_from_json
import joblib
from os import listdir
from os.path import isfile
import pandas as pd

import matplotlib.pyplot as plt

from nkust.steel_price.utilities.describe import Describe
from nkust.steel_price.utilities.save_path import SavePath

JSON = 0
PKL = 1

describe = Describe()
all_path = SavePath()


def save_model(model: Prophet, model_name: str, type=JSON):
    """
        model is after fit Prophet model and model_name didn't need sub file name
    :param model:
    :param model_name:
    :param type:
    :return:
    """


    if type == JSON:
        with open(f'{all_path.MODEL_PATH}/{model_name}.json', 'w') as fout:
            json.dump(model_to_json(model), fout)
    elif type == PKL:
        joblib.dump(model, f'{all_path.MODEL_PATH}/{model_name}.pkl')


def load_model(model_name: str) -> Prophet:

    if model_name.find('json') != -1:
        with open(f'{all_path.MODEL_PATH}/{model_name}', 'rb') as fin:
            m = model_from_json(json.load(fin))
        return m
    elif model_name.find('pkl') != -1:
        return joblib.load(model_name)


def get_files(path: str) -> []:
    files = []
    list_dir = listdir(path)

    for file in list_dir:
        if isfile(f'{path}/{file}'):
            files.append(file)
        else:
            pass
    return files


def make_describe(target_name, target_type: int, train_record=None, validate_record=None):
    path = ''

    if train_record is None and validate_record is None:
        info = describe.record()
    elif train_record is not None:
        data_source = train_record['data_source']
        split_date = train_record['split_date']
        prophet = train_record['prophet']
        info = f'Data Source:\t{data_source}\nSplit Date:\t{split_date}\n'

        if train_record['using_scaler']:
            info += 'Using MinMaxScaler\n'

        info += '\nProphet argument part:\n'

        for arg in prophet:
            info += f' {arg} \t= \t{prophet[arg]}\n'

        if train_record['using_regressor']:
            regressors = train_record['regressor']
            info += f'\nRegressor part:\n'

            for reg_tag in regressors:
                reg = regressors[reg_tag]
                info += f' {reg["name"]}\t\tprior_scale = {reg["prior_scale"]}\n'
    elif validate_record is not None:
        info = validate_record
    if target_type == describe.TYPE_MODEL:
        path = f'{all_path.MODEL_PATH}/info/info.txt'
    elif target_type == describe.TYPE_PREDICT:
        path = f'{all_path.PREDICT_PATH}/info/info.txt'
    elif target_type == describe.TYPE_VALIDATE:
        path = f'{all_path.VALIDATE_PATH}/info/info.txt'

    with open(path, 'a+') as fin:
        if isinstance(target_name, str):
            fin.write(f'Title:{target_name}\n===========\nInfo:\n{info}\n============\n\n')
        else:
            fin.write('Title:\n')

            for name in target_name:
                fin.write(f'\t{name}\n')
            fin.write(f'===========\nInfo:\n{info}\n============\n\n')


def load_csv(file_name:str) -> pd.DataFrame:
    df = pd.read_csv(f'{all_path.TRAIN_PATH}/{file_name}')
    cols = df.columns

    for col in cols:
        # print(f'debug: \n\tcol != ds: {col != "ds"}\n\tcol != Date: {col != "Date"}')

        if col != 'ds' and col != 'Date' and col != 'new_date':
            print(f'debug: {col}')
            df[col] = pd.to_numeric(df[col])
        else:
            df[col] = pd.to_datetime(df[col])

    return df

def load_csv(path:SavePath, name:str) -> pd.DataFrame:
    df = pd.read_csv(f'{all_path.TRAIN_PATH}/{path}/{name}')
    cols = df.columns

    for col in cols:
        # print(f'debug: \n\tcol != ds: {col != "ds"}\n\tcol != Date: {col != "Date"}')

        if col != 'ds' and col != 'c_date' and col != 'new_date':
            print(f'debug: {col}')
            df[col] = pd.to_numeric(df[col])
        else:
            df[col] = pd.to_datetime(df[col])

    return df
