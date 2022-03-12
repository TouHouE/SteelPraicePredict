import numpy as np

from nkust.steel_price.helper.BetterShower import show
from nkust.steel_price.helper.io import save_model, load_model, get_files, make_describe, load_csv
from nkust.steel_price.helper.preprossor import min_max_scaler
from nkust.steel_price.utilities.describe import Describe
from nkust.steel_price.utilities.save_path import SavePath

import matplotlib.pyplot as plt

import pandas as pd

from fbprophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import add_changepoints_to_plot

from sklearn.preprocessing import MinMaxScaler

all_path = SavePath()
describe = Describe()

MODEL_TYPE = ('json', 'pkl')
JSON = 0
PKL = 1
horizon = '365 days'
period = '1 days'
stock_name = ['steel', 'djusst', 'taiwan']
stock_index = 0
freq = ['monthly', 'daily', 'weekly']

hyper_param = {
    'data_source': f'{stock_name[stock_index]}/{stock_name[stock_index]}_train_{freq[2]}',
    'using_scaler': True,
    'split_date': '2000',
    'prophet': {
        'changepoint_range': 1,
        'n_changepoints': 100,
        'changepoint_prior_scale': 0.35,
        'seasonality_prior': 4,
        'interval_width': 0.5
    },
    'drop': [
        'id'
    ],
    'using_regressor': True,
    'regressor': {

        'regressor_1': {
            'name': 'c_high',
            'prior_scale': 10
        },
        'regressor_2': {
            'name': 'c_low',
            'prior_scale': 10
        },
        'regressor_3': {
            'name': 'c_open',
            'prior_scale': 10
        }
    }
}

DATE = 'c_date' # Using Prophet model must change date column name to ds
Y = 'c_price' # Using Prophet model must change price column name to y

TRUE_PATH = f'{all_path.TRAIN_PATH}/djusst/djusst_test_monthly_regressor.csv'

def build_model():
    def build_data(using_regressor_: int, do_preprocessing_: int):

        """Select data sources"""
        if using_regressor_ == 0:
            hyper_param['data_source'] = f"{hyper_param['data_source']}.csv"
            hyper_param['using_regressor'] = False

            df = pd.read_csv(f'{all_path.TRAIN_PATH}/{hyper_param["data_source"]}')
        else:
            hyper_param['data_source'] = f"{hyper_param['data_source']}.csv"
            hyper_param['using_regressor'] = True

            df = pd.read_csv(f'{all_path.TRAIN_PATH}/{hyper_param["data_source"]}')

#        data = df.drop(['Change %'], axis=1)  # Through out Adj Close
        data = df
        data[DATE] = pd.to_datetime(data[DATE])
        data_re = data.rename({
            'c_price': 'y',
            "c_date": "ds"
        }, inplace=False, axis='columns')

        # drop out null
        for index in range(len(data_re)):
            if str(data_re.iloc[index]['ds']) == 'NaT':
                print(f'cut at {index}')
                data_re = data_re.iloc[:index - 1]
                break


        """Decision do data preprocessing or not"""
        if do_preprocessing_ == 1:
            hyper_param['using_scaler'] = True
            date_series = data_re['ds']
            y_series = data_re['y']
            data_re = data_re.drop(['ds', 'y'], axis=1)
            cols = data_re.columns

            for col in cols:
                if data_re[col].dtype != 'float64':
                    continue
                data_re[col] = data_re[col].fillna(data_re[col].mean())
                data_re[col] = MinMaxScaler(feature_range=(-1, 1)).fit_transform(data_re[[col]])

            data_re['ds'] = date_series
            data_re['y'] = y_series
            print(data_re)
        else:
            hyper_param['using_scaler'] = False

        return data_re

    def save(model, name):
        # print('don\'t save the model type \"no\"')
        # name = str(input('Type model name:'))

        if name == 'no':
            print('not save')
            pass
        else:
            make_describe(target_name=name, target_type=describe.TYPE_MODEL, train_record=hyper_param)
            save_model(model, name)
            # plt.savefig(f'{all_path.IMG_PATH}/img_{name.split(",")[0]}.png')

    using_regressor = int(input("Using add regressor?(1/0)\n> "))
    do_preprocessing = int(input("Doing preprocessing? (1/0)\n> "))
    b_data = build_data(using_regressor_=using_regressor, do_preprocessing_=do_preprocessing)
    b_data = b_data.iloc[::-1] # if date series is small to big, delete this line

    # choose how long for training
    for i in range(len(b_data['ds'])):
        print(f'[{i}]: {b_data["ds"].iloc[i]}')
    cut_point = int(input('choose split time\n> '))
    hyper_param['split_date'] = b_data['ds'].iloc[cut_point]  # Nothing, just make record

    train = b_data.iloc[:cut_point]
    prophet_param = hyper_param['prophet']  # Using hyper-variable setting arguments of Prophet

    count = 0
    """training part"""
    while prophet_param['changepoint_prior_scale'] <= float(1.0):
        # prophet_param['changepoint_prior_scale'] = float(cp_p)
        count += 1
        m = Prophet(changepoint_range=prophet_param['changepoint_range'],
                    n_changepoints=prophet_param['n_changepoints'],
                    changepoint_prior_scale=prophet_param['changepoint_prior_scale'])

        # Check if adding regressor into model
        if hyper_param['using_regressor']:
            all_regressor = hyper_param['regressor']

            for reg_tag in all_regressor:
                regressor = all_regressor[reg_tag]
                m.add_regressor(regressor['name'], prior_scale=regressor['prior_scale'])


        m.fit(train)
        '''Training Complete'''
        future_range = b_data if hyper_param['using_regressor'] else m.make_future_dataframe(365 * 1, freq='D', include_history=True)
        """Make model performance Graphics"""
        predicted = m.predict(future_range)
        fig = m.plot(predicted)
        m.plot_components(predicted)
        a = add_changepoints_to_plot(fig.gca(), m, predicted)
        plt.legend()
        plt.show()

        prophet_param['changepoint_prior_scale'] += 0.5
        save(m, hyper_param['data_source'].split('/')[-1].split('.')[0] + str(count))


"""
    Nothing, just a better way for showing graphic
"""
def show_part():
    def get_file_name() -> str:
        files = get_files(all_path.PREDICT_PATH)
        print('choose predict data:')

        for i in range(len(files)):
            print(f'{i}){files[i]}')

        return files[int(input())]

    def make_data(data_name: str):
        true_ = pd.read_csv(f'{all_path.TRAIN_PATH}/djusst/djusst_train_daily.csv')
        true_ = true_.rename({DATE: 'ds', Y: 'y'}, inplace=False, axis=1)
        print(f'debug: loading: {all_path.PREDICT_PATH}/{data_name}')
        predict_ = pd.read_csv(f'{all_path.PREDICT_PATH}/{data_name}')
        # predict_ = predict_.rename({'yhat': 'y'}, inplace=False, axis=1)
        true_['ds'] = pd.to_datetime(true_['ds'])
        predict_['ds'] = pd.to_datetime(predict_['ds'])

        return true_, predict_

    true, predict = make_data(get_file_name())

    show(true, predict)
    plt.legend()
    plt.show()


def def_predict():
    def load() -> {}:
        models = get_files(all_path.MODEL_PATH)
        print('choose model：')

        for i in range(len(models)):
            print(f'{i}){models[i]}')

        model_name = models[int(input())]
        model_ = load_model(model_name)
        model_name = model_name.split('.')[0]

        map = {
            'name': model_name,
            'model': model_
        }

        return map

    def precessing(model_: Prophet, do: int) -> pd.DataFrame:

        if len(model_.extra_regressors) != 0:
            date_ = load_csv('taiwan', 'taiwan_test_monthly_regressor.csv')
            cols = date_.columns
            rename_dict = {}

            for col in cols:
                print(col)
                date_[col] = date_[col].fillna(date_[col].mean())

            rename_dict['c_date'] = 'ds'
            rename_dict['c_price'] = 'y'
            date_ = date_.rename(rename_dict, axis=1)
            date_ = date_.iloc[::-1]
        else:
            date_ = model_.make_future_dataframe(periods=10 * 1, freq='D', include_history=True)

        if do == 1:
            return min_max_scaler(date_)
        else:
            return date_

    model_map = load()
    model = model_map['model']

    do_scale = int(input('Doing data preprocessing?(1/0)\n> '))
    date_range = precessing(model, do_scale)

    predicted = model.predict(date_range)

    model.plot_components(predicted)
    plt.show()

    predicted_file_name = f'predict_{model_map["name"]}.csv'
    predicted.to_csv(f'{all_path.PREDICT_PATH}/{predicted_file_name}', index=False)

def make_validate():
    def load(model_index=None) -> {}:
        if model_index is None:
            models = get_files(all_path.MODEL_PATH)
            print('choose model：')

            for i in range(len(models)):
                print(f'{i}){models[i]}')

            model_name = models[int(input())]
            map = {
                'name': model_name,
                'model': load_model(model_name)
            }
        else:
            map = {
                'name': f'model_new_{model_index}.json',
                'model': load_model(f'model_new_{model_index}.json')
            }

        return map

    model_map = load()
    model = model_map['model']
    model_name = model_map['name']
    pure_model_name = model_name.split('.')[0]
    his_min = model.history['ds'].min()
    his_max = model.history['ds'].max()
    print(f'history range in model are {his_min} ~ {his_max}')

    max = pd.to_datetime(str(input('type end days: ')))
    # max = pd.to_datetime(date)
    initial = max - his_min
    df_cv = cross_validation(model, initial=initial, horizon=horizon, period=period)
    df_p = performance_metrics(df_cv)

    saved_name = [
        f'validate_{pure_model_name}.csv',
        f'performance_{pure_model_name}.csv'
    ]

    make_describe(target_name=saved_name, target_type=describe.TYPE_VALIDATE,
                  validate_record=f'validated by {pure_model_name}.json')
    df_cv.to_csv(f'{all_path.VALIDATE_PATH}/{saved_name[0]}', index=False)
    df_p.to_csv(f'{all_path.VALIDATE_PATH}/{saved_name[1]}', index=False)

# show model each loss function curve in validation
def show_performance():
    files = get_files(all_path.VALIDATE_PATH)
    performance = []

    for file in files:
        if file.find('validate') == -1:
            df = pd.read_csv(f'{all_path.VALIDATE_PATH}/{file}')
            for col in df.columns:
                if col != 'horizon':
                    df[col] = pd.to_numeric(df[col])
            data_map = {
                'df': df,
                'name': file.split('.')[0]
            }
            performance.append(data_map)

    for p in performance:
        time_axis = np.arange(len(p['df']))

        if int(p['name'].split('_')[-1]) > 10:
            plt.plot(time_axis, p['df']['mse'], label=p['name'])

    plt.xlabel('predict day')
    plt.ylabel('loss(MAE)')
    plt.legend()
    plt.show()


def main():
    user_choose = int(input('1)train\n2)validate\n3)predict\n4)show\n5)show performance\n> '))

    if user_choose == 1:
        build_model()
    elif user_choose == 2:
        make_validate()
    elif user_choose == 3:
        def_predict()
    elif user_choose == 4:
        show_part()
    elif user_choose == 5:
        show_performance()
    else:
        print('Not effect input')


if __name__ == "__main__":
    main()
