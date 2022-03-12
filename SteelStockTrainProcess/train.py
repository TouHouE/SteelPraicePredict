import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from fbprophet import Prophet
import matplotlib.pyplot as plt
import make_train_data as md
from nkust.steel_price.helper import BetterShower as bs

hyper_param = {
    'data_source': f'./for_demo/djusst_yahoo_m.csv',
    'using_scaler': True,
    'split_date': '2000',
    'prophet': {
        'changepoint_range': 1,
        'n_changepoints': 100,
        'changepoint_prior_scale': 0.85,
        'seasonality_prior': 4,
        'interval_width': 0.5
    },
    'drop': [
        'id'
    ],
    'using_regressor': True,
    'regressor': {

        'regressor_1': {
            'name': 'High',
            'prior_scale': 10
        },
        'regressor_2': {
            'name': 'Low',
            'prior_scale': 10
        },
        'regressor_3': {
            'name': 'Open',
            'prior_scale': 10
        }
    }
}
DATE = 'Date'
PRICE = 'Close'

def build_data():
    # df = pd.read_csv(f'{hyper_param["data_source"]}')
    map = md.get_data()
    data = map['train']
    data[DATE] = pd.to_datetime(data[DATE])
    data_re = data.rename({
        PRICE: 'y',
        DATE: "ds"
    }, inplace=False, axis='columns')

    # drop out null
    for index in range(len(data_re)):
        if str(data_re.iloc[index]['ds']) == 'NaT':
            print(f'cut at {index}')
            data_re = data_re.iloc[:index - 1]
            break

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

    return {'train': data_re, 'org': map['org']}

def main():
    map = build_data()
    b_data = map['train']
    b_data = b_data.iloc[::-1]  # if date series is small to big, delete this line

    # choose how long for training
    for i in range(len(b_data['ds'])):
        print(f'[{i}]: {b_data["ds"].iloc[i]}')
    cut_point = int(input('choose split time\n> '))
    hyper_param['split_date'] = b_data['ds'].iloc[cut_point]  # Nothing, just make record

    train = b_data.iloc[:cut_point]
    prophet_param = hyper_param['prophet']  # Using hyper-variable setting arguments of Prophet

    count = 0
    """training part"""

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
    future_range = b_data if hyper_param['using_regressor'] else m.make_future_dataframe(365 * 4, freq='D',
                                                                                         include_history=True)
    print(future_range)
    """Make model performance Graphics"""
    predicted = m.predict(future_range)
    bs.show(true=map['org'], predict=predicted)
    # fig = m.plot(predicted)
    # m.plot_components(predicted)
    # a = add_changepoints_to_plot(fig.gca(), m, predicted)
    plt.legend()
    plt.show()

main()