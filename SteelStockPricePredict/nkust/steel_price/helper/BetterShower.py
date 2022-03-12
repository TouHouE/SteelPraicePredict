import matplotlib.pyplot as plt
from matplotlib.dates import AutoDateLocator, AutoDateFormatter
import pandas


def show(true: pandas.DataFrame, predict=None):
    # true, predict = make_data()
    fig = plt.figure(facecolor='w', figsize=(10, 6))
    ax = fig.add_subplot(111)
    print(true)

    if predict is not None:
        # print(f'debug:\n{predict.columns}')
        predict_time = predict['ds'].dt.to_pydatetime()
        ax.plot(predict['ds'].dt.to_pydatetime(), predict['yhat'], '-', label='Predict', c='#FF0000')
        ax.plot(predict_time, predict['yhat_upper'], '-', label='high_p', c='#0072B2')
        ax.plot(predict_time, predict['yhat_lower'], '-', label='low_p', c='#00C2B2')
        ax.fill_between(predict_time, predict['yhat_lower'], predict['yhat_upper'], color='#0072B2', alpha=0.2)

    if 'y' in true:
        ax.plot(true['ds'].dt.to_pydatetime(), true['y'], '-', label='True', c='k')

    locator = AutoDateLocator(interval_multiples=False)
    formatter = AutoDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)
    fig.tight_layout()
    return fig
