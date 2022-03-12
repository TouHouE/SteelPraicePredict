from sklearn.preprocessing import MinMaxScaler
import pandas as pd


def min_max_scaler(df:pd.DataFrame) -> pd.DataFrame:
    y_series = pd.Series()
    drop_list = ['ds']
    if 'y' in df.columns:
        y_series = df['y']
        drop_list.append('y')

    ds_series = df['ds']
    df = df.drop(drop_list, axis=1)
    cols = df.columns

    for col in cols:
        df[col] = MinMaxScaler(feature_range=(-1, 1)).fit_transform(df[[col]])
    if 'y' in df.columns:
        df['y'] = y_series
    df['ds'] = ds_series
    return df


