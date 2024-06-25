import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Function to create rolling window data for time series analysis
def rolling_window(series, window):
    """
    Generate rolling window data for time series analysis.

    Parameters:
    - series: array-like, time series data
    - window: int, size of the rolling window

    Returns:
    - df: pandas DataFrame, containing the rolling window data
    - scaler: MinMaxScaler object, used for normalization
    """
    data = []
    scaler = MinMaxScaler(feature_range=(-1, 1))

    for i in range(len(series) - window):
        example = np.array(series[i:i + window + 1])
        example = scaler.fit_transform(example.reshape(-1, 1)).flatten()
        data.append(example)

    df = pd.DataFrame(data)
    return df, scaler

# Function to split data into training and testing sets for time series forecasting
def train_test_split_window(data, horizon):
    """
    Split data into training and testing sets for time series forecasting.

    Parameters:
    - data: pandas DataFrame, containing features and target
    - horizon: int, number of time steps to forecast into the future

    Returns:
    - X_train: pandas DataFrame, training features
    - X_test: pandas DataFrame, testing features
    - y_train: pandas Series, training target
    - y_test: pandas Series, testing target
    """
    X = data.iloc[:, :-1]  # features
    y = data.iloc[:, -1]   # target

    X_train = X[:-horizon]  # features train
    X_test = X[-horizon:]   # features test

    y_train = y[:-horizon]  # target train
    y_test = y[-horizon:]   # target test

    return X_train, X_test, y_train, y_test

# Function for recursive multi-step forecasting using a trained model
def recursive_multistep_forecasting(X_test, model, horizon):
    """
    Perform recursive multi-step forecasting using a trained model.

    Parameters:
    - X_test: pandas DataFrame, testing features
    - model: trained machine learning model
    - horizon: int, number of time steps to forecast into the future

    Returns:
    - preds: list, predicted values for each time step in the horizon
    """
    # The example consists of the last observed values seen
    # In practice, it is the first example from the test set
    example = X_test.iloc[0].values.reshape(1, -1)

    preds = []
    for i in range(horizon):
        pred = model.predict(example)[0]
        preds.append(pred)

        # Discard the value from the first position of the feature vector
        example = example[:, 1:]

        # Add the predicted value to the last position of the feature vector
        example = np.append(example, pred)
        example = example.reshape(1, -1)

    return preds

# Function to convert a date string into a pandas Timestamp
def convert_date(date_string):
    """
    Convert a date string into a pandas Timestamp.

    Parameters:
    - date_string: str, date in 'YYYYMM' format

    Returns:
    - pd.Timestamp object representing the date
    """
    year_month = date_string.strip()
    year = int(year_month[:4])
    month = int(year_month[4:])
    return pd.Timestamp(year=year, month=month, day=1)
