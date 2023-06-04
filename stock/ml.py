import numpy as np
import pandas as pd
from datetime import date

from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import yfinance as yf

def load_ds():
    return pd.read_csv('./data/stockNames.csv', usecols=['Symbol', 'Name'])

def load_stock(user_input: str):
    stock_info = yf.download(user_input, start='2023-01-01', end=f'{date.today()}', progress=False)
    stock_info.drop('Volume', axis = 1, inplace = True)
    stock_info.index = pd.to_datetime(stock_info.index).date
    stock_info.sort_index(inplace=True, ascending=False)
    return stock_info

def data_preprocess(forecast_days: int, user_input: str):
    stock_info = load_stock(user_input)
    stock_prediction = stock_info[['Adj Close']]
    stock_prediction['Stock Price'] = stock_prediction.loc[:, 'Adj Close'].shift(-forecast_days)
    return stock_prediction

def data_prep(forecast_days: int, user_input: str):
    stock = data_preprocess(forecast_days, user_input)
    # CREATE X DATASET
    X_DATA = np.array(stock.drop('Stock Price', axis=1))
    X_DATA = X_DATA[:-forecast_days]
    # CREATE Y DATASET
    Y_DATA = np.array(stock['Stock Price'])
    Y_DATA = Y_DATA[:-forecast_days]
    # TEST SPLIT TRAIN DATA
    x_train, x_test, y_train, y_test = train_test_split(X_DATA, Y_DATA, test_size = 0.2)
    return x_train, x_test, y_train, y_test

def model(forecast_days: int, user_input: str, m: object):
    stock = data_preprocess(forecast_days, user_input)
    x_train, x_test, y_train, y_test = data_prep(forecast_days, user_input)
    m_fit = m.fit(x_train, y_train)
    stock_price_pred = np.array(stock.drop(columns='Stock Price', axis=1))[forecast_days:]
    pred = m_fit.predict(stock_price_pred) # [[c1, c2.], []]
    return pred[:forecast_days]

algo = {'svm': SVR(kernel='rbf', C = 1000.0, gamma = 0.0001), 'tree': RandomForestRegressor()}