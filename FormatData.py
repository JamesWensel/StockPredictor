import math
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class StockData:
    def __init__ (self, df):
        # Split inital data into training and testing datasets
        high_train, high_test = self.split_data(df.loc[:,'High'].values)
        mid_train, mid_test = self.split_data((df.loc[:,'High'].values+df.loc[:,'Low'].values)/2.0)
        low_train, low_test = self.split_data(df.loc[:,'Low'].values)
        open_train, open_test = self.split_data(df.loc[:,'Open'].values)
        close_train, close_test = self.split_data(df.loc[:,'Close'].values)

        # List of the stock datatypes
        self.data_types = ["High", "Mid", "Low", "Open"]

        # Dictionary to store data
        self.data = {
            "High": {
                "Data": df.loc[:,'High'].values,
                "Train": high_train,
                "Test": high_test
            },
            "Mid": {
                "Data": (df.loc[:,'High'].values+df.loc[:,'Low'].values)/2.0,
                "Train": mid_train,
                "Test": mid_test
            },
            "Low": {
                "Data": df.loc[:,'Low'].values,
                "Train": low_train,
                "Test": low_test
            },
            "Open": {
                "Data": df.loc[:,'Open'].values,
                "Train": open_train,
                "Test": open_test
            },
            "Close": {
                "Data": df.loc[:,'Close'].values,
                "Train": close_train,
                "Test": close_test
            }
        }

        # Dictionary to hold scalers for each data type
        self.scalers = {
            "High": MinMaxScaler(feature_range=(0,1)),
            "Mid": MinMaxScaler(feature_range=(0,1)),
            "Low": MinMaxScaler(feature_range=(0,1)),
            "Open": MinMaxScaler(feature_range=(0,1)),
            "Close": MinMaxScaler(feature_range=(0,1))
        }

        self.prepare_data()

    # Splits data into training and testing set
    def split_data(self, data):
        split = math.ceil(data.shape[0] * .9)
        return data[:split], data[split:]

    # Prepares all data to be in correct form for LSTM
    def prepare_data(self):
         self.prepare_train()
         self.prepare_test()

    # Prepares the training data
    def prepare_train(self):
        self.prep_train("High")
        self.prep_train("Mid")
        self.prep_train("Low")
        self.prep_train("Open")
        self.prep_train("Close")

    # Helper function that puts data into correct for LSTM
    def prep_train(self, d):
        # Reshapes the data and scales the values from 0 to 1
        train = self.data[d]["Train"].reshape(-1,1)
        test = self.data[d]["Test"].reshape(-1,1)
        scaled = self.scalers[d].fit_transform(train)

        # Puts x and y data in form (X, 60, 1)
        x_train = []
        y_train = []
        for i in range (60, scaled.size):
          x_train.append(scaled[i-60:i,0])
          y_train.append(scaled[i,0])
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        # Stores the newly prepared data
        self.data[d]["Train"] = train
        self.data[d]["Test"] = test
        self.data[d]["x_train"] = x_train
        self.data[d]["y_train"] = y_train

    # Prepares test data
    def prepare_test(self):
        self.prep_test("High")
        self.prep_test("Mid")
        self.prep_test("Low")
        self.prep_test("Open")
        self.prep_test("Close")

    # Helper function to prepare the test data for LSTM
    def prep_test(self, d):
        # Reshpaes and scales total dataset
        total = self.data[d]["Data"]
        total = total.reshape(-1,1)
        total = self.scalers[d].transform(total)

        #
        data = []
        for i in range(60, total.size):
          data.append(total[i-60:i, 0])
        data = np.array(data)
        data = np.reshape(data, (data.shape[0], data.shape[1], 1))

        self.data[d]["Test"] = data
