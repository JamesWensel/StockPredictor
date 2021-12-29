import os
import csv
import random
import time
import numpy as np
import pandas as pd
import outputdata
from FormatData import StockData
from argparsing import get_args
from keras.models import load_model

CSV_Data = []

types, output, epoch, batch_size, name, save, demo, train  = get_args()

StocksPath = 'StockData/Stocks/'

# Load the list of stock
stocks = os.listdir("StockData/Stocks")
num_done = 0
stocks = stocks[num_done:]
if demo:
    end = '.us.txt'
    random.seed(time.time())
    stocks = ['aapl'+end, 'amzn'+end, 'fb'+end, 'googl'+end, 'msft'+end]
elif not train:
    s = []
    for i in range(0,500):
        s += [stocks[random.randint(0,len(stocks))]]
    stocks = s

# Load the created and compiled Keras model
os.system('cls')
print("Loading Model...")
model = load_model(name)

for iter, cur_stock in enumerate(stocks):
    for i, type in enumerate(types):
        os.system('cls') ###### IF NOT WINDOWS CHANGE TO 'clear'
        print("Training on {}".format(cur_stock, type))
        if output == 'Verbose':
            print("{} data, iter={}".format(type, iter*len(types)+i))

        if (iter * len(types) + i) % 100 == 0 and iter != 0 and save:
            print("Saving Model after {} iterations...".format(iter))
            model.save(name)

        # Read in and sort first stock csv
        df = pd.read_csv('StockData/Stocks/{}'.format(cur_stock), delimiter=',', usecols=['Date','Open','High','Low','Close'])
        df = df.sort_values('Date')

        # Turn Pandas dataframe into stock object with formatted datasets
        stock = StockData(df)

        if train:
            # Train the model with formatted data
            trained = model.fit(stock.data[type]["x_train"],
                        stock.data[type]["y_train"], epochs = epoch,
                        batch_size = batch_size)

        # Generate Predicted stock value
        predicted_stock_price = model.predict(stock.data[type]["Test"])
        # Reverseve MinMaxScaler transform of data
        predicted_stock_price = stock.scalers[type].inverse_transform(predicted_stock_price)

        # If we are logging data, update the csv data
        if output == 'Verbose' or output == 'Data':
            CSV_Data += outputdata.update_csv(str(cur_stock), type,
                stock.data[type]["Data"][60:], predicted_stock_price.flatten())

        # If we are logging graphs, output the graph
        if output == 'Verbose' or output == 'Graphs':
            outputdata.plotgraph(range(stock.data[type]['Data'].shape[0]),
                stock.data[type]['Data'], predicted_stock_price, cur_stock[:-7], type, name)

# If we are logging data, output the csv
if output == 'Verbose' or output == 'Data':
    outputdata.write_to_csv(CSV_Data, 'Loss', name)

# Save the model
if save:
    os.system('cls')
    print("Saving Model...")
    model.save(name)
