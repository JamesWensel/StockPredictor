# StockPredictor
Creates RNN using basic RNNs, LSTMs, or GRUs to make predictions on stock data. 

### TrainTestModel.py
Main files. Used to train and test a model on provided stock data. Stock data gathered from [Kaggle](https://www.kaggle.com/borismarjanovic/price-volume-data-for-all-us-stocks-etfs)

### DefineModel.py
Creates a model according to provided specifications (through arguments or default values) and saves it to be trained by other files

### FormatData.py
Formats pandas dataframes of each individual stock for training

### SortData.py
Sorts output results according to multiple different metrics to find best results

### outputdata.py
Helper file contatining output functions for data and graphs of each stock

### argparsing.py
Helper file to perform argument parsing for TrainTestModel.py
