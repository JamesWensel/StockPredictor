import csv
from keras import losses
import matplotlib.pyplot as plt

# Loss functions to test data
CONST_LOSS_FUNCTOINS = {
    'MSE': losses.MeanSquaredError(),
    'MAE': losses.MeanAbsoluteError(),
    'MAPE': losses.MeanAbsolutePercentageError(),
    'MSLE':  losses.MeanSquaredLogarithmicError(),
    'H': losses.Huber(),
    'LC': losses.LogCosh()
    }

# Path for graphs and data files
GraphPath = 'Graphs/'
DataPath = 'Data/'

# Plots / outputs graphs to graphs folder
def plotgraph(ran, true, pred, ticker, type, name):
    plt.clf()
    plt.plot(ran, true, label= type + ' Price')
    plt.plot(pred, color = 'red', label = 'Predicted ' + type + ' Price')
    plt.title(ticker + ' Price Prediction')
    plt.xlabel('Time')
    plt.ylabel(ticker + ' Price')
    plt.legend()
    plt.savefig("{}{}.{}.{}.png".format(GraphPath, ticker, type, name))

# Updates csv with closs values for current stock
def update_csv(ticker, type, true, pred):
    metrics = {'Ticker': ticker + '.' + type}

    for m in CONST_LOSS_FUNCTOINS:
        metrics[m] = CONST_LOSS_FUNCTOINS[m](true, pred).numpy()

    return [metrics]

# Prints saved CVS_Data to specified filename
def write_to_csv(CSV_Data, filename, modelname):
    fieldnames = ['Ticker'] + list(CONST_LOSS_FUNCTOINS.keys())

    with open("{}{}.{}.csv".format(DataPath,filename,modelname), 'w', newline ='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(CSV_Data)
