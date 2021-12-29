import argparse
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import SimpleRNN
from keras.layers import GRU
from keras.layers import ConvLSTM2D
from keras.layers import Dropout

def str2bool(v):
    if v == 'False':
        return False
    else:
        return True

CONST_LAYERS = ['LSTM', 'RNN', 'GRU']
CONST_ADD_LAYERS = {
    'LSTM': LSTM,
    'RNN': SimpleRNN,
    'GRU': GRU,
}
CONST_OPTIMIERS = ['adam', 'nadam', 'sgd', 'adagrad', 'adamax', 'rmsprop']
CONST_LOSS = ['mean_squared_error', 'mean_absolute_error', 'mean_absolute_perctage_error',
                'mean_square_logarithmic_error', 'huber', 'logcosh']

parse = argparse.ArgumentParser(description='Detmines what model we will have')
parse.add_argument('--name', type=str, default='LSTMNN')
parse.add_argument('--layers', type=str, default='LSTM')
parse.add_argument('--numlayers', type=int, default=4)
parse.add_argument('--dropout', type=str2bool, default='True')
parse.add_argument('--units', type=int, default=50)
parse.add_argument('--droprate', type=float, default=0.2)
parse.add_argument('--optimizer', type=str, default='nadam')
parse.add_argument('--loss', type=str, default='mean_squared_error')
args = parse.parse_args()

name = args.name
layers = args.layers
numlayers = args.numlayers
dropout = args.dropout
units = args.units
droprate = args.droprate
optimizer = args.optimizer
loss = args.loss

if layers not in CONST_LAYERS: layers = 'LSTM'

model = Sequential()

model.add(CONST_ADD_LAYERS[layers](units = units, return_sequences = numlayers > 1, input_shape = (60, 1)))
if dropout: model.add(Dropout(droprate))

for i in range(1, numlayers):
    model.add(CONST_ADD_LAYERS[layers](units = units, return_sequences = i < numlayers-1))
    if dropout: model.add(Dropout(droprate))

model.add(Dense(units = 1))

model.compile(optimizer = optimizer, loss = loss)

print(model.summary())

model.save(name)
