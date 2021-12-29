import argparse

# Datatype to allow bools as args
def str2bool(v):
    if v == 'False':
        return False
    else:
        return True

# Types of data to test on
CONST_DATA_TYPES = ['High', 'Mid', 'Low', 'Open', 'Close', 'All']
# Output levels (Verbose = all, data = Loss csv, Graphs = true vs pred graphs)
CONST_OUTPUT_LEVELS = ['Verbose', 'Data', 'Graphs', 'None']

# Functions to parse args
def get_args():
    # List of args that are accepted
    parse = argparse.ArgumentParser(description='Process which value to train on')
    parse.add_argument('--data', type=str, default='High')
    parse.add_argument('--output', type=str, default='None')
    parse.add_argument('--epochs', type=int, default=10)
    parse.add_argument('--batch', type=int, default=32)
    parse.add_argument('--modelname', type=str, default='LSTMNN')
    parse.add_argument('--save', type = str2bool, default='True')
    parse.add_argument("--demo", type = str2bool, default='False')
    parse.add_argument("--train", type = str2bool, default='True')
    args = parse.parse_args()

    # get args after parsing
    datatype = args.data
    output = args.output
    epoch = args.epochs
    batch_size = args.batch
    name = args.modelname
    save = args.save
    demo = args.demo
    train = args.train

    if demo:
        output = 'Verbose'
        dynbatch = True
        save = True
        epoch = 1
        batch = 256

    # For verbose printing, output args
    if output == 'Verbose': print("\n\n{}\n\n".format(args))

    # Make sure the datatype was a value we accept
    if datatype not in CONST_DATA_TYPES: datatype = 'High'

    # Make sure we are not saving unnecessarily
    if not train: save = False

    # Create list will all datatypes to be test (either one type or all types)
    types = [datatype]
    if (datatype == 'All'): types = ['High', 'Mid', 'Low', 'Open', 'Close']

    return types, output, epoch, batch_size, name, save, demo, train
