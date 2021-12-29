import csv
import argparse
import numpy as np
import pandas as pd

# Type of function used to scale the values
function_type = {'HiValScaled': np.max,'LowValScaled': np.min}

# Function to sort the data in any of the 3 ways
def Data_Sort(data, type):
    rank = np.zeros(data.shape[0])
    averages = ['averages']

    for i in range(1,data.shape[1]):
        d = data[:,i]
        averages += [np.average(np.array(d).astype('double'))]
        nd = np.array(d)
        if type != 'Rank': nd = scale(nd, function_type[type])
        sort = np.argsort(nd)
        nd = np.sort(nd)
        for id, val in enumerate(sort):
            if type == 'Rank': rank[val] += id
            else: rank[val] += nd[id]

    return sort_ranks(rank, data, averages)

# Function to scale the array d by the largest or smallest value in it depending
# on the function 'func' parameter
def scale (d, func):
    val = func(d)
    return d / val

# Sorts the data array according to the values in rank (The value smalleest
# rank value will be the first ellement, then then next smallest, etc.)
def sort_ranks(rank, data, averages):
    sorted_rank = np.argsort(rank)

    sorted_array = []
    for val in sorted_rank:
        sorted_array += [data[val].tolist() + [rank[val]]]
    sorted_array.append(averages)
    return sorted_array

# Types of organizations that are handled
organization_types = ['Rank', 'HiValScaled', 'LowValScaled', 'All']
# loss_types tracked (for csv header)
loss_types = ['Ticker','MSE', 'MAE', 'MAPE', 'MSLE', 'H', 'LC']

# argument parsing for filename and organization type
parse = argparse.ArgumentParser(description='Process which value to train on')
parse.add_argument('--filename', type=str, default='Data/Loss.csv')
parse.add_argument('--organization', type=str, default='Rank')
args = parse.parse_args()

filename = args.filename
organization = args.organization

# Verify organization is a correct value
if organization not in organization_types: organization = 'Rank'

# Set list for future loop
organizations = [organization]
if organization == 'All': organizations = organization_types[:-1]

# Read in csv file and convert to numpy arry to be sorted
df = pd.read_csv(filename,delimiter=',',usecols=loss_types)
data = df.to_numpy()

# Run for all speciifed organization types
for org in organizations:
    # Sorts numpy array 'data' according to 'organization' strategy
    sorted = Data_Sort(data, org)

    # Writes sorted array to csv file
    with open(filename[:-4] + org + 'Sorted' + filename[-4:], 'w', newline ='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(loss_types + ['Values'])
        writer.writerows(sorted)
