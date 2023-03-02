import os
import sys
import numpy as np
import pandas as pd
import argparse

import pandas as pd
import matplotlib.pyplot as plt



parser = argparse.ArgumentParser(description='Process data from a CSV file')
parser.add_argument('file', type=str, nargs='?', default='RAM_accuracies.csv', help='path to the CSV file')

args = parser.parse_args()

# Get the file path from the arguments
file_path = args.file

font = {'family' : 'serif','size' : 14}
colors = ['#FF0000', '#0000FF', '#00FF00', '#800080']

def save_graph(graph_name):
    # SAVE GRAPH IN PNG
    dir_name = 'graphs'
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    plt.savefig(os.path.join(dir_name, graph_name), bbox_inches='tight',dpi=300)

def parse_csv():
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Define a dictionary to map the old headers to the new headers
    header_map = {
        '# P': 'num_patches',
        '# G': 'num_glimpses',
        'P S': 'patch_size',
        'G S': 'glimpse_scale',
        'size': 'size_ht',
        '# Ep': 'num_epochs',
        'B E': 'best_epoch',
        'B E Ac': 'best_ep_acc',
        'Ac': 'acc',
        'Er': 'err',
        'gt': 'gt_train',
        'ht': 'ht_train',
        'phi': 'phi_train',
        'l_out': 'l_out_train',
        'ht.1': 'ht_test',
        'phi.1': 'phi_test',
        'lay': 'added_layers'
    }

    # Remove empty rows
    df = df.dropna(how='all')

    # Rename the headers using the dictionary
    df = df.rename(columns=header_map)

    # Reset the rows index
    df = df.reset_index(drop=True)

    # Loop through the dataframe and replace percentage strings with numbers
    for col in df:
        if "%" in str(df[col].iloc[0]):
            df[col] = df[col].str.rstrip("%").astype(float)

    # Display the resulting dataframe
    # pd.set_option('display.max_rows', None)
    # print(df)

    return df

def get_first_integer(s):
    return int(s.split('x')[0])

def generate_plot_patch(df, start_rows, cols, legends, x_label, x_ticks, plt_title):

    rows_bunches = [df.iloc[start_row : start_row+7] for start_row in start_rows]

    rows_bunches[1][cols[1]] = rows_bunches[1][cols[1]].apply(get_first_integer)

    # plot the line chart
    for i, rows_bunch in enumerate(rows_bunches):
        plt.plot(rows_bunch[cols[i]], rows_bunch['acc'], c=colors[i], label = legends[i], linewidth = 1, linestyle='-', marker='.')

    plt.xlabel(x_label, font=font)
    plt.xticks(x_ticks, font=font)
    plt.ylabel('Accuracy', font=font)
    plt.ylim((64, 101))
    yticks = list(range(65, 101, 5))
    plt.yticks(yticks, labels=[str(num) + "%" for num in yticks], font=font)
    plt.grid(True, linewidth=0.5, color='gray', linestyle=':')
    plt.legend(fontsize=12)

    # plt.title(plt_title, font=font)


    # SAVE GRAPH IN PDF
    save_graph('patch')

    return plt

def generate_plot_quant(df, start_rows, cols, legends, x_label, x_ticks, plt_title):

    rows_bunches = [df.iloc[start_row : start_row+7] for start_row in start_rows]

    for i, rows_subset in enumerate(rows_bunches):
        rows_subset.loc[start_rows[i], cols[i]] = rows_subset.loc[start_rows[i]+6, cols[i]] + 1
        rows_subset = rows_subset.sort_values(cols[i]) # sort the subset by the specified column
        rows_bunches[i] = rows_subset.reset_index()

    # plot the line chart
    for i, rows_bunch in enumerate(rows_bunches):
        plt.plot(rows_bunch[cols[i]], rows_bunch['acc'], c=colors[i], label = legends[i], linewidth = 1, linestyle='-', marker='.')

    plt.xlabel(x_label, font=font)
    tick_labels = list(x_ticks)
    tick_labels[-1] = 'inf'  # change the last tick label to 'inf'
    plt.xticks(x_ticks, tick_labels, font=font)
    plt.ylabel('Accuracy', font=font)
    plt.ylim((81, 101))
    yticks = list(range(82, 101, 2))
    plt.yticks(yticks, labels=[str(num) + "%" for num in yticks], font=font)
    plt.grid(True, linewidth=0.5, color='gray', linestyle=':')
    plt.legend(fontsize=12, loc="lower right")

    # plt.title(plt_title, font=font)

    save_graph('quantization')

    return plt

def generate_plot_size(df, start_rows, cols, legends, x_labels, x_ticks, plt_title):

    rows_subset0 = df.iloc[start_rows[0] : start_rows[0]+7]
    rows_subset1 = df.iloc[start_rows[1] : start_rows[1]+3]

    # create a figure and an axis object for the first plot
    fig, ax0 = plt.subplots()
    ax0.set_xlabel(x_labels[0], font=font)
    ax0.set_xticks(x_ticks[0], font=font)
    ax0.set_ylabel('Accuracy', font=font)
    ax0.set_ylim((65, 101))
    yticks = list(range(68, 101, 4))
    ax0.set_yticks(yticks, labels=[str(num) + "%" for num in yticks], font=font)
    ax0.grid(True, linewidth=0.5, color='gray', linestyle=':')

    # plot the first line chart on the first axis object
    lns0 = ax0.plot(rows_subset0[cols[0]], rows_subset0['acc'], c=colors[0], label = legends[0], linewidth = 1, linestyle='-', marker='.')

    # create a second axis object that shares the same y-axis as the first axis object
    ax1 = ax0.twiny()
    ax1.set_xlabel(x_labels[1], font=font)
    ax1.set_xticks(x_ticks[1], font=font)
    # set the position of the second x-axis to be at the top of the figure
    ax1.xaxis.set_ticks_position('top')
    ax1.xaxis.set_label_position('top')
    ax1.grid(True, linewidth=0.5, color='gray', linestyle=':')

    # plot the second line chart on the second axis object
    lns1 = ax1.plot(rows_subset1[cols[1]], rows_subset1['acc'], c=colors[1], label = legends[1], linewidth = 1, linestyle='-', marker='.')
    
    # added these three lines
    lns = lns0+lns1
    labs = [l.get_label() for l in lns]
    ax0.legend(lns, labs, loc="lower right")

    # plt.title(plt_title, font=font)

    save_graph('model_size')

    return plt

def generate_plot_datasets(df, start_rows, legends, x_label, x_ticks, plt_title):

    rows_bunches = [df.iloc[start_row : start_row+3] for start_row in start_rows]

    # plot the line chart
    for i, rows_bunch in enumerate(rows_bunches):
        plt.plot([0,1,2], rows_bunch['acc'], c=colors[i], label = legends[i], linewidth = 1, linestyle='-', marker='.')

    # plt.xlabel(x_label, font=font)
    tick_labels = ['MNIST', 'FashionMNIST', 'CIFAR10']
    plt.xticks(x_ticks, tick_labels, font=font)
    plt.ylabel('Accuracy', font=font)
    plt.ylim((28, 102))
    yticks = list(range(30, 101, 10))
    plt.yticks(yticks, labels=[str(num) + "%" for num in yticks], font=font)
    plt.grid(True, linewidth=0.5, color='gray', linestyle=':')
    plt.legend(fontsize=12, loc='lower left')

    plt.title(plt_title, font=font)

    save_graph('datasets')

    return plt


df = parse_csv()

# fig = plt.figure(figsize=(12, 10))

plt = generate_plot_patch(df=df, 
                          start_rows=[7*1,7*2,7*3,7*4],
                          cols=['num_glimpses', 'patch_size', 'glimpse_scale', 'num_patches'],
                          legends=['Glimpses number', 'Patch size', 'Glimpse scale', 'Patches number'],
                          x_label='Parameter value',
                          x_ticks=list(range(2,11,1)), 
                          plt_title='Accuracy with differences in patch vector parameters'
                          )
# plt.show()
plt.clf()

# fig = plt.figure(figsize=(12, 10))

plt = generate_plot_quant(df=df, 
                          start_rows=[7*0,7*5], 
                          cols=['ht_test','phi_train'],
                          legends=['Hidden state vector quantization', 'Patch vector quantization'],
                          x_label='Quantization levels', 
                          x_ticks=list(range(1,10,1)), 
                          plt_title='Accuracy with differences in quantization levels'
                          )
# plt.show()
plt.clf()

# fig = plt.figure(figsize=(12, 10))

plt = generate_plot_size(df=df, 
                          start_rows=[7*6,7*7], 
                          cols=['size_ht','added_layers'],
                          legends=['Hidden state vector length', 'Number of layers added to the original network'],
                          x_labels=['Hidden state vector length', 'Added layers'], 
                          x_ticks=[list(range(16,130,16)), list(range(0,3,1))], 
                          plt_title='Accuracy with differences in the network structure'
                          )
# plt.show()
plt.clf()

# fig = plt.figure(figsize=(12, 10))

plt = generate_plot_datasets(df=df, 
                          start_rows=[7*7 +3*1,7*7 +3*2], 
                          legends=['Datasets quantized', 'Datasets non quantized'],
                          x_label='Datasets', 
                          x_ticks=list(range(0,3,1)), 
                          plt_title='Baseline accuracy for different datasets'
                          )
# plt.show()
plt.clf()