import os
import sys
import numpy as np
import pandas as pd
import argparse

import pandas as pd
import matplotlib.pyplot as plt



parser = argparse.ArgumentParser(description='Process data from a CSV file')
parser.add_argument('file', type=str, nargs='?', default='RAM_accuracies_noise.csv', help='path to the CSV file')

args = parser.parse_args()

# Get the file path from the arguments
file_path = args.file

font = {'family' : 'serif','size' : 14}
legend_fontisze = 12

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
        'acc_50_BO': 'acc_50_bo',
        'acc_100_BO': 'acc_100_bo',
        'noise': 'noise'
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

def generate_plot(df, start_rows, cols, legends, x_label, x_ticks, plt_title):

    rows_subsets = [df.iloc[start_row : start_row+23] for start_row in start_rows]

    colors = ['r', 'b', 'g', 'y']

    # plot the line chart
    for i, rows_subset in enumerate(rows_subsets):
        plt.plot(rows_subset['noise'], rows_subset[cols[i]], c=colors[i], label = legends[i], linewidth = 1, linestyle='-', marker='.')

    plt.xlabel(x_label, font=font)
    plt.xticks(x_ticks, font=font)
    plt.xlim
    plt.ylabel('Accuracy', font=font)
    plt.ylim(65.0, 91.0)
    yticks = list(range(66, 91, 3))
    plt.yticks(yticks, labels=[str(num) + "%" for num in yticks], font=font)
    plt.grid(True, linewidth=0.5, color='gray', linestyle=':')
    plt.legend(fontsize=legend_fontisze)

    # plt.title(plt_title, font=font_title)

    save_graph('noises.pdf')

    return plt


df = parse_csv()

# fig = plt.figure(figsize=(12, 10))

plt = generate_plot(df=df, 
                          start_rows=[0, 0],
                          cols=['acc_50_bo', 'acc_100_bo'],
                          legends=['50% of trained table', '100% of trained table'],
                          x_label='Standard deviation',
                          x_ticks=[0.0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0], 
                          plt_title='Accuracy with Gaussian noise applied to distance metric'
                          )
# plt.show()
plt.clf()