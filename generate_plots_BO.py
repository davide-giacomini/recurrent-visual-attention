import os
import numpy as np
import pandas as pd
import argparse

import pandas as pd
import matplotlib.pyplot as plt



parser = argparse.ArgumentParser(description='Process data from a CSV file')
parser.add_argument('file', type=str, nargs='?', default='RAM_accuracies_BO.csv', help='path to the CSV file')

args = parser.parse_args()

# Get the file path from the arguments
file_path = args.file

font = {'family' : 'serif','size' : 14}

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
        'Training data': 'training_data',
        'BO': 'bo',
        'Acc': 'acc'
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

    rows_subsets = [df.iloc[start_row : start_row+6] for start_row in start_rows]

    colors = ['r', 'b', 'g', 'y']

    # plot the line chart
    for i, rows_subset in enumerate(rows_subsets):
        plt.plot(rows_subset[cols[i]], rows_subset['acc'], c=colors[i], label = legends[i], linewidth = 1, linestyle='-', marker='.')

    plt.xlabel(x_label, font=font)
    plt.xticks(x_ticks, labels=[str(num) + "%" for num in x_ticks], font=font)
    plt.ylabel('Accuracy', font=font)
    yticks = list(range(72, 97, 4))
    plt.yticks(yticks, labels=[str(num) + "%" for num in yticks], font=font)
    plt.grid(True, linewidth=0.5, color='gray', linestyle=':')
    plt.legend(fontsize=12)

    plt.title(plt_title, font=font)

    save_graph('bo')

    return plt



df = parse_csv()

fig = plt.figure(figsize=(12, 10))

plt = generate_plot(df=df, 
                          start_rows=[6*0,6*1],
                          cols=['training_data', 'training_data'],
                          legends=['Function non optimized', 'Function optimized with BO'],
                          x_label='Percentage of trained table',
                          x_ticks=[15,30,50,70,90,100], 
                          plt_title='Distance metric optimization through Bayesian Optimization'
                          )
# plt.show()
plt.clf()