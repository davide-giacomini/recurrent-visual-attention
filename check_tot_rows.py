import os
import pandas as pd


def count_cluster_csv_rows(startpath, K):
    total_rows = 0
    for k in range(K):
        k_dir = os.path.join(startpath, f'{k}')
        if os.path.isdir(k_dir):
            # if k subdirectory exists, keep walking down the path
            total_rows += count_cluster_csv_rows(k_dir, K)
        else:
            # if k subdirectory doesn't exist, count rows in k_cluster.csv files
            for file in os.listdir(startpath):
                if file.endswith(f'{k}_cluster.csv'):
                    df = pd.read_csv(os.path.join(startpath, file))
                    total_rows += len(df)+1
            print(f"{k} has {total_rows} rows in cluster CSV files.")

    return total_rows

tot_rows = count_cluster_csv_rows('training_table_01_clusters', 4)

df = pd.read_csv('training_table_01.csv')

print(tot_rows)
print(len(df))
print(len(df) == tot_rows)