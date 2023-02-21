import os
import pandas as pd


check_file = 'check_100.csv'
origin_file = 'training_table_100.csv'
origin_dir = 'training_table_100_clusters'

df_origin = pd.read_csv(origin_file, header=None)
df_check = pd.DataFrame()

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
                if file == f'{k}_cluster.csv':
                    df = pd.read_csv(os.path.join(startpath, file), header=None)
                    total_rows += len(df)

                    global df_check
                    # Append the DataFrame to check
                    df_check = pd.concat([df_check, df])

            print(f"{k} has {total_rows} rows in cluster CSV files.")

    return total_rows

tot_rows = count_cluster_csv_rows(origin_dir, 32)

print(tot_rows)
print(len(df_origin))
print(len(df_check))
print(len(df_origin) == tot_rows)

# Sort the rows of the first DataFrame based on every column from left to right
df_origin = df_origin.sort_values(by=df_origin.columns.tolist()).reset_index(drop=True)

# Sort the rows of the second DataFrame based on every column from left to right
df_check = df_check.sort_values(by=df_check.columns.tolist()).reset_index(drop=True)

if df_origin.equals(df_check):
    print('EQUAL')
else:
    print('NOT EQUAL')