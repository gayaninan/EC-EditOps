import pandas as pd
import numpy as np
import datasets

def dataset_to_df(datasetpath):
  dataset = datasets.load_dataset(datasetpath, split=None)
  dfs = []
  for split in dataset:
    df = dataset[split].to_pandas()
    dfs.append(df)

  df = pd.concat(dfs, ignore_index=True)
  return df

def csv_to_df(csvpath):
  df = pd.read_csv(csvpath)
  return df


df1 = dataset_to_df('gayanin/kaggle-native')
df2 = dataset_to_df('gayanin/gcd-native')
df3 = dataset_to_df('gayanin/babylon-native')

df = pd.concat([df1, df2,df3], ignore_index=True)

print(df1.shape)
print(df2.shape)
print(df3.shape)
print(df.shape)

df.to_csv('native_all.csv', index=False)