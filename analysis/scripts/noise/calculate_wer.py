import sys
sys.path.append('/Users/gayanin/github/jiwer')  # Add the parent directory to sys.path

import jiwer.measures as measures
import pandas as pd
import numpy as np
import datasets
from datasets import load_dataset, concatenate_datasets

# babylon_aws = '/Users/gayanin/github/RefinedLM/data/babylon/refs_and_aws_all.csv'
# gcd_aws = '/Users/gayanin/github/RefinedLM/data/test/refs_and_trans_aws_gb.csv'
# kaggle_aws = '/Users/gayanin/github/RefinedLM/data/kaggle/refs_and_aws_all_k.csv'
# mixed = 'combined.csv'
test = 'mixed_noise_test.csv'

def dataset_to_df(datasetpath,split):
  dataset = datasets.load_dataset(datasetpath, split)
  dfs = []
  for split in dataset:
    df = dataset[split].to_pandas()
    dfs.append(df)

  df = pd.concat(dfs, ignore_index=True)

  # df = dataset['test'].to_pandas()
  return df

def csv_to_df(csvpath):
  df = pd.read_csv(csvpath)
  return df

def col_to_lst(df, column_name):
    # df = pd.read_csv(csv_file_path)
    df.dropna(inplace=True)
    column_data = df[column_name].tolist()
    return  column_data

# def col_to_lst(csv_file_path, column_name):
#     df = pd.read_csv(csv_file_path)
#     df.dropna(inplace=True)
#     column_data = df[column_name].tolist()
#     return  column_data

print('SUB'+'\t'+'DEL'+'\t'+'INS'+'\t'+'HITS')


# df_ = pd.read_csv('/Users/gayanin/github/RefinedLM/babylon-noised-v8-latest.csv')
# df_ = pd.read_csv('/Users/gayanin/github/RefinedLM/gcd-noised-v8cccc.csv')
# df_ = pd.read_csv('/Users/gayanin/github/RefinedLM/test-gcd.csv')
# df_ = pd.read_csv('/Users/gayanin/github/RefinedLM/gcd-noised-v8-latest.csv')
# df_ = pd.read_csv('/Users/gayanin/github/RefinedLM/kaggle-noised-v8-latest.csv')
# print(measures.wer(col_to_lst(df_, 'refs'), col_to_lst(df_, 'trans')))
# # 0.12996914140497368
# sub: 0.5 del: 0.3333333333333333 ins: 0.16666666666666666


# data3 = load_dataset('gayanin/kaggle-native-v8-noised-test')
# data3 = load_dataset('gayanin/gcd-native-v8-noised-test')
data3 = load_dataset('gayanin/gcd-native-v8')
# data_k = concatenate_datasets([data3['train'],
#                              data3['validation'],
#                              data3['test']])
df_ = data3['test'].to_pandas()
# df_ = data_k.to_pandas()
# print(df_.shape)
print(measures.wer(col_to_lst(df_, 'refs'), col_to_lst(df_, 'trans')))


# result_aws = measures.wer(col_to_lst(babylon_aws, 'refs'), col_to_lst(babylon_aws, 'trans'))
# print(result_aws)
# result_gcd = measures.wer(col_to_lst(gcd_aws, 'refs'), col_to_lst(gcd_aws, 'trans'))
# print(result_gcd)
# result_kaggle = measures.wer(col_to_lst(test, 'refs'), col_to_lst(test, 'trans'))
# print(result_kaggle)


# df_ = dataset_to_df('gayanin/pubmed-mixed-noise', 'prob-0.3')

# df_ = dataset_to_df('gayanin/pubmed-mixed-noise', 'prob-0.3')

# test split WERs
# df_ = dataset_to_df('gayanin/gcd-native','default') #0.20847457627118643
# df_ = dataset_to_df('gayanin/babylon-native','default') #0.08708010335917313
# df_ = dataset_to_df('gayanin/kaggle-native','default') #0.11440922190201729

# all split WERs
# df_ = dataset_to_df('gayanin/gcd-native','default') #0.24378909740840035
# df_ = dataset_to_df('gayanin/babylon-native','default') #0.09810416519839311
# df_ = dataset_to_df('gayanin/kaggle-native','default') #0.08770465859062528

# all datasets WERs after new filtering
# df_ = pd.read_csv('check_new_gcd1.csv') #0.23942639317480485
# df_ = pd.read_csv('check_new_kaggle1.csv') #0.08340357506176428
# df_ = csv_to_df('check_new_babylon1.csv') #0.0.08839785716888765

# data1 = load_dataset('gayanin/pubmed-abstracts-noised-with-prob-dist-v2', 'babylon-prob-01')
# data2 = load_dataset('gayanin/pubmed-abstracts-noised-with-prob-dist-v2', 'gcd-prob-01')
# data3 = load_dataset('gayanin/pubmed-abstracts-noised-with-prob-dist-v2', 'kaggle-prob-01')

# data_k = concatenate_datasets([data1['train'],
#                              data1['validation'],
#                              data1['test']])

# data_g = concatenate_datasets([data2["train"],
#                              data2["validation"],
#                              data2["test"]])

# data_b = concatenate_datasets([data3["train"],
#                              data3["validation"],
#                              data3["test"]])

# df_k = data_k.to_pandas()

# df_g = data_g.to_pandas()

# df_b = data_b.to_pandas()

# print(df_k.shape)
# print(df_g.shape)
# print(df_b.shape)

# print(measures.wer(col_to_lst(df_k, 'refs'), col_to_lst(df_k, 'trans')))
# print(measures.wer(col_to_lst(df_g, 'refs'), col_to_lst(df_g, 'trans')))
# print(measures.wer(col_to_lst(df_b, 'refs'), col_to_lst(df_b, 'trans')))

# print(df_)
# result_ = measures.wer(col_to_lst(df_, 'refs'), col_to_lst(df_, 'trans')) # creates combined_all.txt
# print(result_)
