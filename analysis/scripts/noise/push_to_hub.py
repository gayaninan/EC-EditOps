from datasets import Dataset
import datasets
import pandas as pd
from math import ceil

def custom_train_test_val_split(df, idx1, idx2, idx3, ratio=(8, 1, 1)):
    # Ensure the ratio sums up to 10
    assert sum(ratio) == 10, "The sum of the ratios should be 10"

    # Split each segment
    def split_segment(segment, ratio):
        total = len(segment)
        train_end = ceil(total * ratio[0] / 10)
        val_end = train_end + ceil(total * ratio[1] / 10)
        return segment[:train_end], segment[train_end:val_end], segment[val_end:]

    # Divide the dataframe into three segments
    seg1, seg2, seg3 = df[:idx1], df[idx1:idx1+idx2], df[idx1+idx2:idx1+idx2+idx3]

    # Split each segment into train, validation, and test
    train1, val1, test1 = split_segment(seg1, ratio)
    train2, val2, test2 = split_segment(seg2, ratio)
    train3, val3, test3 = split_segment(seg3, ratio)

    # Combine the splits from each segment
    train_final = pd.concat([train1, train2, train3])
    val_final = pd.concat([val1, val2, val3])
    test_final = pd.concat([test1, test2, test3])

    return train_final, val_final, test_final


def push_to_hub_with_subset_splits(input_csv, dataset_name):
   df = pd.read_csv(input_csv)
   # df = df.dropna()
   df = df[['refs','trans']]
   train_df = df[:int(len(df) * 0.8)]
   test_df = df[int(len(df) * 0.8):int(len(df) * 0.9)]
   val_df = df[int(len(df) * 0.9):]

   # train_df, test_df, val_df = custom_train_test_val_split(df, 6591, 299, 6617)
 
   print(train_df)
   print(test_df.shape)
   print(val_df.shape)
   
   train_dataset = Dataset.from_pandas(train_df)
   test_dataset = Dataset.from_pandas(test_df)
   val_dataset = Dataset.from_pandas(val_df)
   
   dataset = datasets.DatasetDict({
      "train": train_dataset,
      "test": test_dataset,
      "validation": val_dataset
      })
   
   dataset.push_to_hub(dataset_name)

push_to_hub_with_subset_splits("/Users/gayanin/github/RefinedLM/gcd-noised-v8-latest.csv", "gayanin/gcd-native-v8-noised")
push_to_hub_with_subset_splits("/Users/gayanin/github/RefinedLM/kaggle-noised-v8-latest.csv", "gayanin/kaggle-native-v8-noised")
push_to_hub_with_subset_splits("/Users/gayanin/github/RefinedLM/babylon-noised-v8-latest.csv", "gayanin/babylon-native-v8-noised")

# push_to_hub_with_subset_splits("/Users/gayanin/github/RefinedLM/native_all.csv", "gayanin/clinical-all")