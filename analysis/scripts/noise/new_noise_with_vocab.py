import pandas as pd
import numpy as np
import datasets
import random
from datasets import Dataset, load_dataset, concatenate_datasets
import pronouncing
from better_profanity import profanity
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import re
import json

# random.seed(42)  # For reproducibility
# np.random.seed(42)

with open('analysis/outputs/asr_ops/babylon-native-v8-ops.json', 'r') as file:
        vocab_dict = json.load(file)
        
sub_vocab = vocab_dict['list_sub_trans']
del_vocab = vocab_dict['list_del']
ins_vocab = vocab_dict['list_ins']  

def get_sub_word(word):
    shuffled_sub_words = sub_vocab.copy()
    random.shuffle(shuffled_sub_words)
    for sub_word in shuffled_sub_words:
        if word != sub_word:
            return sub_word 
    
def get_ins_word(word):
    shuffled_ins_words = ins_vocab.copy()
    random.shuffle(shuffled_ins_words)
    for ins_word in shuffled_ins_words:
        if word != ins_word:
            return ins_word 

def dataset_to_df(datasetpath,split):
  dataset = datasets.load_dataset(datasetpath, split)
  dfs = []
  for split in dataset:
    df = dataset[split].to_pandas()
    dfs.append(df)

def substitute(word):
    sub_word = get_sub_word(word)
    replacement = re.sub(r'[^a-zA-Z]', '', sub_word)
    return replacement

def delete(word):
    return ''

def insert(word):
    return word + " "+ get_ins_word(word)

def noise_word(word, error_type):
    if error_type == 'sub':
        return substitute(word)
    elif error_type == 'del':
        return delete(word)
    elif error_type == 'ins':
        return insert(word)
    else:
        return word  


def push_to_hub_with_subset_splits(df, savepath, wer, sub, del_, ins, prefix):
  
    word_list, sent_list = new_noise(df, wer, sub, del_, ins)

    trans_list = transform(word_list, sent_list)

    df['trans'] = trans_list

    train_df = df[:int(len(df) * 0.8)]
    test_df = df[int(len(df) * 0.8):int(len(df) * 0.9)]
    val_df = df[int(len(df) * 0.9):]

    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    val_dataset = Dataset.from_pandas(val_df)

    dataset = datasets.DatasetDict({
        "train": train_dataset,
        "test": test_dataset,
        "validation": val_dataset

    })
    df.to_csv('test.csv')
    # dataset.push_to_hub(savepath, prefix)
    dataset.push_to_hub(savepath)

def new_noise(df, wer, sub_rate, del_rate, ins_rate):
    total_words = df['refs'].str.split().str.len().sum()

    # 

    total_errors = int(wer * total_words)

    sub_errors = int(total_errors * sub_rate)
    del_errors = int(total_errors * del_rate)
    ins_errors = total_errors - sub_errors - del_errors

    total_without_errors = total_words - total_errors

    print("total_words: ", total_words) #5509
    print("total_errors: ", total_errors) #550
    print("sub_errors: ", sub_errors) #352
    print("del_errors: ", del_errors) #110
    print("ins_errors: ", ins_errors) #88


    # errors = ['sub'] * sub_errors + ['del'] * del_errors + ['ins'] * ins_errors
    all_words =['sub'] * sub_errors  + ['del'] * del_errors + ['ins'] * ins_errors + ['nochange']*total_without_errors
    print(all_words)

    random.shuffle(all_words)  

    print(all_words)
 
    sent_list = df["refs"].tolist()

    return all_words, sent_list

def transform(all_words, sent_list):
    # for msk in all_words:
    count = 0
    res_list = []
    for s in sent_list:
        s_trns = []
        for w in s.split(): 
            if all_words[count] == 'sub': 
                w = substitute(w)
                s_trns.append(w)
            elif all_words[count] == 'del':
                # s_trns.append(delete(w))
                # w = ""
                # s_trns.append(w)
                None
            elif all_words[count] == 'ins':
                s_trns.append(insert(w))
            else:
                s_trns.append(w)
            count+=1
        res_list.append(" ".join(s_trns))
    return res_list

data_ = load_dataset('gayanin/babylon-native-v8')

data = concatenate_datasets([data_['train'],
                             data_['validation'],
                             data_['test']])

data_df = data.to_pandas()

# push_to_hub_with_subset_splits(data_df, 'gayanin/gcd-native-v8-vocab-noised', 0.2, 0.5, 0.25, 0.25, '') #gcd
# push_to_hub_with_subset_splits(data_df, 'gayanin/kaggle-native-v8-vocab-noised', 0.1, 0.34, 0.33, 0.33, '') #kaggle
push_to_hub_with_subset_splits(data_df, 'gayanin/babylon-native-v8-vocab-noised', 0.1, 0.36, 0.18, 0.46, '') #babylon

# df = pd.read_csv('woz.csv')

# print("start prob1")
# push_to_hub_with_subset_splits(df, 'gayanin/woz-noised', 0.1, 0.34, 0.33, 0.33, 'kaggle-01') #kaggle
# push_to_hub_with_subset_splits(df, 'gayanin/woz-noised', 0.1, 0.5, 0.25, 0.25, 'gcd-01') #gcd
# push_to_hub_with_subset_splits(df, 'gayanin/woz-noised', 0.1, 0.36, 0.18, 0.46, 'babylon-01') #babylon
# print("end prob1")

# print("start prob2")
# # push_to_hub_with_subset_splits(df, 'gayanin/woz-noised', 0.2, 0.34, 0.33, 0.33, 'kaggle-02') #kaggle
# push_to_hub_with_subset_splits(df, 'gayanin/woz-noised', 0.2, 0.5, 0.25, 0.25, 'gcd-02') #gcd
# push_to_hub_with_subset_splits(df, 'gayanin/woz-noised', 0.2, 0.36, 0.18, 0.46, 'babylon-02') #babylon
# print("end prob2")

# print("start prob3")
# push_to_hub_with_subset_splits(df, 'gayanin/woz-noised', 0.3, 0.34, 0.33, 0.33, 'kaggle-03') #kaggle
# push_to_hub_with_subset_splits(df, 'gayanin/woz-noised', 0.3, 0.5, 0.25, 0.25, 'gcd-03') #gcd
# push_to_hub_with_subset_splits(df, 'gayanin/woz-noised', 0.3, 0.36, 0.18, 0.46, 'babylon-03') #babylon
# print("end prob3")

# print("start prob4")
# push_to_hub_with_subset_splits(df, 'gayanin/woz-noised', 0.4, 0.34, 0.33, 0.33, 'kaggle-04') #kaggle
# push_to_hub_with_subset_splits(df, 'gayanin/woz-noised', 0.4, 0.5, 0.25, 0.25, 'gcd-04') #gcd
# push_to_hub_with_subset_splits(df, 'gayanin/woz-noised', 0.4, 0.36, 0.18, 0.46, 'babylon-04') #babylon
# print("end prob4")

# print("start prob5")
# push_to_hub_with_subset_splits(df, 'gayanin/woz-noised', 0.5, 0.34, 0.33, 0.33, 'kaggle-04') #kaggle
# push_to_hub_with_subset_splits(df, 'gayanin/woz-noised', 0.5, 0.5, 0.25, 0.25, 'gcd-05') #gcd
# push_to_hub_with_subset_splits(df, 'gayanin/woz-noised', 0.5, 0.36, 0.18, 0.46, 'babylon-05') #babylon
# print("end prob5")



########
# wer = 0.2
# sub_rate, del_rate, ins_rate = 0.5, 0.25, 0.25
# data3 = load_dataset('gayanin/gcd-native-v8')
    
# wer = 0.1
# sub_rate, del_rate, ins_rate = 0.34, 0.33, 0.33
# data3 = load_dataset('gayanin/kaggle-native-v8')

# wer = 0.1
# sub_rate, del_rate, ins_rate = 0.36, 0.18, 0.46
# data3 = load_dataset('gayanin/babylon-native-v8')

# dataset_ = load_dataset()
# data_k = concatenate_datasets([data3['train'],
#                              data3['validation'],
#                              data3['test']])
# df = data_k.to_pandas()

# df.to_csv('/Users/gayanin/github/RefinedLM/babylon-noised-v8-latest.csv')
# df.to_csv('/Users/gayanin/github/RefinedLM/gcd-noised-v8-latest.csv')
# df.to_csv('/Users/gayanin/github/RefinedLM/kaggle-noised-v8-latest.csv')

##########