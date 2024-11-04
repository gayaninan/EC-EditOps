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

# random.seed(42)  # For reproducibility
# np.random.seed(42)

def is_appropriate(word):
    return not profanity.contains_profanity(word) and ' ' not in word and '_' not in word and '-' not in word

# Precompute the list of all appropriate lemmas from WordNet
# all_lemmas = [lemma.name() for synset in wordnet.all_synsets() for lemma in synset.lemmas() if is_appropriate(lemma.name())]

# def get_random_word():
#     word = random.choice(all_lemmas)
#     if is_appropriate(word):
#         return word
#     else:
#         return get_random_word() 
stop_words_ = stopwords.words('english')
stop_words = [word for word in stop_words_ if len(word) > 1]
 
def get_random_word(word):
    shuffled_stop_words = stop_words.copy()
    random.shuffle(shuffled_stop_words)
    for stop_word in shuffled_stop_words:
        # replacement = stop_word
        replacement = re.sub(r'[^a-zA-Z]', '', stop_word) 
        # print(replacement)
        if word != replacement:
            return replacement 
    return word 
    
def most_probable_rhyme(word, rhyming_words):
    if not rhyming_words:
        return None
    return min(rhyming_words, key=lambda x: abs(len(x) - len(word)))

def get_phonotically_similar_word(word):
    rhyming_words = [w for w in pronouncing.rhymes(word) if is_appropriate(w)]
    if not rhyming_words:  
        return word

    probable_word = most_probable_rhyme(word, rhyming_words)
    return probable_word if probable_word is not None else get_phonotically_similar_word(word) 

def dataset_to_df(datasetpath,split):
  dataset = datasets.load_dataset(datasetpath, split)
  dfs = []
  for split in dataset:
    df = dataset[split].to_pandas()
    dfs.append(df)

def substitute(word):
    sub_word = get_phonotically_similar_word(word)
    replacement = re.sub(r'[^a-zA-Z]', '', sub_word)
    return replacement

def delete(word):
    return ''

def insert(word):
    return word + " "+ get_random_word(word)

# def noise_word(word, error_type):
#     if error_type == 'sub':
#         return substitute(word)
#     elif error_type == 'del':
#         return delete(word)
#     elif error_type == 'ins':
#         return insert(word)
#     else:
#         return word  


def push_to_hub_with_subset_splits(df, savepath, wer, error_type, prefix):
  
    word_list, sent_list = new_noise(df, wer, error_type)

    if error_type == 'del':
        trans_list = transform_del(word_list, sent_list)
    if error_type == 'sub':
       trans_list = transform_sub(word_list, sent_list)
    if error_type == 'ins':
        trans_list = transform_ins(word_list, sent_list)
    
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
    dataset.push_to_hub(savepath, prefix)
    #   dataset.push_to_hub(savepath)

def new_noise(df, wer, error_type):
    total_words = df['refs'].str.split().str.len().sum()

    total_errors = int(wer * total_words)
    total_without_errors = total_words - total_errors

    if error_type == 'del':
        all_words =['del'] * total_errors + ['nochange']*total_without_errors
    if error_type == 'sub':
        all_words =['sub'] * total_errors + ['nochange']*total_without_errors
    if error_type == 'ins':
        all_words =['ins'] * total_errors + ['nochange']*total_without_errors
        
    # print(all_words)
    random.shuffle(all_words)  
    # print(all_words)
    sent_list = df["refs"].tolist()
    return all_words, sent_list

def transform_ins(all_words, sent_list):
    count = 0
    res_list = []
    for s in sent_list:
        s_trns = []
        for w in s.split(): 
            if all_words[count] == 'ins':
                s_trns.append(insert(w))
            else:
                s_trns.append(w)
            count+=1
        res_list.append(" ".join(s_trns))
    return res_list

def transform_sub(all_words, sent_list):
    count = 0
    res_list = []
    for s in sent_list:
        s_trns = []
        for w in s.split(): 
            if all_words[count] == 'sub': 
                w = substitute(w)
                s_trns.append(w)
            else:
                s_trns.append(w)
            count+=1
        res_list.append(" ".join(s_trns))
    return res_list

def transform_del(all_words, sent_list):
    count = 0
    res_list = []
    for s in sent_list:
        s_trns = []
        for w in s.split(): 
            if all_words[count] == 'del':
                # s_trns.append(delete(w))
                # w = ""
                # s_trns.append(w)
                None
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

push_to_hub_with_subset_splits(data_df, 'gayanin/babylon-native-v8-noise-op-wise', 0.1, 'sub', 'sub-0.1') 
push_to_hub_with_subset_splits(data_df, 'gayanin/babylon-native-v8-noise-op-wise', 0.1, 'ins', 'ins-0.1') 
push_to_hub_with_subset_splits(data_df, 'gayanin/babylon-native-v8-noise-op-wise', 0.1, 'del', 'del-0.1') 

push_to_hub_with_subset_splits(data_df, 'gayanin/babylon-native-v8-noise-op-wise', 0.2, 'sub', 'sub-0.2') 
push_to_hub_with_subset_splits(data_df, 'gayanin/babylon-native-v8-noise-op-wise', 0.2, 'del', 'del-0.2') 
push_to_hub_with_subset_splits(data_df, 'gayanin/babylon-native-v8-noise-op-wise', 0.2, 'ins', 'ins-0.2') 

push_to_hub_with_subset_splits(data_df, 'gayanin/babylon-native-v8-noise-op-wise', 0.3, 'sub', 'sub-0.3') 
push_to_hub_with_subset_splits(data_df, 'gayanin/babylon-native-v8-noise-op-wise', 0.3, 'del', 'del-0.3') 
push_to_hub_with_subset_splits(data_df, 'gayanin/babylon-native-v8-noise-op-wise', 0.3, 'ins', 'ins-0.3') 

push_to_hub_with_subset_splits(data_df, 'gayanin/babylon-native-v8-noise-op-wise', 0.4, 'sub', 'sub-0.4') 
push_to_hub_with_subset_splits(data_df, 'gayanin/babylon-native-v8-noise-op-wise', 0.4, 'del', 'del-0.4') 
push_to_hub_with_subset_splits(data_df, 'gayanin/babylon-native-v8-noise-op-wise', 0.4, 'ins', 'ins-0.4') 

push_to_hub_with_subset_splits(data_df, 'gayanin/babylon-native-v8-noise-op-wise', 0.5, 'sub', 'sub-0.5') 
push_to_hub_with_subset_splits(data_df, 'gayanin/babylon-native-v8-noise-op-wise', 0.5, 'del', 'del-0.5') 
push_to_hub_with_subset_splits(data_df, 'gayanin/babylon-native-v8-noise-op-wise', 0.5, 'ins', 'ins-0.5') 












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