import random
import pandas as pd
import numpy as np
import datasets
import re
import inflect

from transformers import BartTokenizer

p = inflect.engine()

tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

def dataset_to_df(datasetpath):
  dataset = datasets.load_dataset(datasetpath, split=None)
  dfs = []
  for split in dataset:
    df = dataset[split].to_pandas()
    dfs.append(df)

  df = pd.concat(dfs, ignore_index=True)
  return df

def contains_number(sentence):
    return bool(re.search(r'\d', sentence))

def filter_cols(df):
    filtered_column1 = df[df['refs'].apply(contains_number)]
    filtered_column2 = df[df['trans'].apply(contains_number)]
    return filtered_column1, filtered_column2

def numbers_to_words_inflect(sentence):
    numbers = re.findall(r'\b\d+\b', sentence)
    for number in numbers:
        word = p.number_to_words(number)
        sentence = re.sub(r'\b{}\b'.format(number), word, sentence)
    return sentence

def convert_(df):
    df['refs_'] = df['refs'].apply(numbers_to_words_inflect)
    df['trans_'] = df['trans'].apply(numbers_to_words_inflect)
    return df

def has_less_than_three_words(sentence):
    return len(sentence.split()) < 3

def filter_sentences_less_than_three_words(row):
    return any(has_less_than_three_words(sentence) for sentence in row)


def sentence_more_than_two_words(sentence):
    return len(sentence.split()) > 3


def apply_all_fn(csv_, dataset_, out_dataset):
    df = dataset_to_df(dataset_)
    print(df.shape)
    df = convert_(df)
    
    sentences_less_than_three_words_column1 = df['refs_'][df['refs_'].apply(sentence_more_than_two_words)]
    sentences_less_than_three_words_column2 = df['trans_'][df['trans_'].apply(sentence_more_than_two_words)]

    filtered_sentences_df = pd.DataFrame({
        'refs': sentences_less_than_three_words_column1,
        'trans': sentences_less_than_three_words_column2
    }).reset_index(drop=True)

    print(filtered_sentences_df.shape)

    filtered_sentences_df_without_na = filtered_sentences_df.dropna()
    print(filtered_sentences_df_without_na.shape)
    for column in filtered_sentences_df_without_na.columns:
        filtered_sentences_df_without_na[column] = filtered_sentences_df_without_na[column].str.replace('"', '', regex=False)
        filtered_sentences_df_without_na[column] = filtered_sentences_df_without_na[column].str.replace('-', ' ', regex=False)

    print('')

    filtered_sentences_df_without_na.to_csv(csv_, header=True, index=False)

def calculate_bart_token_count(sentence):
        tokens = tokenizer.tokenize(sentence)
        return len(tokens)

def check_token_distribution(df):
    df['word_count'] = df['refs'].apply(lambda x: len(x.split()))
    df_sorted = df.sort_values(by='word_count', ascending=False)
    df_sorted = df_sorted [:100]
    df_sorted['bart_token_count'] = df_sorted['refs'].apply(calculate_bart_token_count)

    df_sorted_sorted = df_sorted.sort_values(by='bart_token_count', ascending=False)
    return df_sorted_sorted

df_pubmed = dataset_to_df('gayanin/pubmed-abstracts') #longest 167
df_gcd = pd.read_csv('check_new_gcd1.csv') #longes1 215
df_kaggle =  pd.read_csv('check_new_kaggle1.csv') #longes1 33
df_babylon =  pd.read_csv('check_new_babylon1.csv') #longest 101

print(check_token_distribution(df_pubmed)) 