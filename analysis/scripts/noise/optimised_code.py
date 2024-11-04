import random
import multiprocessing
import pandas as pd
import numpy as np
import pronouncing
import datasets
import nltk
from datasets import Dataset, load_dataset
from datetime import datetime
from joblib import Parallel, delayed
from better_profanity import profanity
import re 
from nltk.corpus import stopwords

random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
               
# nltk.download('wordnet')
from nltk.corpus import wordnet

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
        replacement = re.sub(r'[^a-zA-Z]', '', stop_word) 
        print(replacement)
        if word != replacement:
            return stop_word 
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

def noise_sentence(sentence, prob, strategies):
    words = sentence.split()
    noised_words = []
    for word in words:
        if np.random.random() <= prob:
            error_type = np.random.choice(["sub", "del", "ins"], p=strategies)
            print('error_type: ',error_type)
            if error_type == "sub":
              word = get_phonotically_similar_word(word)
              word = re.sub(r'[^a-zA-Z]', '', word) 
              noised_words.append(word)
            elif error_type == "del":
              None
            if error_type == "ins":
                noised_words.append(word+" "+get_random_word(word))
        else:
            noised_words.append(word)
    noised_sentence = " ".join(noised_words)
    return noised_sentence

def apply_noise_to_sentence(sentence, prob, strategies):
    return noise_sentence(sentence, prob, strategies)

def parallel_apply_noise(df, prob, strategies):
    num_cores = multiprocessing.cpu_count()
    sentences = df["refs"].tolist()
    results = Parallel(n_jobs=num_cores)(delayed(apply_noise_to_sentence)(sentence, prob, strategies) for sentence in sentences)
    df["trans"] = results
    return df

def apply_noise(df, prob, prefix, strategies, savepath):
    df = parallel_apply_noise(df, prob, strategies)
    df.to_csv(savepath)
    # push_to_hub_with_subset_splits(df, prefix, savepath)

def push_to_hub_with_subset_splits(df,prefix, savepath):

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
#   dataset.push_to_hub(savepath, prefix)
  dataset.push_to_hub(savepath)


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


def to_hf(df, save_path, prob, strategies, prefix):
  print(df.shape)
  print("Noising started at"+str(datetime.now().strftime("%H:%M:%S")))
  apply_noise(df, prob, prefix, strategies, save_path)
  print("Noising ended at"+str(datetime.now().strftime("%H:%M:%S")))


def split_dataframe_equally(df):
    thirds = np.array_split(df, 3)
    
    return thirds

# read df from HF datasets
# df_ = dataset_to_df('gayanin/kaggle-native-v8')
df_ = dataset_to_df('gayanin/gcd-native-v8')
# df_ = dataset_to_df('gayanin/babylon-native-v8')

print(df_.shape)

# to_hf(df_, 'gayanin/kaggle-native-v8-noised', 0.1, [0.34, 0.33, 0.33], 'gcd-prob-01') #kaggle
# to_hf(df_, 'gayanin/gcd-native-v8-noised-test', 0.2, [0.5, 0.25, 0.25], 'gcd-prob-01') #gcd
# to_hf(df_, 'gayanin/babylon-native-v8-noised', 0.1, [0.36, 0.18, 0.46], 'babylon-prob-01') #babylon


# to_hf(df_, 'gayanin/kaggle-native-v8-noised', 0.1, [0.34, 0.33, 0.33], 'gcd-prob-01') #kaggle
to_hf(df_, 'test-gcd.csv', 0.2, [0.5, 0.25, 0.25], 'gcd-prob-01') #gcd
# to_hf(df_, 'gayanin/babylon-native-v8-noised', 0.1, [0.36, 0.18, 0.46], 'babylon-prob-01') #babylon

