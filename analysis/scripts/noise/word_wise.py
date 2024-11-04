import random
import time
import pandas as pd
import numpy as np
import pronouncing
import datasets
from datasets import Dataset
from datetime import datetime

import nltk

nltk.download('wordnet')
from nltk.corpus import wordnet

# df = pd.DataFrame({"sentence":["One morning, when Gregor Samsa woke from troubled dreams, he found himself transformed in his bed into a horrible vermin.",
#                                "He lay on his armour-like back, and if he lifted his head a little he could see his brown belly, slightly domed and divided by arches into stiff sections.",
#                                "The bedding was hardly able to cover it and seemed ready to slide off any moment.",
#                                "His many legs, pitifully thin compared with the size of the rest of him, waved about helplessly as he looked."]})

# data = {'sentence': ["Hello world what a fine day", "This is a test for a program", "Sample sentence for testing the code"]}
# df1 = pd.DataFrame(data)

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


def most_probable_rhyme(word, rhyming_words):
    if not rhyming_words:
        return None

    # using the absolute difference in lengths
    return min(rhyming_words, key=lambda x: abs(len(x) - len(word)))

# Get the list of rhyming words
def get_phonotically_similar_word(word):
    rhyming_words = pronouncing.rhymes(word)
    probable_word = most_probable_rhyme(word, rhyming_words)
    return probable_word if probable_word is not None else word


def get_random_word():
    all_lemmas = list(set(lemma.name() for synset in wordnet.all_synsets() for lemma in synset.lemmas()))
    # print(all_lemmas[0:10])
    return random.choice(all_lemmas)

def noise_word(word, error_types):
  error_types = ["sub", "del", "ins"]
  error_probs = [0.57, 0.22, 0.21]

  error_type = np.random.choice(error_types, p=error_probs)
#   print("[DEBUG] error_type: ", error_type)

  if error_type == "sub":
     word = get_phonotically_similar_word(word)
  elif error_type == "del":
    word = ""
  elif error_type == "ins":
        # word = get_random_word()
     return None
  
#   print("[DEBUG] word1: ", word)
  return word


def noise_sentence(sentence, prob):
    words = sentence.split()
    noised_words = []
    for word in words:
        if np.random.random() < prob:
            error_type = np.random.choice(["sub", "del", "ins"], p=[0.57, 0.22, 0.21])
            noised_word = noise_word(word, error_type)
            if noised_word is not None:
                noised_words.append(noised_word)
            if error_type == "ins":
                noised_words.append(get_random_word())
        else:
            noised_words.append(word)
    noised_sentence = " ".join(noised_words)
    return noised_sentence

def push_to_hub_with_subset_splits(df,prefix):

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

  dataset.push_to_hub("gayanin/pubmed-mixed-noise", prefix)

#Run code
df_ = dataset_to_df('gayanin/pubmed-abstracts')

def apply_noise(df, prob, prefix):
    df["trans"] = df["refs"].apply(lambda x: noise_sentence(x, prob))
    push_to_hub_with_subset_splits(df,prefix)

print(df_.shape)
print(datetime.now().strftime("%H:%M:%S"))
apply_noise(df_[:10000], 0.4, "prob-0.1")
print(datetime.now().strftime("%H:%M:%S"))
# apply_noise(df_[:10], 0.2, "prob-0.2")

# apply_noise(df_, 0.3, "prob-0.3")
# apply_noise(df_, 0.4, "prob-0.4")
# apply_noise(df_, 0.5, "prob-0.5")