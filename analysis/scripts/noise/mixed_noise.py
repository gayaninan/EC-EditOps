import pandas as pd
import numpy as np
import random
from nltk.corpus import wordnet
import datasets

def get_random_synonym(word):
    synonyms = wordnet.synsets(word)
    if not synonyms:
        return word
    synonym = synonyms[np.random.randint(0, len(synonyms))].lemmas()[0].name()
    return synonym

def noise_sub(sentence, p):
    words = sentence.split()
    masked_words = [get_random_synonym(word) if np.random.rand() < p else word for word in words]
    return ' '.join(masked_words)

def noise_ins(sentence, p):
    words = sentence.split()
    modified_sentence = []
    for word in words:
        modified_sentence.append(word)
        # Insert a random word from the sentence with probability p
        if np.random.rand() <= p and len(words) > 1:
            random_word = np.random.choice(words)
            modified_sentence.append(random_word)
    return ' '.join(modified_sentence)

def noise_del(sentence, p):
    words = sentence.split()
    modified_sentence = []

    for i, word in enumerate(words):
        if random.random() > p:
            modified_sentence.append(word)
        elif i == len(words) - 1 and not modified_sentence:
            modified_sentence.append(word)
    return ' '.join(modified_sentence)

def apply_noising_to_sentence(sentence, alpha):
    # Generate probabilities for each noising method
    probs = np.random.dirichlet(alpha)

    # Apply noising methods
    noised_sentence = noise_sub(sentence, probs[0])
    noised_sentence = noise_ins(noised_sentence, probs[1])
    noised_sentence = noise_del(noised_sentence, probs[2])

    return noised_sentence

def apply_noising_to_dataset(df, output_file):
    # Read the CSV file
    # df = pd.read_csv(csv_file)

    # Specify the column name here if it's different
    column_name = 'trans'
    alpha = [0.5728717451807007, 0.21661046147205762, 0.21051779334724174]

    # Apply noising to each sentence in the specified column
    df['noised'] = df['refs'].apply(lambda x: apply_noising_to_sentence(x, alpha))

    # Write the processed data to a new CSV file
    df.to_csv(output_file, index=False)

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

# Example usage

# df = dataset_to_df('gayanin/pubmed-abstracts')

data = {
    "refs": [
        "The quick brown fox jumps over the lazy dog.",
        "An apple a day keeps the doctor away.",
        "To be or not to be, that is the question.",
        "All that glitters is not gold.",
        "A bird in the hand is worth two in the bush."
    ],
    "trans": [
        "The rapid brown fox leaps over lazy dog.",
        "An apple a day keeps the doctor away.",
        "To be or not to be, that is the question.",
        "All that glitters is not gold.",
        "A bird in the hand is worth two in the bush."
    ]
}

df = pd.DataFrame(data)
df

output_file = 'mixed_noise_test.csv'
apply_noising_to_dataset(df, output_file)


