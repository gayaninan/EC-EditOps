import pandas as pd
import numpy as np
from nltk.corpus import wordnet
import datasets

# Define the noising functions
def get_random_synonym(word):
    synonyms = wordnet.synsets(word)
    if not synonyms:
        return word
    synonym = synonyms[np.random.randint(0, len(synonyms))].lemmas()[0].name()
    return synonym.replace('_', ' ')

def noise_sub(sentence, p):
    words = sentence.split()
    masked_words = [get_random_synonym(word) if np.random.rand() < p else word for word in words]
    return ' '.join(masked_words)

def noise_ins(sentence, p):
    words = sentence.split()
    modified_sentence = []
    for word in words:
        modified_sentence.append(word)
        if np.random.rand() < p and len(words) > 1:
            random_word = np.random.choice(words)
            modified_sentence.append(random_word)
    return ' '.join(modified_sentence)

def noise_del(sentence, p):
    words = sentence.split()
    modified_sentence = [word for word in words if np.random.rand() > p]
    return ' '.join(modified_sentence)

def process_dataframe(df, output_file, p_sub, p_del, p_ins):
    # Load the dataset
    # df = pd.read_csv(input_file)

    # Apply noising to each sentence in the 'refs' column
    df['noised'] = df['refs'].apply(lambda x: noise_del(noise_ins(noise_sub(x, p_sub), p_ins), p_del))

    # Save the modified dataframe
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

# Creating a DataFrame with example sentences
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

output_csv = 'combined_test.csv'

# Scenario #1
# p_sub = 0.5728717451807007
# p_del = 0.21661046147205762
# p_ins = 0.21051779334724174

# mixed - p_sub: 0.5514603658723305, p_del: 0.24672313373876936, p_ins: 0.20181650038890012

# Scenario #2
p_sub = 0.57
p_del = 0.22
p_ins = 0.21
# mixed - p_sub: 0.5514603658723305, p_del: 0.24672313373876936, p_ins: 0.20181650038890012

process_dataframe(df,output_csv,p_sub, p_del, p_ins)
