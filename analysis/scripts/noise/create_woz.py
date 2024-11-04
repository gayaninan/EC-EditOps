from datasets import load_dataset
import json
import pandas as pd
import re
import inflect

p = inflect.engine()

def extract_utterances(dict_obj):
    # Return the 'utterances' attribute
    return dict_obj.get('utterance', None)  

def contains_phone_number(sentence):
    # Regular expression pattern for matching phone numbers
    pattern = r'\b\d{11}\b'
    # Search for the pattern in the sentence
    if re.search(pattern, sentence):
        return True
    return False

def preprocess_sentence_adjusted(sentence):
    sentence = sentence.lower()
    sentence = re.sub(r'[^a-zA-Z0-9\s]', '', sentence)
    return sentence

def numbers_to_words_inflect(sentence):
    numbers = re.findall(r'\b\d+\b', sentence)
    for number in numbers:
        word = p.number_to_words(number)
        sentence = re.sub(r'\b{}\b'.format(number), word, sentence)
    return sentence

def contains_digit(sentence):
    return any(char.isdigit() for char in sentence)

def sentence_more_than_two_words(sentence):
    return len(sentence.split()) > 3



def filter_sentences_less_than_three_words(row):
    return any(sentence_more_than_two_words(sentence) for sentence in row)



# Load the MultiWOZ dataset
dataset = load_dataset('multi_woz_v22')

df = dataset["train"].to_pandas()

df['utterances'] = df['turns'].apply(extract_utterances)

df_flattened = df['utterances'].explode().reset_index(drop=True).to_frame(name='refs')
df_filtered = df_flattened['refs'].apply(preprocess_sentence_adjusted).to_frame(name='refs')
# df_filtered = df_filtered[~df_filtered['refs'].apply(contains_phone_number)]
# df_filtered = df_filtered['refs'].apply(numbers_to_words_inflect).to_frame(name='refs')

num_rows_with_number = df_filtered['refs'].apply(contains_digit).sum()
df_no_numbers = df_filtered[~df_filtered['refs'].apply(contains_digit)]

df_no_numbers['refs'] = df_no_numbers['refs'].apply(lambda x: x if len(x.split()) >= 3 else None)

df_no_numbers['refs'] = df_no_numbers['refs'].str.replace('-', ' ', regex=False)

final_df = df_no_numbers.dropna()

print(final_df)
final_df.to_csv('woz.csv')

