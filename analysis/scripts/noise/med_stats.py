from datasets import load_dataset
import json
import pandas as pd

# medical_dict = json.loads('meds.json')

with open('/Users/gayanin/github/RefinedLM/analysis/scripts/medical_terms.json', 'r') as file:
    medical_dict = json.load(file)

all_medical_terms = [term for terms in medical_dict.values() for term in terms]

gcd = load_dataset('gayanin/gcd-native-v8')
gcd_df = gcd['test'].to_pandas()

df = gcd_df

def preprocess_text(text):
    return text.lower()

def count_medical_terms(text, medical_terms):
    text = preprocess_text(text)  
    return sum(text.count(term) for term in medical_terms)

def get_category_wise_med_stats(data_path):

    data = load_dataset(data_path)
    df = data['test'].to_pandas()

    for category, terms in medical_dict.items():
        df[f'{category}_count_refs'] = df['refs'].apply(lambda x: count_medical_terms(x, terms))
        df[f'{category}_count_trans'] = df['trans'].apply(lambda x: count_medical_terms(x, terms))

    diseases_total_refs = df['Diseases_count_refs'].sum() 
    diseases_total_trans = df['Diseases_count_trans'].sum()

    chemicals_total_refs = df['Chemicals_count_refs'].sum()
    chemicals_total_trans = df['Chemicals_count_trans'].sum()

    print("diseases_total_refs: ", diseases_total_refs)
    print("diseases_total_trans: ", diseases_total_trans)

    print("chemicals_total_refs: ", chemicals_total_refs)
    print("chemicals_total_trans: ", chemicals_total_trans)

def get_med_stats(data_path):

    data = load_dataset(data_path)
    df = data['test'].to_pandas()

    df['med_count_refs'] = df['refs'].apply(lambda x: count_medical_terms(x, all_medical_terms))
    df['med_count_trans'] = df['trans'].apply(lambda x: count_medical_terms(x, all_medical_terms))

    total_medical_terms_count_refs = df['med_count_refs'].sum()
    total_medical_terms_count_trans = df['med_count_trans'].sum()

    print(f"Total count of all medical terms in refs: {total_medical_terms_count_refs}")
    print(f"Total count of all medical terms in trans: {total_medical_terms_count_trans}")

# get_med_stats('gayanin/gcd-native-v8')
# get_med_stats('gayanin/kaggle-native-v8')
get_category_wise_med_stats('gayanin/babylon-native-v8')