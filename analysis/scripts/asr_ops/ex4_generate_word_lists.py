import pandas as pd
import json

def get_counts(cdv_path, out_file):
    df = pd.read_csv(cdv_path)

    list_sub_ref = []
    list_sub_trans = []
    list_del = []
    list_ins = []

    # Processing substitutions
    for subs in df['sub'].dropna():
        # Splitting each substitution string into pairs
        subs_pairs = subs.split('; ')
        for pair in subs_pairs:
            ref, trans = pair.split(' -> ')
            list_sub_ref.append(ref.strip())
            list_sub_trans.append(trans.strip())

    # Processing deletions
    for dels in df['del'].dropna():
        # Splitting each deletion string into individual deletions
        dels_list = dels.split('; ')
        for del_item in dels_list:
            list_del.append(del_item.strip())

    # Processing insertions
    for inss in df['ins'].dropna():
        # Splitting each insertion string into individual insertions
        inss_list = inss.split('; ')
        for ins_item in inss_list:
            list_ins.append(ins_item.strip())

    # print("list_sub_ref =", list_sub_ref)
    # print("list_sub_trans =", list_sub_trans)
    # print("list_del =", list_del)
    # print("list_ins =", list_ins)
    # print("list_sub_ref =", len(list_sub_ref))
    # print("list_sub_trans =",len(list_sub_trans))
    # print("list_del =", len(list_del))
    # print("list_ins =", len(list_ins))
    # print("list_sub_ref =", len(set((list_sub_ref))))
    # print("list_sub_trans =",len(set(list_sub_trans)))
    # print("list_del =", len(set(list_del)))
    # print("list_ins =", len(set(list_ins)))


    output_dict = {
    'list_sub_ref': list(set(list_sub_ref)),
    'list_sub_trans': list(set(list_sub_trans)),
    'list_del': list(set(list_del)),
    'list_ins': list(set(set(list_ins)))
    }

    with open(out_file, 'w') as json_file:
        json.dump(output_dict, json_file, indent=4)

    # Displaying the dictionary
    # print(output_dict)

# get_counts('gcd-native-v8-ops.csv', 'gcd-native-v8-ops.json')
# get_counts('kaggle-native-v8-ops.csv', 'kaggle-native-v8-ops.json')
# get_counts('babylon-native-v8-ops.csv', 'babylon-native-v8-ops.json')

import json

# Path to your JSON files
file_paths = ['gcd-native-v8-ops.json', 'kaggle-native-v8-ops.json', 'babylon-native-v8-ops.json']

# Initialize an empty list to hold the dictionaries
dicts = []

for file_path in file_paths:
    with open(file_path, 'r') as file:
        # Load the JSON content and append it to the list
        dicts.append(json.load(file))

with open('asr-dictionary.json', 'w') as json_file:
        json.dump(dicts, json_file, indent=4)
