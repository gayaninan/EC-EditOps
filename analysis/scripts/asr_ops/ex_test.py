import json

with open('asr-dictionary.json', 'r') as file:
        combined_dict = json.load(file)

key1_list = combined_dict[0]['list_sub_ref']
key2_list = combined_dict[0]['list_sub_trans']
key3_list = combined_dict[0]['list_del']
key4_list = combined_dict[0]['list_ins']


assert len(key1_list) == len(set(key1_list)), "list_sub_ref contains duplicate elements."
assert len(key2_list) == len(set(key2_list)), "list_sub_trans contains duplicate elements."
assert len(key3_list) == len(set(key3_list)), "list_del contains duplicate elements."
assert len(key4_list) == len(set(key4_list)), "list_ins contains duplicate elements."
