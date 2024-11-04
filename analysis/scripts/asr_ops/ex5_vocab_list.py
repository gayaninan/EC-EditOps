import json

with open('asr-dictionary.json', 'r') as file:
        combined_dict = json.load(file)

key1_list = combined_dict[0]['list_sub_ref']
key2_list = combined_dict[0]['list_sub_trans']
key3_list = combined_dict[0]['list_del']
key4_list = combined_dict[0]['list_ins']

vocab_list = key2_list + key3_list + key4_list

print(len(vocab_list))
print(len(set(vocab_list)))

with open('vocab-list.txt', 'w') as file:
        for word in set(vocab_list):
            file.write(word + '\n')


