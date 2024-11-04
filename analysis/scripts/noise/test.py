# import random
# import re

# stop_words = ['the', 'are', 'a', 'in']


# def get_random_word(word):
#     shuffled_stop_words = stop_words.copy()
#     random.shuffle(shuffled_stop_words)
#     for stop_word in shuffled_stop_words:
#         # replacement = stop_word
#         replacement = re.sub(r'[^a-zA-Z]', '', stop_word) 
#         # print(replacement)
#         if word != replacement:
#             print('replacement: ', replacement) 
#             return replacement
#     print('word: ', word) 
#     return word 
    

# sentence = 'cat sat on the map today'

# list = []
# for word in sentence.split():
#     list.append(get_random_word(word))

# print(list)

from accelerate import Accelerator
a = Accelerator()
print(a.device)