#contains the usefull functions for the generation of the data

import random 

#splits randomly a list into training-testing sets
def split_list(lst):
    random.seed(42)
    lst_copy = lst[:]
    random.shuffle(lst_copy)
    split_index = int(len(lst) * 0.9)
    set1 = lst_copy[:split_index]
    set2 = lst_copy[split_index:]
    return set1, set2


def concat_list(lst):
  new_lst = []
  for l in lst:
    new_lst += l
  return new_lst