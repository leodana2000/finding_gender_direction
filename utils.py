#A QUOI CA SERT

import torch
import random

device = "cuda"
cosim = torch.nn.CosineSimilarity(-1)



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



#ToDo: Implement a better version, using means of median
def get_quantile(leace_eraser, target_activations):
  #We compute the parameters of the hyperplane to project against.
  hyperplane = (leace_eraser.proj_right[0]/torch.norm(leace_eraser.proj_right[0], dim = -1).unsqueeze(-1)).squeeze().to(device)

  #We sort the projection.
  sorted_tensor, indices = torch.sort(torch.einsum('nd, ...d -> n...', target_activations, hyperplane), dim = -1)
  del indices

  Nb_ex = len(target_activations)
  #See above for the quantile formula.
  quantile = (sorted_tensor[Nb_ex//2 + 1] + sorted_tensor[(Nb_ex-1)//2+1])*(hyperplane)/2

  del sorted_tensor
  del hyperplane

  return quantile



def probe_eval(eraser_list):
  def metric(target_activations, layer : int, true_label):
    dir = eraser_list[layer].proj_right[0].to(device)
    bias = eraser_list[layer].bias.to(device)
    Nb_ex = len(target_activations)

    #we compute the accuracy
    acc = torch.sum(cosim(dir, target_activations - bias)*true_label > 0).item()/Nb_ex

    del dir
    del bias

    return acc
  return metric



def token_augmentation(list, text_list, tokenizer):
  for text in text_list:
    for i in range(1, len(text)):
      if (text[:i] != " ") and (text[i:] != " "):
        list.append(tokenizer(text[:i])["input_ids"] + tokenizer(text[i:])["input_ids"])


def show_proba(proba, tokenizer, level = 0.01, nb_tokens = 10, decode = False):
  #From a probability, get the top-{nb_tokens} tokens that have more than {level} probability.

  if decode:
    #to print the words
    proba_token_list = [(proba[i].item(), tokenizer.decode([i])) for i in range(len(proba)) if proba[i] > level]
  else:
    #to print the tokens
    proba_token_list = [(proba[i].item(), i) for i in range(len(proba)) if proba[i] > level]

  proba_token_list.sort()
  return proba_token_list[-nb_tokens:]



#Finds the occurences of the target inside the examples,
#but only the last one for each target.
def finds_indices(example_tokens, target_tokens):

  #stream_indice is the list of all streams where the target was detected
  stream_indices = []
  #example_indice is the list of all example where the target was detected
  example_indices = []
  #stream_example_indices is the list that gives the last stream of each token targeted
  #as well as the example it was seen in
  stream_example_indices = [[],[]]


  for i, (example, target_token) in enumerate(zip(example_tokens, target_tokens)):
    len_target = len(target_token)

    #If there is no target, we hook the last stream.
    if len(target_token) == 0:
      s_indice = torch.Tensor([len(example)])
      e_indice = torch.Tensor([i])
      stream_example_indices[0].append(len(example))
      stream_example_indices[1].append(i)

    #Otherwise, we hook the targeted streams.
    else:
      position = torch.where(torch.Tensor(example) == target_token[0])[0][-1].item() #[-1] means that we take only the last occurence
      if [example[i] for i in range(position, position + len_target)] == target_token:

        s_indice = torch.Tensor([pos for pos in range(position, position + len_target)])
        e_indice = torch.Tensor([i]*len_target)
        stream_example_indices[0].append(position + len_target - 1)
        stream_example_indices[1].append(i)
      else:
        print("Error, no target found.")

    stream_indices.append(s_indice)
    example_indices.append(e_indice)

  return torch.cat(stream_indices, dim = 0).to(int), torch.cat(example_indices, dim = 0).to(int), torch.Tensor(stream_example_indices).to(int)