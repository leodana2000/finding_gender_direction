#code containing all the useful functions for the notebook to run

import torch
from hyperplane_computation.concept_erasure.leace import LeaceEraser
cosim = torch.nn.CosineSimilarity(-1)

Gender = 1 | -1 #1 for male, -1 for female
Label = str #'noun' | 'pronoun' |'name' | 'anatomy'
Data = str


def concat_list(lst):
  new_lst = []
  for l in lst:
    new_lst += l
  return new_lst


def initiate_activations(dataset : list[list[Data, list[int]]], with_label = True, **dict):
  device = dict['device']
  model = dict['model']
  tokenizer = dict['tokenizer']

  indices = []
  activations = []
  labels = []

  for batch in dataset:
    #We keep track the length of each prompt to find the indices on which to measure the probability.
    indices.append(torch.Tensor([len(tokenizer(data[0])["input_ids"])-1 for data in batch]).to(int).to(device))

    #We initiate the activations manually
    tokenized_batch = tokenizer([data[0] for data in batch], padding = True, return_tensors = 'pt')["input_ids"].to(device)
    positions = torch.arange(tokenized_batch.shape[1]).to(int).to(device).to(int).to(device)
    activations.append(model.transformer.wte(tokenized_batch) + model.transformer.wpe(positions))
    if with_label:
      labels.append(torch.Tensor([data[1] for data in batch]).unsqueeze(-1))
  
  #Deletion of all the useless tensor to avoid RAM overload.
  del positions
  del tokenized_batch

  return indices, activations, labels


def gather_update_acts(activations : list[torch.Tensor], layer : int, 
                       post_layer_norm : bool, indices : list[torch.Tensor], 
                       N_batch : int, **dict):
  model = dict['model']
  device = dict['device']

  target_activations = []
  for batch_num in range(N_batch):

    #We update each activation through the next layer.
    if layer != 0:
      activations[batch_num] = model.transformer.h[layer](activations[batch_num])[0]

    #We choose the activation of the targeted tokens and fit the leace estimator.
    if post_layer_norm:
      acts = model.transformer.h[layer].ln_1(activations[batch_num])
    else:
      acts = activations[batch_num]

    target_activations.append(torch.cat([act[ind].unsqueeze(0) for act, ind in zip(acts, indices[batch_num])], dim = 0).to(device))
    del acts
  
  return activations, target_activations

#ToDo: Implement a better version, using means of median
def get_quantile(leace_eraser : LeaceEraser, all_acts : torch.Tensor, **dict):
  device = dict['device']
  #We compute the parameters of the hyperplane to project against.
  hyperplane = (leace_eraser.proj_right[0]/torch.norm(leace_eraser.proj_right[0], dim = -1).unsqueeze(-1)).squeeze().to(device)

  #We sort the projection.
  sorted_tensor, indices = torch.sort(torch.einsum('nd, ...d -> n...', all_acts, hyperplane), dim = -1)
  del indices

  Nb_ex = len(all_acts)
  quantile = (sorted_tensor[Nb_ex//2 + 1] + sorted_tensor[(Nb_ex-1)//2+1])*(hyperplane)/2

  del sorted_tensor

  return quantile.squeeze()



def probe_eval(eraser_list : list[LeaceEraser], **dict):
  device = dict['device']

  def metric(all_acts : torch.Tensor, layer : int, true_label : torch.Tensor):
    '''
    Evaluate the accuracy of each type of data on its hyperplane.
    '''
    dir = eraser_list[layer].proj_right.to(device)
    bias = eraser_list[layer].bias.to(device)
    Nb_ex = len(all_acts)

    acc = torch.sum(cosim(true_label@dir, all_acts - bias) > 0).item()/Nb_ex
    return acc
  
  return metric


def token_augmentation(list : list, text_list : list[str], **dict):
  tokenizer = dict['tokenizer']
  for text in text_list:
    for i in range(1, len(text)):
      if (text[:i] != " ") and (text[i:] != " "):
        list.append(tokenizer(text[:i])["input_ids"] + tokenizer(text[i:])["input_ids"])


def show_proba(proba, level = 0.01, nb_tokens = 10, decode = False, **dict):
  '''From a probability, get the top-{nb_tokens} tokens that have more than {level} probability.'''
  tokenizer = dict['tokenizer']

  if decode:
    #to print the words
    proba_token_list = [(proba[i].item(), tokenizer.decode([i])) for i in range(len(proba)) if proba[i] > level]
  else:
    #to print the tokens
    proba_token_list = [(proba[i].item(), i) for i in range(len(proba)) if proba[i] > level]

  proba_token_list.sort()
  return proba_token_list[-nb_tokens:]


def finds_indices(example_tokens : list[list[int]], target_tokens : list[list[int], list[int]]):
  '''
  Finds the occurences of the target inside the examples, but only the last one for each target.
  stream_indices : list[int], list of all streams where the target was detected.
  example_indices : list[int], list of all example where the target was detected.
  stream_example_indices : list[list[int], list[int]], list that give for each target
  the join example and stream where it was detected.
  '''

  stream_indices = []
  example_indices = []
  stream_example_indices = [[],[]]

  for i, (example, target_token) in enumerate(zip(example_tokens, target_tokens)):
    len_target = len(target_token)

    #If there is no target, we take the last stream.
    if len(target_token) == 0:
      s_indice = torch.Tensor([len(example)])
      e_indice = torch.Tensor([i])
      stream_example_indices[0].append(len(example))
      stream_example_indices[1].append(i)

    #Otherwise, we take the targeted streams.
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