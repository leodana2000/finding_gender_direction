#code containing all the useful functions for the notebook to run

import torch
import numpy as np
from hyperplane_computation.concept_erasure.leace import LeaceEraser
cosim = torch.nn.CosineSimilarity(-1)

Bin = 1 | -1 #1 for male, -1 for female
Label = str #'noun' | 'pronoun' |'name' | 'anatomy'
Data = str
Token = int


def concat_list(lst):
  new_lst = []
  for l in lst:
    new_lst += l
  return new_lst


def transpose_list(lst):
  final_lst = []
  for batch in lst:
    np_batch = np.array(batch)
    np_batch = np.transpose(np_batch, (2, 1, 0))
    final_lst.append(np_batch)
    
  np.concatenate(final_lst, 0)
  final_lst = np.concatenate(final_lst, 0)
  final_lst = np.transpose(final_lst, (2, 1, 0))

  return final_lst


def initiate_activations(dataset : list[list[Data, list[Bin]]], **dict):
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
    labels.append(torch.Tensor([data[1] for data in batch]))
  
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


def show_proba(proba : torch.Tensor, level : float = 0.01, nb_tokens : int = 10, decode : bool = False, **dict):
  '''
  From a probability, get the top-{nb_tokens} tokens that have more than {level} probability.
  '''
  tokenizer = dict['tokenizer']

  if decode:
    #to print the words
    proba_token_list = [(proba[i].item(), tokenizer.decode([i])) for i in range(len(proba)) if proba[i] > level]
  else:
    #to print the tokens
    proba_token_list = [(proba[i].item(), i) for i in range(len(proba)) if proba[i] > level]

  proba_token_list.sort()
  return proba_token_list[-nb_tokens:]


def finds_indices(ex_batch : list[list[Token]], tar_batch : list[list[Token]]):
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

  for i, (ex, tar) in enumerate(zip(ex_batch, tar_batch)):
    len_tar= len(tar)
    len_ex = len(ex)

    #If there is no target, we take the last stream.
    if len_tar == 0:
      s_indice = torch.Tensor([len_ex])
      e_indice = torch.Tensor([i])
      stream_example_indices[0].append(len_ex)
      stream_example_indices[1].append(i)

    #Otherwise, we take the targeted streams.
    else:
      position = torch.where(torch.Tensor(ex) == tar[0])[0][-1].item() #[-1] means that we take only the last occurence
      if [ex[i] for i in range(position, position + len_tar)] == tar:

        s_indice = torch.Tensor([pos for pos in range(position, position + len_tar)])
        e_indice = torch.Tensor([i]*len_tar)
        stream_example_indices[0].append(position + len_tar - 1)
        stream_example_indices[1].append(i)
      else:
        print("Error, no target found.")

    stream_indices.append(s_indice)
    example_indices.append(e_indice)

  return [torch.cat(stream_indices, dim = 0).to(int), 
          torch.cat(example_indices, dim = 0).to(int), 
          torch.Tensor(stream_example_indices).to(int)]