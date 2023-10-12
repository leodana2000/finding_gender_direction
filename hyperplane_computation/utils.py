#code containing all the useful functions for the notebook to run

import torch
device = "cuda"
cosim = torch.nn.CosineSimilarity(-1)

Gender = 1 | -1 #1 for male, -1 for female
Label = str #'noun' | 'pronoun' |'name' | 'anatomy'


def process_labels(gender : Gender, label : Label):
  '''
  Prepares the label to be fed into the concept_erasure module.
  '''
  if label == 'noun':
    return [gender, 0, 0, 0]
  elif label == 'pronoun':
    return [0, gender, 0, 0]
  elif label == 'name':
    return [0, 0, 0, gender]
  else:
    return [0, 0, gender, 0]


def initiate_activations(dataset, with_label = True, **dict):
  device = dict['device']
  model = dict['model']
  tokenizer = dict['tokenizer']

  indices = []
  activations = []
  labels = []

  for batch in dataset:
    #We keep track the length of each prompt to find the right indices on which to measure the probability.
    indices.append(torch.Tensor([len(tokenizer(data[0])["input_ids"])-1 for data in batch]).to(int).to(device))

    #initiate activations
    tokenized_batch = tokenizer([data[0] for data in batch], padding = True, return_tensors = 'pt')["input_ids"].to(device)
    positions = torch.Tensor([i for i in range(tokenized_batch.shape[1])]).to(int).to(device)
    activations.append(model.transformer.wte(tokenized_batch) + model.transformer.wpe(positions))
    if with_label:
      labels.append(torch.Tensor([process_labels(data[1], data[2]) for data in batch]).unsqueeze(-1))
  
  #Deletion of all the useless tensor to avoid RAM overload.
  del positions
  del tokenized_batch

  return indices, activations, labels


def gather_update_acts(activations, layer, post_layer_norm, indices, N_batch, **dict):
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
def get_quantile(leace_eraser, target_activations, **dict):
  device = dict['device']
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



#Finds the occurences of the target inside the examples, but only the last one for each target.
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