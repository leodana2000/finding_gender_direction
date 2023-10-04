#code that contains the functions to store and evaluate hyperplanes and directions
#You can use them in the notebook

import torch
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression

from finding_gender_direction.hyperplane_computation import utils
from finding_gender_direction.hyperplane_computation.concept_erasure import leace

Gender = int #1 for male, -1 for female
Label = str #The label is either 'noun', 'pronoun', 'name', or 'anatomy'.

def process_labels(label : Label, gender : Gender):
  '''
  Prepares the label to be feed into the concept_erasure module.
  '''
  if label == 'noun':
    return [gender, 0, 0, 0]
  elif label == 'pronoun':
    return [0, gender, 0, 0]
  elif label == 'name':
    return [0, 0, 0, gender]
  else:
    return [0, 0, gender, 0]


def storing_hyperplanes(dataset : list[list[str, Label, Gender]], post_layer_norm = True, **dict):
  device = dict['device']
  model = dict['model']
  tokenizer = dict['tokenizer']


  #To save the RAM, we need to compute by batch
  indices = []
  activations = []
  labels = []
  N_batch = len(dataset)
  for batch in dataset:

    #We keep track the length of each prompt to find the right indices on which to measure the probability.
    indices.append(torch.Tensor([len(tokenizer(data[0])["input_ids"])-1 for data in batch]).to(int).to(device))

    #initiate activations
    tokenized_batch = tokenizer([data[0] for data in batch], padding = True, return_tensors = 'pt')["input_ids"].to(device)
    positions = torch.Tensor([i for i in range(tokenized_batch.shape[1])]).to(int).to(device)
    activations.append(model.transformer.wte(tokenized_batch) + model.transformer.wpe(positions))
    labels.append(torch.Tensor([process_labels(data[1], data[2]) for data in batch]).unsqueeze(-1))

  del tokenized_batch
  del positions

  dim_label = labels[0][0].shape[1]
  dim_residual = activations[0][0].shape[-1]

  eraser_mean = []
  eraser_quantile = []
  eraser_probe = []

  for layer in tqdm(range(len(model.transformer.h))):
    #Initiating the leace estimator, default parameters.
    leace_fitter = leace.LeaceFitter(dim_residual, dim_label)

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

    all_target_act = torch.cat(target_activations, dim = 0)
    all_labels = torch.cat(labels, dim = 0).squeeze()
    leace_fitter.update(all_target_act, all_labels)

    #We only keep the eraser. The rest is not useful anymore.
    eraser = leace_fitter.eraser
    del leace_fitter

    #We compute the quantile to have the equal-leace estimator.
    quantile = utils.get_quantile(eraser, all_target_act).unsqueeze(0)

    #ToDo: Update LEACE
    eraser_mean.append(eraser.to(device))
    eraser_quantile.append(leace.LeaceEraser(
        proj_right = eraser.proj_right,
        proj_left = eraser.proj_left,
        bias = quantile,
    ))
    eraser_quantile[-1].to(device)

    #We learn the best LogisticRegression, we need at least 1500 steps to converge
    if post_layer_norm:
      probe = LogisticRegression(random_state=0, max_iter=2000).fit(
              all_target_act.to('cpu'),
              all_labels.to('cpu')
              )

      eraser_probe.append(leace.LeaceEraser(
          proj_right = torch.Tensor(probe.coef_),
          proj_left = ((torch.Tensor(probe.coef_)/(torch.norm(torch.Tensor(probe.coef_), dim = -1)**2).unsqueeze(-1)).T),
          bias = -probe.intercept_[0]*(torch.Tensor(probe.coef_)/(torch.norm(torch.Tensor(probe.coef_), dim = -1).unsqueeze(-1)**2)),
      ))
      eraser_probe[-1].to(device)

  #Deletion of all the useless tensor to avoid RAM overload.
      del probe
  del eraser
  del target_activations
  del all_target_act
  del all_labels
  del activations
  del indices
  del labels
  del positions

  return eraser_mean, eraser_quantile, eraser_probe



def hyperplane_acc(examples : list[list[str]], eval_metric : list, **dict) -> list[list[float]]:
  device = dict['device']
  model = dict['model']
  tokenizer = dict['tokenizer']

  indices_list = []
  activations = []
  for batch in examples:

    #we keep track of the lenght of the prompt to find the right token to read the probability
    indices_list.append(torch.Tensor([len(tokenizer(data[0])["input_ids"])-1 for data in batch]).to(int))

    #initiate the activations
    tokenized_data = tokenizer([data[0] for data in batch], padding = True, return_tensors = 'pt')["input_ids"].to(device)
    positions = torch.arange(tokenized_data.shape[1]).to(int).to(device)
    activations.append(model.transformer.wte(tokenized_data) + model.transformer.wpe(positions))

  del tokenized_data
  del positions

  all_labels = torch.cat(
      [torch.Tensor([data[1] for data in batch]).unsqueeze(-1) for batch in examples]
      , dim = 0
  ).squeeze().to(device)

  acc_list = []
  for layer in tqdm(range(len(model.transformer.h))):

    target_activations = []
    for batch_num in range(len(activations)):

      #we propagade forward through the next layer
      if layer != 0:
        activations[batch_num] = model.transformer.h[layer](activations[batch_num])[0]

      #we store all of the activations
      target_activations.append(torch.cat([
        act[ind].unsqueeze(0) for act, ind in zip(model.transformer.h[layer].ln_1(activations[batch_num]), indices_list[batch_num])
        ], dim = 0).to(device))
    all_target_act = torch.cat(target_activations, dim = 0).to(device)

    #at this layer, we evaluate the accuracy of all of the metrics
    acc = []
    for metric in eval_metric:
      acc.append(metric(all_target_act, layer, all_labels))
    acc_list.append(acc)

  del activations
  del indices_list
  del all_target_act
  del all_labels

  #invert dim 0 and 1 to have [metric, layer]
  acc_list = [[acc[i] for acc in acc_list] for i in range(len(acc_list[0]))]
  return acc_list



#ToDo: doesn't work with newest version
def storing_directions(meta_dataset : list[list], post_layer_norm = True, **dict):
  device = dict['device']
  model = dict['model']
  tokenizer = dict['tokenizer']

  #To save the RAM, we need to compute by batch
  N_batch = len(meta_dataset[0])

  indices = []
  activations = []
  positions = []
  for dataset in meta_dataset:

    #First, we measure the length of each prompt to be able to find the right
    #indices on which to measure the probability.
    tokenized_data = [[tokenizer(data[0], return_tensors = 'pt')["input_ids"].to(device) for data in sub_dataset] for sub_dataset in dataset]
    indices.append([torch.Tensor([len(tokens[0])-1 for tokens in tokenized_sub_data]).to(int).to(device) for tokenized_sub_data in tokenized_data])

    #Now we tokenize for real, to compute the whole sub_dataset at the same time.
    tokenized_data = [tokenizer([word[0] for word in sub_dataset], padding = True, return_tensors = 'pt')["input_ids"].to(device) for sub_dataset in dataset]
    positions.append([torch.Tensor([i for i in range(tokenized_sub_data.shape[1])]).to(int).to(device) for tokenized_sub_data in tokenized_data])

    #We initialise our tensors for each dataset
    activations.append([model.transformer.wte(tokenized_sub_data) + model.transformer.wpe(position) for tokenized_sub_data, position in zip(tokenized_data, positions[-1])])

  del tokenized_data

  dim_label = 1
  dim_residual = activations[0][0][0].shape[-1]

  eraser_mean = []
  eraser_probe = []

  for layer in tqdm(range(len(model.transformer.h))):
    #Initiating the leace estimator, default parameters.
    leace_fitter = leace.LeaceFitter(dim_residual, dim_label)

    target_activations = []
    for b in range(len(activations)):

      super_target_activations = []
      for i in range(N_batch):

        #We update each activation through the next layer.
        if layer != 0:
          activations[b][i] = model.transformer.h[layer](activations[b][i],)[0] #**{'position_ids': positions[b][i]}

        #We choose the activation of the targeted tokens and fit the leace estimator.
        if post_layer_norm:
          super_target_activations.append(torch.cat([act[ind].unsqueeze(0) for act, ind in zip(model.transformer.h[layer].ln_1(activations[b][i]), indices[b][i])], dim = 0).to(device))
        else:
          target_activations.append(torch.cat([act[ind].unsqueeze(0) for act, ind in zip(activations[i], indices[i])], dim = 0).to(device))

      target_activations.append(super_target_activations)

    target_activations = [sub_target_act - sup_target_act for sub_target_act, sup_target_act in zip(target_activations[0], target_activations[1])]
    all_target_act = torch.cat(target_activations, dim = 0)
    all_labels = torch.zeros(all_target_act.shape[0]) + 1

    #We only keep the eraser. The rest is not useful anymore.
    leace_fitter.update(all_target_act, all_labels)
    leace_fitter.update(-all_target_act, -all_labels)
    eraser = leace_fitter.eraser

    eraser_mean.append(leace.LeaceEraser(
        proj_right = eraser.proj_right.to(device),
        proj_left = eraser.proj_left.to(device),
        bias = eraser.bias.to(device),
    ))

    #We learn the best probe, we need at least 1500 steps to converge
    probe = LogisticRegression(random_state=0, max_iter=2000).fit(
            torch.cat([all_target_act, -all_target_act], dim = 0).to('cpu'),
            torch.cat([all_labels, -all_labels], dim = 0).to('cpu')
            )

    eraser_probe.append(leace.LeaceEraser(
        proj_right = torch.Tensor(probe.coef_).to(device),
        proj_left = (eraser.proj_left/(torch.norm(eraser.proj_left, dim = 0).unsqueeze(0)*torch.norm(torch.Tensor(probe.coef_), dim = -1).unsqueeze(-1))).to(device),
        bias = -probe.intercept_[0]*(torch.Tensor(probe.coef_)/(torch.norm(torch.Tensor(probe.coef_), dim = -1).unsqueeze(-1)**2)).to(device),
    ))


    del leace_fitter
    del eraser
    del target_activations
    del all_target_act
    del all_labels
    del probe

  del activations
  del indices
  del positions

  return eraser_mean, eraser_probe



def direction_acc(meta_examples : list[list[str]], eval_metric: list, **dict):
  device = dict['device']
  model = dict['model']
  tokenizer = dict['tokenizer']

  indices_list = []
  activations = []
  positions = []
  for examples in meta_examples:

    tokenized_data = [tokenizer(data)["input_ids"] for data in examples]
    indices_list.append(torch.Tensor([len(tokens)-1 for tokens in tokenized_data]).to(int))

    tokenized_data = tokenizer(examples, padding = True, return_tensors = 'pt')["input_ids"].to(device)

    positions.append(torch.arange(tokenized_data.shape[1]).to(int).to(device))
    activations.append(model.transformer.wte(tokenized_data) + model.transformer.wpe(positions[-1]))

  del tokenized_data

  acc_list = []
  for layer in tqdm(range(len(model.transformer.h))):

    target_activations = []
    for b in range(len(activations)):

      if layer != 0:
        activations[b] = model.transformer.h[layer](activations[b],)[0]

      target_activations.append(torch.cat([act[ind].unsqueeze(0) for act, ind in zip(model.transformer.h[layer].ln_1(activations[b]), indices_list[b])], dim = 0).to(device))

    all_target_act = target_activations[0] - target_activations[1]

    acc = []
    for metric in eval_metric:
      acc.append(metric(all_target_act, layer, 1))
    acc_list.append(acc)

  del activations
  del indices_list
  del positions

  return [[acc[i] for acc in acc_list] for i in range(len(acc_list[0]))]