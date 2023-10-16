#code that contains the functions to store and evaluate hyperplanes and directions
#You can use them in the notebook

import torch
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression

from hyperplane_computation import utils
from hyperplane_computation.concept_erasure import leace
from hyperplane_computation.concept_erasure.leace import LeaceEraser


def storing_hyperplanes(dataset : list[list[str, list[int]]], post_layer_norm=True, learn_probe=False, **dict) -> tuple[list[LeaceEraser]]:
  device = dict['device']
  model = dict['model']

  N_batch = len(dataset)

  indices, activations, labels = utils.initiate_activations(dataset, **dict)
  all_labels = torch.cat(labels, dim = 0).squeeze().unsqueeze(-1)

  dim_label = all_labels.shape[1]
  dim_residual = activations[0].shape[-1]

  eraser_mean = []
  eraser_quantile = []
  eraser_probe = []

  for layer in tqdm(range(len(model.transformer.h))):
    #Initiating the leace estimator, default parameters.
    leace_fitter = leace.LeaceFitter(dim_residual, dim_label)

    activations, target_activations = utils.gather_update_acts(activations, layer, post_layer_norm, indices, N_batch, **dict)

    all_target_act = torch.cat(target_activations, dim = 0)
    leace_fitter.update(all_target_act, all_labels)

    #We only keep the eraser. The rest is not useful anymore.
    eraser = leace_fitter.eraser
    del leace_fitter

    #We compute the quantile to have the equal-leace estimator.
    quantile = utils.get_quantile(eraser, all_target_act, **dict).unsqueeze(0)

    #ToDo: Update LEACE
    eraser_mean.append(eraser.to(device))
    eraser_quantile.append(leace.LeaceEraser(
        proj_right = eraser.proj_right,
        proj_left = eraser.proj_left,
        bias = quantile,
    ))
    del eraser
    eraser_quantile[-1].to(device)

    #Only recognize the gender direction
    if post_layer_norm and learn_probe:
      probe_labels = torch.sum(all_labels, dim=-1)

      probe = LogisticRegression(random_state=0, max_iter=2000,).fit(
              all_target_act.to('cpu'),
              probe_labels.to('cpu')
              )

      eraser_probe.append(leace.LeaceEraser(
          proj_right = torch.Tensor(probe.coef_),
          proj_left = ((torch.Tensor(probe.coef_)/(torch.norm(torch.Tensor(probe.coef_), dim = -1)**2).unsqueeze(-1)).T),
          bias = -probe.intercept_[0]*(torch.Tensor(probe.coef_)/(torch.norm(torch.Tensor(probe.coef_), dim = -1).unsqueeze(-1)**2)),
      ))
      eraser_probe[-1].to(device)

  #Deletion of all the useless tensor to avoid RAM overload.
      del probe
    del target_activations
    del all_target_act
  del all_labels
  del activations
  del indices
  del labels

  return eraser_mean, eraser_quantile, eraser_probe



def hyperplane_acc(dataset : list[list[str, list[int]]], eval_metric : list, **dict) -> list[list[float]]:
  device = dict['device']
  model = dict['model']

  N_batch = len(dataset)

  indices, activations, labels = utils.initiate_activations(dataset, **dict)

  all_labels = torch.cat(labels, dim=0).to(device)

  acc_list = []
  for layer in tqdm(range(len(model.transformer.h))):

    activations, target_activations = utils.gather_update_acts(activations, layer, True, indices, N_batch, **dict)

    all_target_acts = torch.cat(target_activations, dim = 0).to(device)

    #at this layer, we evaluate the accuracy of all of the metrics
    acc = []
    for metric in eval_metric:
      acc.append(metric(all_target_acts, layer, all_labels))
    acc_list.append(acc)

  del activations
  del indices
  del labels
  del all_target_acts
  del all_labels

  #invert dim 0 and 1 to have [metric, layer]
  acc_list = [[acc[i] for acc in acc_list] for i in range(len(acc_list[0]))]
  return acc_list