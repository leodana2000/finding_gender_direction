# code that contains the functions to store and evaluate hyperplanes and directions
import torch
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression

from hyperplane_computation import utils
from hyperplane_computation.concept_erasure import leace
from hyperplane_computation.concept_erasure.leace import LeaceEraser

Bin = 1 | 0 | -1
Label = list[Bin]
Data = str


def storing_hyperplanes(dataset : list[list[Data, Label]], post_layer_norm=True, learn_probe=False, **dict) -> tuple[list[LeaceEraser]]:
  '''
  Computes hyperplanes for the difference in mean, diff-mean quantile and Logisticregression method.
  The dataset is batched, and its elements are composed of:
  - a sentence to learn, but only the last token is used (so if you target word is at the middle of 
    the sentence, you can discard the rest)
  - a list of integers, one of which is 1 or -1, which represents the concept (positive of negative)
    and the index is the class it belongs to.
  
  If you want a faster computation, use learn_probe=False as the LogisticRegression is quite long to run.
  '''
  
  device = dict['device']
  model = dict['model']

  indices, activations, labels = utils.initiate_activations(dataset, **dict)
  all_labels = torch.cat(labels, dim = 0).squeeze().unsqueeze(-1)

  N_batch = len(dataset)
  dim_label = all_labels.shape[1]
  dim_residual = activations[0].shape[-1]

  eraser_mean = []
  eraser_quantile = []
  eraser_probe = []

  for layer in tqdm(range(len(model.transformer.h))):
    leace_fitter = leace.LeaceFitter(dim_residual, dim_label) # default parameters

    activations, target_activations = utils.gather_update_acts(activations, layer, post_layer_norm, indices, N_batch, **dict)

    all_target_act = torch.cat(target_activations, dim = 0)
    leace_fitter.update(all_target_act, all_labels)

    # We only keep the eraser. The rest is not useful anymore and takes dim_residual**2 spaces.
    eraser = leace_fitter.eraser
    del leace_fitter

    eraser_mean.append(eraser.to(device))
    eraser_quantile.append(leace.LeaceEraser(
        proj_right = eraser.proj_right,
        proj_left = eraser.proj_left,
        bias = utils.get_quantile(eraser, all_target_act, **dict),
    ))
    eraser_quantile[-1].to(device)

    # Logistic Regression only works if we are after the layer-norm (otherwise the norm of the hyperplane explodes),
    # and if the data is one-dimensionnal (otherwise it reduces dimensionnality). 
    # ToDo: generalize to these cases.
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

      del probe
    del eraser
    del target_activations
    del all_target_act
  del all_labels
  del activations
  del indices
  del labels
  return eraser_mean, eraser_quantile, eraser_probe



def hyperplane_acc(dataset : list[list[Data, Label]], eval_metric : list, **dict) -> list[list[float]]:
  '''
  As storing_hyperplanes, takes a dataset of sentence to evaluate. 
  The evaluation is a metric that computes on which side of the hyperplane the activation of the last token is.
  It returns the accuracy of the metric (hyperplanes) at each layer.
  We only allow accuracy to be computed after the layer-norm, but one could overwrite this and use it in the residual.
  '''
  device = dict['device']
  model = dict['model']

  indices, activations, labels = utils.initiate_activations(dataset, **dict)
  all_labels = torch.cat(labels, dim=0).to(device)

  N_batch = len(dataset)
  post_layer_norm = True

  acc_list = []
  for layer in tqdm(range(len(model.transformer.h))):

    activations, target_activations = utils.gather_update_acts(activations, layer, post_layer_norm, indices, N_batch, **dict)
    all_target_acts = torch.cat(target_activations, dim = 0).to(device)

    acc = []
    for metric in eval_metric:
      acc.append(metric(all_target_acts, layer, all_labels))
    acc_list.append(acc)

  del activations
  del indices
  del labels
  del all_target_acts
  del all_labels
  return torch.Tensor(acc_list).T.to('cpu')