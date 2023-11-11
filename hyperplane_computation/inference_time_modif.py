# Code that contains all the functions performing the inference-time-modification.

import torch as t 
from tqdm import tqdm #type: ignore
from hyperplane_computation import utils
from hyperplane_computation.concept_erasure.leace import LeaceEraser
from typing import Literal, Tuple, List

Bin = Literal[1, 0, -1]
Data = str
Token = int


def score(
    examples : List[Tuple[List[Data], List[Bin], List[Data]]],
    logit_target : Tuple[List[Token], List[Token]], 
    leace_list : List[LeaceEraser], 
    leace_res_list : List[LeaceEraser], 
    layer_list : List[List[int]], 
    layer_res_list : List[int], 
    lbds : t.Tensor, 
    **dict
 ):
  '''
  This function returns the probabilities of each binary answers, for each examples and for each lambdas.
  examples: batched list containing the questions, their binary label, and the target string of where to modify the attention,
  logit_target: list of logits ids that are considered valid answers for each label,
  leace_list: list of hyperplane to use after the layer-norm,
  leace_res_lsit: list of hyperplanes to use in the residual stream,
  layer_list: list of layers at which to intervene on the attention intervention,
  layer_res_list: list of layers at which to intervene on the residual,
  lbds: list of steering parameters to test,
  '''
  tokenizer = dict['tokenizer']
  device = dict['device']

  assert len(layer_list) == len(layer_res_list), "You should have the same number of experiments in both lists."

  score = []
  for i, ex_bin_tar in enumerate(examples):
    print("Batch {}.".format(i))

    # We measure the length of each example to know where the answer is.
    ex_tokens = [tokenizer(ex).input_ids for ex in ex_bin_tar[0]]
    tar_tokens = [tokenizer(tar).input_ids for tar in ex_bin_tar[2]]

    # We create a list of all the position where to do the hook.
    # We only hook the last example of ex_bin_tar[2].
    indices = utils.finds_indices(ex_tokens, tar_tokens)
    len_examples = t.tensor([len(tokens)-1 for tokens in ex_tokens], dtype=t.int)

    example_tokens = tokenizer(ex_bin_tar[0], padding = True, return_tensors = 'pt').input_ids.to(device)

    # We compute the score for this batch
    lbds_score = []
    meta_hook = hook_attn(indices)
    for lbd in tqdm(lbds):
      hook = hook_wte(lbd, indices)
      lbds_score.append(cache_intervention(example_tokens, logit_target, leace_list, leace_res_list, 
                                           len_examples, meta_hook, hook, layer_list, layer_res_list, **dict))
    
    score.append(t.cat(lbds_score, dim=0))

  del example_tokens
  del len_examples
  del indices
  del ex_tokens
  del tar_tokens

  # score has shape [batch, lbds, experiments, binary, questions]
  return compute_proba_acc(score, examples, **dict)


def hook_wte(lbd, indices):
  '''
  A hook designed to apply concept erasure to the input of the attention module, or a block.
  '''
  def meta_hook(leace_eraser):
    def hook(module, input):
      input[0][indices[1], indices[0]] = leace_eraser(input[0][indices[1], indices[0]], lbd)
      return input
    return hook
  return meta_hook


def hook_attn(indices):
  '''
  A hook designed to change the attention's computation for our custom attention.
  '''
  def meta_hook(hookk):
    def hook(module, input, output):
      return attn_forward(module, input[0], hookk, indices)
    return hook
  return meta_hook


# ToDo: make it more general!
def attn_forward(module,
                 hidden_states,
                 hook,
                 indices):
  
  '''
  This is a custom attention module designed to stop information from reaching certain future layers with minimal intervention.
  See X for a thorough explanation of the technique.
  We define a set of index on which we want to change the past keys and values for some example. 
  Thus we first computes all the modified keys a and values. At the right index, after the attention block at that index, 
  we change the keys and values.
  '''
  
  stream_indices = indices[0]
  example_indices = indices[1]
  stream_example_indices = indices[2]

  # We compute the usual query, key and values.
  query, key, value = module.c_attn(hidden_states).split(module.split_size, dim=2)

  query = module._split_heads(query, module.num_heads, module.head_dim)
  key = module._split_heads(key, module.num_heads, module.head_dim)
  value = module._split_heads(value, module.num_heads, module.head_dim)

  # We compute the modified key and values.
  _ , interv_key, interv_value = module.c_attn(hook(None, (hidden_states,))[0]).split(module.split_size, dim=2)

  interv_key = module._split_heads(interv_key, module.num_heads, module.head_dim)
  interv_value = module._split_heads(interv_value, module.num_heads, module.head_dim)

  last_stream = 0
  a = []

  for stream in range(hidden_states.shape[1]):
    if stream in stream_example_indices[0]:

      # list of all examples where to hook
      example_ind = stream_example_indices[1][t.where(stream_example_indices[0] == stream)[0]]
      target_indices = t.cat([t.where(example_indices == ex)[0] for ex in example_ind], dim = 0)
      example_ind = example_indices[target_indices]
      stream_ind = stream_indices[target_indices]

      aux_query = query[:, :, :stream + 1]
      aux_key = key[:, :, :stream + 1]
      aux_value = value[:, :, :stream + 1]

      # We compute the attention until {stream}.
      attn_outputs, _ = module._attn(aux_query, aux_key, aux_value)
      a.append(attn_outputs[:, :, last_stream:])

      # We change afterward the cache on the targeted examples.
      key[example_ind, :, stream_ind] = interv_key[example_ind, :, stream_ind]
      value[example_ind, :, stream_ind] = interv_value[example_ind, :, stream_ind]
      last_stream = stream + 1

  attn_outputs, _ = module._attn(query, key, value)
  a.append(attn_outputs[:, :, last_stream:])

  a = t.cat(a, dim = 2)
  a = module._merge_heads(a, module.num_heads, module.head_dim)
  a = module.c_proj(a)

  present = (key, value)

  outputs = (a, present,)
  return outputs


def cache_intervention(example_tokens, logit_target, leace_list, leace_res_list,
                       len_example, meta_hook, hook, layer_list,
                       layer_res_list, **dict):
  '''
  Applies all the hooks to the model before running the forward pass, and then clear them.
  We run len(layer_list) experiments. In each we apply the modified attention hook, and the normal modification hook 
  in the residual stream.
  '''
  
  model = dict['model']

  score = []
  for layers, layers_res in zip(layer_list, layer_res_list):
    for layer in layers:
      model.transformer.h[layer].attn.register_forward_hook(meta_hook(hook(leace_list[layer])))
    for layer_res in layers_res:
      model.transformer.h[layer_res].attn.register_forward_pre_hook(hook(leace_res_list[layer_res]))

    probas = t.softmax(model(example_tokens).logits, dim = -1)

    male_proba = diag_proba(logit_target[0], len_example, probas)
    female_proba = diag_proba(logit_target[1], len_example, probas)

    for layer in layers:
      model.transformer.h[layer].attn._forward_hooks.clear()
    for layer_res in layers_res:
      model.transformer.h[layer_res].attn._forward_pre_hooks.clear()

    score.append(t.cat([male_proba, female_proba], dim = 0).unsqueeze(0))
  score = t.cat(score, dim=0).unsqueeze(0)

  del probas
  del male_proba
  del female_proba
  return score


def diag_proba(logit_target, len_example, proba):
  '''
  Computes the probability of the batch by summing the probabilities of all valid next tokens.
  [diag, len_example] is the diagonal saying for each example, where to look for the probability.
  [:, logit_target] says to consider only the target tokens to compute the probability.
  '''
  diag = t.arange(len(len_example))
  return t.sum(proba[diag, len_example][:, logit_target].squeeze(-1), dim = -1).unsqueeze(0)


def compute_proba_acc(score : List[t.Tensor], examples : List[Tuple[List[Data], List[Bin], List[Data]]], **dict):
  '''
  Computes the probability and accuracy for each lbds, using bin to indicate which gender was the right one.
  '''
  device = dict['device']

  # score has shape [batch, lbds, experiments, binary, questions]
  bin = t.cat([t.tensor(ex_bin_tar[1], dtype=t.int) for ex_bin_tar in examples], dim=0).to(device)
  t_score = t.cat([t.transpose(t_batch, 0, 3) for t_batch in score], dim=0)
  t_score = t.transpose(t.transpose(t_score, 0, 3), 0, 2)

  acc = t.mean(((t_score[0] - t_score[1])*bin > 0).to(t.int), dim=-1, dtype=t.float)
  pos_ex = (bin > 0).to(t.int)
  neg_ex = 1 - pos_ex
  proba = t.cat([(t.mean(t_score[0]*pos_ex, dim=-1) + t.mean(t_score[1]*neg_ex, dim=-1)).unsqueeze(-1), 
                     (t.mean(t_score[0]*neg_ex, dim=-1) + t.mean(t_score[1]*pos_ex, dim=-1)).unsqueeze(-1)], dim=-1)
  
  return t.transpose(proba, 1, 2), acc
