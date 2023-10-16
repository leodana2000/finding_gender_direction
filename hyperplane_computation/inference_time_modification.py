#code that contains the main functions performing the inference-time-modification
#you can use it directly in the notebook, or anywhere else : the only function to call is fast_score

import torch
from tqdm import tqdm
from hyperplane_computation import utils
from hyperplane_computation.concept_erasure.leace import LeaceEraser


def diag_proba(logit_target, len_example, proba):
  '''
  Computes the probability of the batch by summing the probabilities of all valid next tokens.
  [diag, len_example] is the diagonal saying for each example, where to look for the probability.
  [:, logit_target] says to consider only the target tokens to compute the probability.
  '''
  diag = torch.arange(len(len_example))
  return torch.sum(proba[diag, len_example][:, logit_target].squeeze(-1), dim = -1).unsqueeze(0)


def batch_probabilities(example_tokens, len_example, logit_target, **dict):
  '''
  Computes the forward pass for each batch and puts them together.
  '''
  model = dict['model']

  male_batch = []
  female_batch = []
  for ex_batch, len_batch in zip(example_tokens, len_example):

    probas = torch.softmax(model(ex_batch).logits, dim = -1)

    male_batch += diag_proba(logit_target[0], len_batch, probas)
    female_batch += diag_proba(logit_target[1], len_batch, probas)

  del probas

  return male_batch, female_batch


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

    for layer in layers_res:
      model.transformer.h[layer].attn.register_forward_pre_hook(hook(leace_res_list[layer]))

    male_batch, female_batch = batch_probabilities(example_tokens, len_example, logit_target, **dict)

    for layer in layers:
      model.transformer.h[layer].attn._forward_hooks.clear()

    for layer in layers_res:
      model.transformer.h[layer].attn._forward_pre_hooks.clear()

    score.append(torch.cat([male_batch, female_batch], dim = 0).unsqueeze(0))

  del male_batch
  del female_batch

  return score


#Initiate a fast way to compute the score, that doesn't involves looking at tokens of length > 1.
def score(example_prompts : list[list[str]], logit_target : list[list[int]], leace_list : list[LeaceEraser],
          leace_res_list : list[LeaceEraser], modif_target : list[list[int]], layer_list : list[list[int]], 
          layer_res_list : list[int], lbds : torch.Tensor = torch.Tensor([1]), **dict):
  '''
  This function returns the probabilities of each answer for each example and each lambda.
  '''
  tokenizer = dict['tokenizer']
  device = dict['device']

  example_tokens = []
  len_examples = []
  indices = []
  for ex_batch, tar_batch in zip(example_prompts, modif_target):
    #We measure the length of each example to know where the answer is.
    tokens_batch = [tokenizer(example_prompt).input_ids for example_prompt in ex_batch]
    len_examples.append(torch.Tensor([len(tokens)-1 for tokens in tokens_batch]).to(int))

    #We create a list of all the position where to do the hook.
    #We only hook the last example of the modif_target.
    indices.append(utils.finds_indices(tokens_batch, tar_batch)) #stream, example, stream_example

    #We tokenize all example together, with padding, to be faster.
    example_tokens.append(tokenizer(ex_batch, padding = True, return_tensors = 'pt').input_ids.to(device))


  score = []
  for lbd in tqdm(lbds):
    hook = hook_wte(lbd, indices)
    meta_hook = hook_attn(indices)
    score.append(cache_intervention(example_tokens, logit_target, leace_list, leace_res_list, 
                             len_examples, meta_hook, hook, layer_list, layer_res_list, **dict))

  del example_tokens
  del len_examples
  del indices

  return score


#ToDo: make it more general!
def attn_forward(module,
                 hidden_states,
                 hook,
                 indices):
  
  '''
  This is a custom attention module designed to stop information from reaching certain future layers with minimal intervention.
  See X for a thorough explanation of the technique.
  We define a set of index on which we want to change the past keys and values for some example. 
  Thus we first computes all the modified keys a and values. At the right index, after the attention block at tthat index, 
  we change the keys and values.
  '''
  
  stream_indices = indices[0]
  example_indices = indices[1]
  stream_example_indices = indices[2]

  #We compute the usual query, key and values.
  query, key, value = module.c_attn(hidden_states).split(module.split_size, dim=2)

  query = module._split_heads(query, module.num_heads, module.head_dim)
  key = module._split_heads(key, module.num_heads, module.head_dim)
  value = module._split_heads(value, module.num_heads, module.head_dim)

  #We compute the modified key and values.
  _ , interv_key, interv_value = module.c_attn(hook(None, (hidden_states,))[0]).split(module.split_size, dim=2)

  interv_key = module._split_heads(interv_key, module.num_heads, module.head_dim)
  interv_value = module._split_heads(interv_value, module.num_heads, module.head_dim)

  last_stream = 0
  a = []

  for stream in range(hidden_states.shape[1]):
    if stream in stream_example_indices[0]:

      #list of all examples where to hook
      example_ind = stream_example_indices[1][torch.where(stream_example_indices[0] == stream)[0]]
      target_indices = torch.cat([torch.where(example_indices == ex)[0] for ex in example_ind], dim = 0)
      example_ind = example_indices[target_indices]
      stream_ind = stream_indices[target_indices]

      aux_query = query[:, :, :stream + 1]
      aux_key = key[:, :, :stream + 1]
      aux_value = value[:, :, :stream + 1]

      #We compute the attention until {stream}.
      attn_outputs, _ = module._attn(aux_query, aux_key, aux_value)
      a.append(attn_outputs[:, :, last_stream:])

      #We change afterward the cache on the targeted examples.
      key[example_ind, :, stream_ind] = interv_key[example_ind, :, stream_ind]
      value[example_ind, :, stream_ind] = interv_value[example_ind, :, stream_ind]
      last_stream = stream + 1

  attn_outputs, _ = module._attn(query, key, value)
  a.append(attn_outputs[:, :, last_stream:])

  a = torch.cat(a, dim = 2)
  a = module._merge_heads(a, module.num_heads, module.head_dim)
  a = module.c_proj(a)

  present = (key, value)

  outputs = (a, present,)
  return outputs


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