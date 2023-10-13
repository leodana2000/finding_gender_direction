#code that contains the main functions performing the inference-time-modification
#you can use it directly in the notebook, or anywhere else : the only function to call is fast_score

import torch
from tqdm import tqdm
import utils


#Fast way to compute the probability for any nb_tokens but only when the tokens are of length one.
def fast_proba(token_list, len_example, proba):
  return torch.sum(proba[torch.arange(len(len_example)), len_example][:, token_list].squeeze(-1), dim = -1).unsqueeze(0)


#Fast way to intervene on the cache by using the custom attention.
def fast_cache_intervention(tokens, token_lists, leace_list, leace_res_list,
                            len_example, meta_hook, hook, layer_list,
                            layer_res_list, **dict):
    model = dict['model']
    device = dict['device']

    score = []
    for layers, layers_res in zip(layer_list, layer_res_list):

      for layer in layers:
        model.transformer.h[layer].attn.register_forward_hook(meta_hook(hook(leace_list[layer])))

      for layer in layers_res:
        model.transformer.h[layer].attn.register_forward_pre_hook(hook(leace_res_list[layer]))

      logits = model(tokens.to(device)).logits

      for layer in layers:
        model.transformer.h[layer].attn._forward_hooks.clear()

      for layer in layers_res:
        model.transformer.h[layer].attn._forward_pre_hooks.clear()

      probas = torch.softmax(logits, dim = -1)

      proba_male = fast_proba(token_lists[0], len_example, probas)
      proba_female = fast_proba(token_lists[1], len_example, probas)

      score.append(torch.cat([proba_male, proba_female], dim = 0).unsqueeze(0))

      del proba_male
      del proba_female
      del probas

    return score


#Initiate a fast way to compute the score, that doesn't involves looking at tokens of length > 1.
def fast_score(example_prompts : list[str], token_lists : list[list[int]], leace_list,
              leace_res_list, target_tokens : list[list[int]], layer_list : list[list[int]], 
              layer_res_list : list[int], lbds = [1], **dict):
  '''
  This function returns the probabilities of each answer for each example and each lambda.
  - example_prompts : list[str], examples to test,
  - token_lists : list[list[int]], token that will count as an answer, position 0 is male and 1 is female,
  - leace_list : list[leace], all leace estimators in attention (one per layer),
  - leace_res_list : list[leace], all leace estimators in residual (one per layer),
  - target_tokens : list[int] which token to target in the examples,
  - layer_list : list[list[int]] which layer to apply the intervention on attention, you can test multiple in a row,
  - layer_res_list : list[list[int]] which layer to apply the intervention in the residual, you can test multiple in a row,
  - lbds : list[float] parameters lambda to test,
  '''
  tokenizer = dict['tokenizer']
  device = dict['device']

  #We measure the length of each example to know where the answer is.
  example_tokens = [tokenizer(example_prompt).input_ids for example_prompt in example_prompts]
  len_examples = torch.Tensor([len(tokens)-1 for tokens in example_tokens]).to(int)

  #We create a list of all the position where to do the hook.
  #We only hook the last example of the target_tokens.
  stream_indices, example_indices, stream_example_indices = utils.finds_indices(example_tokens, target_tokens)

  #We tokenize all example together, with padding, to be faster.
  example_tokens = tokenizer(example_prompts, padding = True, return_tensors = 'pt').input_ids.to(device)


  score = []
  for lbd in tqdm(lbds):
    hook = hook_wte(lbd, stream_indices, example_indices)
    meta_hook = hook_attn(stream_indices, example_indices, stream_example_indices)
    score.append(fast_cache_intervention(example_tokens, token_lists, leace_list, leace_res_list, 
                             len_examples, meta_hook, hook, layer_list, layer_res_list, **dict))

  del example_tokens
  del stream_indices
  del example_indices
  del stream_example_indices
  del len_examples

  return score



#Custom attention module. It compute sequentially the attention pattern by first
#computing the attention until the first cut (which is the first min of
#stream_example_indices[0]), then it changes ex ante the key and value of the right
#example and stream so that the targeted gendered token will be changed. It then
#continues likewise until the next targeted stream until the list is empty.

#ToDo: make it more general!
def attn_forward(module,
                 hidden_states,
                 hook,
                 stream_indices,
                 example_indices,
                 stream_example_indices):

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



#write in the direction of the difference in mean
def hook_wte(lbd, stream_indices, example_indices):
  def meta_hook(leace_eraser):
    def hook(module, input):
      input[0][example_indices, stream_indices] = leace_eraser(input[0][example_indices, stream_indices], lbd)
      return input
    return hook
  return meta_hook



#Overwrite the usual attention mechanism with a custom one.
def hook_attn(stream_indices, example_indices, stream_example_indices):
  def meta_hook(hookk):
    def hook(module, input, output):
      return attn_forward(module, input[0], hookk, stream_indices, example_indices, stream_example_indices)
    return hook
  return meta_hook