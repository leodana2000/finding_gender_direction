# Code containing functions used to compute the continuation of a sentence with the concept erasure method.

import torch as t
from tqdm import tqdm
from hyperplane_computation import utils
from hyperplane_computation.concept_erasure.leace import LeaceEraser
from hyperplane_computation.inference_time_modif import attn_forward

Bin = 1 | -1
Data = str
Token = int


def hook_spe(lbd, indices):
  '''
  The same hook as in inference_time_modif, but doesn't modify anything if the model is generating next tokens.
  '''
  def meta_hook(leace_eraser):
    def hook(module, input):
      if input[0].shape[1] != 1:
        input[0][indices[1], indices[0]] -= lbd*leace_eraser.proj_left.T*((input[0][indices[1], indices[0]]-leace_eraser.bias)@leace_eraser.proj_right.T)
      return input
    return hook
  return meta_hook


def hook_spe_attn(indices):
  '''
  The same hook as in inference_time_modif, but doesn't modify anything if the model is generating next tokens.
  '''
  def meta_hook(hookk):
    def hook(module, input, output):
      if input[0].shape[1] != 1:
        return attn_forward(module, input[0], hookk, indices)
      else:
        return output
    return hook
  return meta_hook


def continuate(examples : list[list[Data]], 
               targets : list[list[Data]],
               leace_list : list[LeaceEraser], 
               leace_res_list : list[LeaceEraser], 
               layer_list : list[list[int]], 
               layer_res_list : list[list[int]],
               lbds : list[float], 
               nb_seq : int = 1, 
               max_len : int = 20,
               temp : float = 0.01,
               seed : int = 42,
               **dict):
    '''
    This function returns the probabilities of each binary answers, for each examples and for each lambdas.
    examples: batched list containing the sentences to continue,
    target: batched list of words to intervene on, there can be multiple words on which to intervene,
    leace_list: list of hyperplane to use after the layer-norm,
    leace_res_lsit: list of hyperplanes to use in the residual stream,
    layer_list: list of layers at which to intervene on the attention intervention,
    layer_res_list: list of layers at which to intervene on the residual,
    lbds: list of steering parameters to test,
    nb_seq: number of alternative sequences to generate for each initial sentences,
    max_len: maximum number of extra token allowed for the continuation,
    temp: the temperature of the generation,
    '''

    assert len(layer_list) == len(layer_res_list), "You should have the same number of experiments in both lists."
    assert len(examples) == len(targets), "You should have as many sentences as you have targets."
  
    model = dict['model']
    tokenizer = dict['tokenizer']
    device = dict['device']
    t.manual_seed(seed)
  
    # Computation by batch.
    continuations = []
    for i, (ex_batch, tar_batch) in enumerate(zip(examples, targets)):
        print("Batch {}.".format(i))

        # Initializing activations.
        ex_tokens = [tokenizer(ex).input_ids for ex in ex_batch]
        tar_tokens = [tokenizer(tar).input_ids for tar in tar_batch]
        indices = utils.finds_indices(ex_tokens, tar_tokens)

        example_tokens = tokenizer(ex_batch, padding = True, return_tensors = 'pt').to(device)

        meta_hook = hook_spe_attn(indices)
        lbd_cont = []
        for lbd in tqdm(lbds):
            hook = hook_spe(lbd, indices)
    
            layer_cont  =[]
            for layers, layers_res in zip(layer_list, layer_res_list):

                # Install the hooks at each specified layers. 
                for layer, layer_res in zip(layers, layers_res):
                    model.transformer.h[layer].attn.register_forward_hook(meta_hook(hook(leace_list[layer])))
                    model.transformer.h[layer_res].attn.register_forward_pre_hook(hook(leace_res_list[layer_res]))

                # Generate the alternative sequence at these layers, and lambdas.
                repet_cont = model.generate(**example_tokens,
                                            max_length = max_len + example_tokens.input_ids.shape[-1],
                                            do_sample = True,
                                            num_return_sequences = nb_seq, 
                                            top_k = 25,
                                            temperature = temp)
            
                # Retracts all hooks.
                for layer, layer_res in zip(layers, layers_res):
                    model.transformer.h[layer].attn._forward_hooks.clear()
                    model.transformer.h[layer_res].attn._forward_pre_hooks.clear()

                layer_cont.append([tokenizer.decode(cont,  skip_special_tokens=True) for cont in repet_cont])
            lbd_cont.append(layer_cont)
        continuations.append(layer_cont)

    del ex_tokens
    del tar_tokens
    del indices
    del example_tokens

    return continuations # Shape (batch, lbds, experiments, alter-continuations)