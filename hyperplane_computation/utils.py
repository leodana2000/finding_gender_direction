# code containing all the useful functions for the notebook to run

import random
import torch as t

from hyperplane_computation.concept_erasure.leace import LeaceEraser

Bin = 1 | 0 | -1
Label = list[Bin]
Data = str
Token = int


def select_rand(lst, nb_ex, seed):
    """Randomly selects nb_ex elements of the list."""
    assert nb_ex < len(lst)
    random.seed(seed)
    random.shuffle(lst)
    return lst[:nb_ex]


def initiate_activations(dataset: list[list[Data, Label]], **dict):
    """
  Initializes the activations to be fed in the layer of the network.
  Also computes the labels and indices to keep track of where to look for the right logits since we use padding.
  """
    device = dict["device"]
    model = dict["model"]
    tokenizer = dict["tokenizer"]

    indices = []
    activations = []
    labels = []

    for batch in dataset:
        indices.append(
            t.tensor(
                [len(tokenizer(data[0])["input_ids"]) - 1 for data in batch], 
                dtype=t.int, 
                device=device
            )
        )

        tokenized_batch = tokenizer(
            [data[0] for data in batch], padding=True, return_tensors="pt"
        )["input_ids"].to(device)
        positions = (
            t.arange(tokenized_batch.shape[1]).to(int).to(device).to(int).to(device)
        )
        activations.append(
            model.transformer.wte(tokenized_batch) + model.transformer.wpe(positions)
        )
        labels.append(t.tensor([data[1] for data in batch], dtype=t.float))

    del positions
    del tokenized_batch
    return indices, activations, labels


def gather_update_acts(
    activations: list[t.Tensor],
    layer: int,
    post_layer_norm: bool,
    indices: list[t.Tensor],
    N_batch: int,
    **dict
):
    """
    Updates all the activations by passing them throughout the layer.
    Then gathers the activations on which we will learn the hyperplane.
    """

    model = dict["model"]
    device = dict["device"]

    target_activations = []
    for batch_num in range(N_batch):

        # We update each activation through the next layer.
        if layer != 0:
            activations[batch_num] = model.transformer.h[layer](activations[batch_num])[
                0
            ]

        # We choose if we learn the hyperplane in the residual stream or after the layer-norm.
        if post_layer_norm:
            acts = model.transformer.h[layer].ln_1(activations[batch_num])
        else:
            acts = activations[batch_num]

        # We gather the activations that have the right indices, corresponding to the last token of the initial sentence.
        target_activations.append(
            t.cat(
                [act[ind].unsqueeze(0) for act, ind in zip(acts, indices[batch_num])],
                dim=0,
            ).to(device)
        )

    del acts
    return activations, target_activations


# ToDo: Implement a better version, using means of median
def get_quantile(leace_eraser: LeaceEraser, all_acts: t.Tensor, **dict):
    """
  Computes the median bias instead of the mean, and returns it.
  The formula is Quantile = delta-quantile(<hyperplane|all_acts>)*hyperplane ()when hyperplane is normed.
  """
    device = dict["device"]
    Nb_ex = len(all_acts)

    hyperplane = (
        (
            leace_eraser.proj_right[0]
            / t.norm(leace_eraser.proj_right[0], dim=-1).unsqueeze(-1)
        )
        .squeeze()
        .to(device)
    )
    sorted_tensor, _ = t.sort(
        t.einsum("nd, ...d -> n...", all_acts, hyperplane), dim=-1
    )
    quantile = (
        (sorted_tensor[Nb_ex // 2 + 1] + sorted_tensor[(Nb_ex - 1) // 2 + 1])
        * hyperplane
        / 2
    )

    del sorted_tensor
    return quantile


def probe_eval(erasers: list[LeaceEraser], **dict):
    """
  Initiate a function that will evaluate activations using different hyperplanes for each layers.
  """
    device = dict["device"]
    cosim = t.nn.CosineSimilarity(-1)

    def metric(all_acts: t.Tensor, layer: int, true_label: t.Tensor):
        """
    Evaluate on which side of the hyperplane the activations are, and return the accuracy of the hyperplane.
    """
        dir = erasers[layer].proj_right.to(device)
        bias = erasers[layer].bias.to(device)
        Nb_ex = len(all_acts)

        acc = t.sum(cosim(true_label @ dir, all_acts - bias) > 0).item() / Nb_ex
        return acc

    return metric


def show_proba(
    proba: t.Tensor,
    level: float = 0.01,
    nb_tokens: int = 10,
    decode: bool = False,
    **dict
):
    """
  From a probability, get the top-nb_tokens tokens that have a probability greater than level.
  """
    tokenizer = dict["tokenizer"]

    if decode:
        # prints words
        proba_token_list = [
            (proba[i].item(), tokenizer.decode([i]))
            for i in range(len(proba))
            if proba[i] > level
        ]
    else:
        # prints tokens
        proba_token_list = [
            (proba[i].item(), i) for i in range(len(proba)) if proba[i] > level
        ]

    proba_token_list.sort()
    return proba_token_list[-nb_tokens:]


def finds_indices(ex_batch: list[list[Token]], tar_batch: list[list[Token]]):
    """
  Finds the occurrences of the target inside the examples, but only the last one for each target.
  stream_indices : list[int], list of all streams where the target was detected.
  example_indices : list[int], list of all example where the target was detected.
  stream_example_indices : list[list[int], list[int]], list that give for each target
  the join example and stream where it was detected.
  """

    stream_indices = []
    example_indices = []
    stream_example_indices = [[], []]

    for i, (ex, tar) in enumerate(zip(ex_batch, tar_batch)):
        len_tar = len(tar)
        len_ex = len(ex)

        # If there is no target, we take the last stream.
        if len_tar == 0:
            s_indice = t.tensor([len_ex], dtype=t.int)
            e_indice = t.tensor([i], dtype=t.int)
            stream_example_indices[0].append(len_ex)
            stream_example_indices[1].append(i)

        # Otherwise, we take the targeted streams.
        else:
            # [-1] means that we take only the last occurrence in the sentence.
            # ToDo: more general version that could account for any number of occurrences.
            position = t.where(t.tensor(ex, dtype=t.int) == t.tensor([tar[0]], dtype=t.int))[0][
                -1
            ].item()
            if [ex[i] for i in range(position, position + len_tar)] == tar:
                s_indice = t.tensor(
                    [pos for pos in range(position, position + len_tar)],
                    dtype=t.int,
                )
                e_indice = t.tensor([i] * len_tar, dtype=t.int)
                stream_example_indices[0].append(position + len_tar - 1)
                stream_example_indices[1].append(i)
            else:
                print("Error, no target found.")

        stream_indices.append(s_indice)
        example_indices.append(e_indice)

    return [
        t.cat(stream_indices, dim=0, dtype=t.int),
        t.cat(example_indices, dim=0, dtype=t.int),
        t.tensor(stream_example_indices, dtype=t.int),
    ]
