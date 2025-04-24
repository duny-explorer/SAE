import numpy as np
import pandas as pd
import torch
from nnsight import NNsight
from tqdm import tqdm


def extract_activations(model, dataloader, get_activations_fn, layer_number=0, model_name='GPT'):

    model.eval()
    nsight_model = NNsight(model)
    users, items, activations = [], [], []

    for batch in tqdm(dataloader):

        with nsight_model.trace(batch['input_ids'], batch['attention_mask']) as tracer:
            try:
                batch_activations = get_activations_fn(nsight_model, layer_number=layer_number, model_name=model_name)
            except TypeError:
                batch_activations = get_activations_fn(nsight_model)

        batch_activations = batch_activations.view(-1, batch_activations.shape[2])
        batch_users = torch.repeat_interleave(
            batch['user_id'][:, None], repeats=batch['input_ids'].shape[1], dim=1)
        batch_users = batch_users.flatten()
        batch_items = batch['input_ids'].flatten()

        batch_activations = batch_activations[batch_items != 0]
        batch_users = batch_users[batch_items != 0]
        batch_items = batch_items[batch_items != 0]

        activations.append(batch_activations.detach().cpu().numpy())
        users.append(batch_users.detach().cpu().numpy())
        items.append(batch_items.detach().cpu().numpy())

    return pd.DataFrame({'user_id': np.hstack(users),
                         'item_id': np.hstack(items),
                         'activation': list(np.vstack(activations))
                        })


def get_last_layer_activations(nsight_model):

    transformer_outputs = nsight_model.transformer_model.output
    activations = transformer_outputs.last_hidden_state.save()

    return activations


def get_mlp_activations(nsight_model, layer_number=0):

    activations = nsight_model.transformer_model.h[layer_number].mlp.output.save()
    # nsight_model.transformer_model.h[layer_number].mlp.stop()  # possibly work in new version only

    return activations


def get_block_activations(nsight_model, layer_number=0, model_name='GPT'):

    if model == 'GPT':
        activations = nsight_model.transformer_model.h[layer_number].output[0].save()
    elif model == 'Bert':
        activations = nsight_model.transformer_model.encoder.layer[layer_number].output[0].save()
        
    # nsight_model.transformer_model.h[layer_number].output[0].stop()  # possibly work in new version only

    return activations
