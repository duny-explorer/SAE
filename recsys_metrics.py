import os
import sys
os.environ['HYDRA_FULL_ERROR']='1'

import hydra
import mlflow
from tqdm import tqdm
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from nnsight import NNsight
from sklearn.metrics import explained_variance_score, mean_absolute_error, root_mean_squared_error
from torch.utils.data import DataLoader

from src.seqrec.datasets import CausalLMDataset, PaddingCollateFn, CausalLMPredictionDataset
from src.seqrec.modules import SeqRec
from src.activations import extract_activations, get_block_activations, get_last_layer_activations, get_mlp_activations
from src.seqrec.utils import preds2recs, get_last_item, remove_last_item
from src.sae import run_sae, SAEIterableDataset

from replay.metrics import OfflineMetrics, NDCG, HitRate, Novelty, Coverage, MRR, Precision
from jurity.recommenders import InterListDiversity

import warnings
warnings.filterwarnings("ignore")

MLFLOW_TRACKING_URI = 'file:/home/jovyan/mlruns'


@hydra.main(config_path="config", config_name="recsys_metrics", version_base="1.1")
def main(config):

    print(OmegaConf.to_yaml(config, resolve=True))
    OmegaConf.save(config=config, f='config.yaml')
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'

    if config.log_mlflow:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(config.mlflow_experiment)
        mlflow.start_run(run_name=config.run_name)
        mlflow.log_params(
            pd.json_normalize(OmegaConf.to_container(config), sep='.').to_dict(orient='records')[0])
        mlflow.log_artifact('config.yaml')

        if config.mlflow_tag:
            mlflow.set_tag('run_type', config.mlflow_tag)
            
    train_gpt, train, test, model, ae = load_data_and_models(config)
    activations_train, activations_test, std, mean = get_activations(model, train, test, config)
    sae_inference(ae, activations_test, config) 
    
    test_last = get_last_item(test)
    test_pred = remove_last_item(test)
    predict_dataset = CausalLMPredictionDataset(test_pred, max_length=config.max_length, time_col='timestamp')
    predict_loader = DataLoader(predict_dataset, batch_size=config.batch_size, shuffle=False, 
                                num_workers=config.num_workers, collate_fn=PaddingCollateFn())
    
    result_original = get_output_before(model, predict_loader, config)
    metrics(train_gpt, result_original, test_last, 'original', config)
    reconstructed = get_output_after(model, predict_loader, ae, config, std, mean)
    metrics(train_gpt, reconstructed, test_last, 'reconstructed', config)

    if config.save_statistic:
        features = pd.read_csv(config.features)
        features_columns = features.loc[:, config.start_feature:].columns
        result_original = pd.merge(result_original, features, how='left', on='item_id')
        reconstructed = pd.merge(reconstructed, features, how='left', on='item_id')
        d = {feature: 'mean' for feature in features_columns}
        original_table = result_original.groupby(by='user_id').agg(d).rename(columns={i: f'{i}_original' for i in features_columns}).mean(axis=0)
        if config.log_mlflow:
            mlflow.log_metrics(original_table.to_dict())

        reconstructed_table = reconstructed.groupby(by='user_id').agg(d).rename(columns={i: f'{i}_changed' for i in features_columns}).mean(axis=0)

        if config.log_mlflow:
            mlflow.log_metrics(reconstructed_table.to_dict())


    if config.log_mlflow:
        mlflow.end_run()
        
        
def load_data_and_models(config):
    train_gpt = pd.read_csv(os.path.join(config.data_path, 'train.csv'))
    train = pd.read_csv(os.path.join(config.data_path, 'train_sae.csv'))
    test = pd.read_csv(os.path.join(config.data_path, 'test.csv'))
    model = torch.load(config.model_path, weights_only=False)
    model.eval()
    model.to('cuda')
    ae = torch.load(config.ae_path, weights_only=False)
    ae.eval()
    ae.to('cuda')

    return train_gpt, train, test, model, ae


def get_activations(model, train, test, config):
    train_dataset = CausalLMDataset(train, max_length=config.max_length, time_col='timestamp')
    test_dataset = CausalLMDataset(test, max_length=config.max_length, time_col='timestamp')

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size,
        shuffle=False, num_workers=config.num_workers,
        collate_fn=PaddingCollateFn())
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size,
        shuffle=False, num_workers=config.num_workers,
        collate_fn=PaddingCollateFn())

    if config.sae.activations_layer == 'block':
        get_activations_fn = get_block_activations
    elif config.sae.activations_layer == 'mlp':
        get_activations_fn = get_mlp_activations
    elif config.sae.activations_layer == 'last':
        get_activations_fn = get_last_layer_activations

    print(f'Get activations from {config.sae.activations_layer}_{config.sae.layer_number}')
    if config.sae.activations_layer != 'last':
        activations_train = extract_activations(model, train_loader,
                                                get_activations_fn=get_activations_fn,
                                                layer_number=config.sae.layer_number, model_name=config.model_name)
        activations_test = extract_activations(model, test_loader,
                                               get_activations_fn=get_activations_fn,
                                               layer_number=config.sae.layer_number, model_name=config.model_name)
    else:
        activations_train = extract_activations(model, train_loader,
                                                get_activations_fn=get_activations_fn)
        activations_test = extract_activations(model, test_loader,
                                               get_activations_fn=get_activations_fn)

    activations_array = np.array(activations_train.activation.tolist())
    mean, std = activations_array.mean(), activations_array.std()
    activations_train['activation_normed'] = activations_train.activation.map(lambda x: (x - mean) / std)
    activations_test['activation_normed'] = activations_test.activation.map(lambda x: (x - mean) / std)

    return activations_train, activations_test, std, mean


def sae_inference(ae, activations_test, config):
    print('Get SAE activations and reconstructions on test data.')
    reconstruction, hidden = run_sae(ae, activations_test, activation_col=config.sae.activation_col,
                                     batch_size=config.batch_size, num_workers=config.num_workers)

    results = activations_test.copy()
    results['hidden'] = list(hidden)
    results['reconstruction'] = list(reconstruction)

    activation_values = np.array(activations_test[config.sae.activation_col].tolist())
    reconstruction_metrics = {
        'explained variance': explained_variance_score(activation_values, reconstruction),
        'mae': mean_absolute_error(activation_values, reconstruction),
        'rmse': root_mean_squared_error(activation_values, reconstruction)}
    print('reconstruction_metrics', reconstruction_metrics)

    if config.log_mlflow:
        mlflow.log_metrics(reconstruction_metrics)

    return results, hidden


def get_output_after(model, data, ae, config, std=None, mean=None):
    nsight_model = NNsight(model)
    result = pd.DataFrame({'user_id': [], 'item_id': [], 'prediction': []})

    for batch in tqdm(data): 
        rows_ids = torch.arange(batch['input_ids'].shape[0])
        matrix = torch.zeros((batch['input_ids'].shape[0], model.head.out_features))
        last_item_idx = []

        for i, indices in enumerate(batch['full_history']):
            indices = np.trim_zeros(indices.numpy(), trim='b')
            matrix[i, indices] = -float('Inf')
            last_item_idx.append(min(len(indices) - 1, batch['input_ids'].shape[1] - 1))

        with nsight_model.trace(batch['input_ids'], batch['attention_mask']) as tracer:
            if config.sae.activations_layer == 'block':
                layer_output = nsight_model.transformer_model.h[config.sae.layer_number].output[0]
            elif config.sae.activations_layer == 'mlp':
                layer_output = nsight_model.transformer_model.h[config.sae.layer_number].mlp.output
            elif config.sae.activations_layer == 'last':
                layer_output = nsight_model.transformer_model.output.last_hidden_state
                
            if config.all_items:
                activations = layer_output[:, :, :]
            else:
                activations = layer_output[rows_ids, last_item_idx, :]
                
            if config.sae.activation_col == 'activation_normed':
                activations = (activations - mean) / std
                
            latent_features = ae.encode(activations)

            if config.neuron_id and config.all_items:
                latent_features[:, :, config.neuron_id] = config.neuron_value
            elif config.neuron_id:
                latent_features[:, config.neuron_id] = config.neuron_value
                
            reconstructed_activations = ae.decode(latent_features)
            
            if config.sae.activation_col == 'activation_normed':
                reconstructed_activations = reconstructed_activations * std + mean
                
            if config.all_items:
                layer_output[:, :, :] = reconstructed_activations
            else:
                layer_output[rows_ids, last_item_idx, :] = reconstructed_activations

            model_outputs_after = nsight_model.head.output.save()
            
        outputs = model_outputs_after.detach().cpu()
        outputs = outputs[rows_ids, last_item_idx, :]
        
        if config.filter_seen:
            outputs = outputs + matrix

        prediction, item_id = outputs.sort(dim=1, descending=True) 
        prediction = prediction[:, :config.top_k]
        item_id = item_id[:, :config.top_k]

        table = pd.DataFrame({'user_id': np.ravel(np.tile(batch['user_id'], (config.top_k, 1)), order='F'),
                              'item_id': item_id.flatten(),
                              'prediction': prediction.flatten()})
        result = pd.concat([result, table])
        
    result['user_id'] = result['user_id'].astype(int)
    result['item_id'] = result['item_id'].astype(int)
        
    return result
    
    
def get_output_before(model, data, config):
    seqrec_module = SeqRec(model, predict_top_k=config.top_k)
    seqrec_module.filter_seen = config.filter_seen
    seqrec_module.eval()
    trainer = pl.Trainer(callbacks=[])
    
    preds = trainer.predict(model=seqrec_module, dataloaders=data)
    result = preds2recs(preds)
    
    return result


def metrics(train, data, ground_truth, type_event, config):
    all_metrics = [NDCG(config.top_k), HitRate(config.top_k),
                   Novelty(config.top_k), Coverage(config.top_k),
                   MRR(config.top_k), Precision(config.top_k)]
    metric = OfflineMetrics(all_metrics, query_column='user_id',
                            rating_column='prediction')(data, ground_truth, train)
    
    keys = list(metric.keys())
    result = dict()
    
    for key in keys:
        new_key = key.replace('@', '')
        result[f"{new_key}_{type_event}"] = metric.pop(f"{key}")
        
    result[f'Inter-List Diversity{config.top_k}_{type_event}'] = InterListDiversity(click_column='prediction',
                                                                                    k=10).get_score(ground_truth, data)

    if config.log_mlflow:
        mlflow.log_metrics(result)
        
        
if __name__ == '__main__':
    main()
   