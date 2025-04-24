import logging
import os
import sys

import hydra
import mlflow
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from sklearn.metrics import explained_variance_score, mean_absolute_error, root_mean_squared_error
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append('/home/jovyan/klenitskiy/repos/')
sys.path.append('/home/jovyan/klenitskiy/repos/seqrec-experiments')
from dictionary_learning.dictionary import AutoEncoder, GatedAutoEncoder, JumpReluAutoEncoder
from dictionary_learning.trainers import StandardTrainer, GatedSAETrainer, TrainerTopK
from dictionary_learning.trainers.top_k import AutoEncoderTopK
from dictionary_learning.training import trainSAE

from seqrec_experiments.lightning.datasets import CausalLMDataset, PaddingCollateFn
from src.activations import extract_activations, get_block_activations, get_last_layer_activations, get_mlp_activations
from src.analyze import compute_hidden_stats, features_vs_neuron_stats
from src.sae import run_sae, SAEIterableDataset


MLFLOW_TRACKING_URI = 'file:/home/jovyan/mlruns'

logging.getLogger('nnsight').setLevel(logging.WARNING)


@hydra.main(config_path="config", config_name="l1_sae_ml-1m")
def main(config):

    print(OmegaConf.to_yaml(config, resolve=True))
    OmegaConf.save(config=config, f='config.yaml')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.cuda_visible_devices)

    if config.log_mlflow:

        if config.run_name:
            run_name = config.run_name
        else:
            run_name = f'{config.sae_type}_{config.sae_params.dict_size}_{config.activations_layer}_{config.layer_number}'

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(config.mlflow_experiment)
        mlflow.start_run(run_name=run_name)
        mlflow.log_params(
            pd.json_normalize(OmegaConf.to_container(config), sep='.').to_dict(orient='records')[0])
        mlflow.log_artifact('config.yaml')
        if config.mlflow_tag:
            mlflow.set_tag('run_type', config.mlflow_tag)

    train, test, item_features, model = load_data_and_models(config)
    activations_train, activations_test = get_activations(model, train, test, config)
    ae, logs = sae_training(activations_train, config)
    results, hidden = sae_inference(ae, activations_test, config)                                    
    compute_neuron_stats(hidden, config)
    compute_feature_neuron_stats(activations_test, item_features, hidden, config)

    if config.save_sae:

        torch.save(ae, 'sae.pt')

        os.mkdir('top_activations')
        dead_neurons = save_top_examples(results, path='top_activations',
                                         num_examples=config.num_top_examples,
                                         min_activation=config.min_top_activation)

        if config.save_activations_sample:
            results = results.sample(config.save_activations_sample, replace=False)
        results.to_parquet('activations.parquet', index=False)

        if config.log_mlflow:
            mlflow.log_artifact('sae.pt')
            mlflow.log_artifact('top_activations')
            mlflow.log_artifact('activations.parquet')
            mlflow.set_tag('tag', 'saved_sae')

    if config.log_mlflow:
        mlflow.end_run()


def load_data_and_models(config):

    try:
        train = pd.read_csv(os.path.join(config.data_path, 'train_sae.csv'))
    except FileNotFoundError:
        train = pd.read_csv(os.path.join(config.data_path, 'validation.csv'))
    test = pd.read_csv(os.path.join(config.data_path, 'test.csv'))
    item_features = pd.read_csv(config.item_features_path)
    model = torch.load(config.model_path, weights_only=False)
    model.eval()
    model.to('cuda')

    return train, test, item_features, model


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

    if config.activations_layer == 'block':
        get_activations_fn = get_block_activations
    elif config.activations_layer == 'mlp':
        get_activations_fn = get_mlp_activations
    elif config.activations_layer == 'last':
        get_activations_fn = get_last_layer_activations

    print(f'Get activations from {config.activations_layer}_{config.layer_number}')

    activations_train = extract_activations(model, train_loader,
                                            get_activations_fn=get_activations_fn,
                                            layer_number=config.layer_number)
    activations_test = extract_activations(model, test_loader,
                                           get_activations_fn=get_activations_fn,
                                           layer_number=config.layer_number)

    activations_array = np.array(activations_train.activation.tolist())
    mean, std = activations_array.mean(), activations_array.std()
    activations_train['activation_normed'] = activations_train.activation.map(lambda x: (x - mean) / std)
    activations_test['activation_normed'] = activations_test.activation.map(lambda x: (x - mean) / std)

    return activations_train, activations_test


def sae_training(activations_train, config):

    sae_params = OmegaConf.to_container(config)['sae_params']
    sae_params['layer'] = 'layer'
    sae_params['lm_name'] = 'GPT'

    sae_train_dataset = SAEIterableDataset(activations_train, activation_col=config.activation_col)
    sae_train_loader = DataLoader(sae_train_dataset, batch_size=config.batch_size,
                                  num_workers=config.num_workers)

    if config.sae_type == 'vanilla':
        sae_params['trainer'] = StandardTrainer
        sae_params['dict_class'] = AutoEncoder
    elif config.sae_type == 'jumprelu':
        sae_params['trainer']= TrainerJumpRelu
        sae_params['dict_class'] = JumpReluAutoEncoder 
    elif config.sae_type == 'gated':
        sae_params['trainer'] = GatedSAETrainer
        sae_params['dict_class'] = GatedAutoEncoder
    elif config.sae_type == 'top_k':
        sae_params['trainer'] = TrainerTopK
        sae_params['dict_class'] = AutoEncoderTopK

    print(f'Training {config.sae_type} SAE. Hidden size: {sae_params["dict_size"]}')

    ae, logs = trainSAE(
            data=sae_train_loader,
            trainer_configs=[sae_params],
            steps=config.steps,
            log_steps=config.log_steps)

    logs = pd.DataFrame(logs)
    logs.reset_index().to_csv('train_logs.csv', index=False)
    if config.log_mlflow:
        mlflow.log_artifact('train_logs.csv')

    return ae, logs


def sae_inference(ae, activations_test, config):

    print('Get SAE activations and reconstructions on test data.')
    reconstruction, hidden = run_sae(ae, activations_test, activation_col=config.activation_col,
                                     batch_size=config.batch_size, num_workers=config.num_workers)

    results = activations_test.copy()
    results['hidden'] = list(hidden)
    results['reconstruction'] = list(reconstruction)

    activation_values = np.array(activations_test[config.activation_col].tolist())
    reconstruction_metrics = {
        'explained variance': explained_variance_score(activation_values, reconstruction),
        'mae': mean_absolute_error(activation_values, reconstruction),
        'rmse': root_mean_squared_error(activation_values, reconstruction)}
    print('reconstruction_metrics', reconstruction_metrics)

    if config.log_mlflow:
        mlflow.log_metrics(reconstruction_metrics)

    return results, hidden


def compute_neuron_stats(hidden, config):

    num_active_neurons, neuron_fired_fraction = compute_hidden_stats(
        hidden, thresholds=config.thresholds)
    num_dead_neurons = (neuron_fired_fraction[0] <= config.dead_threshold).sum()

    num_active_neurons.describe().reset_index().to_csv('statistics_active_neurons.csv', index=False)
    neuron_fired_fraction.describe().reset_index().to_csv('statistics_fired_fraction.csv', index=False)

    if config.log_mlflow:
        mlflow.log_metrics({'num_dead_neurons': num_dead_neurons,
                            'num_active_neurons': num_active_neurons[0].mean(),
                            'neuron_fired_fraction': neuron_fired_fraction[0].mean()})
        mlflow.log_artifact('statistics_active_neurons.csv')
        mlflow.log_artifact('statistics_fired_fraction.csv')


def compute_feature_neuron_stats(activations_test, item_features, hidden, config):

    features = pd.merge(activations_test, item_features, how='left')
    feature_cols = features.loc[:, config.start_feature:].columns
    features = features[feature_cols]
    mean_neuron_activation, fraction_neuron_fired, neuron_feature_corr, neuron_feature_rocauc, \
        neuron_feature_precision = features_vs_neuron_stats(
            hidden, features, sample_size=config.roc_auc_sample_size, n_jobs=config.num_workers)

    if config.log_detailed_tables:
        mean_neuron_activation.reset_index().to_csv('mean_neuron_activation.csv', index=False)
        fraction_neuron_fired.reset_index().to_csv('fraction_neuron_fired.csv', index=False)
        neuron_feature_corr.reset_index().to_csv('neuron_feature_corr.csv', index=False)
        neuron_feature_rocauc.reset_index().to_csv('neuron_feature_rocauc.csv', index=False)
        neuron_feature_precision.reset_index().to_csv('neuron_feature_precision.csv', index=False)
        if config.log_mlflow:
            mlflow.log_artifact('mean_neuron_activation.csv')
            mlflow.log_artifact('fraction_neuron_fired.csv')
            mlflow.log_artifact('neuron_feature_corr.csv')
            mlflow.log_artifact('neuron_feature_rocauc.csv')
            mlflow.log_artifact('neuron_feature_precision.csv')

    roc_auc_max = neuron_feature_rocauc.max(axis=0).round(3).rename('max roc auc')
    correlation_max = neuron_feature_corr.max(axis=0).round(3).rename('max correlation')
    fraction_max = fraction_neuron_fired.max(axis=0).round(3).rename('max fraction')
    mean_activation_max = mean_neuron_activation.max(axis=0).round(3).rename('max mean activation')
    precision_max = neuron_feature_precision.max(axis=0).round(3).rename('max precision')
    max_by_feature = roc_auc_max.to_frame().join(correlation_max).join(fraction_max).join(
        mean_activation_max).join(precision_max)
    max_by_feature.reset_index().to_csv('max_by_feature.csv', index=False)
    print('max_by_feature', max_by_feature.mean())
    if config.log_mlflow:
        mlflow.log_metrics(max_by_feature.mean())
        mlflow.log_artifact('max_by_feature.csv')


def save_top_examples(results, path, num_examples=50, min_activation=0.1):

    hidden = np.array(results.hidden.tolist())
    dead_neurons = []

    for neuron_id in tqdm(range(hidden.shape[1]), desc='Saving top examples'):

        neuron = results[['user_id', 'item_id']].copy()
        neuron['activation'] = hidden[:, neuron_id]

        active = neuron[neuron.activation > min_activation]
        if len(active) > 0:
            user_ids = active.groupby('user_id').activation.max().sort_values(
                ascending=False).head(num_examples).index
            top_examples = neuron[neuron.user_id.isin(user_ids)]
            top_examples.to_csv(
                f'{path}/neuron_{neuron_id}.csv',
                index=False)
        else:
            dead_neurons.append(neuron_id)

    return dead_neurons


if __name__ == "__main__":

    main()
