import os

import hydra
import mlflow
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, ModelSummary, TQDMProgressBar
from torch.utils.data import DataLoader
from replay.metrics import Coverage, MRR, NDCG, OfflineMetrics, Recall

from src.seqrec.datasets import CausalLMDataset, CausalLMPredictionDataset, PaddingCollateFn, MaskedLMDataset, MaskedLMPredictionDataset 
from src.seqrec.models import GPT4Rec, BERT4Rec
from src.seqrec.modules import SeqRec
from src.seqrec.utils import get_last_item, remove_last_item, preds2recs


MLFLOW_TRACKING_URI = 'file:/home/jovyan/mlruns'


@hydra.main(config_path="config", config_name="gpt_music4all")
def main(config):

    print(OmegaConf.to_yaml(config, resolve=True))
    OmegaConf.save(config=config, f='config.yaml')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.cuda_visible_devices)

    if config.log_mlflow:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(config.mlflow_experiment)
        mlflow.start_run(run_name=config.run_name)
        mlflow.log_params(
            pd.json_normalize(OmegaConf.to_container(config), sep='.').to_dict(orient='records')[0])
        mlflow.log_artifact('config.yaml')
        mlflow.pytorch.autolog(log_models=False, checkpoint=False) 

    train, validation, test = load_data(config)
    train_loader, eval_loader = create_dataloaders(train, validation, config)

    max_item_id = max(train.item_id.max(), validation.item_id.max(), test.item_id.max())

    if config.model_name == 'GPT':
        model = GPT4Rec(config.gpt_config, vocab_size=max_item_id + 1)
    if config.model_name == 'BERT':
        model = BERT4Rec(max_item_id + 1, config.bert_config)

    trainer, seqrec_module, model = training(model, train_loader, eval_loader, config)

    test_inputs = remove_last_item(test)
    test_last_item = get_last_item(test)
    recs_test = predict(trainer, seqrec_module, test_inputs, config)
    evaluate(recs_test, test_last_item, train, config)

    if config.save_model:
        torch.save(seqrec_module.model, 'model.pt')
        if config.log_mlflow:
            mlflow.log_artifact('model.pt')

    if config.log_mlflow:
        mlflow.end_run()


def load_data(config):

    train = pd.read_csv(os.path.join(config.data_path, 'train.csv'))
    validation = pd.read_csv(os.path.join(config.data_path, 'train_sae.csv'))
    test = pd.read_csv(os.path.join(config.data_path, 'test.csv'))

    return train, validation, test


def create_dataloaders(train, validation, config):

    validation_size = config.dataloader.validation_size
    validation_users = validation.user_id.unique()
    if validation_size and (validation_size < len(validation_users)):
        validation_users = np.random.choice(validation_users, size=validation_size, replace=False)
        validation = validation[validation.user_id.isin(validation_users)]

    if config.model_name == 'GPT':
        train_dataset = CausalLMDataset(train,  **config['dataset_params'])
        eval_dataset = CausalLMPredictionDataset(
            validation, validation_mode=True, **config['dataset_params'])
    elif config.model_name == 'BERT':
        train_dataset = MaskedLMDataset(train, mlm_probability=0.2, 
                                        force_last_item_masking_prob=0, **config['dataset_params'])
        eval_dataset = MaskedLMPredictionDataset(validation, validation_mode=True, **config['dataset_params'])

    train_loader = DataLoader(train_dataset, batch_size=config.dataloader.batch_size,
                              shuffle=True, num_workers=config.dataloader.num_workers,
                              collate_fn=PaddingCollateFn())
    eval_loader = DataLoader(eval_dataset, batch_size=config.dataloader.test_batch_size,
                             shuffle=False, num_workers=config.dataloader.num_workers,
                             collate_fn=PaddingCollateFn())

    return train_loader, eval_loader


def training(model, train_loader, eval_loader, config):

    seqrec_module = SeqRec(model, **config['seqrec_module'])

    early_stopping = EarlyStopping(monitor="val_ndcg", mode="max",
                                   patience=config.patience, verbose=False)
    model_summary = ModelSummary(max_depth=4)
    checkpoint = ModelCheckpoint(save_top_k=1, monitor="val_ndcg",
                                 mode="max", save_weights_only=True)
    progress_bar = TQDMProgressBar(refresh_rate=100)
    callbacks=[early_stopping, model_summary, checkpoint, progress_bar]

    trainer = pl.Trainer(callbacks=callbacks, enable_checkpointing=True,
                         **config['trainer_params'])

    trainer.fit(model=seqrec_module,
                train_dataloaders=train_loader,
                val_dataloaders=eval_loader)

    seqrec_module.load_state_dict(torch.load(checkpoint.best_model_path)['state_dict'])

    return trainer, seqrec_module, model


def predict(trainer, seqrec_module, data, config):

    predict_dataset = CausalLMPredictionDataset(data, **config['dataset_params'])
    predict_loader = DataLoader(
        predict_dataset, shuffle=False,
        collate_fn=PaddingCollateFn(),
        batch_size=config.dataloader.test_batch_size,
        num_workers=config.dataloader.num_workers)

    preds = trainer.predict(model=seqrec_module, dataloaders=predict_loader)
    recs = preds2recs(preds)

    return recs


def evaluate(recs, test_last_item, train, config):

    top_k = config.seqrec_module.predict_top_k

    metrics_list = [Coverage(top_k), NDCG(top_k), MRR(top_k), Recall(top_k)]
    offline_metrics = OfflineMetrics(metrics_list, query_column='user_id',
                                     rating_column='prediction')
    metrics = offline_metrics(recs, test_last_item, train)
    print('metrics', metrics)

    if config.log_mlflow:
        metrics = {metric_name.replace('@', '_'): metric_value
                   for metric_name, metric_value in metrics.items()}
        mlflow.log_metrics(metrics)


if __name__ == "__main__":

    main()
