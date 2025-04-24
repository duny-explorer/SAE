import logging

import hydra
import mlflow
import pandas as pd
from omegaconf import OmegaConf
from replay.splitters import ColdUserRandomSplitter, NewUsersSplitter


MLFLOW_TRACKING_URI = 'file:/home/jovyan/mlruns'

log = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="data_split_music4all")
def main(config):

    log.info(OmegaConf.to_yaml(config, resolve=True))
    OmegaConf.save(config=config, f='config.yaml')

    if config.log_mlflow:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(config.mlflow_experiment)
        mlflow.start_run(run_name=config.run_name)
        mlflow.log_params(
            pd.json_normalize(OmegaConf.to_container(config), sep='.').to_dict(orient='records')[0])
        mlflow.log_artifact('config.yaml')

    df = pd.read_csv(config.data_path)
    df = df.rename(columns={'userId': 'user_id', 'movieId': 'item_id'})
    
    if config.run_name == 'ml-1m':
        df = df.rename(columns={'userId': 'user_id', 'movieId': 'item_id'})
        df.timestamp = pd.to_datetime(df.timestamp).astype(int) / 10**9
    elif config.run_name == 'ml-20m':
        df = df.rename(columns={'userId': 'user_id', 'movieId': 'item_id'})
        
    df = df.sort_values(['user_id', 'timestamp'])
    log.info(f'df shape {df.shape}')
    log.info(f'unique users {df.user_id.nunique()}')
    log.info(f'unique items {df.item_id.nunique()}')

    test_splitter = NewUsersSplitter(
        test_size=config.test_size, drop_cold_items=True, query_column="user_id")
    train, test = test_splitter.split(df)
    
    # need at least 2 items in a sequence for model training
    user_counts = train.user_id.value_counts()
    user_ids = user_counts[user_counts > 1].index
    train = train[train.user_id.isin(user_ids)]

    validation_splitter = ColdUserRandomSplitter(
        test_size=config.train_sae_size, drop_cold_items=True, query_column="user_id", seed=42)
    train, train_sae = validation_splitter.split(train)
    test = test[test.item_id.isin(train.item_id.unique())]

    log.info(f'train shape {train.shape}, unique users {train.user_id.nunique()}, unique items {train.item_id.nunique()}')
    log.info(f'train_sae shape {train_sae.shape}, unique users {train_sae.user_id.nunique()}, unique items {train_sae.item_id.nunique()}')
    log.info(f'test shape {test.shape}, unique users {test.user_id.nunique()}, unique items {test.item_id.nunique()}')
    
    log.info(f"train min max date {pd.to_datetime(train.timestamp.min(), unit='s').strftime('%Y-%m-%d %X')} {pd.to_datetime(train.timestamp.max(), unit='s').strftime('%Y-%m-%d %X')}")
    log.info(f"train_sae min max date {pd.to_datetime(train_sae.timestamp.min(), unit='s').strftime('%Y-%m-%d %X')} {pd.to_datetime(train_sae.timestamp.max(), unit='s').strftime('%Y-%m-%d %X')}")
    log.info(f"test min max date {pd.to_datetime(test.timestamp.min(), unit='s').strftime('%Y-%m-%d %X')} {pd.to_datetime(test.timestamp.max(), unit='s').strftime('%Y-%m-%d %X')}")

    train.to_csv('train.csv', index=False)
    train_sae.to_csv('train_sae.csv', index=False)
    test.to_csv('test.csv', index=False)

    if config.log_mlflow:
            mlflow.log_artifact('train.csv')
            mlflow.log_artifact('train_sae.csv')
            mlflow.log_artifact('test.csv')
            mlflow.log_artifact('data_split.log')


if __name__ == "__main__":

    main()
