import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset, IterableDataset


def run_sae(ae, activations_df, activation_col='activation', batch_size=1024, num_workers=8):

    sae_dataset = SAEDataset(activations_df, activation_col)
    sae_loader = DataLoader(
        sae_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers)

    reconstruction, hidden  = [], []
    for batch in sae_loader:

        reconstructed_activations, features = ae(batch.to('cuda'), output_features=True)
        reconstruction.append(reconstructed_activations.detach().cpu().numpy())
        hidden.append(features.detach().cpu().numpy())

    reconstruction = np.vstack(reconstruction)
    hidden = np.vstack(hidden)
    
    return reconstruction, hidden


class SAEDataset(Dataset):

    def __init__(self, activations, activation_col='activation'):

        if isinstance(activations, pd.DataFrame):
            activations = np.array(activations[activation_col].tolist())

        self.activations = activations

    def __len__(self):

        return len(self.activations)

    def __getitem__(self, idx):

        return self.activations[idx]


class SAEIterableDataset(IterableDataset):

    def __init__(self, activations, activation_col='activation'):

        if isinstance(activations, pd.DataFrame):
            activations = np.array(activations[activation_col].tolist())

        self.activations = activations

    def __iter__(self):

        return iter(self.generate())

    def generate(self):

        while True:
            idx = np.random.randint(low=0, high=len(self.activations))
            x = self.activations[idx]
            yield x
