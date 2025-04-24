import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.feature_selection import chi2
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


def compute_hidden_stats(hidden, thresholds=[0, 0.1, 0.5, 1, 2, 3]):

    num_dead_neurons = (hidden.sum(axis=0) == 0).sum()

    num_active_neurons = {}
    for threshold in thresholds:
        num_active_neurons[threshold] = (hidden > threshold).sum(axis=1)
    num_active_neurons = pd.DataFrame(num_active_neurons)

    neuron_fired_fraction = {}
    for threshold in thresholds:
        neuron_fired_fraction[threshold] = (hidden > threshold).mean(axis=0)
    neuron_fired_fraction = pd.DataFrame(neuron_fired_fraction)

    return num_active_neurons, neuron_fired_fraction


def analyze_feature(hidden, features, feature_name):

    feature = features[feature_name]

    print('Analyze feature:', feature_name)
    print('=' * 70)
    print('Number of examples with feature', feature.sum())
    print('Fraction of examples with feature', feature.mean())

    print('=' * 70)
    print('Top neurons according to chi2 statistics')
    chi2_stats, p_values = chi2(hidden, feature)
    chi2_stats = pd.Series(chi2_stats, name='chi2').sort_values(ascending=False)
    print(chi2_stats.head(10))

    print('=' * 70)
    print('Stats for top neuron')
    neuron_id = chi2_stats.index[0]
    print('Neuron id', neuron_id)

    neuron_activations = hidden[:, neuron_id]
    print('When feature is on:')
    print('Average neuron activation', neuron_activations[feature == 1].mean())
    print('Number of examples with fired neuron', (neuron_activations[feature == 1] > 0).sum())
    print('Fraction of examples with fired neuron',
          (neuron_activations[feature == 1] > 0).sum() / feature.sum())
    print('When feature is off:')
    print('Average neuron activation', neuron_activations[feature == 0].mean())

    print('=' * 70)
    print('Average top neuron activation for other features')
    all_mean_activations = {}
    for feature_col in features.columns:
        all_mean_activations[feature_col] = neuron_activations[features[feature_col] == 1].mean()
    all_mean_activations = pd.Series(all_mean_activations).sort_values(ascending=False)
    print(all_mean_activations.head(10))

    return chi2_stats


def features_vs_neuron_stats(hidden, features, threshold=0, corr_type='pearson',
                             sample_size=20000, n_jobs=8, random_state=42):

    mean_neuron_activation = compute_mean_activation(features, hidden)
    fraction_neuron_fired = compute_fraction_fired(features, hidden, threshold)
    neuron_feature_corr = compute_correlation(features, hidden, corr_type)
    neuron_feature_rocauc = compute_rocauc(features, hidden, sample_size=sample_size,
                                           n_jobs=n_jobs, random_state=random_state)
    neuron_feature_precision = compute_precision(features, hidden, threshold)

    return (mean_neuron_activation, fraction_neuron_fired,
            neuron_feature_corr, neuron_feature_rocauc, neuron_feature_precision)


def compute_mean_activation(features, hidden):
    
    mean_neuron_activation = []
    for feature_col in tqdm(features.columns, desc='mean activation computation'):
        mean_neuron_activation.append(
            pd.Series(hidden[features[feature_col] == 1].mean(axis=0), name=feature_col))

    mean_neuron_activation = pd.DataFrame(mean_neuron_activation).T
    
    return mean_neuron_activation


def compute_fraction_fired(features, hidden, threshold=0):

    fraction_neuron_fired = []
    for feature_col in tqdm(features.columns, desc='fraction computation'):
        fraction_neuron_fired.append(pd.Series(
            (hidden[features[feature_col] == 1] > threshold).sum(axis=0) / features[feature_col].sum(),
            name=feature_col))

    fraction_neuron_fired = pd.DataFrame(fraction_neuron_fired).T
    
    return fraction_neuron_fired


def compute_precision(features, hidden, threshold=0):

    precision = []
    for neuron_id in tqdm(range(hidden.shape[1]), desc='precision computation'):
        precision.append(
            features[hidden[:, neuron_id] > threshold].sum() / (hidden[:, neuron_id] > threshold).sum())

    precision = pd.DataFrame(precision).fillna(0)
    
    return precision


def compute_correlation(features, hidden, corr_type='pearson', sample_size=200000, random_state=42):
    
    # if corr_type == 'pearson':
    #     corr = np.corrcoef(hidden.T, features.T)
    # elif corr_type == 'spearman':
    #     corr = spearmanr(hidden, features, axis=0).correlation

    # corr = corr[:hidden.shape[1], hidden.shape[1]:]
    # neuron_feature_corr = pd.DataFrame(corr, columns=features.columns)

    features_sample, hidden_sample = sample_activations(
        features, hidden, sample_size=sample_size, random_state=random_state)

    corr = 1 - cdist(hidden_sample.T, features_sample.T, metric='correlation')
    neuron_feature_corr = pd.DataFrame(corr, columns=features.columns)

    return neuron_feature_corr


def compute_rocauc(features, hidden, sample_size=100000, n_jobs=8, random_state=42):

    feature_cols = features.columns
    features_sample, hidden_sample = sample_activations(
        features, hidden, sample_size=sample_size, random_state=random_state)
    
    n_jobs = min(n_jobs, len(feature_cols))

    if n_jobs > 1:
        parallel = ProgressParallel(n_jobs=n_jobs, use_tqdm=True, total=len(feature_cols),
                                    desc='ROC AUC computation')
        roc_aucs = parallel(delayed(compute_rocauc_for_feature)
                            (feature_col, features=features_sample, hidden=hidden_sample)
                            for feature_col in feature_cols)
    else:
        roc_aucs = []
        for feature_col in tqdm(feature_cols):
            roc_aucs.append(compute_rocauc_for_feature(
                feature_col, features_sample, hidden_sample))

    roc_aucs = pd.DataFrame(roc_aucs).T
    roc_aucs.columns = feature_cols

    return roc_aucs


def sample_activations(features, hidden, sample_size=100000, random_state=42):

    if sample_size < features.shape[0]:
        features_sample = features.sample(sample_size, replace=False, random_state=random_state)
    else:
        features_sample = features
    hidden_sample  = hidden[features_sample.index]

    return features_sample, hidden_sample


def compute_rocauc_for_feature(feature_col, features, hidden):

    roc_aucs = []
    for neuron_id in range(hidden.shape[1]):
        roc_aucs.append(roc_auc_score(features[feature_col], hidden[:, neuron_id]))

    return roc_aucs


# https://stackoverflow.com/questions/37804279/how-can-we-use-tqdm-in-a-parallel-execution-with-joblib
class ProgressParallel(Parallel):
    def __init__(self, use_tqdm=True, total=None, desc='', *args, **kwargs):
        self._use_tqdm = use_tqdm
        self._total = total
        self._desc = desc
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm(disable=not self._use_tqdm, total=self._total, desc=self._desc) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()
