import numpy as np
import pandas as pd


def get_logit_diff_for_neuron(model, ae, item_features, neuron_id, feature_cols, title_col='title'):

    decoder_weights = ae.decoder.weight.detach().cpu().numpy()
    unembed = model.head.weight.detach().cpu().numpy()

    neuron_direction = decoder_weights[:, neuron_id]
    logit_diff = np.dot(unembed, neuron_direction)

    logit_diff = pd.DataFrame({
        'logit_diff': logit_diff[item_features.item_id.values],
        'title': item_features[title_col].values,
        'features': item_features[feature_cols].apply(lambda x: '|'.join(x[x == 1].index), axis=1).values})

    return logit_diff


def get_logit_diff_matrix(model, ae, item_features, features):

    decoder_weights = ae.decoder.weight.detach().cpu().numpy()
    unembed = model.head.weight.detach().cpu().numpy()

    logit_diff_matrix = np.dot(unembed, decoder_weights)

    logit_diffs = []
    for feature_name in features.columns:
        indices = item_features[item_features[feature_name] == 1].item_id.values
        logit_diff_for_feature = logit_diff_matrix[indices].mean(axis=0)
        logit_diff_for_feature = pd.Series(logit_diff_for_feature, name=feature_name)
        logit_diffs.append(logit_diff_for_feature)
    logit_diffs = pd.DataFrame(logit_diffs).T

    return logit_diffs
