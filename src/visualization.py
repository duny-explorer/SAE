import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_activation_distr(activations, low_value=0.001, clip_quantile=0.99,
                          sample_size=1000, figsize=(8, 3)):

    activations = activations[np.random.randint(low=0, high=len(activations), size=sample_size)]
    activations_series = pd.Series(activations.flatten())
    activations_series = activations_series[activations_series > low_value]
    activations_series = activations_series.clip(upper=activations_series.quantile(clip_quantile))

    fig = plt.figure(figsize=figsize)
    activations_series.hist(bins=50, range=(low_value, activations_series.max()), figure=fig)

    return fig


def plot_num_active_neurons(num_active_neurons, col_wrap=6, height=3,
                            sharey=False, sharex=False):

    num_active_neurons2 = num_active_neurons.stack().reset_index()
    num_active_neurons2 = num_active_neurons2[['level_1', 0]]
    num_active_neurons2.columns = ['threshold', 'num_active_neurons']

    grid = sns.displot(data=num_active_neurons2, x="num_active_neurons", col="threshold",
                       col_wrap=col_wrap, height=height, binwidth=1,
                       common_bins=sharex, facet_kws={'sharey': sharey, 'sharex': sharex})

    return grid.figure


def plot_neuron_fired_fraction(neuron_fired_fraction, col_wrap=6, height=3,
                               cumulative=True, sharey=False, sharex=False):

    neuron_fired_fraction2 = neuron_fired_fraction.stack().reset_index()
    neuron_fired_fraction2 = neuron_fired_fraction2[['level_1', 0]]
    neuron_fired_fraction2.columns = ['threshold', 'neuron_fired_fraction']

    grid = sns.displot(data=neuron_fired_fraction2, x="neuron_fired_fraction", col="threshold",
                       col_wrap=col_wrap, height=height, binwidth=0.01, cumulative=cumulative,
                       common_bins=sharex, facet_kws={'sharey': sharey, 'sharex': sharex})

    return grid.figure


def plot_feature_vs_neuron(hidden, features, feature_name, neuron_id,
                           figsize=(14, 4), threshold=0):

    df_for_plot = features[[feature_name]]
    df_for_plot['hidden'] = hidden[:, neuron_id]
    df_for_plot_activated = df_for_plot[df_for_plot.hidden > threshold]

    fig, ax = plt.subplots(1, 2, figsize=figsize)
    # element="poly" or "step"
    sns.histplot(df_for_plot_activated, x='hidden', hue=feature_name, stat='count',
             multiple="stack", element="poly", ax=ax[0])
    sns.ecdfplot(data=df_for_plot, x="hidden", hue=feature_name, ax=ax[1])

    return fig
