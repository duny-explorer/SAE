import numpy as np
import pandas as pd


def remove_last_item(data, user_id='user_id', item_id='item_id', timestamp='timestamp'):
    """Remove last item from each user sequence."""

    data.sort_values([user_id, timestamp], inplace=True)
    short_data = data.groupby(user_id)[item_id].agg(list).apply(
        lambda x: x[:-1]).reset_index().explode(item_id)
    short_data[timestamp] = data.groupby(user_id)[timestamp].agg(list).apply(
        lambda x: x[:-1]).reset_index().explode(timestamp)[timestamp]

    return short_data


def get_last_item(data, user_id='user_id', item_id='item_id', timestamp='timestamp'):
    """Get last item from each user sequence."""

    data.sort_values([user_id, timestamp], inplace=True)
    data_last = data.groupby(user_id)[item_id].agg(list).apply(lambda x: x[-1]).reset_index()

    return data_last


def preds2recs(preds, item_mapping=None):

    user_ids = np.hstack([pred['user_ids'] for pred in preds])
    scores = np.vstack([pred['scores'] for pred in preds])
    preds = np.vstack([pred['preds'] for pred in preds])

    user_ids = np.repeat(user_ids[:, None], repeats=scores.shape[1], axis=1)

    recs = pd.DataFrame({'user_id': user_ids.flatten(),
                         'item_id': preds.flatten(),
                         'prediction': scores.flatten()})

    if item_mapping is not None:
        recs.item_id = recs.item_id.map(item_mapping)

    return recs
