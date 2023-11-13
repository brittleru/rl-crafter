import numpy as np
import warnings


def compute_success_rates(runs, sort_by=None):
    algorithms = sorted(set(run['algorithm'] for run in runs))
    seeds = sorted(set(run['seed'] for run in runs))
    tasks = sorted(key for key in runs[0] if key.startswith('achievement_'))
    percents = np.empty((len(algorithms), len(seeds), len(tasks)))
    percents[:] = np.nan
    for run in runs:
        i = algorithms.index(run['algorithm'])
        j = seeds.index(run['seed'])
        for key, values in run.items():
            if key in tasks:
                k = tasks.index(key)
                percent = 100 * (np.array(values) >= 1).mean()
                percents[i, j, k] = percent
    if isinstance(sort_by, (str, int)):
        if isinstance(sort_by, str):
            sort_by = algorithms.index(sort_by)
        order = np.argsort(-np.nanmean(percents[sort_by], 0), -1)
        percents = percents[:, :, order]
        tasks = np.array(tasks)[order].tolist()
    return percents, algorithms, seeds, tasks


def compute_scores(percents):
    # Geometric mean with an offset of 1%.
    assert (0 <= percents).all() and (percents <= 100).all()
    if (percents <= 1.0).all():
        print('Warning: The input may not be in the right range.')
    with warnings.catch_warnings():  # Empty seeds become NaN.
        warnings.simplefilter('ignore', category=RuntimeWarning)
        scores = np.exp(np.nanmean(np.log(1 + percents), -1)) - 1
    return scores


def binning(xs, ys, borders, reducer=np.nanmean, fill='nan'):
    xs, ys = np.array(xs), np.array(ys)
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    binned = []
    with warnings.catch_warnings():  # Empty buckets become NaN.
        warnings.simplefilter('ignore', category=RuntimeWarning)
        for start, stop in zip(borders[:-1], borders[1:]):
            left = (xs <= start).sum()
            right = (xs <= stop).sum()
            if left < right:
                value = reducer(ys[left:right])
            elif binned:
                value = {'nan': np.nan, 'last': binned[-1]}[fill]
            else:
                value = np.nan
            binned.append(value)
    return borders[1:], np.array(binned)
