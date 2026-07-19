"""Deterministic dataset splits used for checkpoint selection."""

import math

import torch


def build_train_holdout_indices(
    num_samples,
    holdout_fraction,
    seed,
    *,
    strategy='random',
    sample_counts=None,
    num_strata=5,
):
    """Return deterministic train and validation indices.

    Count stratification keeps validation representative on long-tailed crowd
    datasets without inspecting image pixels or benchmark-test annotations.
    """
    if num_samples < 2:
        raise ValueError('train-holdout split requires at least 2 samples')
    holdout_fraction = float(holdout_fraction)
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError('train_holdout_fraction must be in (0, 1)')
    num_val = max(1, min(num_samples - 1, int(round(num_samples * holdout_fraction))))
    generator = torch.Generator().manual_seed(int(seed))

    if strategy == 'random':
        permutation = torch.randperm(num_samples, generator=generator).tolist()
        return sorted(permutation[num_val:]), sorted(permutation[:num_val])
    if strategy != 'count_stratified':
        raise ValueError(f'unsupported train holdout strategy: {strategy!r}')
    if sample_counts is None or len(sample_counts) != num_samples:
        raise ValueError(
            'count_stratified train holdout requires one sample count per training image'
        )

    counts = [float(value) for value in sample_counts]
    ordered = sorted(range(num_samples), key=lambda index: (counts[index], index))
    n_strata = min(int(num_strata), num_samples)
    strata = []
    for index in range(n_strata):
        start = (num_samples * index) // n_strata
        stop = (num_samples * (index + 1)) // n_strata
        strata.append(ordered[start:stop])

    raw_allocations = [len(stratum) * holdout_fraction for stratum in strata]
    allocations = [int(math.floor(value)) for value in raw_allocations]
    remaining = num_val - sum(allocations)
    for index in sorted(
        range(n_strata),
        key=lambda item: (raw_allocations[item] - allocations[item], -item),
        reverse=True,
    )[:remaining]:
        allocations[index] += 1

    validation = []
    for stratum, allocation in zip(strata, allocations):
        if allocation:
            selected = torch.randperm(len(stratum), generator=generator)[:allocation].tolist()
            validation.extend(stratum[item] for item in selected)
    validation = sorted(validation)
    validation_set = set(validation)
    train = [index for index in range(num_samples) if index not in validation_set]
    return train, validation
