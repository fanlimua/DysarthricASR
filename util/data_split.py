from collections import defaultdict
from typing import Dict, List, Tuple
import numpy as np
from datasets import Dataset, DatasetDict


def split_data(n: int, ratios: Tuple[float, float, float]) -> Tuple[int, int, int]:
    train_ratio, val_ratio, test_ratio = ratios
    if n <= 0:
        return 0, 0, 0
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val
    if n >= 3:
        if n_train == 0:
            n_train = 1
            n_test -= 1
        if n_val == 0:
            n_val = 1
            n_test -= 1
        if n_test == 0:
            n_test = 1
            if n_train > n_val:
                n_train -= 1
            else:
                n_val -= 1
    else:
        n_train = min(1, n)
        n_val = min(1, max(0, n - n_train))
        n_test = max(0, n - n_train - n_val)
        
    return n_train, n_val, n_test

def split_per_speaker_ratio(
    dataset: Dataset,
    speaker_column: str,
    seed: int,
    ratios: Tuple[float, float, float],
) -> Tuple[DatasetDict, Dict]:
    # group by speaker
    indices_by_speaker: Dict[str, List[int]] = defaultdict(list)
    for idx, spk in enumerate(dataset[speaker_column]):
        indices_by_speaker[str(spk)].append(idx)

    rng = np.random.default_rng(seed)
    train_indices: List[int] = []
    val_indices: List[int] = []
    test_indices: List[int] = []

    # split each speaker's data into train, val, test
    for spk, indices in indices_by_speaker.items():
        indices = list(indices)
        rng.shuffle(indices)
        n_train, n_val, n_test = split_data(len(indices), ratios)
        train_indices.extend(indices[:n_train])
        val_indices.extend(indices[n_train : n_train + n_val])
        test_indices.extend(indices[n_train + n_val : n_train + n_val + n_test])

    indices_dict = {
        "train": sorted(train_indices),
        "validation": sorted(val_indices),
        "test": sorted(test_indices),
        "split_method": "ratio",
        "seed": seed,
        "ratios": list(ratios),
    }
    return (
        DatasetDict(
            train=dataset.select(train_indices),
            validation=dataset.select(val_indices),
            test=dataset.select(test_indices),
        ),
        indices_dict,
    )

def leave_one_speaker_out(
    dataset: Dataset,
    speaker_column: str,
    loso_test_speaker: str,
    loso_val_speaker: str,
) -> Tuple[DatasetDict, Dict]:

    # Speaker-independent LOSO
    # test = one speaker, val = one speaker, train = the others
    indices_by_speaker: Dict[str, List[int]] = defaultdict(list)
    for idx, spk in enumerate(dataset[speaker_column]):
        indices_by_speaker[str(spk)].append(idx)

    test_indices = indices_by_speaker[loso_test_speaker]
    val_indices = indices_by_speaker[loso_val_speaker]
    train_indices = []
    for spk, indices in indices_by_speaker.items():
        if spk != loso_test_speaker and spk != loso_val_speaker:
            train_indices.extend(indices)

    indices_dict = {
        "train": sorted(train_indices),
        "validation": sorted(val_indices),
        "test": sorted(test_indices),
        "split_method": "loso",
        "loso_test_speaker": loso_test_speaker,
        "loso_val_speaker": loso_val_speaker,
    }
    return (
        DatasetDict(
            train=dataset.select(train_indices),
            validation=dataset.select(val_indices),
            test=dataset.select(test_indices),
        ),
        indices_dict,
    )

def split_dataset(
    dataset: Dataset,
    method: str,
    speaker_column: str,
    seed: int,
    ratios: Tuple[float, float, float],
    loso_test_speaker: str = None,
    loso_val_speaker: str = None,
) -> Tuple[DatasetDict, Dict]:

    # Split dataset by method: ratio (per-speaker ratios) or loso
    if method == "ratio":
        return split_per_speaker_ratio(
            dataset=dataset,
            speaker_column=speaker_column,
            seed=seed,
            ratios=ratios,
        )
    elif method == "loso":
        return leave_one_speaker_out(
            dataset=dataset,
            speaker_column=speaker_column,
            loso_test_speaker=loso_test_speaker,
            loso_val_speaker=loso_val_speaker,
        )
    else:
        raise ValueError("split_method must be 'ratio' or 'loso', got: %s" % method)
