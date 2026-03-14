import os
import glob
from typing import Any, Dict, List, Optional
import numpy as np
import torch
from transformers import TrainerCallback
from audiomentations import (
    Compose,
    TimeStretch,
    AddGaussianSNR,
    ApplyImpulseResponse,
    PitchShift,
)

# speed factors
DEFAULT_SPEED_FACTORS = (0.9, 1.0, 1.1)


def list_rir_paths(root_dir: str, exts: Optional[List[str]] = None) -> List[str]:
    # list all RIR files in the directory
    if exts is None:
        exts = [".wav", ".flac"]

    root_dir = os.path.abspath(root_dir)
    paths: List[str] = []
    for ext in exts:
        pattern = os.path.join(root_dir, "**", f"*{ext}")
        paths.extend(glob.glob(pattern, recursive=True))
    return sorted(set(paths))

class AugmentedDataset(torch.utils.data.Dataset):
    # apply augmentation 
    def __init__(
        self,
        hf_dataset, 
        feature_extractor,
        tokenizer,
        text_column: str,
        augment_seed: int,
        snr_db_range: tuple = (5.0, 20.0),
        rir_paths: list = None,
        speed_factors: tuple = DEFAULT_SPEED_FACTORS,
        noise_prob: float = 0.5,
        reverb_prob: float = 0.5,
        pitch_prob: float = 0.5,
        pitch_steps_range: tuple = (-2.0, 2.0),
    ):
        self.hf_dataset = hf_dataset
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.text_column = text_column
        self.augment_seed = augment_seed
        self.snr_db_range = snr_db_range or (5.0, 20.0)
        self.rir_paths = rir_paths or []
        self.speed_factors = speed_factors
        self.noise_prob = noise_prob
        self.reverb_prob = reverb_prob
        self.pitch_prob = pitch_prob
        self.pitch_steps_range = pitch_steps_range
        self.epoch = 0
        self._length = len(self.hf_dataset)
        self.augment_pipeline = self._build_pipeline()

    def _build_pipeline(self):
        transforms = [
            AddGaussianSNR(
                min_snr_db=self.snr_db_range[0],
                max_snr_db=self.snr_db_range[1],
                p=self.noise_prob,
            ),
            PitchShift(
                min_semitones=self.pitch_steps_range[0],
                max_semitones=self.pitch_steps_range[1],
                p=self.pitch_prob,
            ),
        ]
        if self.rir_paths:
            transforms.append(
                ApplyImpulseResponse(
                    ir_path=self.rir_paths,
                    p=self.reverb_prob,
                    leave_length_unchanged=True,
                )
            )
        return Compose(transforms)

    def __len__(self):
        return self._length

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.hf_dataset[int(idx)]
        audio = row["audio"]
        arr = np.asarray(audio["array"], dtype=np.float32)
        sr = audio["sampling_rate"]
        
        # set seed for augmentation
        text = row.get(self.text_column, "")
        sample_seed = int(self.augment_seed + self.epoch * self._length + idx)
        np.random.seed(sample_seed)
        
        # normalize audio waveform 
        if arr.max() > 1.0 or arr.min() < -1.0:
            arr = np.clip(arr, -1.0, 1.0)

        # speed
        rate = float(np.random.choice(self.speed_factors))
        if rate != 1.0:
            arr = TimeStretch(min_rate=rate, max_rate=rate, p=1.0)(samples=arr, sample_rate=sr)
        
        # apply augmentation pipeline
        arr = self.augment_pipeline(samples=arr, sample_rate=sr)
        arr = arr.astype(np.float32)
        
        # convert audio to input_features
        feat = self.feature_extractor(arr, sampling_rate=sr).input_features[0]
        label_ids = self.tokenizer(text).input_ids
        return {"input_features": feat, "labels": label_ids}

class SetEpochCallback(TrainerCallback):
       def on_epoch_begin(self, args, state, control, **kwargs):
           train_dataloader = kwargs.get("train_dataloader")
           if train_dataloader is not None:
               ds = getattr(train_dataloader, "dataset", None)
               if ds is not None and hasattr(ds, "epoch"):
                   ds.epoch = int(state.epoch)
