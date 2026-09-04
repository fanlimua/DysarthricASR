import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple
import numpy as np
import torch
from jiwer import cer as jiwer_cer
from jiwer import wer as jiwer_wer
from torch.utils.data import Dataset as TorchDataset
from transformers.models.whisper.english_normalizer import BasicTextNormalizer


def parse_layer_spec(spec: str, num_layers: int, argument_name: str) -> List[int]:
    # Parse layer selections such as ``8-11``, ``0,2,4-6``, ``all``, or ``none``.
    value = spec.strip().lower()
    if value == "all":
        return list(range(num_layers))
    if value == "none":
        return []
    if not value:
        raise ValueError(f"{argument_name} cannot be empty")

    selected = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            raise ValueError(f"Invalid {argument_name} value: {spec!r}")
        if "-" in part:
            bounds = part.split("-")
            if len(bounds) != 2 or not all(bound.isdigit() for bound in bounds):
                raise ValueError(f"Invalid {argument_name} range: {part!r}")
            start, end = map(int, bounds)
            if start > end:
                raise ValueError(f"Invalid descending {argument_name} range: {part!r}")
            selected.update(range(start, end + 1))
        elif part.isdigit():
            selected.add(int(part))
        else:
            raise ValueError(f"Invalid {argument_name} layer: {part!r}")

    invalid = sorted(layer for layer in selected if layer >= num_layers)
    if invalid:
        raise ValueError(
            f"{argument_name} contains out-of-range layers {invalid}; "
            f"valid layer indices are 0-{num_layers - 1}"
        )
    return sorted(selected)


def parse_target_modules(spec: str, argument_name: str) -> List[str]:
    # Parse a comma separated module list such as ``q_proj,v_proj``.
    modules: List[str] = []
    for part in spec.split(","):
        name = part.strip()
        if not name:
            raise ValueError(f"Invalid {argument_name} value: {spec!r}")
        if name not in modules:
            modules.append(name)
    if not modules:
        raise ValueError(f"{argument_name} cannot be empty")
    return modules


def load_severity_map(path: str) -> Tuple[Dict[str, str], List[str]]:
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    order = list(raw.get("order_for_lora", []))
    mapping = {
        speaker: severity
        for speaker, severity in raw.items()
        if not speaker.startswith("_") and speaker != "order_for_lora"
    }
    return mapping, order


def resolve_loso_speakers(
    severity_map: Mapping[str, str],
    requested: Sequence[str] | None = None,
    include_normal: bool = False,
) -> List[str]:
    if requested:
        unknown = sorted(set(requested) - set(severity_map))
        if unknown:
            raise ValueError(f"Speakers missing from severity map: {unknown}")
        return list(dict.fromkeys(requested))
    return [
        speaker
        for speaker, severity in severity_map.items()
        if include_normal or severity.lower() != "normal"
    ]


def audio_to_numpy(audio: Any, target_sample_rate: int = 16000) -> np.ndarray:
    if hasattr(audio, "get_all_samples"):
        decoded = audio.get_all_samples()
        samples = decoded.data
        sample_rate = int(decoded.sample_rate)
    elif isinstance(audio, Mapping):
        samples = audio["array"]
        sample_rate = int(audio["sampling_rate"])
    else:
        raise TypeError(f"Unsupported audio value: {type(audio)!r}")

    if isinstance(samples, torch.Tensor):
        samples = samples.detach().cpu().numpy()
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim == 2:
        samples = samples.mean(axis=0)
    samples = np.squeeze(samples)
    if samples.ndim != 1:
        raise ValueError(f"Expected mono audio, got shape {samples.shape}")

    if sample_rate != target_sample_rate:
        old_positions = np.arange(samples.size, dtype=np.float64)
        new_size = int(round(samples.size * target_sample_rate / sample_rate))
        new_positions = np.linspace(0, max(samples.size - 1, 0), new_size)
        samples = np.interp(new_positions, old_positions, samples).astype(np.float32)
    return samples


def filter_by_duration(dataset, max_seconds: float, sample_rate: int = 16000):
    if max_seconds <= 0:
        return dataset

    def is_short_enough(row: Dict[str, Any]) -> bool:
        samples = audio_to_numpy(row["audio"], target_sample_rate=sample_rate)
        return samples.size <= int(max_seconds * sample_rate)

    return dataset.filter(is_short_enough, desc=f"Filtering audio <= {max_seconds:g}s")


def speaker_indices(dataset) -> Dict[str, List[int]]:
    result: Dict[str, List[int]] = defaultdict(list)
    for index, speaker in enumerate(dataset["speaker"]):
        result[str(speaker)].append(index)
    return dict(result)


def make_loso_fold(dataset, indices: Mapping[str, Sequence[int]], held_out: str):
    # create a leave-one-speaker-out fold from a dataset and speaker indices.
    if held_out not in indices:
        raise ValueError(f"Speaker {held_out!r} is not present in the dataset")
    validation_indices = list(indices[held_out])
    validation_set = set(validation_indices)
    train_indices = [idx for idx in range(len(dataset)) if idx not in validation_set]
    return dataset.select(train_indices), dataset.select(validation_indices)


class WaveformAugmentedDataset(TorchDataset):
    def __init__(self, dataset, transform, sample_rate: int = 16000):
        self.dataset = dataset
        self.transform = transform
        self.sample_rate = sample_rate

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = dict(self.dataset[index])
        samples = audio_to_numpy(row["audio"], self.sample_rate)
        row["audio"] = {
            "array": self.transform(samples=samples, sample_rate=self.sample_rate),
            "sampling_rate": self.sample_rate,
        }
        return row


class WhisperRawAudioCollator:
    def __init__(self, processor, sample_rate: int = 16000):
        self.processor = processor
        self.sample_rate = sample_rate

    def __call__(self, rows: Sequence[Mapping[str, Any]]) -> Dict[str, torch.Tensor]:
        waveforms = [audio_to_numpy(row["audio"], self.sample_rate) for row in rows]
        batch = self.processor.feature_extractor(
            waveforms,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
        )
        tokenized = self.processor.tokenizer(
            [str(row["text"]) for row in rows],
            padding=True,
            return_tensors="pt",
        )
        labels = tokenized.input_ids.masked_fill(tokenized.attention_mask.ne(1), -100)
        decoder_start = self.processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
        if labels.shape[1] and (labels[:, 0] == decoder_start).all().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def decode_prediction_output(prediction_output, processor) -> Tuple[List[str], List[str]]:
    predictions = prediction_output.predictions
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    labels = np.array(prediction_output.label_ids, copy=True)
    labels[labels == -100] = processor.tokenizer.pad_token_id
    predicted_text = processor.tokenizer.batch_decode(predictions, skip_special_tokens=True)
    reference_text = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
    return predicted_text, reference_text


def text_metrics(predictions: Sequence[str], references: Sequence[str]) -> Dict[str, float | int]:
    normalizer = BasicTextNormalizer()
    normalized_predictions = [normalizer(text) for text in predictions]
    normalized_references = [normalizer(text) for text in references]
    return {
        "wer": 100.0 * float(jiwer_wer(normalized_references, normalized_predictions)),
        "cer": 100.0 * float(jiwer_cer(normalized_references, normalized_predictions)),
        "num_samples": len(normalized_references),
    }


def aggregate_speaker_metrics(
    per_speaker: Mapping[str, Mapping[str, float | int | str]],
    severity_map: Mapping[str, str],
    severity_order: Iterable[str] = (),
) -> Dict[str, Any]:
    if not per_speaker:
        raise ValueError("Cannot aggregate an empty speaker metric mapping")
    overall = {
        key: float(np.mean([float(metrics[key]) for metrics in per_speaker.values()]))
        for key in ("wer", "cer")
    }
    grouped: Dict[str, List[Mapping[str, float | int | str]]] = defaultdict(list)
    for speaker, metrics in per_speaker.items():
        grouped[severity_map[speaker]].append(metrics)
    ordered_levels = [level for level in severity_order if level in grouped]
    ordered_levels.extend(sorted(set(grouped) - set(ordered_levels)))
    severity = {
        level: {
            "wer": float(np.mean([float(item["wer"]) for item in grouped[level]])),
            "cer": float(np.mean([float(item["cer"]) for item in grouped[level]])),
            "num_speakers": len(grouped[level]),
            "num_samples": int(sum(int(item["num_samples"]) for item in grouped[level])),
        }
        for level in ordered_levels
    }
    return {"overall_macro": overall, "severity_macro": severity}


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()
