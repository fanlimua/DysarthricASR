"""Evaluate Whisper models on TORGO LOSO folds and LibriSpeech retention.

The script evaluates each ``held-out-*`` model on:

1. that model's held-out TORGO speaker; and/or
2. a shared LibriSpeech set used to measure general-ASR retention.

Examples
--------
Evaluate every fold in a fine-tuning method::

    pixi run python evaluate_loso_retention.py \
      --evaluation_mode loso \
      --method_dir results/partial_ft/partial-finetune-enc6-11-dec6-11 \
      --eval_targets both \
      --batch_size 128 \
      --output_dir results/partial_ft/partial-finetune-enc6-11-dec6-11

Evaluate an official pretrained Whisper model on all TORGO and LibriSpeech::

    pixi run python evaluate_loso_retention.py \
      --evaluation_mode single \
      --model_path openai/whisper-small \
      --eval_targets both \
      --batch_size 128

Evaluate one fine-tuned fold on its explicitly selected held-out speaker::

    pixi run python evaluate_loso_retention.py \
      --evaluation_mode single \
      --model_path results/full_ft/full-finetune/held-out-F04/final-model \
      --held_out_speaker F04 \
      --eval_targets both
"""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import torch
from datasets import Dataset, load_dataset
from peft import PeftModel
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

from util.loso import audio_to_numpy, filter_by_duration, text_metrics


SCHEMA_VERSION = 1
DEFAULT_TORGO_DATASET = "extraordinarylab/torgo"
DEFAULT_TORGO_SPLIT = "test"
DEFAULT_LIBRISPEECH_DATASET = "SPRINGLab/LibriSpeech-Test"
DEFAULT_LIBRISPEECH_SPLIT = "train"


@dataclass(frozen=True)
class FoldModel:
    speaker: str
    fold_dir: Path
    model_dir: Path
    run_config: Mapping[str, Any]
    model_type: str
    base_model: str


@dataclass
class WhisperRetentionCollator:
    """Create Whisper inputs and always include a reliable attention mask."""

    processor: WhisperProcessor
    sample_rate: int = 16000

    def __call__(self, rows: Sequence[Mapping[str, Any]]) -> Dict[str, torch.Tensor]:
        waveforms = [audio_to_numpy(row["audio"], self.sample_rate) for row in rows]
        batch = self.processor.feature_extractor(
            waveforms,
            sampling_rate=self.sample_rate,
            return_attention_mask=True,
            return_tensors="pt",
        )
        tokenized = self.processor.tokenizer(
            [str(row["text"]) for row in rows],
            padding=True,
            return_tensors="pt",
        )
        labels = tokenized.input_ids.masked_fill(tokenized.attention_mask.ne(1), -100)
        decoder_start = self.processor.tokenizer.convert_tokens_to_ids(
            "<|startoftranscript|>"
        )
        if labels.shape[1] and (labels[:, 0] == decoder_start).all().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def checkpoint_type(model_dir: Path) -> str:
    if (model_dir / "adapter_config.json").is_file():
        return "lora_adapter"
    if (model_dir / "config.json").is_file():
        return "whisper_finetuned"
    raise ValueError(
        f"{model_dir} is neither a full Whisper checkpoint nor a PEFT adapter"
    )


def model_source_type(model_source: str) -> str:
    """Classify a local checkpoint/adapter or a Hugging Face model identifier."""
    path = Path(model_source).expanduser()
    if path.exists():
        if not path.is_dir():
            raise ValueError(f"Model path is not a directory: {path}")
        return checkpoint_type(path)
    if path.is_absolute() or model_source.startswith(("./", "../")):
        raise ValueError(f"Local model path does not exist: {path}")
    return "whisper_pretrained"


def adapter_base_model(model_dir: Path) -> str:
    config = read_json(model_dir / "adapter_config.json")
    base_model = config.get("base_model_name_or_path")
    if not base_model:
        raise ValueError(
            f"LoRA adapter {model_dir} does not declare base_model_name_or_path; "
            "pass --base_model explicitly"
        )
    return str(base_model)


def discover_fold_models(
    method_dir: Path,
    requested_speakers: Sequence[str] | None = None,
) -> List[FoldModel]:
    method_dir = method_dir.resolve()
    if not method_dir.is_dir():
        raise ValueError(f"Fine-tuning method directory does not exist: {method_dir}")

    requested = set(requested_speakers or [])
    found: List[FoldModel] = []
    for fold_dir in sorted(method_dir.glob("held-out-*")):
        if not fold_dir.is_dir():
            continue
        directory_speaker = fold_dir.name.removeprefix("held-out-")
        if requested and directory_speaker not in requested:
            continue

        config_path = fold_dir / "run_config.json"
        model_dir = fold_dir / "final-model"
        if not config_path.is_file():
            raise ValueError(f"Missing run_config.json for fold: {fold_dir}")
        if not model_dir.is_dir():
            raise ValueError(f"Missing final-model for fold: {fold_dir}")

        run_config = read_json(config_path)
        configured_speaker = str(
            run_config.get("held_out_speaker", directory_speaker)
        ).strip()
        if configured_speaker != directory_speaker:
            raise ValueError(
                f"Fold directory says {directory_speaker!r}, but {config_path} says "
                f"{configured_speaker!r}"
            )
        parameters = run_config.get("parameters") or {}
        base_model = str(parameters.get("model_name") or "openai/whisper-small")
        found.append(
            FoldModel(
                speaker=directory_speaker,
                fold_dir=fold_dir,
                model_dir=model_dir,
                run_config=run_config,
                model_type=checkpoint_type(model_dir),
                base_model=base_model,
            )
        )

    if requested:
        missing = sorted(requested - {fold.speaker for fold in found})
        if missing:
            raise ValueError(
                f"Requested speakers are missing from {method_dir}: {', '.join(missing)}"
            )
    if not found:
        raise ValueError(f"No held-out-* models found in {method_dir}")
    return found


def from_pretrained_with_cache_fallback(factory, source: str, **kwargs):
    """Prefer normal Hub behavior, then use an existing cache if the Hub is down."""
    try:
        return factory.from_pretrained(source, **kwargs)
    except (OSError, RuntimeError) as online_error:
        try:
            return factory.from_pretrained(source, local_files_only=True, **kwargs)
        except (OSError, RuntimeError):
            raise online_error


def load_fold_model(fold: FoldModel):
    model_source = str(fold.model_dir)
    processor = from_pretrained_with_cache_fallback(
        WhisperProcessor, model_source, language="en", task="transcribe"
    )
    if fold.model_type == "lora_adapter":
        backbone = from_pretrained_with_cache_fallback(
            WhisperForConditionalGeneration, fold.base_model
        )
        model = PeftModel.from_pretrained(backbone, str(fold.model_dir), is_trainable=False)
    else:
        model = from_pretrained_with_cache_fallback(
            WhisperForConditionalGeneration, model_source
        )

    model.eval()
    model.generation_config.language = "en"
    model.generation_config.task = "transcribe"
    return model, processor


def build_single_model(
    model_source: str,
    held_out_speaker: str | None,
    base_model: str | None,
) -> FoldModel:
    model_type = model_source_type(model_source)
    source_path = Path(model_source).expanduser()
    if source_path.exists():
        source_path = source_path.resolve()
    if model_type == "lora_adapter":
        resolved_base = base_model or adapter_base_model(source_path)
    else:
        resolved_base = base_model or model_source
    return FoldModel(
        speaker=held_out_speaker or "all-torgo",
        fold_dir=source_path.parent if source_path.exists() else Path("."),
        model_dir=source_path,
        run_config={},
        model_type=model_type,
        base_model=resolved_base,
    )


def load_filtered_dataset(
    dataset_name: str,
    split: str,
    max_audio_seconds: float,
) -> Dataset:
    loaded = load_dataset(dataset_name)
    if split not in loaded:
        raise ValueError(
            f"Split {split!r} not found in {dataset_name!r}; available: {list(loaded)}"
        )
    return filter_by_duration(loaded[split], max_audio_seconds)


def select_torgo_speaker(
    dataset: Dataset,
    speaker: str,
    speaker_column: str,
) -> Dataset:
    if speaker_column not in dataset.column_names:
        raise ValueError(
            f"TORGO speaker column {speaker_column!r} is absent; "
            f"available columns: {dataset.column_names}"
        )
    values = [
        index
        for index, value in enumerate(dataset[speaker_column])
        if str(value).strip() == speaker
    ]
    if not values:
        raise ValueError(f"No TORGO samples found for held-out speaker {speaker!r}")
    return dataset.select(values)


def limit_dataset(dataset: Dataset, max_samples: int | None) -> Dataset:
    if max_samples is None:
        return dataset
    return dataset.select(range(min(max_samples, len(dataset))))


def decode_prediction_output(prediction_output, processor) -> Dict[str, List[str]]:
    prediction_ids = prediction_output.predictions
    if isinstance(prediction_ids, tuple):
        prediction_ids = prediction_ids[0]
    label_ids = np.array(prediction_output.label_ids, copy=True)
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    predictions = processor.tokenizer.batch_decode(
        prediction_ids, skip_special_tokens=True
    )
    references = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    normalizer = BasicTextNormalizer()
    return {
        "predictions": predictions,
        "references": references,
        "normalized_predictions": [normalizer(text) for text in predictions],
        "normalized_references": [normalizer(text) for text in references],
    }


def save_predictions(path: Path, decoded: Mapping[str, Sequence[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        rows = zip(
            decoded["predictions"],
            decoded["references"],
            decoded["normalized_predictions"],
            decoded["normalized_references"],
        )
        for index, (prediction, reference, norm_prediction, norm_reference) in enumerate(rows):
            record = {
                "idx": index,
                "prediction": prediction,
                "reference": reference,
                "normalized_prediction": norm_prediction,
                "normalized_reference": norm_reference,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def evaluate_dataset(
    model,
    processor,
    dataset: Dataset,
    output_dir: Path,
    dataset_label: str,
    batch_size: int,
    num_workers: int,
    generation_max_length: int,
    save_prediction_rows: bool,
) -> Dict[str, Any]:
    required = {"audio", "text"}
    missing = sorted(required - set(dataset.column_names))
    if missing:
        raise ValueError(
            f"{dataset_label} is missing required columns {missing}; "
            f"available columns: {dataset.column_names}"
        )

    trainer_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir / "trainer-output"),
        per_device_eval_batch_size=batch_size,
        dataloader_num_workers=num_workers,
        predict_with_generate=True,
        generation_max_length=generation_max_length,
        remove_unused_columns=False,
        report_to=[],
        do_train=False,
        do_eval=True,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=trainer_args,
        data_collator=WhisperRetentionCollator(processor),
        processing_class=processor,
    )
    prediction_output = trainer.predict(dataset, metric_key_prefix=dataset_label)
    decoded = decode_prediction_output(prediction_output, processor)
    metrics = text_metrics(
        decoded["normalized_predictions"], decoded["normalized_references"]
    )
    result: Dict[str, Any] = {
        **metrics,
        "runtime": prediction_output.metrics.get(f"{dataset_label}_runtime"),
        "samples_per_second": prediction_output.metrics.get(
            f"{dataset_label}_samples_per_second"
        ),
    }
    if save_prediction_rows:
        predictions_path = output_dir / f"{dataset_label}_predictions.jsonl"
        save_predictions(predictions_path, decoded)
        result["predictions_path"] = str(predictions_path.resolve())
    return result


def macro_average_models(
    models: Mapping[str, Mapping[str, Any]],
    target: str,
) -> Dict[str, Any] | None:
    values = [
        result[target]
        for result in models.values()
        if isinstance(result.get(target), Mapping)
    ]
    if not values:
        return None
    return {
        "wer": sum(float(value["wer"]) for value in values) / len(values),
        "cer": sum(float(value["cer"]) for value in values) / len(values),
        "num_models": len(values),
        "num_samples_per_model": sorted(
            {int(value["num_samples"]) for value in values}
        ),
        "aggregation": "unweighted macro average over LOSO models",
    }


def refresh_averages(report: Dict[str, Any]) -> None:
    models = report["models"]
    if report.get("evaluation_mode", "loso") == "single":
        only_model = next(iter(models.values()), {})
        report["single_model_result"] = {
            target: only_model[target]
            for target in ("torgo", "librispeech")
            if isinstance(only_model.get(target), Mapping)
        }
        report["method_average"] = {}
        report["updated_at"] = utc_timestamp()
        return
    report["method_average"] = {
        target: average
        for target in ("torgo", "librispeech")
        if (average := macro_average_models(models, target)) is not None
    }
    report["updated_at"] = utc_timestamp()


def markdown_summary(report: Mapping[str, Any]) -> str:
    is_loso = report.get("evaluation_mode", "loso") == "loso"
    title = "LOSO retention evaluation" if is_loso else "Single-model evaluation"
    first_column = "Held-out model" if is_loso else "Model selection"
    lines = [
        f"# {title}: {report['method_name']}",
        "",
        (
            "WER/CER are percentages. The method average is an unweighted macro "
            "average over completed LOSO models."
            if is_loso
            else "WER/CER are percentages. Exactly one model is evaluated."
        ),
        "",
        f"| {first_column} | Model type | TORGO samples | TORGO WER | TORGO CER | "
        "LibriSpeech samples | LibriSpeech WER | LibriSpeech CER |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for speaker, result in sorted(report["models"].items()):
        torgo = result.get("torgo") or {}
        libri = result.get("librispeech") or {}
        display_selection = (
            speaker
            if is_loso
            else result.get("held_out_speaker") or "all-torgo"
        )
        lines.append(
            "| {speaker} | {model_type} | {tn} | {tw} | {tc} | {ln} | {lw} | {lc} |".format(
                speaker=display_selection,
                model_type=result["model_type"],
                tn=torgo.get("num_samples", "—"),
                tw=_format_metric(torgo.get("wer")),
                tc=_format_metric(torgo.get("cer")),
                ln=libri.get("num_samples", "—"),
                lw=_format_metric(libri.get("wer")),
                lc=_format_metric(libri.get("cer")),
            )
        )

    averages = report.get("method_average") or {}
    torgo_average = averages.get("torgo") or {}
    libri_average = averages.get("librispeech") or {}
    if is_loso:
        lines.append(
            "| **Method macro average** | — | — | **{tw}** | **{tc}** | — | "
            "**{lw}** | **{lc}** |".format(
                tw=_format_metric(torgo_average.get("wer")),
                tc=_format_metric(torgo_average.get("cer")),
                lw=_format_metric(libri_average.get("wer")),
                lc=_format_metric(libri_average.get("cer")),
            )
        )
    lines.extend(
        [
            "",
            (
                "- TORGO: each model is evaluated only on the speaker named by its "
                "`held-out-*` fold."
                if is_loso
                else "- TORGO: the model is evaluated on `held_out_speaker`, or on "
                "all filtered TORGO samples when no speaker is specified."
            ),
            "- LibriSpeech: every model is evaluated on the same retention set; "
            "this is not a speaker hold-out split within LibriSpeech.",
            "",
        ]
    )
    return "\n".join(lines)


def _format_metric(value: Any) -> str:
    return "—" if value is None else f"{float(value):.3f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate either all models in a TORGO LOSO method or one explicitly "
            "selected Whisper model on TORGO and LibriSpeech retention."
        )
    )
    parser.add_argument(
        "--evaluation_mode",
        choices=("loso", "single"),
        default="loso",
        help=(
            "loso scans METHOD_DIR/held-out-*; single loads exactly MODEL_PATH."
        ),
    )
    parser.add_argument(
        "--method_dir",
        type=Path,
        default=None,
        help="Directory containing held-out-*/final-model folds.",
    )
    parser.add_argument(
        "--model_path",
        default=None,
        help=(
            "Single-mode Hugging Face model id, full local checkpoint, or local "
            "LoRA adapter directory."
        ),
    )
    parser.add_argument(
        "--base_model",
        default=None,
        help="Optional base model override when MODEL_PATH is a LoRA adapter.",
    )
    parser.add_argument(
        "--held_out_speaker",
        default=None,
        help=(
            "Single-mode TORGO speaker, e.g. F04. If omitted, the single model "
            "is evaluated on all filtered TORGO samples."
        ),
    )
    parser.add_argument(
        "--eval_targets",
        choices=("both", "torgo", "librispeech"),
        default="both",
    )
    parser.add_argument("--speakers", nargs="*", default=None)
    parser.add_argument("--output_dir", type=Path, default=None)

    parser.add_argument("--torgo_dataset_name", default=DEFAULT_TORGO_DATASET)
    parser.add_argument("--torgo_split", default=DEFAULT_TORGO_SPLIT)
    parser.add_argument("--speaker_column", default="speaker")
    parser.add_argument(
        "--librispeech_dataset_name", default=DEFAULT_LIBRISPEECH_DATASET
    )
    parser.add_argument("--librispeech_split", default=DEFAULT_LIBRISPEECH_SPLIT)

    parser.add_argument("--max_audio_seconds", type=float, default=30.0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--generation_max_length", type=int, default=225)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--save_predictions", action="store_true")
    parser.add_argument(
        "--skip_completed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse a model/dataset result already present in retention_results.json.",
    )
    args = parser.parse_args()
    if args.evaluation_mode == "loso":
        if args.method_dir is None:
            parser.error("--method_dir is required when --evaluation_mode loso")
        if args.model_path is not None or args.held_out_speaker is not None:
            parser.error(
                "--model_path and --held_out_speaker require --evaluation_mode single"
            )
        if args.base_model is not None:
            parser.error("--base_model requires --evaluation_mode single")
    else:
        if args.model_path is None:
            parser.error("--model_path is required when --evaluation_mode single")
        if args.method_dir is not None or args.speakers:
            parser.error(
                "--method_dir and --speakers cannot be used with --evaluation_mode single"
            )
    if args.max_audio_seconds <= 0:
        parser.error("--max_audio_seconds must be positive")
    if args.batch_size <= 0:
        parser.error("--batch_size must be positive")
    if args.num_workers < 0:
        parser.error("--num_workers cannot be negative")
    if args.generation_max_length <= 0:
        parser.error("--generation_max_length must be positive")
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max_samples must be positive")
    return args


def selected_targets(value: str) -> Iterable[str]:
    return ("torgo", "librispeech") if value == "both" else (value,)


def initial_report(
    args: argparse.Namespace,
    report_name: str,
    method_dir: Path | None = None,
) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_timestamp(),
        "updated_at": utc_timestamp(),
        "evaluation_mode": args.evaluation_mode,
        "method_name": report_name,
        "method_dir": str(method_dir) if method_dir is not None else None,
        "model_source": args.model_path if args.evaluation_mode == "single" else None,
        "protocol": protocol_config(args),
        "models": {},
        "method_average": {},
    }


def protocol_config(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "evaluation_mode": args.evaluation_mode,
        "torgo": {
            "dataset": args.torgo_dataset_name,
            "split": args.torgo_split,
            "selection": (
                "speaker matching the held-out-* model"
                if args.evaluation_mode == "loso"
                else args.held_out_speaker or "all filtered TORGO speakers"
            ),
        },
        "librispeech": {
            "dataset": args.librispeech_dataset_name,
            "split": args.librispeech_split,
            "selection": (
                "same retention set for every LOSO model"
                if args.evaluation_mode == "loso"
                else "retention set for the single selected model"
            ),
        },
        "max_audio_seconds": args.max_audio_seconds,
        "max_samples": args.max_samples,
        "metric_normalization": "Whisper BasicTextNormalizer",
        "result_aggregation": (
            "unweighted macro average over LOSO models"
            if args.evaluation_mode == "loso"
            else "single model; no model-level averaging"
        ),
    }


def output_slug(model_source: str, held_out_speaker: str | None) -> str:
    name = Path(model_source.rstrip("/")).name or "whisper-model"
    safe_name = "".join(
        character if character.isalnum() or character in "-_" else "-"
        for character in name
    ).strip("-") or "whisper-model"
    selection = held_out_speaker or "all-torgo"
    return f"{safe_name}-{selection}"


def main() -> None:
    args = parse_args()
    method_dir = args.method_dir.resolve() if args.method_dir is not None else None
    if args.evaluation_mode == "loso":
        assert method_dir is not None
        folds = discover_fold_models(method_dir, args.speakers)
        report_name = method_dir.name
        default_output_dir = method_dir / "retention-eval"
    else:
        assert args.model_path is not None
        folds = [
            build_single_model(
                args.model_path,
                args.held_out_speaker,
                args.base_model,
            )
        ]
        report_name = f"single:{args.model_path}"
        default_output_dir = (
            Path(__file__).resolve().parent
            / "results"
            / "evaluation"
            / output_slug(args.model_path, args.held_out_speaker)
        )
    output_dir = args.output_dir.resolve() if args.output_dir else default_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "retention_results.json"
    summary_path = output_dir / "summary.md"

    if report_path.is_file() and args.skip_completed:
        report = read_json(report_path)
        if report.get("evaluation_mode") != args.evaluation_mode:
            raise ValueError(f"Existing {report_path} uses a different evaluation mode")
        if args.evaluation_mode == "loso":
            if Path(report.get("method_dir", "")).resolve() != method_dir:
                raise ValueError(
                    f"Existing {report_path} belongs to a different method directory"
                )
        elif report.get("model_source") != args.model_path:
            raise ValueError(f"Existing {report_path} belongs to a different model")
        if report.get("protocol") != protocol_config(args):
            raise ValueError(
                f"Existing {report_path} uses a different dataset/evaluation "
                "protocol. Choose another --output_dir or pass --no-skip_completed "
                "to replace it."
            )
    else:
        report = initial_report(args, report_name, method_dir)

    targets = tuple(selected_targets(args.eval_targets))
    torgo = None
    librispeech = None
    if "torgo" in targets:
        print(f"Loading TORGO: {args.torgo_dataset_name}[{args.torgo_split}]")
        torgo = load_filtered_dataset(
            args.torgo_dataset_name, args.torgo_split, args.max_audio_seconds
        )
    if "librispeech" in targets:
        print(
            "Loading LibriSpeech retention set: "
            f"{args.librispeech_dataset_name}[{args.librispeech_split}]"
        )
        librispeech = limit_dataset(
            load_filtered_dataset(
                args.librispeech_dataset_name,
                args.librispeech_split,
                args.max_audio_seconds,
            ),
            args.max_samples,
        )

    for model_number, fold in enumerate(folds, start=1):
        result_key = fold.speaker if args.evaluation_mode == "loso" else "single"
        existing = report["models"].get(result_key, {})
        pending = [
            target
            for target in targets
            if not (args.skip_completed and isinstance(existing.get(target), Mapping))
        ]
        if not pending:
            print(f"[{model_number}/{len(folds)}] {result_key}: already complete")
            continue

        print(
            f"[{model_number}/{len(folds)}] Loading {fold.model_type} model for "
            f"evaluation selection {fold.speaker}"
        )
        model, processor = load_fold_model(fold)
        model_result = {
            **existing,
            "held_out_speaker": (
                fold.speaker if args.evaluation_mode == "loso" else args.held_out_speaker
            ),
            "fold_dir": str(fold.fold_dir) if args.evaluation_mode == "loso" else None,
            "model_path": str(fold.model_dir),
            "model_type": fold.model_type,
            "base_model": fold.base_model,
        }
        model_output_dir = (
            output_dir / "models" / f"held-out-{fold.speaker}"
            if args.evaluation_mode == "loso"
            else output_dir / "models" / "single"
        )

        if "torgo" in pending:
            assert torgo is not None
            if args.evaluation_mode == "single" and args.held_out_speaker is None:
                fold_torgo = limit_dataset(torgo, args.max_samples)
                torgo_selection = "all speakers"
            else:
                selected_speaker = (
                    args.held_out_speaker
                    if args.evaluation_mode == "single"
                    else fold.speaker
                )
                assert selected_speaker is not None
                fold_torgo = limit_dataset(
                    select_torgo_speaker(
                        torgo, selected_speaker, args.speaker_column
                    ),
                    args.max_samples,
                )
                torgo_selection = selected_speaker
            print(f"  TORGO {torgo_selection}: {len(fold_torgo)} samples")
            model_result["torgo"] = evaluate_dataset(
                model,
                processor,
                fold_torgo,
                model_output_dir,
                "torgo",
                args.batch_size,
                args.num_workers,
                args.generation_max_length,
                args.save_predictions,
            )

        if "librispeech" in pending:
            assert librispeech is not None
            print(f"  LibriSpeech retention: {len(librispeech)} samples")
            model_result["librispeech"] = evaluate_dataset(
                model,
                processor,
                librispeech,
                model_output_dir,
                "librispeech",
                args.batch_size,
                args.num_workers,
                args.generation_max_length,
                args.save_predictions,
            )

        report["models"][result_key] = model_result
        refresh_averages(report)
        write_json(report_path, report)
        summary_path.write_text(markdown_summary(report), encoding="utf-8")

        del model, processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    refresh_averages(report)
    write_json(report_path, report)
    summary_path.write_text(markdown_summary(report), encoding="utf-8")
    print(f"Saved aggregate JSON: {report_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
