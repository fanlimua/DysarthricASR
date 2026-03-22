import argparse
import glob
import os
import re
import csv
import itertools
import json
import evaluate
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple
from datasets import Audio, Dataset, DatasetDict, load_dataset, load_from_disk
from transformers import pipeline, WhisperProcessor
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from util.data_split import split_dataset
from peft import PeftModel
from transformers import WhisperForConditionalGeneration

AUDIO_EXTENSIONS = ("*.wav", "*.flac", "*.mp3", "*.ogg", "*.m4a")

TORGO_SPEAKERS = {
    "F01",
    "F03",
    "F04",
    "FC01",
    "FC02",
    "FC03",
    "M01",
    "M02",
    "M03",
    "M04",
    "M05",
    "MC01",
    "MC02",
    "MC03",
    "MC04",
}


def remote_filesystem_fixed(fs) -> bool:
    protocol = getattr(fs, "protocol", None)
    if isinstance(protocol, (tuple, list, set)):
        return "file" not in protocol
    return protocol != "file"


def fix_datasets_fs() -> None:
    import datasets.arrow_dataset as ds_arrow
    import datasets.builder as ds_builder
    import datasets.dataset_dict as ds_dict
    import datasets.filesystems as ds_fs
    import datasets.info as ds_info

    ds_fs.is_remote_filesystem = remote_filesystem_fixed
    ds_builder.is_remote_filesystem = remote_filesystem_fixed
    ds_arrow.is_remote_filesystem = remote_filesystem_fixed
    ds_dict.is_remote_filesystem = remote_filesystem_fixed
    ds_info.is_remote_filesystem = remote_filesystem_fixed
    

def dedup_per_speaker(
    dataset: Dataset,
    speaker_column: str,
    text_column: str,
    short_word_max_words: int,
) -> List[int]:

    # Per-speaker dedup: 
    # among short utterances, keep unique normalized text
    # among sentences, keep unique normalized text
    by_speaker: Dict[str, List[Tuple[int, str, int]]] = defaultdict(list)
    for idx in range(len(dataset)):
        row = dataset[int(idx)]
        trans = (row.get(text_column) or "").strip()
        spk = str(row.get(speaker_column) or "")
        norm = " ".join(trans.lower().split()) if trans else ""
        wc = len(trans.split()) if trans else 0
        by_speaker[spk].append((idx, norm, wc))

    keep: List[int] = []
    for spk in sorted(by_speaker.keys()):
        short_seen: set = set()
        sent_seen: set = set()
        for idx, norm, wc in by_speaker[spk]:
            if not norm:
                continue
            if wc <= short_word_max_words:
                if norm in short_seen:
                    continue
                short_seen.add(norm)
            else:
                if norm in sent_seen:
                    continue
                sent_seen.add(norm)
            keep.append(idx)
    return keep


def append_speaker(example: Dict[str, Dict[str, str]]) -> Dict[str, str]:
    audio = example.get("audio", {})
    path = audio.get("path")

    # derive speaker from audio path
    basename = os.path.basename(path)
    match = re.match(r"^([A-Z]C?\d{2})_", basename)
    speaker = match.group(1) if match else basename.split("_")[0]
    if speaker not in TORGO_SPEAKERS:
        raise ValueError(f"Unknown speaker derived from path: {speaker} ({basename})")

    # append speaker column 
    return {"speaker": speaker}


def load_custom_audio_dataset(audio_dir: str) -> Dataset:
   # Build a Dataset from a directory of audio files.
    audio_dir = os.path.abspath(audio_dir)

    paths = []
    for ext in AUDIO_EXTENSIONS:
        paths.extend(glob.glob(os.path.join(audio_dir, "**", ext), recursive=True))
    paths = sorted(paths)

    dataset = Dataset.from_dict({
        "audio": [{"path": p} for p in paths],
        "audio_path": [os.path.relpath(p, audio_dir) for p in paths],
    })
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    return dataset

def inference_custom_audio(
    dataset: Dataset,
    model_name: str,
    batch_size: int,
    max_new_tokens: int,
    language: str,
    task: str,
    output_path: str,
    load_lora: bool = False,
) -> None:

    device = 0 if __import__("torch").cuda.is_available() else -1
    model_path = os.path.abspath(model_name)
    is_checkpoint_subdir = os.path.isdir(model_path) and "checkpoint-" in os.path.basename(model_path)
    # Load processor and ASR pipeline from checkpoint
    if is_checkpoint_subdir:
        processor_path = "openai/whisper-small"
        processor = WhisperProcessor.from_pretrained(processor_path)
        if load_lora:
            # Load LoRA adapter from checkpoint
            adapter_config_path = os.path.join(model_path, "adapter_config.json")
            with open(adapter_config_path, "r") as f:
                adapter_cfg = json.load(f)
            base_name = adapter_cfg.get("base_model_name_or_path") 
            base_model = WhisperForConditionalGeneration.from_pretrained(base_name)
            model = PeftModel.from_pretrained(base_model, model_path)
            asr = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                device=device,
            )
            print(f"Loading ASR (LoRA): base={base_name}, adapter={model_path}")
        else:
            asr = pipeline(
                "automatic-speech-recognition",
                model=model_path,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                device=device,
            )
            print(asr.model.name_or_path)
    else:
        # Load ASR pipeline from model_name
        asr = pipeline("automatic-speech-recognition", model=model_name, device=device)

    generate_kwargs = {"max_new_tokens": max_new_tokens}
    is_english_only = ".en" in model_name.lower() or model_name.lower().endswith("-en")
    if not is_english_only:
        if language and language.lower() != "auto":
            generate_kwargs["language"] = language
        if task:
            generate_kwargs["task"] = task

    def _predict(batch: Dict[str, List[Dict[str, np.ndarray]]]) -> Dict[str, List[str]]:
        audio_arrays = [item["array"] for item in batch["audio"]]
        outputs = asr(audio_arrays, batch_size=batch_size, generate_kwargs=generate_kwargs)
        if isinstance(outputs, dict):
            outputs = [outputs]
        return {"prediction": [out["text"] for out in outputs]}

    with_predictions = dataset.map(_predict, batched=True, batch_size=batch_size)
    out = [
        {"audio_path": p, "prediction": str(pred) if not isinstance(pred, str) else pred}
        for p, pred in zip(with_predictions["audio_path"], with_predictions["prediction"])
    ]
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("Saved %d predictions to: %s" % (len(out), output_path))

def inference(
    dataset: Dataset,
    model_name: str,
    batch_size: int,
    max_new_tokens: int,
    language: str,
    task: str,
    prediction_path: str,
    output_dir: str,
    load_lora: bool = False,
) -> None:
    device = 0 if __import__("torch").cuda.is_available() else -1
    print(device)

    # Load checkpoint for inference
    model_path = os.path.abspath(model_name)
    is_checkpoint_subdir = os.path.isdir(model_path) and "checkpoint-" in os.path.basename(model_path)
   
    if is_checkpoint_subdir:
        # Load processor and ASR pipeline from checkpoint
        processor_path = "openai/whisper-small"
        processor = WhisperProcessor.from_pretrained(processor_path)
        if load_lora:
            # Load LoRA adapter from checkpoint
            adapter_config_path = os.path.join(model_path, "adapter_config.json")
            with open(adapter_config_path, "r") as f:
                adapter_cfg = json.load(f)
            # lora wights only stored in adapter_config.json
            base_name = adapter_cfg.get("base_model_name_or_path") 
            base_model = WhisperForConditionalGeneration.from_pretrained(base_name)
            model = PeftModel.from_pretrained(base_model, model_path)
            asr = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                device=device,
            )
            print(f"Loading ASR model (LoRA): base={base_name}, adapter={model_path}")
        else:
            # Load full model from checkpoint
            asr = pipeline(
                "automatic-speech-recognition",
                model=model_path,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                device=device,
            )
            print(f"Loading ASR model from: {model_path}")
    else:
        # Load ASR pipeline from model_name
        asr = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            device=device,
        )

    generate_kwargs = {"max_new_tokens": max_new_tokens}
    # English-only models (whisper-small.en) do not support task or language
    is_english_only = ".en" in model_name.lower() or model_name.lower().endswith("-en")
    if not is_english_only:
        if language and language.lower() != "auto":
            generate_kwargs["language"] = language
        if task:
            generate_kwargs["task"] = task

    # predict transcription for each batch
    def _predict(batch: Dict[str, List[Dict[str, np.ndarray]]]) -> Dict[str, List[str]]:
        audio_arrays = [item["array"] for item in batch["audio"]]
        outputs = asr(
            audio_arrays,
            batch_size=batch_size,
            generate_kwargs=generate_kwargs,
        )
        if isinstance(outputs, dict):
            outputs = [outputs]
        texts = [out["text"] for out in outputs]
        return {"prediction": texts}

    # map _predict to the dataset
    with_predictions = dataset.map(_predict, batched=True, batch_size=batch_size)
    # json_ready = with_predictions.select_columns(["speaker", "transcription", "prediction"])
    # json_ready.to_json(prediction_path)
    # print(f"Saved test predictions to: {prediction_path}")

    normalizer = BasicTextNormalizer()
    def add_normalized(example):
        return {
            "trans_normalized": normalizer(example["transcription"]),
            "pred_normalized": normalizer(example["prediction"]),
        }

    with_predictions = with_predictions.map(add_normalized)
    json_ready = with_predictions.select_columns([
        "speaker", "transcription", "prediction",
        "pred_normalized", "trans_normalized",
    ])
    json_ready.to_json(prediction_path)
    print(f"Saved test predictions to: {prediction_path}")

    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")
    speakers = with_predictions["speaker"]
    preds = with_predictions["prediction"]
    refs = with_predictions["transcription"]
    normalizer = BasicTextNormalizer()

    by_speaker = defaultdict(lambda: {"preds": [], "refs": []})
    for spk, pred, ref in zip(speakers, preds, refs):
        by_speaker[str(spk)]["preds"].append(normalizer(pred))
        by_speaker[str(spk)]["refs"].append(normalizer(ref))

    # compute WER for each speaker
    wer_results = {}

    for spk in sorted(by_speaker.keys()):
        spk_data = by_speaker[spk]
        wer_results[spk] = 100 * wer_metric.compute(
            predictions=spk_data["preds"],
            references=spk_data["refs"],
        )
    # compute overall WER, flatten outer
    wer_results["overall"] = 100 * wer_metric.compute(
        predictions=[normalizer(p) for p in preds],
        references=[normalizer(r) for r in refs],
    )

    # compute CER for each speaker
    cer_results = {}
    for spk in sorted(by_speaker.keys()):
        spk_data = by_speaker[spk]
        cer_results[spk] = 100 * cer_metric.compute(
            predictions=spk_data["preds"],
            references=spk_data["refs"],
        )

    # compute overall CER
    cer_results["overall"] = 100 * cer_metric.compute(
        predictions=[normalizer(p) for p in preds],
        references=[normalizer(r) for r in refs],
    )

    os.makedirs(output_dir, exist_ok=True)
    wer_path = os.path.join(output_dir, "wer.json")
    cer_path = os.path.join(output_dir, "cer.json")

    with open(wer_path, "w", encoding="utf-8") as f:
        json.dump(wer_results, f, ensure_ascii=True, indent=2)
    print(f"Saved WER results to: {wer_path}")

    with open(cer_path, "w", encoding="utf-8") as f:
        json.dump(cer_results, f, ensure_ascii=True, indent=2)
    print(f"Saved CER results to: {cer_path}")


def save_speaker_counts_table(dataset_dict: DatasetDict, output_path: str) -> None:
    speaker_set = set()
    counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"train": 0, "validation": 0, "test": 0})

    for split in ["train", "validation", "test"]:
        for spk in dataset_dict[split]["speaker"]:
            spk = str(spk)
            speaker_set.add(spk)
            counts[spk][split] += 1

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["speaker", "train", "validation", "test", "total"])
        for spk in sorted(speaker_set):
            train_c = counts[spk]["train"]
            val_c = counts[spk]["validation"]
            test_c = counts[spk]["test"]
            writer.writerow([spk, train_c, val_c, test_c, train_c + val_c + test_c])
    print(f"Saved speaker counts table to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference on TOGOR dataset.")
    # split datasets settings
    parser.add_argument("--data_source", choices=("torgo", "custom"), default="torgo", help="torgo: TORGO dataset, custom: custom audio directory.")
    parser.add_argument("--audio_dir", type=str, default="/home/fan/project/whisper/data/audio_data/mp3", help="Directory of audio files.")
    parser.add_argument("--speaker_counts_path", default="torgo_results/speaker_counts.csv")
    parser.add_argument("--speaker_column", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_method", choices=("ratio", "loso"), default="loso", help="Dataset split method.")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--loso_test_speaker", type=str, default="M01", help="Speaker ID for test set.")
    parser.add_argument("--loso_val_speaker", type=str, default="M05", help="Speaker ID for validation set.")
    parser.add_argument("--short_word_max_words", type=int, default=2, help="The length of utterances.")
    parser.add_argument("--dedup", action="store_true", help="Enable per-speaker deduplication.")
    # inference settings
    parser.add_argument("--model_name", default="openai/whisper-small", help="HuggingFace model or path to local checkpoint")
    parser.add_argument("--checkpoint_dir", default="results/train/augment_lora_r32/checkpoint-39000", help="Load checkpoint for inference.")
    parser.add_argument("--language", default="en", help="Whisper language code")
    parser.add_argument("--task", default="transcribe")
    parser.add_argument("--max_new_tokens", type=int, default=225)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--run_inference", action="store_true")
    parser.add_argument("--load_lora", action="store_true", help="Load checkpoint as LoRA adapter (base model + adapter from checkpoint dir).")
    # output settings
    parser.add_argument("--predictions_path", default="results/train/augment_lora_r32/val/test_predictions.json")
    parser.add_argument("--output_dir", default="results/train/augment_lora_r32/val")
    parser.add_argument("--split_indices", type=str, default=None, help="Save train/val/test indices.")
    args = parser.parse_args()

    if args.data_source == "custom":
        dataset = load_custom_audio_dataset(args.audio_dir)
        print("Custom audio: %d files." % (len(dataset)))
        if args.run_inference:
            model_for_inference = args.checkpoint_dir if args.checkpoint_dir else args.model_name
            inference_custom_audio(
                dataset=dataset,
                model_name=model_for_inference,
                batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
                language=args.language,
                task=args.task,
                output_path=args.predictions_path,
                load_lora=args.load_lora,
            )
        return

    ratios = (args.train_ratio, args.val_ratio, args.test_ratio)
    dataset_with_speaker = "/home/fan/project/dataset/Huggingface_TORGO"
    # dataset_with_speaker = "/media/justin/SSD Ubuntu Stora/datasets/TORGO"
    
    if os.path.exists(dataset_with_speaker):
        dataset = load_from_disk(dataset_with_speaker)
    else:
        fix_datasets_fs()
        dataset_dict = load_dataset("abnerh/TORGO-database")
        dataset = dataset_dict["train"].cast_column("audio", Audio(sampling_rate=16000))
        
        # TORGO dataset from huggingface doesn't provide a speaker column, 
        # derive it from audio.path and append speaker column to the dataset
        dataset = dataset.map(append_speaker)
        dataset.save_to_disk(dataset_with_speaker)
    
    speaker_column = "speaker"
    text_column = "transcription"
    # Per-speaker dedup: unique short phrases and unique sentences
    if args.dedup:
        orig_size = len(dataset)
        dedup_indices = dedup_per_speaker(
            dataset=dataset,
            speaker_column=speaker_column,
            text_column=text_column,
            short_word_max_words=args.short_word_max_words,
        )
        dataset = dataset.select(dedup_indices)
        print("After deduplication: %d -> %d samples" % (orig_size, len(dataset)))

    # dataset splitting
    dataset_dict, split_indices = split_dataset(
        dataset=dataset,
        method=args.split_method,
        speaker_column=speaker_column,
        seed=args.seed,
        ratios=ratios,
        loso_test_speaker=args.loso_test_speaker,
        loso_val_speaker=args.loso_val_speaker,
    )

    if args.split_indices:
        os.makedirs(os.path.dirname(args.split_indices) or ".", exist_ok=True)
        with open(args.split_indices, "w", encoding="utf-8") as f:
            json.dump(split_indices, f, indent=2)
        print(f"Saved split indices to {args.split_indices}")

    # dataset_dict.save_to_disk(args.output_dir)
    # save_speaker_counts_table(dataset_dict, args.speaker_counts_path)
    print(dataset_dict)

    if args.run_inference:
        # Use checkpoint_dir if provided, else model_name
        model_for_inference = args.checkpoint_dir if args.checkpoint_dir else args.model_name
        inference(
            dataset=dataset_dict["test"],
            # dataset=dataset_dict["validation"],
            model_name=model_for_inference,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            language=args.language,
            task=args.task,
            prediction_path=args.predictions_path,
            output_dir=args.output_dir,
            load_lora=args.load_lora,
        )


if __name__ == "__main__":
    main()

# python main.py --checkpoint_dir results/whisper-torgo/checkpoint-25000 --data_source custom --audio_dir /home/fan/project/whisper/data/audio_data/mp3 --run_inference --predictions_path results/kirk/predictions.json
# python main.py --checkpoint_dir results/whisper-torgo/checkpoint-25000 --data_source torgo --run_inference --predictions_path results/whisper-torgo/test/predictions.json --output_dir results/whisper-torgo/test