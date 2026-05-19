import os
import json
import torch
import torch.nn as nn
import numpy as np 
import evaluate
import argparse
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import Audio, Dataset, DatasetDict, load_dataset, load_from_disk
from transformers import (
    AutoModelForSpeechSeq2Seq,
    GenerationConfig,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from util.augment import AugmentedDataset, SetEpochCallback, list_rir_paths
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift, RoomSimulator, AddGaussianSNR, TimeMask

warnings.filterwarnings("ignore", category=FutureWarning)

torch.autograd.set_detect_anomaly(True)
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

@dataclass
# Pads audio features and label sequences so each batch has the same shape
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": self.processor.feature_extractor(f["input_features"],
                                                sampling_rate=16000, return_tensors="pt")["input_features"].squeeze() } for f in features]
        # print(input_features[0]["input_features"])
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        # print(features[0]["labels"])
        labels = [{"input_ids": f["labels"]["input_ids"], "attention_mask": f["labels"]["attention_mask"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(labels, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch["attention_mask"].ne(1), -100)

        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def compute_metrics(pred, tokenizer):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    normalizer = BasicTextNormalizer()
    wer = 100 * wer_metric.compute(
        predictions=[normalizer(p) for p in pred_str], 
        references=[normalizer(l) for l in label_str],
    )
    cer = 100 * cer_metric.compute(
        predictions=[normalizer(p) for p in pred_str], 
        references=[normalizer(l) for l in label_str],
    )
    return {"wer": wer, "cer": cer }

def local_channel_shuffle(x, window_size=16):
    """
    Randomly shuffle channels locally within fixed windows.

    Args:
        x (torch.Tensor):
            Input tensor of shape [B, C, T]
        window_size (int):
            Number of neighboring channels to shuffle together.

    Returns:
        torch.Tensor:
            Tensor with locally shuffled channels, shape [B, C, T]
    """
    B, C, T = x.shape
    device = x.device

    # Output tensor
    out = x.clone()

    # Process channel windows
    for start in range(0, C, window_size):
        end = min(start + window_size, C)
        curr_window = end - start

        # Independent random permutation per batch
        perms = torch.stack([
            torch.randperm(curr_window, device=device)
            for _ in range(B)
        ])  # [B, curr_window]

        # Convert local indices -> global channel indices
        perms = perms + start  # [B, curr_window]

        # Batch indexing
        batch_idx = torch.arange(B, device=device).unsqueeze(1)

        # Apply shuffle
        out[:, start:end, :] = x[batch_idx, perms, :]

    return out

def random_mask(x, max_mask_ratio=0.25):
    """
    Efficiently masks a sequential block of channels in a [B, C, T] tensor.
    All batch samples share the same mask length, but the start index is random per sample.

    Args:
        x (torch.Tensor): Input tensor of shape [B, C, T].
        max_mask_ratio (float): Maximum fraction of channels to mask.

    Returns:
        torch.Tensor: Tensor with sequential channels masked.
    """
    B, C, T = x.shape
    masked_x = x.clone()

    # Mask length (same for all batch samples)
    mask_len = torch.randint(1, max(1, int(C * max_mask_ratio)) + 1, (1,)).item()

    # Random start indices per batch
    start_indices = torch.randint(0, C - mask_len + 1, (B,), device=x.device)

    # Create a mask of shape [B, C] initialized to 1
    mask = torch.ones(B, C, device=x.device)

    # Generate a tensor of channel indices [C]
    channel_indices = torch.arange(C, device=x.device).unsqueeze(0)  # [1, C]

    # Broadcast start and length to [B, 1]
    start_indices_broadcast = start_indices.unsqueeze(1)  # [B, 1]

    # Mask sequential channels
    mask *= ~((channel_indices >= start_indices_broadcast) & 
              (channel_indices < start_indices_broadcast + mask_len))

    # Expand mask to [B, C, T] and apply
    masked_x = masked_x * mask.unsqueeze(-1)
    return masked_x

class SudoTrainer(Seq2SeqTrainer):
    def training_step(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        num_items_in_batch: torch.Tensor | int | None = None,
    ) -> torch.Tensor:
        try:
            inputs["input_features"] = local_channel_shuffle(inputs["input_features"])
            val = super().training_step(model, inputs, num_items_in_batch)
            # print(val)
            # assert(False)
            return val
        except RuntimeError:
            device = next(iter(inputs.values())).device
            return torch.tensor(0.0).to(device=device)

def run_training(
    processor: WhisperProcessor, 
    train_ds: Dataset,
    val_ds: Dataset,
    model_name: str = "openai/whisper-small",
    output_dir: str = "models/whisper-torgo",
    language: str = "en",
    task: str = "transcribe",
    batch_size: int = 8,
    learning_rate: float = 1e-5,
    num_train_epochs: int = 3,
    use_lora: bool = False,
    **kwargs,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    max_new_tokens = kwargs.get("max_new_tokens", 225)
    num_proc = kwargs.get("num_proc", 4)
    text_column = kwargs.get("text_column", "transcription")

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    print(f"Loading model {model_name}...")

    # Training settings
    save_steps = kwargs.get("save_steps", 500)
    eval_steps = kwargs.get("eval_steps", 500)
    logging_steps = kwargs.get("logging_steps", 50)
    gradient_accumulation_steps = kwargs.get("gradient_accumulation_steps", 1)
    warmup_ratio = kwargs.get("warmup_ratio", 0.1)
    fp16 = kwargs.get("fp16", True)
    save_total_limit = kwargs.get("save_total_limit", 2) 

    distill_whisper = False
    if "distil" in model_name:
        distill_whisper = True 

    # WhisperProcessor includes the feature extractor (audio -> log-Mel) and the tokenizer (text <-> ids).
    # model = WhisperForConditionalGeneration.from_pretrained(
    #     model_name,
    #     torch_dtype=torch.float32,
    #     device_map="cuda" if torch.cuda.is_available() else "cpu",
    # )
    # model.config.forced_decoder_ids = None
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model_id = model_name
    if distill_whisper:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, low_cpu_mem_usage=False, use_safetensors=True
        )
        model.to(device).float()
    else:
        model = WhisperForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="cuda" if torch.cuda.is_available() else "cpu",
        )
        model.config.forced_decoder_ids = None
        if getattr(model, "generation_config", None) is not None:
            model.generation_config.language = language
            model.generation_config.task = task

    # model.config.apply_spec_augment = True
    print(model)

    # processor = WhisperProcessor.from_pretrained("distil-whisper/distil-small.en")
    # processor = AutoProcessor.from_pretrained(model_id)
    # Force single language for evaluation (avoid "Multiple languages detected" in batch)
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.language = language
        model.generation_config.task = task

    if use_lora:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=kwargs.get("lora_r", 16),
            lora_alpha=kwargs.get("lora_alpha", 32),
            target_modules=["q_proj", "k_proj"],
            lora_dropout=kwargs.get("lora_dropout", 0.05),
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    use_augmentation = kwargs.get("use_augmentation", False)
    augment_seed = kwargs.get("seed", 42)
    augment_snr_db_range = kwargs.get("augment_snr_db_range", (5.0, 20.0))
    augment_rir_dir = kwargs.get("augment_rir_dir") or ""
    
    # Padding: ensures all audio features and token sequences in a batch have equal length
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    ) 

    if distill_whisper:
        for n, p in model.named_parameters():
            if "decoder" in n:
                p.requires_grad_(True)
            else:
                p.requires_grad_(False)
        # print(n)
        # print(p.size())

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_steps=warmup_ratio, # Float for ratio, int for steps 
        num_train_epochs=num_train_epochs,
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        eval_steps=eval_steps,
        logging_steps=logging_steps,
        fp16=fp16,
        predict_with_generate=True,
        generation_max_length=max_new_tokens,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
        remove_unused_columns=False,
        label_names=["labels"],
        eval_on_start=False,
        # resume_from_checkpoint="results/train/new_ds4/checkpoint-27000",
    )

    callbacks = [SetEpochCallback()] if use_augmentation else []
    trainer = SudoTrainer(
        args=training_args,
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=lambda pred: compute_metrics(pred, processor.tokenizer),
        processing_class=processor,
        # tokenizer=processor.feature_extractor,
        callbacks=callbacks,
    )
    # Training
    trainer.train(resume_from_checkpoint=False)
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    print(f"Model and processor saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    # model settings
    parser.add_argument("--model_name", type=str, default="openai/whisper-small", choices=["openai/whisper-small", "distil-whisper/distil-large-v3"])
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--task", type=str, default="transcribe")
    # training settings
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--max_new_tokens", type=int, default=225)
    parser.add_argument("--num_proc", type=int, default=2)
    parser.add_argument("--num_train_epochs", type=int, default=30)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--logging_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=2, help="Max number of checkpoint dirs to keep.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--fp16", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--dataset_path", type=str, default=None, help="Path to TORGO dataset on disk")
    # split datasets settings
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_method", choices=("ratio", "loso"), default="loso", help="Dataset split method.")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--loso_test_speaker", type=str, default="M01", help="Speaker ID for test set.")
    parser.add_argument("--loso_val_speaker", type=str, default="M05", help="Speaker ID for validation set.")
    parser.add_argument("--short_word_max_words", type=int, default=2, help="The length of utterances.")
    parser.add_argument("--dedup", action="store_true", help="Enable per-speaker deduplication.")
    parser.add_argument("--use_augmentation", action="store_true", help="Apply audio augmentation on train.")
    parser.add_argument("--augment_snr_db_min", type=float, default=5.0, help="Min SNR (dB) for noise augmentation.")
    parser.add_argument("--augment_snr_db_max", type=float, default=20.0, help="Max SNR (dB) for noise augmentation.")
    parser.add_argument("--augment_rir_dir", type=str, default="data/RIR/RIRS_NOISES/real_rirs_isotropic_noises", help="Directory containing real RIR files.")
    # output settings
    parser.add_argument("--output_dir", type=str, default="results/train/new_ds4")
    parser.add_argument("--split_indices", type=str, default="results/data_split/split_indices.json", help="Save train/val/test indices.")
    args = parser.parse_args()

    # load TORGO dataset 
    # dataset_path = args.dataset_path or os.environ.get("DATASET_PATH", "/home/fan/project/dataset/Huggingface_TORGO")
    # ratios = (args.train_ratio, args.val_ratio, args.test_ratio)

    dataset = load_dataset("extraordinarylab/torgo")["test"]
    if len(dataset.cache_files) > 32: 
        dataset.cleanup_cache_files() # Only cleans up parquet and not downloads 

    dataset = dataset.map(lambda x: {"length": x["audio"].get_all_samples().duration_seconds }, num_proc=4)
    dataset = dataset.filter(lambda x: x["length"] <= 30, num_proc=4) # Manually confirmed the 2 samples above this are garbage

    # First experiment done with language unset in DistilWhisper 
    processor = WhisperProcessor.from_pretrained(args.model_name, language=args.language, task=args.task)
    # dataset Deduplication 
    speaker_column = "speaker"
    text_column = "text"
    if args.dedup:
        assert(False)

    val_speaker = args.loso_val_speaker
    test_speaker = args.loso_test_speaker 
    train_ds = dataset.filter(lambda x: x[speaker_column] != val_speaker and x[speaker_column] != test_speaker, num_proc=4)
    val_ds = dataset.filter(lambda x: x[speaker_column] == val_speaker, num_proc=4)
    tokenizer = processor.tokenizer 
    def convert(ds, num_shards=32, iterable=True):
        if iterable:
            ds = ds.to_iterable_dataset(num_shards=num_shards)
        return ds.map(
            lambda x: {"input_features": x["audio"].get_all_samples().data.squeeze(), "labels": tokenizer(text=x[text_column]) }).remove_columns(
                ["audio", "speaker", "speech_status", "microphone", "length"])
    
    train = convert(train_ds, iterable=False)

    transform_fn = Compose([
        AddGaussianSNR(min_snr_db=args.augment_snr_db_min, max_snr_db=args.augment_snr_db_max, p=0.5),
        # AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
        # RoomSimulator(),
        TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
        PitchShift(min_semitones=-2, max_semitones=2, p=0.5),
        TimeMask()
        # Shift(p=0.5, shift_unit="seconds"),
    ])

    def transform(batch):
        try:
            batch["input_features"] = [transform_fn(x.numpy(), sample_rate=16000) for x in batch["input_features"]]
        except Exception:
            pass
        return batch

    if args.use_augmentation:
        train.set_transform(transform=transform)
    val = convert(val_ds, num_shards=16, iterable=False)
    
    # training
    run_training(
        processor=processor,
        train_ds=train,
        val_ds=val,
        model_name=args.model_name,
        output_dir=args.output_dir,
        task=args.task,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        use_lora=args.use_lora,
        max_new_tokens=args.max_new_tokens,
        num_proc=args.num_proc,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        fp16=args.fp16,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_augmentation=args.use_augmentation,
        augment_snr_db_range=(args.augment_snr_db_min, args.augment_snr_db_max),
        augment_rir_dir=args.augment_rir_dir or None,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
