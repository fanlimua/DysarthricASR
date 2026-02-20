import os
import json
import torch
import evaluate
import argparse
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import Audio, Dataset, DatasetDict, load_dataset, load_from_disk
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from main import split_per_speaker, append_speaker, fix_datasets_fs
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
warnings.filterwarnings("ignore", category=FutureWarning)

@dataclass
# Pads audio features and label sequences so each batch has the same shape
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        labels = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(labels, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def prepare_dataset(examples: Dict[str, Any], feature_extractor, tokenizer, text_column: str = "transcription") -> Dict[str, Any]:
    # Convert audio to input_features and text to label ids
    audios = examples["audio"]
    texts = examples[text_column]
    input_features = []
    labels_list = []
    for audio, text in zip(audios, texts):
        input_features.append(
            feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        )
        labels_list.append(tokenizer(text).input_ids)
    return {"input_features": input_features, "labels": labels_list}


def compute_metrics(pred, tokenizer):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    wer_metric = evaluate.load("wer")
    normalizer = BasicTextNormalizer()
    wer = 100 * wer_metric.compute(
        predictions=[normalizer(p) for p in pred_str], 
        references=[normalizer(l) for l in label_str],
    )
    return {"wer": wer}


def run_training(
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

    # WhisperProcessor includes the feature extractor (audio -> log-Mel) and the tokenizer (text <-> ids).
    processor = WhisperProcessor.from_pretrained(model_name, language=language, task=task)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
    )
    model.config.forced_decoder_ids = None

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

    def _map_fn(examples):
        return prepare_dataset(examples, processor.feature_extractor, processor.tokenizer, text_column)

    train_ds = train_ds.map(_map_fn, batched=True, remove_columns=train_ds.column_names, num_proc=num_proc)
    val_ds = val_ds.map(_map_fn, batched=True, remove_columns=val_ds.column_names, num_proc=num_proc)

    # Padding: ensures all audio features and token sequences in a batch have equal length
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # Training settings
    save_steps = kwargs.get("save_steps", 500)
    eval_steps = kwargs.get("eval_steps", 500)
    logging_steps = kwargs.get("logging_steps", 50)
    gradient_accumulation_steps = kwargs.get("gradient_accumulation_steps", 1)
    warmup_ratio = kwargs.get("warmup_ratio", 0.1)
    fp16 = kwargs.get("fp16", True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        num_train_epochs=num_train_epochs,
        evaluation_strategy="steps",
        save_steps=save_steps,
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
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=lambda pred: compute_metrics(pred, processor.tokenizer),
        tokenizer=processor.feature_extractor,
    )

    # Training
    trainer.train()
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    print(f"Model and processor saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    # model settings
    parser.add_argument("--model_name", type=str, default="openai/whisper-small")
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--task", type=str, default="transcribe")
    # training settings
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_train_epochs", type=int, default=30)
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=225)
    parser.add_argument("--num_proc", type=int, default=2)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--logging_steps", type=int, default=500)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--fp16", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--dataset_path", type=str, default=None, help="Path to TORGO dataset on disk")
    # split datasets settings
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    # output settings
    parser.add_argument("--output_dir", type=str, default="results/whisper-torgo")
    parser.add_argument("--split_indices", type=str, default="results/whisper-torgo/split_indices.json", help="Save train/val/test indices.")
    args = parser.parse_args()

    # load TORGO dataset and split into train, validation, and test sets
    dataset_path = args.dataset_path or os.environ.get("DATASET_PATH", "/home/fan/project/dataset/Huggingface_TORGO")
    ratios = (args.train_ratio, args.val_ratio, args.test_ratio)

    if os.path.exists(dataset_path):
        dataset = load_from_disk(dataset_path)
    else:
        fix_datasets_fs()
        hf_dict = load_dataset("abnerh/TORGO-database")
        dataset = hf_dict["train"].cast_column("audio", Audio(sampling_rate=16000))
        dataset = dataset.map(append_speaker)
        os.makedirs(os.path.dirname(dataset_path) or ".", exist_ok=True)
        dataset.save_to_disk(dataset_path)

    dataset_dict, split_indices = split_per_speaker(
        dataset=dataset,
        speaker_column="speaker",
        seed=args.seed,
        ratios=ratios,
    )
    train_ds = dataset_dict["train"]
    # train_ds = dataset_dict["test"]
    val_ds = dataset_dict["validation"]

    # save split indices
    if args.split_indices:
        os.makedirs(os.path.dirname(args.split_indices) or ".", exist_ok=True)
        with open(args.split_indices, "w", encoding="utf-8") as f:
            json.dump(split_indices, f, indent=2)
        print(f"Saved split indices to {args.split_indices}")
    
    run_training(
        train_ds=train_ds,
        val_ds=val_ds,
        model_name=args.model_name,
        output_dir=args.output_dir,
        language=args.language,
        task=args.task,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        use_lora=args.use_lora,
        max_new_tokens=args.max_new_tokens,
        num_proc=args.num_proc,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        fp16=args.fp16,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )


if __name__ == "__main__":
    main()
