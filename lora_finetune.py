import os
import torch
import argparse
import warnings
from datasets import load_dataset, Audio
from transformers import (
    WhisperForConditionalGeneration,
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
import evaluate
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Union
warnings.filterwarnings("ignore", category=FutureWarning)
from peft import LoraConfig, get_peft_model

# Data Collator
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Audio features
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Labels
        labels = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(labels, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        # print(batch.keys())
        return batch

# Preprocessing function
class PrepareDataset:
    def __init__(self, feature_extractor, tokenizer):
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer

    def __call__(self, batch):
        audio = batch["audio"]
        # Compute log-Mel spectrogram features
        batch["input_features"] = self.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        batch["labels"] = self.tokenizer(batch["sentence"]).input_ids
        return batch

# Evaluation metric
def compute_metrics(pred, tokenizer):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer_metric = evaluate.load("wer")
    wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

def show_forced_decoder_ids(model, tokenizer):
    forced_ids = model.config.forced_decoder_ids
    print(f"Raw value: {forced_ids}")

    if forced_ids is None:
        print("forced_decoder_ids is None")
        return

    print("\nDecoded tokens:")
    for position, token_id in forced_ids:
        token_str = tokenizer.decode([token_id])
        print(f"position={position}, token_id={token_id}, token='{token_str}'")

def main():
    parser = argparse.ArgumentParser()

    # Model setups
    parser.add_argument("--model_name", type=str, default="openai/whisper-small")
    parser.add_argument("--language", type=str, default="Portuguese")
    parser.add_argument("--task", type=str, default="transcribe")
    parser.add_argument("--max_new_tokens", default=225, type=int)
    # LoRA setups
    parser.add_argument("--r", default=16, type=int, help="LoRA rank, number of trainable low-rank dimensions.")
    parser.add_argument("--lora_alpha", default=32, type=int, help="LoRA scaling factor")
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    # Finetuning setups
    parser.add_argument("--num_proc", default=1, type=int)
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--save_steps", default=500, type=int)
    parser.add_argument("--eval_steps", default=500, type=int)
    parser.add_argument("--logging_steps", default=50, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=2, type=int)
    parser.add_argument("--output_dir", type=str, default="./whisper-lora")
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    # dataset_train = load_dataset("fsicoli/common_voice_15_0", "pt", split="train[:10]")
    # dataset_test = load_dataset("fsicoli/common_voice_15_0", "pt", split="test[:10]")
    dataset_train = load_dataset("fsicoli/common_voice_15_0", "pt", split="train")
    dataset_test = load_dataset("fsicoli/common_voice_15_0", "pt", split="test")

    dataset_train = dataset_train.cast_column("audio", Audio(sampling_rate=16000))
    dataset_test = dataset_test.cast_column("audio", Audio(sampling_rate=16000))

    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.model_name)
    tokenizer = WhisperTokenizer.from_pretrained(args.model_name, language=args.language, task=args.task)
    processor = WhisperProcessor.from_pretrained(args.model_name, language=args.language, task=args.task)

    # Load base Whisper
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=torch.float32,
        device_map=device,
    )
    # show_forced_decoder_ids(model, tokenizer)
    model.config.forced_decoder_ids = None
    # show_forced_decoder_ids(model, tokenizer)

    # LoRA injection
    lora_config = LoraConfig(
        r=args.r,   # r controls how many new trainable parameters LoRA adds.
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj"],
        lora_dropout=args.lora_dropout,
        bias="none"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Preprocessing dataset
    print("\nPreprocessing traning data")
    prep_dataset = PrepareDataset(feature_extractor, tokenizer)
    dataset_train = dataset_train.map(
        prep_dataset,
        remove_columns=dataset_train.column_names,
        num_proc=args.num_proc,
    )

    # Process test set
    print("\nPreprocessing testing data")
    dataset_test = dataset_test.map(
        prep_dataset,
        remove_columns=dataset_test.column_names,
        num_proc=args.num_proc,
    )

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        evaluation_strategy="steps",
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        fp16=True,
        predict_with_generate=True,
        generation_max_length=args.max_new_tokens,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
        # disable_tqdm=True,
        remove_unused_columns=False,
        label_names=["labels"]
    )

    #   Trainer
    print("\nStarting training with LoRA")
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset_train,
        eval_dataset=dataset_test,
        data_collator=data_collator,
        compute_metrics=lambda pred: compute_metrics(pred, tokenizer),
        tokenizer=processor.feature_extractor,
    )

    trainer.train()
    model.save_pretrained(args.output_dir + "/lora")


if __name__ == "__main__":
    main()
