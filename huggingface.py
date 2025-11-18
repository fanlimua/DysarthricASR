from datasets import load_dataset, Audio
from transformers import (
    WhisperForConditionalGeneration,
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, RandomSampler
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate
import numpy as np
from pprint import pprint
warnings.filterwarnings("ignore", category=FutureWarning)
# Parameter-Efficient Fine-Tuning
# from peft import LoraConfig, get_peft_model

# load dataset, https://huggingface.co/datasets/fsicoli/common_voice_15_0
dataset_train = load_dataset("fsicoli/common_voice_15_0", "pt", split="train[:1000]")
dataset_test = load_dataset("fsicoli/common_voice_15_0", "pt", split="test[:100]")
# dataset_train = load_dataset("fsicoli/common_voice_15_0", "pt", split="train")
# dataset_test = load_dataset("fsicoli/common_voice_15_0", "pt", split="test")
# print(dataset_train)
# print(dataset_train[0]['audio']['sampling_rate'])

sample = dataset_train[0]
# pprint(sample)
# print("Audio Path:", sample["audio"]["path"])
# print("Sampling Rate:", sample["audio"]["sampling_rate"])
print("Waveform Shape:", sample["audio"]["array"].shape)
# print("Sentence:", sample["sentence"])

# batch_sampler = BatchSampler(RandomSampler(dataset_train), batch_size=32, drop_last=False)
# dataloader = DataLoader(dataset_train, batch_sampler=batch_sampler)

# Resample datasets to 16kHz
dataset_train = dataset_train.cast_column("audio", Audio(sampling_rate=16000))
dataset_test = dataset_test.cast_column("audio", Audio(sampling_rate=16000))
# print("Resampled Sampling Rate:", dataset_train[0]['audio']['sampling_rate'])

# Load model and processor
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
# Converts text into numerical token IDs (each ID represents a specific subword)
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="Portuguese", task="transcribe")
processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="Portuguese", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    label_ids[label_ids == -100] = tokenizer.pad_token_id
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

# preprocessing
def prepare_dataset(batch):
    audio = batch["audio"]
    # Compute log-Mel spectrogram features
    batch["input_features"] = feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    # Encode text labels
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch

# Data padding
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Handle input audio features
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Handle label token sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # Remove BOS token if already included
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

# Preprocessing (extract log-mel features and prepare for token id)
dataset_train = dataset_train.map(prepare_dataset, remove_columns=dataset_train.column_names, num_proc=4)
dataset_test = dataset_test.map(prepare_dataset, remove_columns=dataset_test.column_names, num_proc=4)
sample = dataset_train[0]
features = sample["input_features"]
print("\nFeature shape:", np.array(features).shape)

labels = sample["labels"]
print("Labels:", labels)
print("Number of tokens:", len(labels))
# print(dataset_train)
# print(dataset_train[0]['labels'])

# Padding: ensures all audio features and token sequences in a batch have equal length
data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

# print("data_collator info:")
# samples = [dataset_train[i] for i in range(3)] 
# batch = data_collator(samples)
# print("Batch keys:", batch.keys())
# print("Input feature shape:", batch["input_features"].shape)
# print("Labels shape:", batch["labels"].shape)
# print("Labels:", batch["labels"])

training_args = Seq2SeqTrainingArguments(
    output_dir="./models/whisper-small-pt",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    warmup_steps=10,
    max_steps=100,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=10,
    eval_steps=5,
    logging_steps=1,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
)


trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset_train,
    eval_dataset=dataset_test,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

trainer.train()