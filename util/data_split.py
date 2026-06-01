from collections import defaultdict
from typing import Dict, List, Tuple
import numpy as np
from datasets import Dataset, DatasetDict, load_dataset
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

def get_torgo(val_speaker, test_speaker, tokenizer):
    dataset = load_dataset("extraordinarylab/torgo")["test"]
    if len(dataset.cache_files) > 32: 
        dataset.cleanup_cache_files() # Only cleans up parquet and not downloads 

    dataset = dataset.map(lambda x: {"length": x["audio"].get_all_samples().duration_seconds }, num_proc=4)
    dataset = dataset.filter(lambda x: x["length"] <= 30, num_proc=4) # Manually confirmed the 2 samples above this are garbage
    speaker_column = "speaker"
    text_column = "text"
    train_ds = dataset.filter(lambda x: x[speaker_column] != val_speaker and x[speaker_column] != test_speaker, num_proc=4)
    val_ds = dataset.filter(lambda x: x[speaker_column] == val_speaker, num_proc=4)
    def convert(ds, num_shards=32, iterable=False):
        if iterable:
            ds = ds.to_iterable_dataset(num_shards=num_shards)
        return ds.map(
            lambda x: {"input_features": x["audio"].get_all_samples().data.squeeze(), "labels": tokenizer(text=x[text_column]) }).remove_columns(
                ["audio", "speech_status", "microphone", "length"])
    
    train = convert(train_ds, iterable=False)
    val = convert(val_ds, num_shards=16, iterable=False)
    return train, val

def partition_torgo_on_phrase(tokenizer, val_count=10):
    dataset = load_dataset("extraordinarylab/torgo")["test"]
    if len(dataset.cache_files) > 32: 
        dataset.cleanup_cache_files() # Only cleans up parquet and not downloads 
    text_column = "text"
    dataset = dataset.map(lambda x: {"length": x["audio"].get_all_samples().duration_seconds }, num_proc=4)
    dataset = dataset.filter(lambda x: x["length"] <= 30, num_proc=4) # Manually confirmed the 2 samples above this are garbage
    data = defaultdict(list)
    norm = BasicTextNormalizer()
    all_text = defaultdict(list)
    for idx in range(len(dataset)):
        d = dataset[idx]
        speaker = d["speaker"]
        text = norm(d["text"])
        data[speaker].append(idx)
        all_text[text].append(idx)
    data_set = {k: i for (i, k) in enumerate(all_text)}

    val_ds = dataset.filter(lambda x: data_set[norm(x["text"])] % val_count == 0, num_proc=4)
    train_ds = dataset.filter(lambda x: data_set[norm(x["text"])] % val_count != 0, num_proc=4)

    def convert(ds, num_shards=32, iterable=False):
        if iterable:
            ds = ds.to_iterable_dataset(num_shards=num_shards)
        return ds.map(
            lambda x: {"input_features": x["audio"].get_all_samples().data.squeeze(), "labels": tokenizer(text=x[text_column]) }).remove_columns(
                ["audio", "speech_status", "microphone", "length"])
    
    train = convert(train_ds, iterable=False)
    val = convert(val_ds, num_shards=16, iterable=False)
    return train, val

def get_libri_test(tokenizer):
    dataset = load_dataset("SPRINGLab/LibriSpeech-Test")["train"]
    text_column = "text"
    if len(dataset.cache_files) > 16: 
        dataset.cleanup_cache_files() # Only cleans up parquet and not downloads 

    dataset = dataset.map(lambda x: {"length": x["audio"].get_all_samples().duration_seconds }, num_proc=4)
    dataset = dataset.filter(lambda x: x["length"] <= 30, num_proc=4) # Manually confirmed the 2 samples above this are garbage

    def convert(ds):
        return ds.map(
            lambda x: {"input_features": x["audio"].get_all_samples().data.squeeze(), "labels": tokenizer(text=x[text_column]) }).remove_columns(
                ["audio", "id", "length"])
    
    test = convert(dataset)
    return test


