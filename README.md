## Overview

This repo adapts Whisper to dysarthric speech. It uses the TORGO dataset and Leave-One-Speaker-Out (LOSO) evaluation. The main metrics are Word Error Rate (WER) and Character Error Rate (CER).

The project also evaluates models on LibriSpeech. This test measures the retention of general speech recognition ability.

## Methods

| Method | Description |
|---|---|
| Zero-shot | Evaluate pretrained Whisper without training |
| Full fine-tuning | Train all model parameters |
| Partial fine-tuning | Train selected encoder and decoder blocks |
| Component fine-tuning | Train selected attention, FFN, or normalization modules |
| LoRA | Train low-rank updates for selected linear layers |
| Bottleneck Adapter | Train small residual adapters and freeze Whisper |
| Waveform augmentation | Add noise, time stretch, pitch shift, and time masking |

### Zero-shot evaluation

```bash
pixi run python loso_whisper.py \
  --mode zero \
  --model_name openai/whisper-small \
  --output_dir results/loso
```

### Fine-tuning example

This example trains the last six encoder and decoder blocks:

```bash
pixi run python loso_whisper.py \
  --mode finetune \
  --finetune_scope partial \
  --train_encoder_layers 6-11 \
  --train_decoder_layers 6-11 \
  --num_train_epochs 30 \
  --output_dir results/loso
```

### Retention evaluation

This command evaluates one LOSO method on TORGO and LibriSpeech:

```bash
pixi run python evaluate_loso_retention.py \
  --evaluation_mode loso \
  --method_dir results/loso/partial-finetune-enc6-11-dec6-11 \
  --eval_targets both \
  --batch_size 8
```

