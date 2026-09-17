import argparse
import gc
import json
import platform
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple
import datasets
import peft
import torch
import transformers
from audiomentations import AddGaussianSNR, Compose, PitchShift, TimeMask, TimeStretch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    set_seed,
)

from models.whisper_bottleneck_adapter import adapter_summary, attach_adapters
from train_torgo import WhisperConsistencyTrainer
from util.loso import (
    WaveformAugmentedDataset,
    WhisperRawAudioCollator,
    aggregate_speaker_metrics,
    decode_prediction_output,
    filter_by_duration,
    load_severity_map,
    make_loso_fold,
    parse_layer_spec,
    parse_target_modules,
    resolve_loso_speakers,
    speaker_indices,
    text_metrics,
    utc_timestamp,
    write_json,
)


COMPONENT_CHOICES = (
    "attention",
    "self_attention",
    "cross_attention",
    "ffn",
    "normalization",
)
COMPONENT_ALIASES = {
    "normalisation": "normalization",
}


def parse_train_components(
    spec: str | None,
    argument_name: str = "--train_components",
) -> List[str]:
    # Parse and validate the component groups used by component fine-tuning.
    if spec is None or not spec.strip():
        raise ValueError(f"{argument_name} must not be empty")

    components: List[str] = []
    for raw_component in spec.split(","):
        component = raw_component.strip().lower().replace("-", "_")
        component = COMPONENT_ALIASES.get(component, component)
        if component not in COMPONENT_CHOICES:
            choices = ", ".join(COMPONENT_CHOICES)
            raise ValueError(
                f"Invalid component {raw_component.strip()!r} in {argument_name}; "
                f"choose from: {choices}"
            )
        if component not in components:
            components.append(component)
    return components


def resolved_train_components(components: List[str]) -> List[str]:
    # Expand ``attention`` into the two attention types it represents.
    resolved = set(components)
    if "attention" in resolved:
        resolved.update(("self_attention", "cross_attention"))
        resolved.remove("attention")
    return [
        component
        for component in ("self_attention", "cross_attention", "ffn", "normalization")
        if component in resolved
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run zero-shot evaluation or fine-tuning LOSO on TORGO."
    )
    parser.add_argument("--mode", choices=("zero", "finetune"), required=True)
    # Fine-tuning options
    parser.add_argument(
        "--finetune_scope",
        choices=("full", "partial", "components", "lora", "adapter"),
        default="full",
        help=(
            "Train all model parameters, whole selected layers, selected components "
            "inside selected layers, LoRA adapters, or bottleneck Adapter modules on "
            "top of a frozen backbone."
        ),
    )
    parser.add_argument(
        "--train_encoder_layers",
        default=None,
        help=(
            "Encoder layers used by partial/components fine-tuning, e.g. "
            "8-11, 0,2,4-6, all, or none."
        ),
    )
    parser.add_argument(
        "--train_decoder_layers",
        default=None,
        help=(
            "Decoder layers used by partial/components fine-tuning, e.g. "
            "8-11, 0,2,4-6, all, or none."
        ),
    )
    parser.add_argument(
        "--train_components",
        default=None,
        help=(
            "Comma-separated component groups for --finetune_scope components: "
            "attention (all self/cross attention), self_attention, "
            "cross_attention, ffn, and/or normalization."
        ),
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=16,
        help="LoRA rank; controls how many trainable low-rank dimensions are added.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA scaling factor; the adapter update is scaled by alpha / rank.",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="Dropout applied to the LoRA input projection.",
    )
    parser.add_argument(
        "--lora_target_modules",
        default="q_proj,v_proj",
        help=(
            "Comma separated projections to adapt, e.g. q_proj,v_proj or "
            "q_proj,k_proj,v_proj,out_proj,fc1,fc2."
        ),
    )
    parser.add_argument(
        "--adapter_dim",
        type=int,
        default=64,
        help=(
            "Bottleneck dimension of the sequential Adapter modules "
            "(hidden_size -> adapter_dim -> hidden_size)."
        ),
    )
    parser.add_argument(
        "--adapter_dropout",
        type=float,
        default=0.0,
        help="Dropout applied to the Adapter's up-projection output.",
    )
    parser.add_argument(
        "--adapter_encoder_layers",
        default=None,
        help=(
            "Encoder layers that get an Adapter for --finetune_scope adapter, e.g. "
            "8-11, 0,2,4-6, all, or none."
        ),
    )
    parser.add_argument(
        "--adapter_decoder_layers",
        default=None,
        help=(
            "Decoder layers that get an Adapter for --finetune_scope adapter, e.g. "
            "8-11, 0,2,4-6, all, or none."
        ),
    )
    parser.add_argument("--model_name", default="openai/whisper-small")
    parser.add_argument("--dataset_name", default="extraordinarylab/torgo")
    parser.add_argument("--dataset_split", default="test")
    parser.add_argument("--severity_path", default="TORGO/speaker_severity.json")
    parser.add_argument("--output_dir", default="results/loso")
    parser.add_argument("--speakers", nargs="*", default=None)
    parser.add_argument("--include_normal_speakers", action="store_true")
    parser.add_argument("--max_audio_seconds", type=float, default=30.0)
    parser.add_argument("--language", default="en")
    parser.add_argument("--task", default="transcribe")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--generation_max_length", type=int, default=225)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--save_predictions", action="store_true")
    # Fine-tuning hyperparameters
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--save_total_limit", type=int, default=1)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--save_model", action=argparse.BooleanOptionalAction, default=True)
    # Augmentation options
    parser.add_argument("--use_waveform_augmentation", action="store_true")
    parser.add_argument("--augment_snr_db_min", type=float, default=5.0)
    parser.add_argument("--augment_snr_db_max", type=float, default=20.0)
    parser.add_argument("--augment_time_stretch_min", type=float, default=0.8)
    parser.add_argument("--augment_time_stretch_max", type=float, default=1.25)
    parser.add_argument("--augment_pitch_min", type=float, default=-2.0)
    parser.add_argument("--augment_pitch_max", type=float, default=2.0)
    parser.add_argument("--augmentation_probability", type=float, default=0.5)
    # Bitwise augmentation options
    parser.add_argument("--use_bitwise_augmentation", action="store_true")
    parser.add_argument("--bitwise_mask_bits", type=int, default=4)
    parser.add_argument("--consistency_weight", type=float, default=0.5)
    parser.add_argument("--consistency_layer", type=int, default=-2)
    args = parser.parse_args()
    if args.bitwise_mask_bits <= 0 or 80 % args.bitwise_mask_bits != 0:
        parser.error("--bitwise_mask_bits must be a positive divisor of 80")
    if not 0.0 <= args.augmentation_probability <= 1.0:
        parser.error("--augmentation_probability must be between 0 and 1")
    if args.eval_steps <= 0 or args.save_steps <= 0:
        parser.error("--eval_steps and --save_steps must be positive")
    if args.save_model and args.save_steps % args.eval_steps != 0:
        parser.error("--save_steps must be a multiple of --eval_steps when saving the best model")
    # Validate mode and fine-tuning options
    if args.mode == "zero" and (
        args.use_waveform_augmentation or args.use_bitwise_augmentation
    ):
        parser.error("Augmentation options apply to training and require --mode finetune")
    if args.mode == "zero" and args.finetune_scope != "full":
        parser.error(f"--finetune_scope {args.finetune_scope} requires --mode finetune")
    if args.mode == "zero" and any(
        option is not None
        for option in (
            args.train_encoder_layers,
            args.train_decoder_layers,
            args.train_components,
            args.adapter_encoder_layers,
            args.adapter_decoder_layers,
        )
    ):
        parser.error("Layer/component selection options require --mode finetune")
    if args.mode == "finetune" and args.finetune_scope in ("partial", "components"):
        if args.train_encoder_layers is None or args.train_decoder_layers is None:
            parser.error(
                f"{args.finetune_scope.capitalize()} fine-tuning requires both "
                "--train_encoder_layers and "
                "--train_decoder_layers; use 'none' to freeze one side"
            )
    if args.finetune_scope not in ("partial", "components") and (
        args.train_encoder_layers is not None or args.train_decoder_layers is not None
    ):
        parser.error(
            "Layer selection options require --finetune_scope partial or components"
        )
    if args.finetune_scope == "components":
        try:
            parse_train_components(args.train_components)
        except ValueError as exc:
            parser.error(str(exc))
    elif args.train_components is not None:
        parser.error("--train_components requires --finetune_scope components")
    if args.finetune_scope == "lora":
        if args.lora_rank <= 0:
            parser.error("--lora_rank must be positive")
        if args.lora_alpha <= 0:
            parser.error("--lora_alpha must be positive")
        if not 0.0 <= args.lora_dropout < 1.0:
            parser.error("--lora_dropout must be in [0, 1)")
        try:
            parse_target_modules(args.lora_target_modules, "--lora_target_modules")
        except ValueError as exc:
            parser.error(str(exc))
    if args.finetune_scope == "adapter":
        if args.adapter_encoder_layers is None or args.adapter_decoder_layers is None:
            parser.error(
                "Adapter fine-tuning requires both --adapter_encoder_layers and "
                "--adapter_decoder_layers; use 'none' to skip one side"
            )
        if args.adapter_dim <= 0:
            parser.error("--adapter_dim must be positive")
        if not 0.0 <= args.adapter_dropout < 1.0:
            parser.error("--adapter_dropout must be in [0, 1)")
    elif args.adapter_encoder_layers is not None or args.adapter_decoder_layers is not None:
        parser.error(
            "--adapter_encoder_layers/--adapter_decoder_layers require "
            "--finetune_scope adapter"
        )
    return args


def runtime_metadata() -> Dict[str, Any]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "datasets": datasets.__version__,
        "peft": peft.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def load_model_and_processor(args: argparse.Namespace):
    processor = WhisperProcessor.from_pretrained(
        args.model_name,
        language=args.language,
        task=args.task,
    )
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name)
    model.config.forced_decoder_ids = None
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.language = args.language
        model.generation_config.task = args.task
    return model, processor


def configure_finetuning(model, args: argparse.Namespace) -> Tuple[Any, Dict[str, Any]]:
    encoder_layers = model.model.encoder.layers
    decoder_layers = model.model.decoder.layers
    lora_configuration = None
    adapter_configuration = None
    requested_components = None
    resolved_components = None

    if args.finetune_scope == "adapter":
        try:
            selected_encoder = parse_layer_spec(
                args.adapter_encoder_layers,
                len(encoder_layers),
                "--adapter_encoder_layers",
            )
            selected_decoder = parse_layer_spec(
                args.adapter_decoder_layers,
                len(decoder_layers),
                "--adapter_decoder_layers",
            )
        except ValueError as exc:
            raise ValueError(f"Invalid adapter fine-tuning configuration: {exc}") from exc
        if not selected_encoder and not selected_decoder:
            raise ValueError(
                "Adapter fine-tuning must select at least one encoder or decoder layer"
            )

        model = attach_adapters(
            model,
            args.adapter_dim,
            args.adapter_dropout,
            selected_encoder,
            selected_decoder,
        )
        model.freeze_backbone()
        if args.gradient_checkpointing:
            # Checkpointed blocks need an input that requires grad; the backbone is frozen.
            model.enable_input_require_grads()
        adapter_configuration = adapter_summary(model)
    elif args.finetune_scope == "lora":
        target_modules = parse_target_modules(
            args.lora_target_modules, "--lora_target_modules"
        )
        model = get_peft_model(
            model,
            LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=target_modules,
                bias="none",
            ),
        )
        if args.gradient_checkpointing:
            # Checkpointed blocks need an input that requires grad; the backbone is frozen.
            model.enable_input_require_grads()
        selected_encoder = list(range(len(encoder_layers)))
        selected_decoder = list(range(len(decoder_layers)))
        lora_configuration = {
            "rank": args.lora_rank,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": target_modules,
        }
    elif args.finetune_scope == "full":
        for parameter in model.parameters():
            parameter.requires_grad = True
        selected_encoder = list(range(len(encoder_layers)))
        selected_decoder = list(range(len(decoder_layers)))
    else:
        try:
            selected_encoder = parse_layer_spec(
                args.train_encoder_layers,
                len(encoder_layers),
                "--train_encoder_layers",
            )
            selected_decoder = parse_layer_spec(
                args.train_decoder_layers,
                len(decoder_layers),
                "--train_decoder_layers",
            )
        except ValueError as exc:
            raise ValueError(
                f"Invalid {args.finetune_scope} fine-tuning configuration: {exc}"
            ) from exc
        if not selected_encoder and not selected_decoder:
            raise ValueError(
                f"{args.finetune_scope.capitalize()} fine-tuning must select at least "
                "one encoder or decoder layer"
            )

        for parameter in model.parameters():
            parameter.requires_grad = False

        if args.finetune_scope == "partial":
            for layer_index in selected_encoder:
                for parameter in encoder_layers[layer_index].parameters():
                    parameter.requires_grad = True
            for layer_index in selected_decoder:
                for parameter in decoder_layers[layer_index].parameters():
                    parameter.requires_grad = True
        elif args.finetune_scope == "components":
            requested_components = parse_train_components(args.train_components)
            resolved_components = resolved_train_components(requested_components)

            encoder_module_names = []
            decoder_module_names = []
            if "self_attention" in resolved_components:
                encoder_module_names.append("self_attn")
                decoder_module_names.append("self_attn")
            if "cross_attention" in resolved_components:
                # Whisper encoder blocks have no cross-attention. Decoder cross-attention
                # is named encoder_attn in Hugging Face's Whisper implementation.
                decoder_module_names.append("encoder_attn")
            if "ffn" in resolved_components:
                encoder_module_names.extend(("fc1", "fc2"))
                decoder_module_names.extend(("fc1", "fc2"))
            if "normalization" in resolved_components:
                encoder_module_names.extend(
                    ("self_attn_layer_norm", "final_layer_norm")
                )
                decoder_module_names.extend(
                    (
                        "self_attn_layer_norm",
                        "encoder_attn_layer_norm",
                        "final_layer_norm",
                    )
                )

            for layer_index in selected_encoder:
                layer = encoder_layers[layer_index]
                for module_name in encoder_module_names:
                    for parameter in getattr(layer, module_name).parameters():
                        parameter.requires_grad = True
            for layer_index in selected_decoder:
                layer = decoder_layers[layer_index]
                for module_name in decoder_module_names:
                    for parameter in getattr(layer, module_name).parameters():
                        parameter.requires_grad = True
        else:
            raise ValueError(f"Unsupported fine-tuning scope: {args.finetune_scope}")

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if trainable_parameters == 0:
        raise ValueError(
            "The selected layers/components contain no trainable parameters; "
            "cross_attention applies only to selected decoder layers"
        )
    return model, {
        "scope": args.finetune_scope,
        "requested_encoder_layers": args.train_encoder_layers,
        "requested_decoder_layers": args.train_decoder_layers,
        "resolved_encoder_layers": selected_encoder,
        "resolved_decoder_layers": selected_decoder,
        "requested_components": requested_components,
        "resolved_components": resolved_components,
        "lora": lora_configuration,
        "adapter": adapter_configuration,
        "trainable_parameters": trainable_parameters,
        "total_parameters": total_parameters,
        "trainable_percent": 100.0 * trainable_parameters / total_parameters,
    }


def layer_spec_slug(spec: str) -> str:
    return spec.strip().lower().replace(",", "_")


def finetune_output_dir(args: argparse.Namespace) -> Path:
    if args.finetune_scope == "full":
        return Path(args.output_dir) / "full-finetune"
    if args.finetune_scope == "lora":
        modules = "_".join(
            parse_target_modules(args.lora_target_modules, "--lora_target_modules")
        )
        return (
            Path(args.output_dir)
            / f"lora-r{args.lora_rank}-a{args.lora_alpha}-{modules}"
        )
    if args.finetune_scope == "adapter":
        encoder = layer_spec_slug(args.adapter_encoder_layers)
        decoder = layer_spec_slug(args.adapter_decoder_layers)
        return (
            Path(args.output_dir)
            / f"adapter-dim{args.adapter_dim}-enc{encoder}-dec{decoder}"
        )
    encoder = layer_spec_slug(args.train_encoder_layers)
    decoder = layer_spec_slug(args.train_decoder_layers)
    if args.finetune_scope == "components":
        components = "_".join(parse_train_components(args.train_components))
        return (
            Path(args.output_dir)
            / f"component-finetune-{components}-enc{encoder}-dec{decoder}"
        )
    return Path(args.output_dir) / f"partial-finetune-enc{encoder}-dec{decoder}"


def waveform_transform(args: argparse.Namespace):
    probability = args.augmentation_probability
    return Compose(
        [
            AddGaussianSNR(
                min_snr_db=args.augment_snr_db_min,
                max_snr_db=args.augment_snr_db_max,
                p=probability,
            ),
            TimeStretch(
                min_rate=args.augment_time_stretch_min,
                max_rate=args.augment_time_stretch_max,
                p=probability,
            ),
            PitchShift(
                min_semitones=args.augment_pitch_min,
                max_semitones=args.augment_pitch_max,
                p=probability,
            ),
            TimeMask(p=probability),
        ]
    )


def metrics_callback(processor):
    def compute_metrics(prediction_output):
        predictions, references = decode_prediction_output(prediction_output, processor)
        values = text_metrics(predictions, references)
        return {"wer": values["wer"], "cer": values["cer"]}

    return compute_metrics


def training_arguments(
    args: argparse.Namespace,
    output_dir: Path,
    training: bool,
) -> Seq2SeqTrainingArguments:
    fp16 = bool(args.fp16 and torch.cuda.is_available())
    bf16 = bool(args.bf16 and torch.cuda.is_available())
    common = dict(
        output_dir=str(output_dir),
        per_device_eval_batch_size=args.eval_batch_size,
        dataloader_num_workers=args.num_workers,
        predict_with_generate=True,
        generation_max_length=args.generation_max_length,
        remove_unused_columns=False,
        report_to=["tensorboard"],
        fp16=fp16,
        bf16=bf16,
        seed=args.seed,
        data_seed=args.seed,
    )
    if not training:
        return Seq2SeqTrainingArguments(**common)
    return Seq2SeqTrainingArguments(
        **common,
        per_device_train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        save_strategy="steps" if args.save_model else "no",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=args.save_model,
        metric_for_best_model="wer",
        greater_is_better=False,
        gradient_checkpointing=args.gradient_checkpointing,
    )


def build_trainer(
    args: argparse.Namespace,
    model,
    processor,
    output_dir: Path,
    train_dataset=None,
    eval_dataset=None,
):
    is_training = train_dataset is not None
    trainer_class = WhisperConsistencyTrainer if args.use_bitwise_augmentation else Seq2SeqTrainer
    kwargs = dict(
        model=model,
        args=training_arguments(args, output_dir, training=is_training),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=WhisperRawAudioCollator(processor),
        compute_metrics=metrics_callback(processor),
        processing_class=processor,
    )
    if args.use_bitwise_augmentation:
        kwargs.update(
            consistency_weight=args.consistency_weight,
            consistency_layer=args.consistency_layer,
            bitwise_mask_bits=args.bitwise_mask_bits,
            use_bitwise_augmentation=True,
        )
    return trainer_class(**kwargs)


def evaluate_speaker(trainer, processor, dataset, prefix: str):
    output = trainer.predict(dataset, metric_key_prefix=prefix)
    predictions, references = decode_prediction_output(output, processor)
    return text_metrics(predictions, references), predictions, references


def save_predictions(path: Path, predictions: List[str], references: List[str]) -> None:
    rows = [
        {"prediction": prediction, "reference": reference}
        for prediction, reference in zip(predictions, references)
    ]
    write_json(path, {"samples": rows})


def last_eval_metrics(log_history: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    for entry in reversed(log_history):
        if "eval_wer" in entry:
            return {
                "wer": float(entry["eval_wer"]),
                "cer": float(entry["eval_cer"]),
                "eval_loss": float(entry["eval_loss"]) if "eval_loss" in entry else None,
                "epoch": entry.get("epoch"),
                "step": entry.get("step"),
            }
    return None


def base_result(args, selected_speakers, severity_map) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "created_at": utc_timestamp(),
        "mode": args.mode,
        "model_name": args.model_name,
        "dataset": {"name": args.dataset_name, "split": args.dataset_split},
        "selected_speakers": selected_speakers,
        "speaker_severity": {speaker: severity_map[speaker] for speaker in selected_speakers},
        "parameters": vars(args),
        "runtime": runtime_metadata(),
        "speaker_metrics": {},
        "folds": {},
    }


def run_zero(args, dataset, index, selected_speakers, severity_map, severity_order):
    output_dir = Path(args.output_dir) / "zero-shot"
    model, processor = load_model_and_processor(args)
    trainer = build_trainer(args, model, processor, output_dir)
    result = base_result(args, selected_speakers, severity_map)
    for speaker in selected_speakers:
        speaker_dataset = dataset.select(index[speaker])
        metrics, predictions, references = evaluate_speaker(
            trainer, processor, speaker_dataset, f"eval_{speaker}"
        )
        result["speaker_metrics"][speaker] = {
            **metrics,
            "severity": severity_map[speaker],
        }
        if args.save_predictions:
            save_predictions(output_dir / "predictions" / f"{speaker}.json", predictions, references)
        write_json(output_dir / "metrics.json", {
            **result,
            **aggregate_speaker_metrics(result["speaker_metrics"], severity_map, severity_order),
        })
    result.update(aggregate_speaker_metrics(result["speaker_metrics"], severity_map, severity_order))
    write_json(output_dir / "metrics.json", result)
    return result


def run_finetune(args, dataset, index, selected_speakers, severity_map, severity_order):
    root = finetune_output_dir(args)
    result = base_result(args, selected_speakers, severity_map)
    last_epoch = {
        "schema_version": 1,
        "created_at": utc_timestamp(),
        "mode": args.mode,
        "model_name": args.model_name,
        "source": "trainer_state.log_history",
        "note": "Validation metrics of the final training epoch, read from the log; no extra inference pass.",
        "folds": {},
    }
    for fold_number, speaker in enumerate(selected_speakers, start=1):
        set_seed(args.seed + fold_number - 1)
        fold_dir = root / f"held-out-{speaker}"
        # create the train/validation split for this fold
        train_dataset, validation_dataset = make_loso_fold(dataset, index, speaker)
        if args.use_waveform_augmentation:
            train_dataset = WaveformAugmentedDataset(
                train_dataset,
                waveform_transform(args),
            )
        model, processor = load_model_and_processor(args)
        model, finetune_configuration = configure_finetuning(model, args)
        trainer = build_trainer(
            args,
            model,
            processor,
            fold_dir,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
        )
        fold_config = {
            "held_out_speaker": speaker,
            "severity": severity_map[speaker],
            "train_samples": len(train_dataset),
            "validation_samples": len(validation_dataset),
            "parameters": vars(args),
            "finetune_configuration": finetune_configuration,
        }
        write_json(fold_dir / "run_config.json", fold_config)
        train_output = trainer.train()
        if args.save_model:
            trainer.save_model(str(fold_dir / "final-model"))
            processor.save_pretrained(str(fold_dir / "final-model"))
        trainer.state.save_to_json(str(fold_dir / "trainer_state.json"))
        for checkpoint in fold_dir.glob("checkpoint-*"):
            if checkpoint.is_dir():
                shutil.rmtree(checkpoint, ignore_errors=True)
        final_epoch_metrics = last_eval_metrics(trainer.state.log_history)
        last_epoch["folds"][speaker] = (
            {**final_epoch_metrics, "severity": severity_map[speaker]}
            if final_epoch_metrics
            else None
        )
        write_json(root / "last_epoch_metrics.json", last_epoch)
        metrics, predictions, references = evaluate_speaker(
            trainer, processor, validation_dataset, f"eval_{speaker}"
        )
        result["speaker_metrics"][speaker] = {
            **metrics,
            "severity": severity_map[speaker],
        }
        result["folds"][speaker] = {
            **fold_config,
            "train_metrics": train_output.metrics,
            "model_path": str(fold_dir / "final-model") if args.save_model else None,
        }
        if args.save_predictions:
            save_predictions(fold_dir / "predictions.json", predictions, references)
        partial = {
            **result,
            **aggregate_speaker_metrics(result["speaker_metrics"], severity_map, severity_order),
        }
        write_json(root / "metrics.json", partial)
        del trainer, model, processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    result.update(aggregate_speaker_metrics(result["speaker_metrics"], severity_map, severity_order))
    write_json(root / "metrics.json", result)
    return result


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    severity_map, severity_order = load_severity_map(args.severity_path)
    selected_speakers = resolve_loso_speakers(
        severity_map,
        requested=args.speakers,
        include_normal=args.include_normal_speakers,
    )
    dataset = load_dataset(args.dataset_name, split=args.dataset_split)
    dataset = filter_by_duration(dataset, args.max_audio_seconds)
    index = speaker_indices(dataset)
    missing = sorted(set(selected_speakers) - set(index))
    if missing:
        raise ValueError(f"Selected speakers missing from dataset: {missing}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print(json.dumps({"mode": args.mode, "speakers": selected_speakers}, indent=2))
    if args.mode == "zero":
        result = run_zero(
            args, dataset, index, selected_speakers, severity_map, severity_order
        )
    else:
        result = run_finetune(
            args, dataset, index, selected_speakers, severity_map, severity_order
        )
    print(json.dumps({
        "overall_macro": result["overall_macro"],
        "severity_macro": result["severity_macro"],
    }, indent=2))


if __name__ == "__main__":
    main()
