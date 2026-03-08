#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import shutil
import sys
import warnings
from dataclasses import dataclass
from typing import Any

import onnx
import onnxruntime as ort
import torch


DEFAULT_BEST_MODELS_ROOT = pathlib.Path("models/best_models")
DEFAULT_TOKENIZERS_ROOT = pathlib.Path("models/tokenizers")
DEFAULT_ONNX_BUILDS_ROOT = pathlib.Path("models/onnx_builds")
DEFAULT_CHECKPOINT_NAME = "best.pt"
DEFAULT_METRICS_NAME = "metrics.json"
DEFAULT_TOKENIZER_MODEL_NAME = "tokenizer.model"
DEFAULT_TOKENIZER_VOCAB_NAME = "tokenizer.vocab"
DEFAULT_ONNX_MODEL_NAME = "model.onnx"
DEFAULT_CONFIG_NAME = "config.json"
DEFAULT_OPSET_VERSION = 18
DEFAULT_NUM_HEADS = 4
DEFAULT_BOS_ID = 1
DEFAULT_EOS_ID = 2


@dataclass(frozen=True)
class ExportPaths:
    checkpoint_path: pathlib.Path
    metrics_path: pathlib.Path
    tokenizer_model_path: pathlib.Path
    tokenizer_vocab_path: pathlib.Path
    onnx_build_dir: pathlib.Path
    onnx_model_path: pathlib.Path
    config_path: pathlib.Path
    exported_metrics_path: pathlib.Path
    exported_tokenizer_model_path: pathlib.Path
    exported_tokenizer_vocab_path: pathlib.Path


class OnnxExportWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a trained seq2seq checkpoint to an ONNX build "
            "under models/onnx_builds/{language}."
        )
    )
    parser.add_argument(
        "language_positional",
        nargs="?",
        help="Language directory under models/best_models, models/tokenizers, and models/onnx_builds.",
    )
    parser.add_argument(
        "--language",
        dest="language_flag",
        help="Language directory under models/best_models, models/tokenizers, and models/onnx_builds.",
    )
    parser.add_argument(
        "--best-models-root",
        default=str(DEFAULT_BEST_MODELS_ROOT),
        help="Checkpoint root directory (default: models/best_models)",
    )
    parser.add_argument(
        "--tokenizers-root",
        default=str(DEFAULT_TOKENIZERS_ROOT),
        help="Tokenizer root directory (default: models/tokenizers)",
    )
    parser.add_argument(
        "--onnx-builds-root",
        default=str(DEFAULT_ONNX_BUILDS_ROOT),
        help="ONNX build root directory (default: models/onnx_builds)",
    )
    parser.add_argument(
        "--checkpoint-name",
        default=DEFAULT_CHECKPOINT_NAME,
        help="Checkpoint filename (default: best.pt)",
    )
    parser.add_argument(
        "--metrics-name",
        default=DEFAULT_METRICS_NAME,
        help="Metrics filename (default: metrics.json)",
    )
    parser.add_argument(
        "--tokenizer-model-name",
        default=DEFAULT_TOKENIZER_MODEL_NAME,
        help="Tokenizer model filename (default: tokenizer.model)",
    )
    parser.add_argument(
        "--tokenizer-vocab-name",
        default=DEFAULT_TOKENIZER_VOCAB_NAME,
        help="Tokenizer vocab filename (default: tokenizer.vocab)",
    )
    parser.add_argument(
        "--onnx-model-name",
        default=DEFAULT_ONNX_MODEL_NAME,
        help="Exported ONNX filename (default: model.onnx)",
    )
    parser.add_argument(
        "--config-name",
        default=DEFAULT_CONFIG_NAME,
        help="Exported config filename (default: config.json)",
    )
    parser.add_argument(
        "--opset-version",
        type=int,
        default=DEFAULT_OPSET_VERSION,
        help=f"ONNX opset version (default: {DEFAULT_OPSET_VERSION})",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=DEFAULT_NUM_HEADS,
        help=(
            "Attention head count used during training. "
            f"Default: {DEFAULT_NUM_HEADS}"
        ),
    )
    parser.add_argument(
        "--bos-id",
        type=int,
        default=DEFAULT_BOS_ID,
        help=f"Beginning-of-sequence token id (default: {DEFAULT_BOS_ID})",
    )
    parser.add_argument(
        "--eos-id",
        type=int,
        default=DEFAULT_EOS_ID,
        help=f"End-of-sequence token id (default: {DEFAULT_EOS_ID})",
    )
    args = parser.parse_args()
    args.language = args.language_flag or args.language_positional

    if not args.language:
        parser.error("the following arguments are required: language")
    if args.language_flag and args.language_positional:
        parser.error("use either positional language or --language, not both")

    return args


def validate_args(args: argparse.Namespace) -> None:
    if args.opset_version < 13:
        raise SystemExit("--opset-version must be at least 13")
    if args.num_heads < 1:
        raise SystemExit("--num-heads must be a positive integer")


def resolve_paths(args: argparse.Namespace) -> ExportPaths:
    best_models_root = pathlib.Path(args.best_models_root).resolve()
    tokenizers_root = pathlib.Path(args.tokenizers_root).resolve()
    onnx_builds_root = pathlib.Path(args.onnx_builds_root).resolve()

    checkpoint_dir = best_models_root / args.language
    tokenizer_dir = tokenizers_root / args.language
    onnx_build_dir = onnx_builds_root / args.language

    return ExportPaths(
        checkpoint_path=checkpoint_dir / args.checkpoint_name,
        metrics_path=checkpoint_dir / args.metrics_name,
        tokenizer_model_path=tokenizer_dir / args.tokenizer_model_name,
        tokenizer_vocab_path=tokenizer_dir / args.tokenizer_vocab_name,
        onnx_build_dir=onnx_build_dir,
        onnx_model_path=onnx_build_dir / args.onnx_model_name,
        config_path=onnx_build_dir / args.config_name,
        exported_metrics_path=onnx_build_dir / args.metrics_name,
        exported_tokenizer_model_path=onnx_build_dir / args.tokenizer_model_name,
        exported_tokenizer_vocab_path=onnx_build_dir / args.tokenizer_vocab_name,
    )


def require_file(file_path: pathlib.Path, label: str) -> None:
    if not file_path.exists():
        raise SystemExit(f"{label} does not exist: {file_path}")
    if not file_path.is_file():
        raise SystemExit(f"{label} is not a file: {file_path}")


def load_training_module() -> Any:
    script_path = (
        pathlib.Path(__file__).resolve().parents[1]
        / "overfit"
        / "train_seq2seq_overfit.py"
    )
    module_name = "train_seq2seq_overfit"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Failed to load training module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_checkpoint(checkpoint_path: pathlib.Path) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise SystemExit(f"Unexpected checkpoint format: {checkpoint_path}")
    if "model_state_dict" not in checkpoint:
        raise SystemExit(f"Checkpoint is missing model_state_dict: {checkpoint_path}")
    return checkpoint


def count_layer_prefixes(state_dict: dict[str, torch.Tensor], prefix: str) -> int:
    indices: set[int] = set()
    needle = f"{prefix}."

    for key in state_dict:
        if not key.startswith(needle):
            continue
        remainder = key[len(needle) :]
        head, _, _tail = remainder.partition(".")
        if head.isdigit():
            indices.add(int(head))

    if not indices:
        raise SystemExit(f"Could not infer layer count for prefix '{prefix}'")

    return len(indices)


def infer_model_config(
    checkpoint: dict[str, Any],
    *,
    num_heads: int,
    bos_id: int,
    eos_id: int,
) -> dict[str, Any]:
    state_dict = checkpoint["model_state_dict"]
    if not isinstance(state_dict, dict):
        raise SystemExit("Checkpoint model_state_dict is not a dictionary")

    vocab_size = int(state_dict["output_projection.weight"].shape[0])
    d_model = int(state_dict["token_embedding.weight"].shape[1])
    max_source_positions = int(state_dict["source_position_embedding.weight"].shape[0])
    max_target_positions = int(state_dict["target_position_embedding.weight"].shape[0])
    ffn_dim = int(state_dict["encoder.layers.0.linear1.weight"].shape[0])
    num_encoder_layers = count_layer_prefixes(state_dict, "encoder.layers")
    num_decoder_layers = count_layer_prefixes(state_dict, "decoder.layers")
    pad_id = vocab_size

    if d_model % num_heads != 0:
        raise SystemExit(
            f"Inferred d_model={d_model} is not divisible by num_heads={num_heads}. "
            "Pass the training head count with --num-heads."
        )

    return {
        "vocab_size": vocab_size,
        "pad_id": pad_id,
        "d_model": d_model,
        "num_heads": num_heads,
        "num_encoder_layers": num_encoder_layers,
        "num_decoder_layers": num_decoder_layers,
        "ffn_dim": ffn_dim,
        "dropout": 0.0,
        "max_source_positions": max_source_positions,
        "max_target_positions": max_target_positions,
        "bos_id": bos_id,
        "eos_id": eos_id,
    }


def build_model(module: Any, model_config: dict[str, Any], state_dict: dict[str, Any]) -> torch.nn.Module:
    model = module.TinySeq2SeqTransformer(
        vocab_size=model_config["vocab_size"],
        pad_id=model_config["pad_id"],
        d_model=model_config["d_model"],
        num_heads=model_config["num_heads"],
        num_encoder_layers=model_config["num_encoder_layers"],
        num_decoder_layers=model_config["num_decoder_layers"],
        ffn_dim=model_config["ffn_dim"],
        dropout=model_config["dropout"],
        max_source_positions=model_config["max_source_positions"],
        max_target_positions=model_config["max_target_positions"],
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def build_dummy_inputs(
    model_config: dict[str, Any],
    *,
    source_length: int | None = None,
    target_length: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_source_positions = model_config["max_source_positions"]
    max_target_positions = model_config["max_target_positions"]

    source_length = source_length or min(max_source_positions, 16)
    target_length = target_length or min(max_target_positions, 8)

    if source_length < 1 or source_length > max_source_positions:
        raise SystemExit(
            f"Invalid source length {source_length}; expected 1 <= length <= {max_source_positions}"
        )
    if target_length < 1 or target_length > max_target_positions:
        raise SystemExit(
            f"Invalid target length {target_length}; expected 1 <= length <= {max_target_positions}"
        )

    input_ids = torch.full((1, source_length), 1, dtype=torch.long)
    attention_mask = torch.ones((1, source_length), dtype=torch.long)
    decoder_input_ids = torch.full((1, target_length), model_config["bos_id"], dtype=torch.long)
    return input_ids, attention_mask, decoder_input_ids


def build_dynamic_shapes(model_config: dict[str, Any]) -> tuple[dict[int, Any], dict[int, Any], dict[int, Any]]:
    if not hasattr(torch, "export") or not hasattr(torch.export, "Dim"):
        raise SystemExit(
            "Dynamic ONNX export requires torch.export.Dim. Upgrade PyTorch to a version "
            "that supports the dynamo-based ONNX exporter."
        )

    batch_size = torch.export.Dim("batch_size", min=1)
    source_length = torch.export.Dim(
        "source_length",
        min=1,
        max=model_config["max_source_positions"],
    )
    target_length = torch.export.Dim(
        "target_length",
        min=1,
        max=model_config["max_target_positions"],
    )

    return (
        {0: batch_size, 1: source_length},
        {0: batch_size, 1: source_length},
        {0: batch_size, 1: target_length},
    )


def build_validation_cases(model_config: dict[str, Any]) -> list[dict[str, int | str]]:
    max_source_length = model_config["max_source_positions"]
    max_target_length = model_config["max_target_positions"]
    candidates = [
        {
            "name": "example",
            "source_length": min(max_source_length, 16),
            "target_length": min(max_target_length, 8),
        },
        {
            "name": "max_source_single_decoder",
            "source_length": max_source_length,
            "target_length": 1,
        },
        {
            "name": "max_lengths",
            "source_length": max_source_length,
            "target_length": max_target_length,
        },
    ]

    seen: set[tuple[int, int]] = set()
    cases: list[dict[str, int | str]] = []

    for candidate in candidates:
        key = (candidate["source_length"], candidate["target_length"])
        if key in seen:
            continue
        seen.add(key)
        cases.append(candidate)

    return cases


def export_onnx_model(
    *,
    model: torch.nn.Module,
    onnx_model_path: pathlib.Path,
    model_config: dict[str, Any],
    opset_version: int,
) -> None:
    wrapper = OnnxExportWrapper(model)
    wrapper.eval()
    dummy_inputs = build_dummy_inputs(model_config)
    dynamic_shapes = build_dynamic_shapes(model_config)

    onnx_model_path.parent.mkdir(parents=True, exist_ok=True)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="You are using the legacy TorchScript-based ONNX export.",
            category=DeprecationWarning,
        )
        torch.onnx.export(
            wrapper,
            dummy_inputs,
            onnx_model_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            dynamo=True,
            fallback=False,
            external_data=True,
            input_names=["input_ids", "attention_mask", "decoder_input_ids"],
            output_names=["logits"],
            dynamic_shapes=dynamic_shapes,
        )


def validate_export(
    *,
    model: torch.nn.Module,
    onnx_model_path: pathlib.Path,
    model_config: dict[str, Any],
) -> dict[str, Any]:
    onnx_model = onnx.load(str(onnx_model_path))
    onnx.checker.check_model(onnx_model)

    session = ort.InferenceSession(
        str(onnx_model_path),
        providers=["CPUExecutionProvider"],
    )
    validation_cases: list[dict[str, Any]] = []

    for case in build_validation_cases(model_config):
        input_ids, attention_mask, decoder_input_ids = build_dummy_inputs(
            model_config,
            source_length=int(case["source_length"]),
            target_length=int(case["target_length"]),
        )

        with torch.no_grad():
            torch_logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
            ).cpu()

        ort_inputs = {
            "input_ids": input_ids.numpy(),
            "attention_mask": attention_mask.numpy(),
            "decoder_input_ids": decoder_input_ids.numpy(),
        }
        ort_logits = session.run(["logits"], ort_inputs)[0]
        ort_tensor = torch.from_numpy(ort_logits)

        if ort_tensor.shape != torch_logits.shape:
            raise SystemExit(
                f"ONNX Runtime output shape mismatch for case '{case['name']}': "
                f"expected {tuple(torch_logits.shape)}, got {tuple(ort_tensor.shape)}"
            )

        validation_cases.append(
            {
                "name": case["name"],
                "inputShape": {
                    "input_ids": list(input_ids.shape),
                    "attention_mask": list(attention_mask.shape),
                    "decoder_input_ids": list(decoder_input_ids.shape),
                },
                "outputShape": list(torch_logits.shape),
                "maxAbsDiff": float(torch.max(torch.abs(torch_logits - ort_tensor)).item()),
            }
        )

    return {
        "inputs": [item.name for item in session.get_inputs()],
        "outputs": [item.name for item in session.get_outputs()],
        "providers": session.get_providers(),
        "cases": validation_cases,
    }


def copy_artifacts(paths: ExportPaths) -> None:
    paths.onnx_build_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(paths.tokenizer_model_path, paths.exported_tokenizer_model_path)
    shutil.copy2(paths.tokenizer_vocab_path, paths.exported_tokenizer_vocab_path)
    if paths.metrics_path.exists():
        shutil.copy2(paths.metrics_path, paths.exported_metrics_path)


def write_config(
    *,
    config_path: pathlib.Path,
    args: argparse.Namespace,
    model_config: dict[str, Any],
    validation: dict[str, Any],
) -> None:
    config = {
        "language": args.language,
        "format": "onnx-seq2seq-build",
        "modelFile": args.onnx_model_name,
        "tokenizerModelFile": args.tokenizer_model_name,
        "tokenizerVocabFile": args.tokenizer_vocab_name,
        "metricsFile": args.metrics_name,
        "inputNames": ["input_ids", "attention_mask", "decoder_input_ids"],
        "outputNames": ["logits"],
        "tokenIds": {
            "bos": model_config["bos_id"],
            "eos": model_config["eos_id"],
            "pad": model_config["pad_id"],
        },
        "limits": {
            "maxInputLength": model_config["max_source_positions"],
            "maxDecoderLength": model_config["max_target_positions"],
        },
        "model": {
            "vocabSize": model_config["vocab_size"],
            "dModel": model_config["d_model"],
            "numHeads": model_config["num_heads"],
            "numEncoderLayers": model_config["num_encoder_layers"],
            "numDecoderLayers": model_config["num_decoder_layers"],
            "ffnDim": model_config["ffn_dim"],
            "opsetVersion": args.opset_version,
        },
        "validation": validation,
    }
    config_path.write_text(f"{json.dumps(config, indent=2)}\n", encoding="utf8")


def main() -> None:
    args = parse_args()
    validate_args(args)
    paths = resolve_paths(args)

    require_file(paths.checkpoint_path, "Checkpoint")
    require_file(paths.tokenizer_model_path, "Tokenizer model")
    require_file(paths.tokenizer_vocab_path, "Tokenizer vocab")

    module = load_training_module()
    checkpoint = load_checkpoint(paths.checkpoint_path)
    model_config = infer_model_config(
        checkpoint,
        num_heads=args.num_heads,
        bos_id=args.bos_id,
        eos_id=args.eos_id,
    )
    model = build_model(module, model_config, checkpoint["model_state_dict"])

    export_onnx_model(
        model=model,
        onnx_model_path=paths.onnx_model_path,
        model_config=model_config,
        opset_version=args.opset_version,
    )
    validation = validate_export(
        model=model,
        onnx_model_path=paths.onnx_model_path,
        model_config=model_config,
    )
    copy_artifacts(paths)
    write_config(
        config_path=paths.config_path,
        args=args,
        model_config=model_config,
        validation=validation,
    )

    print(f"Language: {args.language}")
    print(f"Checkpoint: {paths.checkpoint_path}")
    print(f"ONNX build dir: {paths.onnx_build_dir}")
    print(f"ONNX model: {paths.onnx_model_path}")
    print(f"Config: {paths.config_path}")
    print(f"Tokenizer model: {paths.exported_tokenizer_model_path}")
    print(f"Tokenizer vocab: {paths.exported_tokenizer_vocab_path}")
    if paths.metrics_path.exists():
        print(f"Metrics: {paths.exported_metrics_path}")
    for case in validation["cases"]:
        print(
            f"Validation case {case['name']}: "
            f"output_shape={case['outputShape']} "
            f"max_abs_diff={case['maxAbsDiff']:.8f}"
        )


if __name__ == "__main__":
    main()
