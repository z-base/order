#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import pathlib
import shutil
from dataclasses import dataclass
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort


DEFAULT_ONNX_BUILDS_ROOT = pathlib.Path("models/onnx_builds")
DEFAULT_QUANTIZED_MODELS_ROOT = pathlib.Path("models/quantized_models")
DEFAULT_ONNX_MODEL_NAME = "model.onnx"
DEFAULT_QUANTIZED_MODEL_NAME = "model.int4.onnx"
DEFAULT_CONFIG_NAME = "config.json"
DEFAULT_METRICS_NAME = "metrics.json"
DEFAULT_TOKENIZER_MODEL_NAME = "tokenizer.model"
DEFAULT_TOKENIZER_VOCAB_NAME = "tokenizer.vocab"
DEFAULT_BACKEND = "int4"
DEFAULT_BLOCK_SIZE = 128
DEFAULT_INT4_ACCURACY_LEVEL = None
DEFAULT_BNB4_QUANT_TYPE = "nf4"


@dataclass(frozen=True)
class QuantizationPaths:
    source_model_path: pathlib.Path
    source_config_path: pathlib.Path
    source_metrics_path: pathlib.Path
    source_tokenizer_model_path: pathlib.Path
    source_tokenizer_vocab_path: pathlib.Path
    quantized_dir: pathlib.Path
    quantized_model_path: pathlib.Path
    quantized_config_path: pathlib.Path
    quantized_metrics_path: pathlib.Path
    quantized_tokenizer_model_path: pathlib.Path
    quantized_tokenizer_vocab_path: pathlib.Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Quantize ONNX models from models/onnx_builds/{language} into "
            "4-bit builds under models/quantized_models/{language}."
        )
    )
    parser.add_argument(
        "language_positional",
        nargs="?",
        help="Language directory under models/onnx_builds and models/quantized_models.",
    )
    parser.add_argument(
        "--language",
        dest="language_flag",
        help="Language directory under models/onnx_builds and models/quantized_models.",
    )
    parser.add_argument(
        "--onnx-builds-root",
        default=str(DEFAULT_ONNX_BUILDS_ROOT),
        help="Source ONNX build root directory (default: models/onnx_builds)",
    )
    parser.add_argument(
        "--quantized-models-root",
        default=str(DEFAULT_QUANTIZED_MODELS_ROOT),
        help="Quantized output root directory (default: models/quantized_models)",
    )
    parser.add_argument(
        "--onnx-model-name",
        default=DEFAULT_ONNX_MODEL_NAME,
        help="Source ONNX model filename (default: model.onnx)",
    )
    parser.add_argument(
        "--quantized-model-name",
        default=DEFAULT_QUANTIZED_MODEL_NAME,
        help="Quantized ONNX model filename (default: model.int4.onnx)",
    )
    parser.add_argument(
        "--config-name",
        default=DEFAULT_CONFIG_NAME,
        help="Build config filename (default: config.json)",
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
        "--backend",
        choices=["auto", "int4", "bnb4"],
        default=DEFAULT_BACKEND,
        help="Quantization backend preference (default: int4)",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=DEFAULT_BLOCK_SIZE,
        help=f"Block size for 4-bit quantization (default: {DEFAULT_BLOCK_SIZE})",
    )
    parser.add_argument(
        "--int4-accuracy-level",
        type=int,
        default=DEFAULT_INT4_ACCURACY_LEVEL,
        help="Optional ONNX Runtime MatMulNBits accuracy level.",
    )
    parser.add_argument(
        "--bnb4-quant-type",
        choices=["fp4", "nf4"],
        default=DEFAULT_BNB4_QUANT_TYPE,
        help=f"BNB4 quantization data type (default: {DEFAULT_BNB4_QUANT_TYPE})",
    )
    parser.add_argument(
        "--nodes-to-exclude",
        nargs="*",
        default=[],
        help="Optional list of node names to exclude from quantization.",
    )
    parser.add_argument(
        "--nodes-to-include",
        nargs="*",
        default=[],
        help="Optional list of node names to include for quantization. Only valid for the int4 backend.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing quantized model artifacts.",
    )
    args = parser.parse_args()
    args.language = args.language_flag or args.language_positional

    if not args.language:
        parser.error("the following arguments are required: language")
    if args.language_flag and args.language_positional:
        parser.error("use either positional language or --language, not both")

    return args


def validate_args(args: argparse.Namespace) -> None:
    if args.block_size < 1:
        raise SystemExit("--block-size must be a positive integer")
    if args.int4_accuracy_level is not None and args.int4_accuracy_level < 0:
        raise SystemExit("--int4-accuracy-level must be zero or greater")
    if args.nodes_to_include and args.backend == "bnb4":
        raise SystemExit("--nodes-to-include is only supported by the int4 backend")


def resolve_paths(args: argparse.Namespace) -> QuantizationPaths:
    onnx_builds_root = pathlib.Path(args.onnx_builds_root).resolve()
    quantized_models_root = pathlib.Path(args.quantized_models_root).resolve()
    source_dir = onnx_builds_root / args.language
    quantized_dir = quantized_models_root / args.language

    return QuantizationPaths(
        source_model_path=source_dir / args.onnx_model_name,
        source_config_path=source_dir / args.config_name,
        source_metrics_path=source_dir / args.metrics_name,
        source_tokenizer_model_path=source_dir / args.tokenizer_model_name,
        source_tokenizer_vocab_path=source_dir / args.tokenizer_vocab_name,
        quantized_dir=quantized_dir,
        quantized_model_path=quantized_dir / args.quantized_model_name,
        quantized_config_path=quantized_dir / args.config_name,
        quantized_metrics_path=quantized_dir / args.metrics_name,
        quantized_tokenizer_model_path=quantized_dir / args.tokenizer_model_name,
        quantized_tokenizer_vocab_path=quantized_dir / args.tokenizer_vocab_name,
    )


def require_file(path: pathlib.Path, label: str) -> None:
    if not path.exists():
        raise SystemExit(f"{label} does not exist: {path}")
    if not path.is_file():
        raise SystemExit(f"{label} is not a file: {path}")


def prepare_output_paths(paths: QuantizationPaths, force: bool) -> None:
    paths.quantized_dir.mkdir(parents=True, exist_ok=True)
    targets = [
        paths.quantized_model_path,
        paths.quantized_config_path,
        paths.quantized_metrics_path,
        paths.quantized_tokenizer_model_path,
        paths.quantized_tokenizer_vocab_path,
    ]
    for target in targets:
        if target.exists():
            if not force:
                raise SystemExit(
                    f"Output already exists: {target}. Pass --force to overwrite."
                )
            if target.is_file():
                target.unlink()
            else:
                raise SystemExit(f"Expected file output path, got directory: {target}")


def resolve_backend(args: argparse.Namespace) -> str:
    if args.backend == "bnb4":
        return "bnb4"
    if args.backend == "int4":
        ensure_int4_backend_available()
        return "int4"
    try:
        ensure_int4_backend_available()
        return "int4"
    except SystemExit:
        return "bnb4"


def ensure_int4_backend_available() -> None:
    try:
        importlib.import_module("onnx_ir")
        importlib.import_module("onnxruntime.quantization.matmul_nbits_quantizer")
    except Exception as error:
        raise SystemExit(
            "The int4 backend requires both onnxruntime.quantization.matmul_nbits_quantizer "
            f"and the 'onnx_ir' package. Install the missing dependency first. Detail: {error}"
        ) from error


def quantize_with_int4(args: argparse.Namespace, paths: QuantizationPaths) -> dict[str, Any]:
    quant_module = importlib.import_module("onnxruntime.quantization.matmul_nbits_quantizer")
    quant_utils = importlib.import_module("onnxruntime.quantization.quant_utils")

    model = onnx.load(str(paths.source_model_path))
    quant_config = quant_module.DefaultWeightOnlyQuantConfig(
        block_size=args.block_size,
        is_symmetric=True,
        accuracy_level=args.int4_accuracy_level,
        quant_format=quant_utils.QuantFormat.QOperator,
        op_types_to_quantize=("MatMul",),
        bits=4,
    )
    quantizer = quant_module.MatMulNBitsQuantizer(
        model=model,
        bits=4,
        accuracy_level=args.int4_accuracy_level,
        nodes_to_exclude=args.nodes_to_exclude,
        nodes_to_include=args.nodes_to_include or None,
        algo_config=quant_config,
    )
    quantizer.process()
    quantizer.model.save_model_to_file(str(paths.quantized_model_path), True)

    return {
        "backend": "int4",
        "algorithm": "MatMulNBits",
        "bits": 4,
        "blockSize": args.block_size,
        "accuracyLevel": args.int4_accuracy_level,
        "nodesExcluded": list(args.nodes_to_exclude),
        "nodesIncluded": list(args.nodes_to_include),
    }


def quantize_with_bnb4(args: argparse.Namespace, paths: QuantizationPaths) -> dict[str, Any]:
    quant_module = importlib.import_module("onnxruntime.quantization.matmul_bnb4_quantizer")
    model = onnx.load(str(paths.source_model_path))
    quant_type = (
        quant_module.MatMulBnb4Quantizer.NF4
        if args.bnb4_quant_type == "nf4"
        else quant_module.MatMulBnb4Quantizer.FP4
    )
    quantizer = quant_module.MatMulBnb4Quantizer(
        model=model,
        quant_type=quant_type,
        block_size=args.block_size,
        nodes_to_exclude=args.nodes_to_exclude,
    )
    quantizer.process()
    quantizer.model.save_model_to_file(str(paths.quantized_model_path), True)

    return {
        "backend": "bnb4",
        "algorithm": "MatMulBnb4",
        "bits": 4,
        "blockSize": args.block_size,
        "quantType": args.bnb4_quant_type,
        "nodesExcluded": list(args.nodes_to_exclude),
    }


def build_validation_cases(source_config: dict[str, Any]) -> list[dict[str, int | str]]:
    limits = source_config.get("limits", {})
    max_source_length = int(limits.get("maxInputLength", 16))
    max_target_length = int(limits.get("maxDecoderLength", 8))
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


def validate_quantized_model(model_path: pathlib.Path, source_config: dict[str, Any]) -> dict[str, Any]:
    model = onnx.load(str(model_path))
    onnx.checker.check_model(model)
    session = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )
    input_names = source_config.get(
        "inputNames",
        ["input_ids", "attention_mask", "decoder_input_ids"],
    )
    output_names = source_config.get("outputNames", ["logits"])
    token_ids = source_config.get("tokenIds", {})
    bos_id = int(token_ids.get("bos", 1))
    vocab_size = int(source_config.get("model", {}).get("vocabSize", 0))
    validation_cases: list[dict[str, Any]] = []

    for case in build_validation_cases(source_config):
        source_length = int(case["source_length"])
        target_length = int(case["target_length"])
        ort_inputs = {
            input_names[0]: np.ones((1, source_length), dtype=np.int64),
            input_names[1]: np.ones((1, source_length), dtype=np.int64),
            input_names[2]: np.full((1, target_length), bos_id, dtype=np.int64),
        }
        outputs = session.run(output_names, ort_inputs)
        logits = outputs[0]

        expected_shape = (1, target_length)
        actual_prefix_shape = tuple(logits.shape[:2])
        if actual_prefix_shape != expected_shape:
            raise SystemExit(
                f"Quantized model output shape mismatch for case '{case['name']}': "
                f"expected prefix {expected_shape}, got {actual_prefix_shape}"
            )
        if vocab_size and logits.shape[-1] != vocab_size:
            raise SystemExit(
                f"Quantized model vocab axis mismatch for case '{case['name']}': "
                f"expected {vocab_size}, got {logits.shape[-1]}"
            )

        validation_cases.append(
            {
                "name": case["name"],
                "inputShape": {
                    input_names[0]: list(ort_inputs[input_names[0]].shape),
                    input_names[1]: list(ort_inputs[input_names[1]].shape),
                    input_names[2]: list(ort_inputs[input_names[2]].shape),
                },
                "outputShape": list(logits.shape),
            }
        )

    return {
        "inputs": [item.name for item in session.get_inputs()],
        "outputs": [item.name for item in session.get_outputs()],
        "providers": session.get_providers(),
        "cases": validation_cases,
    }


def copy_support_artifacts(paths: QuantizationPaths) -> None:
    shutil.copy2(paths.source_tokenizer_model_path, paths.quantized_tokenizer_model_path)
    shutil.copy2(paths.source_tokenizer_vocab_path, paths.quantized_tokenizer_vocab_path)
    if paths.source_metrics_path.exists():
        shutil.copy2(paths.source_metrics_path, paths.quantized_metrics_path)


def write_quantized_config(
    *,
    paths: QuantizationPaths,
    quantization: dict[str, Any],
    validation: dict[str, Any],
) -> None:
    if paths.source_config_path.exists():
        source_config = json.loads(paths.source_config_path.read_text(encoding="utf8"))
    else:
        source_config = {}

    source_config["format"] = "onnx-4bit-quantized-build"
    source_config["modelFile"] = paths.quantized_model_path.name
    source_config["quantization"] = quantization
    source_config["validation"] = validation

    paths.quantized_config_path.write_text(
        f"{json.dumps(source_config, indent=2)}\n",
        encoding="utf8",
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    paths = resolve_paths(args)

    require_file(paths.source_model_path, "Source ONNX model")
    require_file(paths.source_config_path, "Source config")
    require_file(paths.source_tokenizer_model_path, "Tokenizer model")
    require_file(paths.source_tokenizer_vocab_path, "Tokenizer vocab")
    prepare_output_paths(paths, args.force)
    source_config = json.loads(paths.source_config_path.read_text(encoding="utf8"))

    backend = resolve_backend(args)
    if backend == "int4":
        quantization = quantize_with_int4(args, paths)
    else:
        quantization = quantize_with_bnb4(args, paths)

    validation = validate_quantized_model(paths.quantized_model_path, source_config)
    copy_support_artifacts(paths)
    write_quantized_config(
        paths=paths,
        quantization=quantization,
        validation=validation,
    )

    print(f"Language: {args.language}")
    print(f"Source model: {paths.source_model_path}")
    print(f"Quantized model: {paths.quantized_model_path}")
    print(f"Quantized config: {paths.quantized_config_path}")
    print(f"Backend: {quantization['backend']}")
    print(f"Inputs: {', '.join(validation['inputs'])}")
    print(f"Outputs: {', '.join(validation['outputs'])}")
    for case in validation["cases"]:
        print(f"Validation case {case['name']}: output_shape={case['outputShape']}")


if __name__ == "__main__":
    main()
