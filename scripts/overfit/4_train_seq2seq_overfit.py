#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import random
from dataclasses import dataclass

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except ImportError:
    torch = None
    F = None
    nn = None
    DataLoader = None
    Dataset = object


DEFAULT_DATASETS_ROOT = pathlib.Path("models/datasets")
DEFAULT_TOKENIZERS_ROOT = pathlib.Path("models/tokenizers")
DEFAULT_BEST_MODELS_ROOT = pathlib.Path("models/best_models")
DEFAULT_TRAIN_FILE = "train.jsonl"
DEFAULT_VALIDATION_FILE = "validation.jsonl"
DEFAULT_VOCAB_FILE = "tokenizer.vocab"
DEFAULT_BATCH_SIZE = 1
DEFAULT_EPOCHS = 40
DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_D_MODEL = 128
DEFAULT_NUM_HEADS = 4
DEFAULT_NUM_ENCODER_LAYERS = 2
DEFAULT_NUM_DECODER_LAYERS = 2
DEFAULT_FFN_DIM = 256
DEFAULT_DROPOUT = 0.1
DEFAULT_LABEL_PAD_ID = -100
DEFAULT_BOS_ID = 1
DEFAULT_EOS_ID = 2
DEFAULT_LOG_EVERY = 10
DEFAULT_EXACT_MATCH_EVERY = 0
DEFAULT_SEED = 7
DEFAULT_GRAD_CLIP = 1.0
DEFAULT_MAX_INPUT_LENGTH = 256
DEFAULT_MAX_LABEL_LENGTH = 512
DEFAULT_EARLY_STOPPING_PATIENCE = 8
DEFAULT_EARLY_STOPPING_MIN_DELTA = 1e-4


@dataclass(frozen=True)
class SplitPaths:
    train_path: pathlib.Path
    validation_path: pathlib.Path
    vocab_path: pathlib.Path


@dataclass(frozen=True)
class EffectiveSequenceLengths:
    max_input_length: int
    max_label_length: int
    max_source_positions: int
    max_target_positions: int


class TokenizedJsonlDataset(Dataset):
    def __init__(self, file_path: pathlib.Path, vocab_size: int) -> None:
        self.file_path = file_path
        self.records = self._load_records(file_path, vocab_size)

    @staticmethod
    def _load_records(file_path: pathlib.Path, vocab_size: int) -> list[dict]:
        if not file_path.exists():
            raise SystemExit(f"Dataset file does not exist: {file_path}")
        if not file_path.is_file():
            raise SystemExit(f"Dataset path is not a file: {file_path}")

        records: list[dict] = []

        with file_path.open("r", encoding="utf8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue

                try:
                    parsed = json.loads(line)
                except json.JSONDecodeError as error:
                    raise SystemExit(
                        f"Invalid JSONL in {file_path} at line {line_number}: {error}"
                    ) from error

                TokenizedJsonlDataset._validate_record(
                    parsed=parsed,
                    vocab_size=vocab_size,
                    file_path=file_path,
                    line_number=line_number,
                )
                records.append(parsed)

        if not records:
            raise SystemExit(f"Dataset file is empty: {file_path}")

        return records

    @staticmethod
    def _validate_record(
        parsed: object,
        vocab_size: int,
        file_path: pathlib.Path,
        line_number: int,
    ) -> None:
        if not isinstance(parsed, dict):
            raise SystemExit(
                f"Expected JSON object in {file_path} at line {line_number}"
            )

        required_keys = ("sample_id", "input_ids", "labels")
        for key in required_keys:
            if key not in parsed:
                raise SystemExit(
                    f"Missing key '{key}' in {file_path} at line {line_number}"
                )

        for key in ("input_ids", "labels"):
            value = parsed[key]
            if not isinstance(value, list) or not value:
                raise SystemExit(
                    f"Expected non-empty list '{key}' in {file_path} at line {line_number}"
                )
            if not all(isinstance(token_id, int) for token_id in value):
                raise SystemExit(
                    f"Expected integer token ids in '{key}' for {file_path} at line {line_number}"
                )
            max_token_id = max(value)
            min_token_id = min(value)
            if min_token_id < 0 or max_token_id >= vocab_size:
                raise SystemExit(
                    f"Token id out of range in '{key}' for {file_path} at line {line_number}: "
                    f"expected 0 <= id < {vocab_size}, got range [{min_token_id}, {max_token_id}]"
                )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        return self.records[index]


class Seq2SeqCollator:
    def __init__(
        self,
        *,
        pad_id: int,
        bos_id: int,
        eos_id: int,
        label_pad_id: int,
        max_input_length: int | None = None,
        max_label_length: int | None = None,
    ) -> None:
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.label_pad_id = label_pad_id
        self.max_input_length = max_input_length
        self.max_label_length = max_label_length

    def __call__(self, items: list[dict]) -> dict:
        if torch is None:
            raise SystemExit(
                "PyTorch is required to collate batches. Install torch first."
            )

        truncated_inputs = [
            self._truncate_input(item["input_ids"]) for item in items
        ]
        truncated_labels = [
            self._truncate_labels(item["labels"]) for item in items
        ]

        decoder_inputs = [[self.bos_id, *labels] for labels in truncated_labels]
        decoder_targets = [[*labels, self.eos_id] for labels in truncated_labels]

        input_ids = self._pad_tokens(truncated_inputs, pad_value=self.pad_id)
        attention_mask = input_ids.ne(self.pad_id).to(dtype=torch.long)
        decoder_input_ids = self._pad_tokens(
            decoder_inputs,
            pad_value=self.pad_id,
        )
        decoder_attention_mask = decoder_input_ids.ne(self.pad_id).to(
            dtype=torch.long
        )
        labels = self._pad_tokens(
            decoder_targets,
            pad_value=self.label_pad_id,
        )

        return {
            "sample_ids": [item["sample_id"] for item in items],
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "labels": labels,
            "target_token_ids": [list(tokens) for tokens in truncated_labels],
            "input_lengths": torch.tensor(
                [len(tokens) for tokens in truncated_inputs],
                dtype=torch.long,
            ),
            "label_lengths": torch.tensor(
                [len(tokens) for tokens in decoder_targets],
                dtype=torch.long,
            ),
        }

    def _truncate_input(self, token_ids: list[int]) -> list[int]:
        if self.max_input_length is None:
            return list(token_ids)
        return list(token_ids[: self.max_input_length])

    def _truncate_labels(self, token_ids: list[int]) -> list[int]:
        if self.max_label_length is None:
            return list(token_ids)
        if self.max_label_length < 1:
            raise SystemExit("--max-label-length must be at least 1")
        return list(token_ids[: self.max_label_length])

    @staticmethod
    def _pad_tokens(sequences: list[list[int]], pad_value: int) -> torch.Tensor:
        max_length = max(len(sequence) for sequence in sequences)
        padded = [
            sequence + [pad_value] * (max_length - len(sequence))
            for sequence in sequences
        ]
        return torch.tensor(padded, dtype=torch.long)


class TinySeq2SeqTransformer(nn.Module if nn is not None else object):
    def __init__(
        self,
        *,
        vocab_size: int,
        pad_id: int,
        d_model: int,
        num_heads: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        ffn_dim: int,
        dropout: float,
        max_source_positions: int,
        max_target_positions: int,
    ) -> None:
        super().__init__()
        if nn is None:
            raise SystemExit("PyTorch is required to build the model.")

        embedding_vocab_size = vocab_size + 1
        self.pad_id = pad_id
        self.d_model = d_model
        self.token_embedding = nn.Embedding(
            embedding_vocab_size,
            d_model,
            padding_idx=pad_id,
        )
        self.source_position_embedding = nn.Embedding(
            max_source_positions,
            d_model,
        )
        self.target_position_embedding = nn.Embedding(
            max_target_positions,
            d_model,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
            enable_nested_tensor=False,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers,
        )
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
    ) -> torch.Tensor:
        source_padding_mask = attention_mask.eq(0)
        target_padding_mask = decoder_input_ids.eq(self.pad_id)

        source_embeddings = self._embed_source(input_ids)
        target_embeddings = self._embed_target(decoder_input_ids)
        target_mask = self._causal_mask(
            length=decoder_input_ids.size(1),
            device=decoder_input_ids.device,
        )
        memory = self.encoder(
            source_embeddings,
            src_key_padding_mask=source_padding_mask,
        )
        hidden = self.decoder(
            tgt=target_embeddings,
            memory=memory,
            tgt_mask=target_mask,
            tgt_is_causal=True,
            tgt_key_padding_mask=target_padding_mask,
            memory_key_padding_mask=source_padding_mask,
        )

        return self.output_projection(hidden)

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source_padding_mask = attention_mask.eq(0)
        source_embeddings = self._embed_source(input_ids)
        memory = self.encoder(
            source_embeddings,
            src_key_padding_mask=source_padding_mask,
        )
        return memory, source_padding_mask

    def decode_step(
        self,
        *,
        decoder_input_ids: torch.Tensor,
        memory: torch.Tensor,
        source_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        target_embeddings = self._embed_target(decoder_input_ids)
        target_mask = self._causal_mask(
            length=decoder_input_ids.size(1),
            device=decoder_input_ids.device,
        )
        hidden = self.decoder(
            tgt=target_embeddings,
            memory=memory,
            tgt_mask=target_mask,
            tgt_is_causal=True,
            tgt_key_padding_mask=decoder_input_ids.eq(self.pad_id),
            memory_key_padding_mask=source_padding_mask,
        )
        return self.output_projection(hidden)

    def _embed_source(self, token_ids: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(
            token_ids.size(1),
            device=token_ids.device,
        ).unsqueeze(0)
        token_embeddings = self.token_embedding(token_ids) * (self.d_model**0.5)
        embeddings = token_embeddings + self.source_position_embedding(positions)
        return self.dropout(embeddings)

    def _embed_target(self, token_ids: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(
            token_ids.size(1),
            device=token_ids.device,
        ).unsqueeze(0)
        token_embeddings = self.token_embedding(token_ids) * (self.d_model**0.5)
        embeddings = token_embeddings + self.target_position_embedding(positions)
        return self.dropout(embeddings)

    @staticmethod
    def _causal_mask(length: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.ones((length, length), device=device, dtype=torch.bool),
            diagonal=1,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load tokenized seq2seq datasets, pad them with a collator, and "
            "run a tiny random-init Transformer overfit experiment."
        )
    )
    parser.add_argument(
        "language_positional",
        nargs="?",
        help="Language directory under models/datasets and models/tokenizers.",
    )
    parser.add_argument(
        "--language",
        dest="language_flag",
        help="Language directory under models/datasets and models/tokenizers.",
    )
    parser.add_argument(
        "--datasets-root",
        default=str(DEFAULT_DATASETS_ROOT),
        help="Dataset root directory (default: models/datasets)",
    )
    parser.add_argument(
        "--tokenizers-root",
        default=str(DEFAULT_TOKENIZERS_ROOT),
        help="Tokenizer root directory (default: models/tokenizers)",
    )
    parser.add_argument(
        "--train-file",
        default=DEFAULT_TRAIN_FILE,
        help="Train split filename (default: train.jsonl)",
    )
    parser.add_argument(
        "--validation-file",
        default=DEFAULT_VALIDATION_FILE,
        help="Validation split filename (default: validation.jsonl)",
    )
    parser.add_argument(
        "--vocab-file",
        default=DEFAULT_VOCAB_FILE,
        help="Tokenizer vocab filename (default: tokenizer.vocab)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Training epochs (default: {DEFAULT_EPOCHS})",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help=f"AdamW learning rate (default: {DEFAULT_LEARNING_RATE})",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
        help=f"AdamW weight decay (default: {DEFAULT_WEIGHT_DECAY})",
    )
    parser.add_argument(
        "--d-model",
        type=int,
        default=DEFAULT_D_MODEL,
        help=f"Transformer embedding width (default: {DEFAULT_D_MODEL})",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=DEFAULT_NUM_HEADS,
        help=f"Attention heads (default: {DEFAULT_NUM_HEADS})",
    )
    parser.add_argument(
        "--num-encoder-layers",
        type=int,
        default=DEFAULT_NUM_ENCODER_LAYERS,
        help=f"Encoder layer count (default: {DEFAULT_NUM_ENCODER_LAYERS})",
    )
    parser.add_argument(
        "--num-decoder-layers",
        type=int,
        default=DEFAULT_NUM_DECODER_LAYERS,
        help=f"Decoder layer count (default: {DEFAULT_NUM_DECODER_LAYERS})",
    )
    parser.add_argument(
        "--ffn-dim",
        type=int,
        default=DEFAULT_FFN_DIM,
        help=f"Feed-forward width (default: {DEFAULT_FFN_DIM})",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=DEFAULT_DROPOUT,
        help=f"Dropout rate (default: {DEFAULT_DROPOUT})",
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
    parser.add_argument(
        "--label-pad-id",
        type=int,
        default=DEFAULT_LABEL_PAD_ID,
        help=f"Loss ignore index for padded labels (default: {DEFAULT_LABEL_PAD_ID})",
    )
    parser.add_argument(
        "--max-input-length",
        type=int,
        default=DEFAULT_MAX_INPUT_LENGTH,
        help=f"Input truncation length (default: {DEFAULT_MAX_INPUT_LENGTH})",
    )
    parser.add_argument(
        "--max-label-length",
        type=int,
        default=DEFAULT_MAX_LABEL_LENGTH,
        help=(
            "Label truncation length before appending EOS "
            f"(default: {DEFAULT_MAX_LABEL_LENGTH})"
        ),
    )
    parser.add_argument(
        "--max-generation-length",
        type=int,
        default=None,
        help="Optional greedy decoding length override.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=DEFAULT_LOG_EVERY,
        help=f"Epoch logging interval (default: {DEFAULT_LOG_EVERY})",
    )
    parser.add_argument(
        "--exact-match-every",
        type=int,
        default=DEFAULT_EXACT_MATCH_EVERY,
        help=(
            "Greedy exact-match evaluation interval in epochs "
            f"(default: {DEFAULT_EXACT_MATCH_EVERY}, disabled)"
        ),
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=DEFAULT_GRAD_CLIP,
        help=f"Gradient clipping max norm (default: {DEFAULT_GRAD_CLIP})",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to use: auto, cpu, cuda, or mps (default: auto)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=DEFAULT_EARLY_STOPPING_PATIENCE,
        help=(
            "Stop when validation loss does not improve for this many epochs "
            f"(default: {DEFAULT_EARLY_STOPPING_PATIENCE})"
        ),
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=DEFAULT_EARLY_STOPPING_MIN_DELTA,
        help=(
            "Minimum validation loss improvement to reset patience "
            f"(default: {DEFAULT_EARLY_STOPPING_MIN_DELTA})"
        ),
    )
    parser.add_argument(
        "--save-dir",
        default=None,
        help="Override output directory (default: models/best_models/{language}).",
    )
    args = parser.parse_args()
    args.language = args.language_flag or args.language_positional

    if not args.language:
        parser.error("the following arguments are required: language")
    if args.language_flag and args.language_positional:
        parser.error("use either positional language or --language, not both")

    return args


def require_torch() -> None:
    if torch is None or nn is None or F is None or DataLoader is None:
        raise SystemExit(
            "PyTorch is required for overfit training. Install torch in this "
            "Python environment first."
        )


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be a positive integer")
    if args.epochs < 1:
        raise SystemExit("--epochs must be a positive integer")
    if args.learning_rate <= 0:
        raise SystemExit("--learning-rate must be greater than zero")
    if args.weight_decay < 0:
        raise SystemExit("--weight-decay must be zero or greater")
    if args.d_model < 1:
        raise SystemExit("--d-model must be a positive integer")
    if args.num_heads < 1:
        raise SystemExit("--num-heads must be a positive integer")
    if args.d_model % args.num_heads != 0:
        raise SystemExit("--d-model must be divisible by --num-heads")
    if args.num_encoder_layers < 1:
        raise SystemExit("--num-encoder-layers must be a positive integer")
    if args.num_decoder_layers < 1:
        raise SystemExit("--num-decoder-layers must be a positive integer")
    if args.ffn_dim < args.d_model:
        raise SystemExit("--ffn-dim must be at least --d-model")
    if not 0 <= args.dropout < 1:
        raise SystemExit("--dropout must be in the range [0, 1)")
    if args.max_input_length is not None and args.max_input_length < 1:
        raise SystemExit("--max-input-length must be at least 1")
    if args.max_label_length is not None and args.max_label_length < 1:
        raise SystemExit("--max-label-length must be at least 1")
    if args.max_generation_length is not None and args.max_generation_length < 1:
        raise SystemExit("--max-generation-length must be at least 1")
    if args.log_every < 1:
        raise SystemExit("--log-every must be a positive integer")
    if args.exact_match_every < 0:
        raise SystemExit("--exact-match-every must be zero or greater")
    if args.grad_clip <= 0:
        raise SystemExit("--grad-clip must be greater than zero")
    if args.early_stopping_patience < 1:
        raise SystemExit("--early-stopping-patience must be a positive integer")
    if args.early_stopping_min_delta < 0:
        raise SystemExit("--early-stopping-min-delta must be zero or greater")


def resolve_paths(args: argparse.Namespace) -> SplitPaths:
    datasets_root = pathlib.Path(args.datasets_root).resolve()
    tokenizers_root = pathlib.Path(args.tokenizers_root).resolve()
    language_dataset_dir = datasets_root / args.language
    language_tokenizer_dir = tokenizers_root / args.language

    return SplitPaths(
        train_path=language_dataset_dir / args.train_file,
        validation_path=language_dataset_dir / args.validation_file,
        vocab_path=language_tokenizer_dir / args.vocab_file,
    )


def resolve_save_dir(args: argparse.Namespace) -> pathlib.Path:
    if args.save_dir:
        return pathlib.Path(args.save_dir).resolve()
    return (DEFAULT_BEST_MODELS_ROOT / args.language).resolve()


def read_vocab_size(vocab_path: pathlib.Path) -> int:
    if not vocab_path.exists():
        raise SystemExit(f"Tokenizer vocab does not exist: {vocab_path}")
    if not vocab_path.is_file():
        raise SystemExit(f"Tokenizer vocab path is not a file: {vocab_path}")

    with vocab_path.open("r", encoding="utf8") as handle:
        size = sum(1 for line in handle if line.strip())

    if size < 3:
        raise SystemExit(
            f"Tokenizer vocab looks invalid or too small: {vocab_path}"
        )

    return size


def build_device(device_name: str) -> str:
    require_torch()
    if device_name == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device_name


def set_seed(seed: int) -> None:
    random.seed(seed)
    if torch is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_effective_sequence_lengths(
    *,
    train_dataset: TokenizedJsonlDataset,
    validation_dataset: TokenizedJsonlDataset,
    max_input_length: int,
    max_label_length: int,
) -> EffectiveSequenceLengths:
    observed_max_input_length = max(
        max(len(record["input_ids"]) for record in train_dataset.records),
        max(len(record["input_ids"]) for record in validation_dataset.records),
    )
    observed_max_label_length = max(
        max(len(record["labels"]) for record in train_dataset.records),
        max(len(record["labels"]) for record in validation_dataset.records),
    )

    effective_input_length = min(observed_max_input_length, max_input_length)
    effective_label_length = min(observed_max_label_length, max_label_length)

    return EffectiveSequenceLengths(
        max_input_length=effective_input_length,
        max_label_length=effective_label_length,
        max_source_positions=effective_input_length,
        max_target_positions=effective_label_length + 1,
    )


def count_parameters(model: TinySeq2SeqTransformer) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def greedy_generate(
    model: TinySeq2SeqTransformer,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    bos_id: int,
    eos_id: int,
    max_generation_length: int,
) -> list[list[int]]:
    model.eval()
    memory, source_padding_mask = model.encode(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    batch_size = input_ids.size(0)
    generated = torch.full(
        (batch_size, 1),
        fill_value=bos_id,
        dtype=torch.long,
        device=input_ids.device,
    )
    finished = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)

    for _ in range(max_generation_length):
        logits = model.decode_step(
            decoder_input_ids=generated,
            memory=memory,
            source_padding_mask=source_padding_mask,
        )
        next_token = logits[:, -1, :].argmax(dim=-1)
        next_token = torch.where(
            finished,
            torch.full_like(next_token, eos_id),
            next_token,
        )
        generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
        finished = finished | next_token.eq(eos_id)
        if bool(finished.all()):
            break

    outputs: list[list[int]] = []
    for sequence in generated[:, 1:].tolist():
        trimmed: list[int] = []
        for token_id in sequence:
            if token_id == eos_id:
                break
            trimmed.append(token_id)
        outputs.append(trimmed)

    return outputs


def compute_exact_match(
    model: TinySeq2SeqTransformer,
    loader: DataLoader,
    *,
    device: str,
    bos_id: int,
    eos_id: int,
    max_generation_length: int,
) -> float:
    matches = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            predictions = greedy_generate(
                model,
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                bos_id=bos_id,
                eos_id=eos_id,
                max_generation_length=max_generation_length,
            )
            targets = batch["target_token_ids"]
            for predicted, target in zip(predictions, targets):
                total += 1
                if predicted == target:
                    matches += 1

    if total == 0:
        return 0.0

    return matches / total


def compute_loss(
    model: TinySeq2SeqTransformer,
    batch: dict,
    *,
    device: str,
    label_pad_id: int,
) -> torch.Tensor:
    logits = model(
        input_ids=batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
        decoder_input_ids=batch["decoder_input_ids"].to(device),
    )
    labels = batch["labels"].to(device)
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=label_pad_id,
    )


def evaluate_loss(
    model: TinySeq2SeqTransformer,
    loader: DataLoader,
    *,
    device: str,
    label_pad_id: int,
) -> float:
    model.eval()
    total_loss = 0.0
    batch_count = 0

    with torch.no_grad():
        for batch in loader:
            loss = compute_loss(
                model,
                batch,
                device=device,
                label_pad_id=label_pad_id,
            )
            total_loss += float(loss.item())
            batch_count += 1

    if batch_count == 0:
        return 0.0

    return total_loss / batch_count


def train_epoch(
    model: TinySeq2SeqTransformer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: str,
    label_pad_id: int,
    grad_clip: float,
) -> float:
    model.train()
    total_loss = 0.0
    batch_count = 0

    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        loss = compute_loss(
            model,
            batch,
            device=device,
            label_pad_id=label_pad_id,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += float(loss.item())
        batch_count += 1

    return total_loss / batch_count


def save_artifacts(
    *,
    save_dir: pathlib.Path,
    metrics: dict,
    model: TinySeq2SeqTransformer,
    optimizer: torch.optim.Optimizer,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = save_dir / "metrics.json"
    checkpoint_path = save_dir / "best.pt"

    metrics_path.write_text(f"{json.dumps(metrics, indent=2)}\n", encoding="utf8")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        checkpoint_path,
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    require_torch()

    split_paths = resolve_paths(args)
    save_dir = resolve_save_dir(args)
    vocab_size = read_vocab_size(split_paths.vocab_path)
    pad_id = vocab_size

    train_dataset = TokenizedJsonlDataset(split_paths.train_path, vocab_size=vocab_size)
    validation_dataset = TokenizedJsonlDataset(
        split_paths.validation_path,
        vocab_size=vocab_size,
    )

    sequence_lengths = resolve_effective_sequence_lengths(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        max_input_length=args.max_input_length,
        max_label_length=args.max_label_length,
    )

    collator = Seq2SeqCollator(
        pad_id=pad_id,
        bos_id=args.bos_id,
        eos_id=args.eos_id,
        label_pad_id=args.label_pad_id,
        max_input_length=sequence_lengths.max_input_length,
        max_label_length=sequence_lengths.max_label_length,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    evaluation_train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    device = build_device(args.device)
    set_seed(args.seed)

    model = TinySeq2SeqTransformer(
        vocab_size=vocab_size,
        pad_id=pad_id,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
        max_source_positions=sequence_lengths.max_source_positions,
        max_target_positions=sequence_lengths.max_target_positions,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    if args.max_generation_length is None:
        max_generation_length = sequence_lengths.max_target_positions
    else:
        max_generation_length = args.max_generation_length

    best_metrics: dict | None = None
    best_validation_loss: float | None = None
    epochs_without_improvement = 0

    print(f"Language: {args.language}")
    print(f"Train split: {split_paths.train_path}")
    print(f"Validation split: {split_paths.validation_path}")
    print(f"Tokenizer vocab: {split_paths.vocab_path}")
    print(f"Vocab size: {vocab_size}")
    print(f"Pad id: {pad_id}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(validation_dataset)}")
    print(f"Observed max input length: {max(len(record['input_ids']) for record in train_dataset.records + validation_dataset.records)}")
    print(f"Observed max label length: {max(len(record['labels']) for record in train_dataset.records + validation_dataset.records)}")
    print(f"Effective max input length: {sequence_lengths.max_input_length}")
    print(f"Effective max label length: {sequence_lengths.max_label_length}")
    print(f"Max source positions: {sequence_lengths.max_source_positions}")
    print(f"Max target positions: {sequence_lengths.max_target_positions}")
    print(f"Parameter count: {count_parameters(model)}")
    print(f"Device: {device}")
    print(f"Save dir: {save_dir}")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            label_pad_id=args.label_pad_id,
            grad_clip=args.grad_clip,
        )
        validation_loss = evaluate_loss(
            model,
            validation_loader,
            device=device,
            label_pad_id=args.label_pad_id,
        )
        should_run_exact_match = (
            args.exact_match_every > 0
            and (
                epoch % args.exact_match_every == 0
                or epoch == args.epochs
            )
        )
        train_exact_match: float | None = None
        validation_exact_match: float | None = None
        if should_run_exact_match:
            train_exact_match = compute_exact_match(
                model,
                evaluation_train_loader,
                device=device,
                bos_id=args.bos_id,
                eos_id=args.eos_id,
                max_generation_length=max_generation_length,
            )
            validation_exact_match = compute_exact_match(
                model,
                validation_loader,
                device=device,
                bos_id=args.bos_id,
                eos_id=args.eos_id,
                max_generation_length=max_generation_length,
            )

        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "train_exact_match": train_exact_match,
            "validation_exact_match": validation_exact_match,
        }

        improved_validation = (
            best_validation_loss is None
            or validation_loss < best_validation_loss - args.early_stopping_min_delta
        )

        if improved_validation:
            best_validation_loss = validation_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if best_metrics is None:
            best_metrics = metrics
            save_artifacts(
                save_dir=save_dir,
                metrics=metrics,
                model=model,
                optimizer=optimizer,
            )
        elif improved_validation:
            best_metrics = metrics
            save_artifacts(
                save_dir=save_dir,
                metrics=metrics,
                model=model,
                optimizer=optimizer,
            )

        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            log_message = (
                "epoch="
                f"{epoch} "
                f"train_loss={train_loss:.4f} "
                f"validation_loss={validation_loss:.4f}"
            )
            if train_exact_match is not None and validation_exact_match is not None:
                log_message = (
                    f"{log_message} "
                    f"train_exact_match={train_exact_match:.3f} "
                    f"validation_exact_match={validation_exact_match:.3f}"
                )
            print(log_message)

        if train_exact_match is not None and train_exact_match >= 1.0:
            print(f"Stopping early at epoch {epoch}: train_exact_match reached 1.000")
            break

        if epochs_without_improvement >= args.early_stopping_patience:
            print(
                "Stopping early at epoch "
                f"{epoch}: validation loss did not improve for "
                f"{args.early_stopping_patience} epochs"
            )
            break

    if best_metrics is not None:
        print("Best train exact match snapshot:")
        print(json.dumps(best_metrics, indent=2))


if __name__ == "__main__":
    main()
