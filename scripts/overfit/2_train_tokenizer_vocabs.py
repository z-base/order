#!/usr/bin/env python3
import argparse
import pathlib

try:
    import sentencepiece as spm
except ImportError as error:
    raise SystemExit(
        "sentencepiece is required to train tokenizer vocabs. "
        "Install it in your Python environment first."
    ) from error


DEFAULT_TOKENIZERS_ROOT = pathlib.Path("models/tokenizers")
DEFAULT_CORPUS_NAME = "corpus.jsonl"
DEFAULT_MODEL_PREFIX = "tokenizer"
DEFAULT_MAX_SENTENCE_LENGTH = 16384


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a SentencePiece tokenizer for one language corpus under "
            "models/tokenizers/{lang}/corpus.jsonl."
        )
    )
    parser.add_argument(
        "language_positional",
        nargs="?",
        help="Language directory name under models/tokenizers, for example: eng",
    )
    parser.add_argument(
        "--language",
        dest="language_flag",
        help="Language directory name under models/tokenizers, for example: eng",
    )
    parser.add_argument(
        "--tokenizers-root",
        default=str(DEFAULT_TOKENIZERS_ROOT),
        help="Tokenizer root directory (default: models/tokenizers)",
    )
    parser.add_argument(
        "--corpus-name",
        default=DEFAULT_CORPUS_NAME,
        help="Corpus filename inside the language directory (default: corpus.jsonl)",
    )
    parser.add_argument(
        "--model-prefix",
        default=DEFAULT_MODEL_PREFIX,
        help="Model prefix written into the language directory (default: tokenizer)",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=8000,
        help="SentencePiece vocab size (default: 8000)",
    )
    parser.add_argument(
        "--model-type",
        choices=["unigram", "bpe", "char", "word"],
        default="unigram",
        help="SentencePiece model type (default: unigram)",
    )
    parser.add_argument(
        "--character-coverage",
        type=float,
        default=1.0,
        help="SentencePiece character coverage (default: 1.0)",
    )
    parser.add_argument(
        "--input-sentence-size",
        type=int,
        default=0,
        help="Optional SentencePiece input_sentence_size (default: 0, disabled)",
    )
    parser.add_argument(
        "--shuffle-input-sentence",
        action="store_true",
        help="Enable SentencePiece shuffle_input_sentence",
    )
    parser.add_argument(
        "--max-sentence-length",
        type=int,
        default=DEFAULT_MAX_SENTENCE_LENGTH,
        help=(
            "Maximum corpus line length accepted by SentencePiece "
            f"(default: {DEFAULT_MAX_SENTENCE_LENGTH})"
        ),
    )
    parser.add_argument(
        "--hard-vocab-limit",
        action="store_true",
        help=(
            "Require the trained vocabulary to match --vocab-size exactly. "
            "By default the script allows smaller vocabs for tiny corpora."
        ),
    )
    args = parser.parse_args()
    args.language = args.language_flag or args.language_positional

    if not args.language:
        parser.error("the following arguments are required: language")
    if args.language_flag and args.language_positional:
        parser.error("use either positional language or --language, not both")

    return args


def resolve_paths(args: argparse.Namespace) -> tuple[pathlib.Path, pathlib.Path]:
    tokenizers_root = pathlib.Path(args.tokenizers_root).resolve()
    language_dir = tokenizers_root / args.language
    corpus_path = language_dir / args.corpus_name
    return language_dir, corpus_path


def validate_args(args: argparse.Namespace, language_dir: pathlib.Path, corpus_path: pathlib.Path) -> None:
    if args.vocab_size < 1:
        raise SystemExit("--vocab-size must be a positive integer")
    if not 0.0 < args.character_coverage <= 1.0:
        raise SystemExit("--character-coverage must be between 0 and 1")
    if args.input_sentence_size < 0:
        raise SystemExit("--input-sentence-size must be zero or greater")
    if args.max_sentence_length < 1:
        raise SystemExit("--max-sentence-length must be a positive integer")
    if not language_dir.exists():
        raise SystemExit(f"Language directory does not exist: {language_dir}")
    if not corpus_path.exists():
        raise SystemExit(f"Corpus file does not exist: {corpus_path}")
    if not corpus_path.is_file():
        raise SystemExit(f"Corpus path is not a file: {corpus_path}")
    if corpus_path.stat().st_size == 0:
        raise SystemExit(f"Corpus file is empty: {corpus_path}")


def build_trainer_kwargs(
    args: argparse.Namespace, corpus_path: pathlib.Path, language_dir: pathlib.Path
) -> dict:
    model_prefix_path = language_dir / args.model_prefix
    trainer_kwargs = {
        "input": str(corpus_path),
        "model_prefix": str(model_prefix_path),
        "vocab_size": args.vocab_size,
        "model_type": args.model_type,
        "character_coverage": args.character_coverage,
        "max_sentence_length": args.max_sentence_length,
        "hard_vocab_limit": args.hard_vocab_limit,
    }

    if args.input_sentence_size > 0:
        trainer_kwargs["input_sentence_size"] = args.input_sentence_size
        trainer_kwargs["shuffle_input_sentence"] = args.shuffle_input_sentence

    return trainer_kwargs


def main() -> None:
    args = parse_args()
    language_dir, corpus_path = resolve_paths(args)
    validate_args(args, language_dir, corpus_path)
    trainer_kwargs = build_trainer_kwargs(args, corpus_path, language_dir)

    spm.SentencePieceTrainer.train(**trainer_kwargs)

    model_prefix_path = language_dir / args.model_prefix
    print(f"Trained tokenizer for language: {args.language}")
    print(f"Corpus: {corpus_path}")
    print(f"Model: {model_prefix_path}.model")
    print(f"Vocab: {model_prefix_path}.vocab")


if __name__ == "__main__":
    main()
