# Models

This directory is the self-contained receipt-to-`schema.org/Order` workspace.

## Layout

- `data/`
  - `image_sources/` image dataset manifests
  - `pdf_sources/` PDF dataset manifests
  - `text_sources/` text dataset manifests and generators
  - `downloaded_images/` raw downloaded image assets
  - `downloaded_pdfs/` raw downloaded PDF assets
  - `downloaded_texts/` raw downloaded text assets and generated JSONL shards
- `samples/`
  - `inputs/` normalized `.txt` annotation inputs grouped by language and source
  - `outputs/` hand-annotated JSON-LD targets
- `tokenizers/` per-language corpora and SentencePiece artifacts
- `datasets/` tokenized train/validation JSONL splits
- `best_models/` PyTorch checkpoints from training
- `onnx_builds/` exported ONNX builds
- `quantized_models/` 4-bit inference artifacts
- `scripts/`
  - `data/` acquisition, extraction, corpus, tokenizer, and dataset preparation
  - `overfit/` the current PyTorch training baseline
  - `production/` ONNX export and quantization

## Pipelines

- `npm run pipeline:data -- --language eng`
- `npm run pipeline:overfit -- --language eng`
- `npm run pipeline:production -- --language eng`

What they do:

1. `pipeline:data`
   Builds annotation-ready inputs and tokenized training data for one language:
   - download and extract raw receipt sources
   - build `tokenizers/{lang}/corpus.jsonl`
   - train `tokenizers/{lang}/tokenizer.{model,vocab}`
   - build `datasets/{lang}/{train,validation}.jsonl`
2. `pipeline:overfit`
   Runs the current baseline trainer and writes checkpoints into `best_models/{lang}/`.
3. `pipeline:production`
   Exports the best checkpoint to `onnx_builds/{lang}/` and quantizes it into `quantized_models/{lang}/`.

Advanced options still live on the underlying scripts in `models/scripts/*`.

## Input Extraction Goal

`extract:inputs` is the end-to-end ingestion command for raw receipt material.

It is expected to:

1. download image, PDF, and text sources
2. extract all supported source material into `samples/inputs/{lang}/...`
3. write only `.txt` files as annotation inputs

The extraction pipeline is intentionally format-agnostic on the text side. CSV, JSON, YAML, XML, HTML, `.receipt`, Markdown, and other text-like files are accepted and flattened into `.txt`.

## Worker Usage

Heavy extraction scripts use worker-thread concurrency:

- `scripts/data/download-receipt-assets.mjs`
- `scripts/data/download-receipt-images.mjs`
- `scripts/data/download-receipt-texts.mjs`
- `scripts/data/extract-inputs-from-receipt-images.mjs`
- `scripts/data/extract-inputs-from-receipt-pdfs.mjs`
- `scripts/data/extract-inputs-from-receipt-texts.mjs`
- `scripts/data/extract-inputs.mjs`

## Main Commands

- `npm run extract:inputs`
- `npm run download:receipts`
- `npm run extract:receipt-images`
- `npm run extract:receipt-pdfs`
- `npm run extract:receipt-texts`
- `npm run corpus:build`
- `npm run tokenizer:train -- --language eng`
- `npm run dataset:build -- --language eng`
- `npm run train:overfit -- --language eng`
- `npm run export:onnx -- --language eng`
- `npm run quantize:onnx -- --language eng --force`

## Conventions

- Source manifests are language-first: `{lang}.json`
- Inputs are language-first: `samples/inputs/{lang}/{source}/...`
- Outputs are language-first: `samples/outputs/{lang}/{source}/...`
- Training and export artifacts are language-first
