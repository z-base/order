# Model Scripts

The `models/scripts/` tree is organized by stage:

- `data/`
  - source download
  - OCR and text extraction
  - corpus generation
  - tokenizer training
  - tokenized dataset generation
- `overfit/`
  - the current baseline training script
- `production/`
  - ONNX export
  - post-training quantization

Convenience entrypoints live in `package.json`:

- `npm run pipeline:data -- --language eng`
- `npm run pipeline:overfit -- --language eng`
- `npm run pipeline:production -- --language eng`

Use the underlying scripts directly when you need fine-grained flags beyond the convenience pipeline wrappers.
