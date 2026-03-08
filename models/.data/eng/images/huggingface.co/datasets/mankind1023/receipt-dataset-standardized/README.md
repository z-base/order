---
language:
- en
task_categories:
- document-question-answering
- optical-character-recognition
- information-extraction
license: other
---

# Receipt Dataset — Standardized & AI-Assisted Annotations

This dataset contains **2800 receipt images** with paired JSON annotations, curated and standardized from multiple publicly available receipt datasets.

## Structure

- `receipts/` — receipt images
- `annotations/` — JSON annotations (paired by filename)
- `MANIFEST.csv` — sample index
- `NOTICE.md` — attribution and methodology
- `licenses/` — upstream license texts

## Annotation methodology

Annotations were created using:
- original dataset annotations (where available)
- AI-assisted OCR and information extraction using commercial AI models
- manual review and normalization

Annotations are **model-assisted labels** intended for training and evaluation.

## Source breakdown

  sroie: 971
  zenodo: 813
  cord: 798
  express: 198

## Full dataset download

The full dataset is provided as a single archive:

- `receipt_dataset_2780.tgz` (contains `receipts/` and `annotations/`)
- `receipt_dataset_2780.tgz.sha256` (checksum)

Extract:
```bash
tar -xzf receipt_dataset_2780.tgz

## Licensing

This dataset redistributes upstream datasets under their original licenses.
See `NOTICE.md` and `licenses/` for full details.
