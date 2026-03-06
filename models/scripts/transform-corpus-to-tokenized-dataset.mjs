#!/usr/bin/env node
import crypto from 'node:crypto'
import fs from 'node:fs/promises'
import path from 'node:path'
import process from 'node:process'
import { createRequire } from 'node:module'

const require = createRequire(import.meta.url)
const { SentencePieceProcessor } = require('@agnai/sentencepiece-js')

const DEFAULT_OPTIONS = {
  language: '',
  tokenizers: 'models/tokenizers',
  datasets: 'models/datasets',
  corpusName: 'corpus.txt',
  modelName: 'tokenizer.model',
  validationRatio: 0.25,
  validationCount: null,
}

function parseArgs(argv) {
  const parsed = { ...DEFAULT_OPTIONS }

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i]

    if (arg === '--help' || arg === '-h') {
      parsed.help = true
      continue
    }
    if ((arg === '--language' || arg === '--lang') && argv[i + 1]) {
      parsed.language = argv[i + 1]
      i += 1
      continue
    }
    if (!arg.startsWith('-') && !parsed.language) {
      parsed.language = arg
      continue
    }
    if (arg === '--tokenizers' && argv[i + 1]) {
      parsed.tokenizers = argv[i + 1]
      i += 1
      continue
    }
    if (arg === '--datasets' && argv[i + 1]) {
      parsed.datasets = argv[i + 1]
      i += 1
      continue
    }
    if (arg === '--corpus-name' && argv[i + 1]) {
      parsed.corpusName = argv[i + 1]
      i += 1
      continue
    }
    if (arg === '--model-name' && argv[i + 1]) {
      parsed.modelName = argv[i + 1]
      i += 1
      continue
    }
    if (arg === '--validation-ratio' && argv[i + 1]) {
      const value = Number(argv[i + 1])
      if (!Number.isFinite(value) || value < 0 || value >= 1) {
        throw new Error(`Invalid --validation-ratio value: ${argv[i + 1]}`)
      }
      parsed.validationRatio = value
      i += 1
      continue
    }
    if (arg === '--validation-count' && argv[i + 1]) {
      const value = Number(argv[i + 1])
      if (!Number.isInteger(value) || value < 0) {
        throw new Error(`Invalid --validation-count value: ${argv[i + 1]}`)
      }
      parsed.validationCount = value
      i += 1
      continue
    }
  }

  return parsed
}

function printHelp() {
  console.log(`Usage:
node models/scripts/transform-corpus-to-tokenized-dataset.mjs --language <lang> [options]

Options:
  --help                   Show this help text
  --language, --lang       Language directory under models/tokenizers
  --tokenizers <path>      Tokenizer root (default: models/tokenizers)
  --datasets <path>        Dataset root (default: models/datasets)
  --corpus-name <name>     Corpus filename (default: corpus.txt)
  --model-name <name>      SentencePiece model filename (default: tokenizer.model)
  --validation-ratio <n>   Validation split ratio in [0,1) (default: 0.25)
  --validation-count <n>   Validation sample count override
`)
}

function stableJsonValue(value) {
  if (Array.isArray(value)) {
    return value.map((item) =>
      item === undefined ? null : stableJsonValue(item),
    )
  }
  if (value && typeof value === 'object') {
    const normalized = {}
    const keys = Object.keys(value).sort((a, b) => a.localeCompare(b))
    for (const key of keys) {
      const item = value[key]
      if (item === undefined) continue
      normalized[key] = stableJsonValue(item)
    }
    return normalized
  }
  return value
}

function stableStringify(value) {
  return JSON.stringify(stableJsonValue(value))
}

function hashSample(inputText, outputText) {
  return crypto
    .createHash('sha256')
    .update(inputText)
    .update('\n')
    .update(outputText)
    .digest('hex')
}

function percentile(sortedNumbers, fraction) {
  if (sortedNumbers.length === 0) return 0
  const index = Math.min(
    sortedNumbers.length - 1,
    Math.max(0, Math.ceil(sortedNumbers.length * fraction) - 1),
  )
  return sortedNumbers[index]
}

function summarizeLengths(values) {
  const sorted = [...values].sort((a, b) => a - b)
  const total = sorted.reduce((sum, value) => sum + value, 0)

  return {
    count: sorted.length,
    min: sorted[0] ?? 0,
    max: sorted[sorted.length - 1] ?? 0,
    avg: sorted.length === 0 ? 0 : total / sorted.length,
    p50: percentile(sorted, 0.5),
    p95: percentile(sorted, 0.95),
  }
}

function resolveValidationCount(total, options) {
  if (total <= 1) return 0

  if (options.validationCount !== null) {
    return Math.min(total - 1, options.validationCount)
  }

  const derived = Math.round(total * options.validationRatio)
  return Math.min(total - 1, Math.max(1, derived))
}

async function loadCorpusLines(corpusPath) {
  const raw = await fs.readFile(corpusPath, 'utf8')
  return raw
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
}

function parseCorpusLine(line, lineNumber) {
  let parsed
  try {
    parsed = JSON.parse(line)
  } catch (error) {
    throw new Error(
      `Invalid JSONL at line ${lineNumber}: ${error?.message ?? String(error)}`,
    )
  }

  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    throw new Error(`Expected JSON object at line ${lineNumber}`)
  }
  if (typeof parsed.input !== 'string') {
    throw new Error(`Expected string input at line ${lineNumber}`)
  }
  if (!Object.prototype.hasOwnProperty.call(parsed, 'output')) {
    throw new Error(`Missing output at line ${lineNumber}`)
  }

  return parsed
}

async function loadTokenizer(modelPath) {
  const processor = new SentencePieceProcessor()
  await processor.load(modelPath)
  return processor
}

function tokenizeSample(processor, parsed, lineNumber) {
  const inputText = parsed.input
  const outputText = stableStringify(parsed.output)
  const inputIds = processor.encodeIds(inputText)
  const labels = processor.encodeIds(outputText)

  return {
    sample_id: hashSample(inputText, outputText),
    source_line: lineNumber,
    input_text: inputText,
    output_text: outputText,
    input_ids: inputIds,
    attention_mask: inputIds.map(() => 1),
    labels,
    input_length: inputIds.length,
    label_length: labels.length,
  }
}

function splitSamples(samples, options) {
  const ordered = [...samples].sort((a, b) =>
    a.sample_id.localeCompare(b.sample_id),
  )
  const validationCount = resolveValidationCount(ordered.length, options)
  const validation = ordered.slice(0, validationCount)
  const train = ordered.slice(validationCount)

  return { train, validation }
}

async function writeJsonl(filePath, records) {
  const lines = records.map((record) => JSON.stringify(record))
  const content = lines.length === 0 ? '' : `${lines.join('\n')}\n`
  await fs.writeFile(filePath, content)
}

async function run() {
  const options = parseArgs(process.argv.slice(2))
  if (options.help) {
    printHelp()
    return
  }
  if (!options.language) {
    throw new Error('Missing required --language value')
  }

  const tokenizersRoot = path.resolve(process.cwd(), options.tokenizers)
  const datasetsRoot = path.resolve(process.cwd(), options.datasets)
  const languageRoot = path.join(tokenizersRoot, options.language)
  const corpusPath = path.join(languageRoot, options.corpusName)
  const modelPath = path.join(languageRoot, options.modelName)
  const outputRoot = path.join(datasetsRoot, options.language)

  const [corpusLines] = await Promise.all([
    loadCorpusLines(corpusPath),
    fs.access(modelPath),
  ])

  const processor = await loadTokenizer(modelPath)
  const tokenizedSamples = corpusLines.map((line, index) => {
    const lineNumber = index + 1
    const parsed = parseCorpusLine(line, lineNumber)
    return tokenizeSample(processor, parsed, lineNumber)
  })

  const { train, validation } = splitSamples(tokenizedSamples, options)
  const stats = {
    language: options.language,
    corpusPath,
    modelPath,
    sampleCount: tokenizedSamples.length,
    trainCount: train.length,
    validationCount: validation.length,
    inputLengths: summarizeLengths(tokenizedSamples.map((sample) => sample.input_length)),
    labelLengths: summarizeLengths(tokenizedSamples.map((sample) => sample.label_length)),
  }

  await fs.mkdir(outputRoot, { recursive: true })
  await Promise.all([
    writeJsonl(path.join(outputRoot, 'train.jsonl'), train),
    writeJsonl(path.join(outputRoot, 'validation.jsonl'), validation),
    fs.writeFile(path.join(outputRoot, 'stats.json'), `${JSON.stringify(stats, null, 2)}\n`),
  ])

  console.log(`Done. Tokenized dataset written to ${outputRoot}`)
  console.log(`Samples: ${tokenizedSamples.length}`)
  console.log(`Train: ${train.length}`)
  console.log(`Validation: ${validation.length}`)
  console.log(`Input max length: ${stats.inputLengths.max}`)
  console.log(`Label max length: ${stats.labelLengths.max}`)
}

run().catch((error) => {
  console.error(error)
  process.exit(1)
})
