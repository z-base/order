#!/usr/bin/env node
import fs from 'node:fs/promises'
import path from 'node:path'
import process from 'node:process'

const DEFAULT_OPTIONS = {
  inputs: 'models/samples/inputs',
  outputs: 'models/samples/outputs',
  tokenizers: 'models/tokenizers',
}

function parseArgs(argv) {
  const parsed = { ...DEFAULT_OPTIONS }

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i]

    if (arg === '--help' || arg === '-h') {
      parsed.help = true
      continue
    }
    if (arg === '--inputs') {
      parsed.inputs = argv[i + 1] ?? parsed.inputs
      i += 1
      continue
    }
    if (arg === '--outputs') {
      parsed.outputs = argv[i + 1] ?? parsed.outputs
      i += 1
      continue
    }
    if (arg === '--tokenizers') {
      parsed.tokenizers = argv[i + 1] ?? parsed.tokenizers
      i += 1
      continue
    }
  }

  return parsed
}

function printHelp() {
  console.log(`Usage:
node models/scripts/data/transform-samples-to-corpus.mjs [options]

Options:
  --help                 Show this help text
  --inputs <path>        Sample inputs root (default: models/samples/inputs)
  --outputs <path>       Sample outputs root (default: models/samples/outputs)
  --tokenizers <path>    Tokenizer root (default: models/tokenizers)
`)
}

async function findFiles(rootPath, extension) {
  const files = []
  const stack = [rootPath]

  while (stack.length > 0) {
    const current = stack.pop()
    const entries = await fs.readdir(current, { withFileTypes: true })

    for (const entry of entries) {
      const candidate = path.join(current, entry.name)
      if (entry.isDirectory()) {
        stack.push(candidate)
        continue
      }
      if (!entry.isFile()) continue
      if (path.extname(entry.name).toLowerCase() !== extension) continue
      files.push(candidate)
    }
  }

  return files.sort((a, b) => a.localeCompare(b))
}

function collapseWhitespace(value) {
  return value.replace(/\s+/g, ' ').trim()
}

async function buildInputIndex(inputsRoot) {
  const inputFiles = await findFiles(inputsRoot, '.txt')
  const index = new Map()

  for (const inputPath of inputFiles) {
    const relativeInputPath = path.relative(inputsRoot, inputPath)
    const parts = relativeInputPath.split(path.sep)
    const language = parts[0]
    const dataset = parts[1]
    const basename = path.basename(inputPath)
    const key = `${language}/${dataset}/${basename}`
    const existing = index.get(key) ?? []

    existing.push(inputPath)
    index.set(key, existing)
  }

  return index
}

function findMatchingInput(relativeOutputPath, inputIndex) {
  const withoutExt = relativeOutputPath.slice(0, -'.jsonld'.length)
  const parsed = path.parse(withoutExt)
  const parts = parsed.dir.split(path.sep)
  const language = parts[0]
  const dataset = parts[1]
  const key = `${language}/${dataset}/${parsed.name}.txt`
  const matches = inputIndex.get(key) ?? []

  if (matches.length === 0) {
    throw new Error(`Missing matching input for ${relativeOutputPath}`)
  }
  if (matches.length > 1) {
    throw new Error(
      `Ambiguous matching input for ${relativeOutputPath}: ${matches.join(', ')}`
    )
  }

  return matches[0]
}

async function buildCorpusEntries(outputFiles, outputsRoot, inputsRoot) {
  const corpora = new Map()
  const inputIndex = await buildInputIndex(inputsRoot)

  for (const outputPath of outputFiles) {
    const relativeOutputPath = path.relative(outputsRoot, outputPath)
    const parts = relativeOutputPath.split(path.sep)
    const language = parts[0]

    if (!language) continue

    const inputPath = findMatchingInput(relativeOutputPath, inputIndex)
    const [inputText, outputJson] = await Promise.all([
      fs.readFile(inputPath, 'utf8'),
      fs.readFile(outputPath, 'utf8'),
    ])

    const normalizedInput = collapseWhitespace(inputText)
    if (!normalizedInput) continue

    const normalizedOutput = JSON.parse(outputJson)
    const entry = JSON.stringify({
      input: normalizedInput,
      output: normalizedOutput,
    })

    const current = corpora.get(language) ?? []
    current.push(entry)
    corpora.set(language, current)
  }

  return corpora
}

async function writeCorpora(corpora, tokenizersRoot) {
  const languages = [...corpora.keys()].sort((a, b) => a.localeCompare(b))

  for (const language of languages) {
    const directory = path.join(tokenizersRoot, language)
    const corpusPath = path.join(directory, 'corpus.jsonl')
    const lines = corpora.get(language) ?? []

    await fs.mkdir(directory, { recursive: true })
    await fs.writeFile(corpusPath, lines.join('\n') + '\n')
  }

  return languages
}

async function run() {
  const options = parseArgs(process.argv.slice(2))
  if (options.help) {
    printHelp()
    return
  }

  const inputsRoot = path.resolve(process.cwd(), options.inputs)
  const outputsRoot = path.resolve(process.cwd(), options.outputs)
  const tokenizersRoot = path.resolve(process.cwd(), options.tokenizers)
  const outputFiles = await findFiles(outputsRoot, '.jsonld')

  if (outputFiles.length === 0) {
    console.log(`No .jsonld files found in ${outputsRoot}`)
    return
  }

  const corpora = await buildCorpusEntries(outputFiles, outputsRoot, inputsRoot)
  const languages = await writeCorpora(corpora, tokenizersRoot)
  const totalEntries = [...corpora.values()].reduce(
    (sum, entries) => sum + entries.length,
    0
  )

  console.log(`Done. Corpus files written to ${tokenizersRoot}`)
  console.log(`Annotated samples processed: ${totalEntries}`)
  console.log(`Languages written: ${languages.length}`)

  for (const language of languages) {
    console.log(` - ${language}: ${corpora.get(language)?.length ?? 0}`)
  }
}

run().catch((error) => {
  console.error(error)
  process.exit(1)
})
