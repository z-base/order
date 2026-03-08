#!/usr/bin/env node
import fs from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import process from 'node:process'
import {
  Worker,
  isMainThread,
  parentPort,
  workerData,
} from 'node:worker_threads'

const BINARY_EXTS = new Set([
  '.jpg',
  '.jpeg',
  '.png',
  '.gif',
  '.webp',
  '.bmp',
  '.tif',
  '.tiff',
  '.ico',
  '.pdf',
  '.zip',
  '.parquet',
  '.woff',
  '.woff2',
  '.ttf',
  '.otf',
  '.mp3',
  '.mp4',
  '.avi',
  '.mov',
  '.exe',
  '.dll',
  '.so',
  '.dylib',
])
const SKIP_DIRECTORIES = new Set(['licenses', 'node_modules'])

const DEFAULT_OPTIONS = {
  input: 'models/data/downloaded_texts',
  output: 'models/samples/inputs',
  workers: Math.max(1, Math.min(os.availableParallelism(), 8)),
  languageFilters: null,
  sourceFilters: null,
}

function parseArgs(argv) {
  const parsed = { ...DEFAULT_OPTIONS }

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index]

    if (arg === '--help' || arg === '-h') {
      parsed.help = true
      continue
    }
    if (arg === '--input') {
      parsed.input = argv[index + 1] ?? parsed.input
      index += 1
      continue
    }
    if (arg === '--output') {
      parsed.output = argv[index + 1] ?? parsed.output
      index += 1
      continue
    }
    if (arg === '--workers') {
      const value = Number(argv[index + 1])
      if (!Number.isFinite(value) || value < 1) {
        throw new Error(`Invalid --workers value: ${argv[index + 1]}`)
      }
      parsed.workers = Math.max(1, Math.floor(value))
      index += 1
      continue
    }
    if (arg === '--source') {
      parsed.sourceFilters = parsed.sourceFilters ?? []
      parsed.sourceFilters.push((argv[index + 1] ?? '').trim().toLowerCase())
      index += 1
      continue
    }
    if (arg === '--languages') {
      const raw = argv[index + 1]
      if (!raw) {
        throw new Error('--languages requires a comma-separated list')
      }
      parsed.languageFilters = raw
        .split(',')
        .map((value) => value.trim().toLowerCase())
        .filter(Boolean)
      index += 1
    }
  }

  return parsed
}

function printHelp() {
  console.log(`Usage:
node models/scripts/data/extract-inputs-from-receipt-texts.mjs [options]

Options:
  --help                 Show this help text
  --input <path>         Text download root (default: models/data/downloaded_texts)
  --output <path>        Input sample root (default: models/samples/inputs)
  --workers <n>          Worker-thread concurrency
  --source <slug>        Restrict to one source (repeatable)
  --languages <codes>    Restrict to languages (comma-separated)

Notes:
  Structured text documents are converted into plain .txt inputs under samples/inputs/{lang}/{source}/texts/...
  JSONL shards with { content } entries are exploded into one input .txt per record.
  Any text-like file is accepted; binary files are skipped by content sniffing.
`)
}

async function isProbablyTextFile(filePath) {
  const ext = path.extname(filePath).toLowerCase()
  if (BINARY_EXTS.has(ext)) {
    return false
  }

  const handle = await fs.open(filePath, 'r')
  try {
    const buffer = Buffer.alloc(4096)
    const { bytesRead } = await handle.read(buffer, 0, buffer.length, 0)
    if (bytesRead === 0) {
      return false
    }

    let suspicious = 0
    for (let index = 0; index < bytesRead; index += 1) {
      const byte = buffer[index]
      if (byte === 0) {
        return false
      }
      if (
        (byte < 7 || (byte > 14 && byte < 32)) &&
        byte !== 9 &&
        byte !== 10 &&
        byte !== 13
      ) {
        suspicious += 1
      }
    }

    return suspicious / bytesRead < 0.1
  } finally {
    await handle.close()
  }
}

async function findInputFiles(rootPath, options) {
  const absoluteRoot = path.resolve(process.cwd(), rootPath)
  const stack = [absoluteRoot]
  const files = []

  while (stack.length > 0) {
    const current = stack.pop()
    const entries = await fs.readdir(current, { withFileTypes: true })

    for (const entry of entries) {
      const candidate = path.join(current, entry.name)
      if (entry.isDirectory()) {
        if (SKIP_DIRECTORIES.has(entry.name.toLowerCase())) continue
        stack.push(candidate)
        continue
      }
      if (!entry.isFile()) continue

      if (!(await isProbablyTextFile(candidate))) continue

      const relative = path.relative(absoluteRoot, candidate)
      const parts = relative.split(path.sep).filter(Boolean)
      const language = parts[0]?.toLowerCase()
      const sourceSlug = parts[1]?.toLowerCase()

      if (
        options.languageFilters &&
        options.languageFilters.length > 0 &&
        !options.languageFilters.includes(language)
      ) {
        continue
      }

      if (
        options.sourceFilters &&
        options.sourceFilters.length > 0 &&
        !options.sourceFilters.includes(sourceSlug)
      ) {
        continue
      }

      files.push(candidate)
    }
  }

  return files.sort((a, b) => a.localeCompare(b))
}

function chunk(items, chunkCount) {
  const groups = Array.from({ length: chunkCount }, () => [])
  for (let index = 0; index < items.length; index += 1) {
    groups[index % chunkCount].push(items[index])
  }
  return groups.filter((group) => group.length > 0)
}

function cleanText(value) {
  return value.replace(/\r\n/g, '\n').trim()
}

function safeName(value) {
  return String(value || '')
    .replace(/[^\w.-]+/g, '-')
    .replace(/-+/g, '-')
    .replace(/^-+|-+$/g, '')
}

async function ensureDirectory(filePath) {
  await fs.mkdir(path.dirname(filePath), { recursive: true })
}

async function writeTextFile(filePath, content) {
  await ensureDirectory(filePath)
  await fs.writeFile(filePath, `${cleanText(content)}\n`)
}

function outputPathForRecord(
  outputRoot,
  language,
  sourceSlug,
  shardName,
  recordId
) {
  return path.join(
    outputRoot,
    language,
    sourceSlug,
    'texts',
    safeName(shardName),
    `${safeName(recordId)}.txt`
  )
}

function outputPathForFile(outputRoot, language, sourceSlug, relativeFilePath) {
  const parsed = path.parse(relativeFilePath)
  return path.join(
    outputRoot,
    language,
    sourceSlug,
    'texts',
    parsed.dir,
    `${parsed.name}.txt`
  )
}

async function processJsonLines(
  filePath,
  relativePath,
  outputRoot,
  language,
  sourceSlug
) {
  const raw = await fs.readFile(filePath, 'utf8')
  const lines = raw
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)

  let written = 0
  const shardName = path.basename(relativePath, path.extname(relativePath))

  for (const line of lines) {
    const record = JSON.parse(line)
    if (typeof record?.content !== 'string' || !record.content.trim()) {
      continue
    }
    const recordId =
      record.recordId || `${shardName}-${String(written + 1).padStart(6, '0')}`
    const outputPath = outputPathForRecord(
      outputRoot,
      language,
      sourceSlug,
      shardName,
      recordId
    )
    await writeTextFile(outputPath, record.content)
    written += 1
  }

  return written
}

async function processJson(
  filePath,
  relativePath,
  outputRoot,
  language,
  sourceSlug
) {
  const raw = await fs.readFile(filePath, 'utf8')
  const payload = JSON.parse(raw)

  if (
    Array.isArray(payload) &&
    payload.every((entry) => typeof entry?.content === 'string')
  ) {
    let written = 0
    const stem = path.basename(relativePath, path.extname(relativePath))
    for (const entry of payload) {
      const recordId =
        entry.recordId || `${stem}-${String(written + 1).padStart(6, '0')}`
      const outputPath = outputPathForRecord(
        outputRoot,
        language,
        sourceSlug,
        stem,
        recordId
      )
      await writeTextFile(outputPath, entry.content)
      written += 1
    }
    return written
  }

  const outputPath = outputPathForFile(
    outputRoot,
    language,
    sourceSlug,
    relativePath
  )
  await writeTextFile(outputPath, JSON.stringify(payload, null, 2))
  return 1
}

async function processTextFile(
  filePath,
  relativePath,
  outputRoot,
  language,
  sourceSlug
) {
  const raw = await fs.readFile(filePath, 'utf8')
  const outputPath = outputPathForFile(
    outputRoot,
    language,
    sourceSlug,
    relativePath
  )
  await writeTextFile(outputPath, raw)
  return 1
}

async function processFile(filePath, inputRoot, outputRoot) {
  const relativePath = path.relative(inputRoot, filePath)
  const parts = relativePath.split(path.sep).filter(Boolean)
  const language = parts[0]
  const sourceSlug = parts[1]
  const filePathInsideSource = parts.slice(2).join(path.sep)
  const ext = path.extname(filePath).toLowerCase()

  if (!language || !sourceSlug || !filePathInsideSource) {
    return 0
  }

  if (ext === '.jsonl') {
    return processJsonLines(
      filePath,
      filePathInsideSource,
      outputRoot,
      language,
      sourceSlug
    )
  }

  if (ext === '.json') {
    return processJson(
      filePath,
      filePathInsideSource,
      outputRoot,
      language,
      sourceSlug
    )
  }

  return processTextFile(
    filePath,
    filePathInsideSource,
    outputRoot,
    language,
    sourceSlug
  )
}

async function runProcessFilesWorker() {
  const { files, inputRoot, outputRoot } = workerData
  let written = 0

  for (const file of files) {
    written += await processFile(file, inputRoot, outputRoot)
  }

  parentPort?.postMessage({ written })
}

function runWorker(workerPayload) {
  return new Promise((resolve, reject) => {
    const worker = new Worker(new URL(import.meta.url), {
      workerData: {
        mode: 'process-files',
        ...workerPayload,
      },
    })
    worker.once('message', resolve)
    worker.once('error', reject)
    worker.once('exit', (code) => {
      if (code !== 0) {
        reject(new Error(`Worker stopped with exit code ${code}`))
      }
    })
  })
}

async function main() {
  const options = parseArgs(process.argv.slice(2))
  if (options.help) {
    printHelp()
    return
  }

  const inputRoot = path.resolve(process.cwd(), options.input)
  const outputRoot = path.resolve(process.cwd(), options.output)
  const files = await findInputFiles(inputRoot, options)

  if (files.length === 0) {
    console.log(`No supported text files found in ${inputRoot}`)
    return
  }

  const groups = chunk(files, Math.min(options.workers, files.length))
  const results = await Promise.all(
    groups.map((group) =>
      runWorker({
        files: group,
        inputRoot,
        outputRoot,
      })
    )
  )

  const written = results.reduce((sum, result) => sum + result.written, 0)
  console.log(`Done. Text-derived inputs written to ${outputRoot}`)
  console.log(`Source files processed: ${files.length}`)
  console.log(`Input .txt files written: ${written}`)
}

if (isMainThread) {
  main().catch((error) => {
    console.error(error)
    process.exit(1)
  })
} else {
  runProcessFilesWorker().catch((error) => {
    throw error
  })
}
