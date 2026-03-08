#!/usr/bin/env node
import { spawn } from 'node:child_process'
import fs from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import process from 'node:process'
import { createWorker } from 'tesseract.js'
import {
  Worker,
  isMainThread,
  parentPort,
  workerData,
} from 'node:worker_threads'

const PDF_EXTS = new Set(['.pdf'])
const ZIP_EXTS = new Set(['.zip'])
const SKIP_DIRECTORIES = new Set(['licenses', 'node_modules'])
const OCR_LANGUAGE_BY_DATASET_LANGUAGE = {
  ara: 'ara',
  deu: 'deu',
  eng: 'eng',
  fra: 'fra',
  ind: 'ind',
  ita: 'ita',
  kor: 'kor',
  msa: 'msa',
  nld: 'nld',
  por: 'por',
  rus: 'rus',
  spa: 'spa',
  zho: 'chi_sim',
}

const DEFAULT_OPTIONS = {
  input: 'models/data/downloaded_pdfs',
  output: 'models/samples/inputs',
  includeZips: true,
  defaultLanguage: null,
  languageFilters: null,
  ocrFallback: true,
  failOnError: false,
  workers: Math.max(1, Math.min(os.availableParallelism(), 4)),
}
const TESSERACT_ARTIFACTS_PATH = path.resolve(
  process.cwd(),
  'tesseract-artifacts'
)

function parseArgs(argv) {
  const parsed = { ...DEFAULT_OPTIONS }

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i]

    if (arg === '--help' || arg === '-h') {
      parsed.help = true
      continue
    }
    if (arg === '--input') {
      parsed.input = argv[i + 1] || parsed.input
      i += 1
      continue
    }
    if (arg === '--output') {
      parsed.output = argv[i + 1] || parsed.output
      i += 1
      continue
    }
    if (arg === '--language') {
      parsed.defaultLanguage = (argv[i + 1] || '').trim().toLowerCase() || null
      i += 1
      continue
    }
    if (arg === '--languages') {
      const raw = argv[i + 1]
      if (!raw) {
        throw new Error('--languages requires a comma-separated list')
      }
      parsed.languageFilters = raw
        .split(',')
        .map((value) => value.trim().toLowerCase())
        .filter(Boolean)
      i += 1
      continue
    }
    if (arg === '--workers') {
      const parsedValue = Number(argv[i + 1])
      if (!Number.isFinite(parsedValue) || parsedValue < 1) {
        throw new Error(`Invalid --workers value: ${argv[i + 1]}`)
      }
      parsed.workers = Math.max(1, Math.floor(parsedValue))
      i += 1
      continue
    }
    if (arg === '--with-zip') {
      parsed.includeZips = true
      continue
    }
    if (arg === '--no-zip') {
      parsed.includeZips = false
      continue
    }
    if (arg === '--with-ocr-fallback') {
      parsed.ocrFallback = true
      continue
    }
    if (arg === '--no-ocr-fallback') {
      parsed.ocrFallback = false
      continue
    }
    if (arg === '--fail-on-error') {
      parsed.failOnError = true
      continue
    }
  }

  return parsed
}

function printHelp() {
  console.log(`Usage:
node models/scripts/data/extract-inputs-from-receipt-pdfs.mjs [options]

Options:
  --help                 Show this help text
  --input <path>         PDF download root (default: models/data/downloaded_pdfs)
  --output <path>        Text output root (default: models/samples/inputs)
  --language <code>      Force one dataset language for the whole run
  --languages <codes>    Restrict to top-level dataset languages (comma-separated)
  --workers <n>          Worker-thread concurrency across file shards
  --with-zip             Extract and process PDFs from zip archives (default)
  --no-zip               Skip zip archives
  --with-ocr-fallback    OCR scanned PDF pages when no text layer is found (default)
  --no-ocr-fallback      Disable OCR fallback and keep text-layer extraction only
  --fail-on-error        Exit non-zero if any file fails

Notes:
  Relative paths are preserved, so downloaded_pdfs/{lang}/... becomes samples/inputs/{lang}/...
  This script uses pdfjs-dist for PDF parsing. OCR fallback for scanned PDFs also needs @napi-rs/canvas or canvas.
  Tesseract traineddata cache is written under tesseract-artifacts/ at repo root.
`)
}

function cleanText(value) {
  return value
    .replace(/\r\n/g, '\n')
    .replace(/[ \t]+\n/g, '\n')
    .trim()
}

function pathParts(value) {
  return value.split(path.sep).filter(Boolean)
}

function stripExtension(filePath) {
  const parsed = path.parse(filePath)
  if (!parsed.ext) return filePath
  return path.join(parsed.dir, parsed.name)
}

function outputTextPathFromRelative(relativeInputPath, outputRoot) {
  const parsed = path.parse(relativeInputPath)
  return path.join(outputRoot, parsed.dir, `${parsed.name}.txt`)
}

async function ensureDirectory(filePath) {
  await fs.mkdir(path.dirname(filePath), { recursive: true })
}

function inferDatasetLanguage(relativeInputPath, inputRoot, options) {
  if (options.defaultLanguage) {
    return options.defaultLanguage
  }

  const parts = pathParts(relativeInputPath)
  const firstSegment = parts[0]?.toLowerCase()
  if (firstSegment && OCR_LANGUAGE_BY_DATASET_LANGUAGE[firstSegment]) {
    return firstSegment
  }

  const rootBase = path.basename(inputRoot).toLowerCase()
  if (OCR_LANGUAGE_BY_DATASET_LANGUAGE[rootBase]) {
    return rootBase
  }

  return 'eng'
}

function ensureLanguagePrefixedRelativePath(
  relativeInputPath,
  datasetLanguage
) {
  const parts = pathParts(relativeInputPath)
  if (parts[0]?.toLowerCase() === datasetLanguage) {
    return relativeInputPath
  }
  return path.join(datasetLanguage, relativeInputPath)
}

function shouldIncludeLanguage(relativeInputPath, inputRoot, options) {
  if (!options.languageFilters || options.languageFilters.length === 0) {
    return true
  }
  const datasetLanguage = inferDatasetLanguage(
    relativeInputPath,
    inputRoot,
    options
  )
  return options.languageFilters.includes(datasetLanguage)
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
      const lowerName = entry.name.toLowerCase()

      if (entry.isDirectory()) {
        if (SKIP_DIRECTORIES.has(lowerName)) continue
        stack.push(candidate)
        continue
      }
      if (!entry.isFile()) continue

      const relativeCandidate = path.relative(absoluteRoot, candidate)
      if (!shouldIncludeLanguage(relativeCandidate, absoluteRoot, options)) {
        continue
      }

      const ext = path.extname(lowerName)
      if (PDF_EXTS.has(ext)) {
        files.push(candidate)
        continue
      }
      if (options.includeZips && ZIP_EXTS.has(ext)) {
        files.push(candidate)
      }
    }
  }

  return files.sort((a, b) => a.localeCompare(b))
}

async function extractZipWithPython(zipPath, extractTo) {
  const script = `
import json
import pathlib
import sys
import zipfile

zip_path = pathlib.Path(sys.argv[1]).resolve()
output_dir = pathlib.Path(sys.argv[2]).resolve()
written = []

output_dir.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(zip_path, 'r') as archive:
    for name in sorted(archive.namelist()):
        lower = name.lower()
        if lower.endswith('/'):
            continue
        if pathlib.Path(lower).suffix != '.pdf':
            continue

        target = (output_dir / name).resolve()
        if not str(target).startswith(str(output_dir)):
            continue

        target.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(name) as source, target.open('wb') as destination:
            destination.write(source.read())
        written.append(str(target))

print(json.dumps(written))
`

  const args = ['-c', script, zipPath, extractTo]
  const child = spawn('python', args, { stdio: ['ignore', 'pipe', 'pipe'] })

  let stdout = ''
  let stderr = ''
  child.stdout.on('data', (chunk) => {
    stdout += chunk.toString()
  })
  child.stderr.on('data', (chunk) => {
    stderr += chunk.toString()
  })

  const code = await new Promise((resolve, reject) => {
    child.on('error', reject)
    child.on('close', resolve)
  })

  if (code !== 0) {
    throw new Error(`zip extraction failed: ${stderr}`)
  }

  return JSON.parse(stdout || '[]')
}

async function resolvePdfSupport() {
  const pdfjs = await import('pdfjs-dist/legacy/build/pdf.mjs')
  const pdfjsRoot = path.resolve(process.cwd(), 'node_modules/pdfjs-dist')
  let createCanvas = null

  try {
    const canvasLib = await import('@napi-rs/canvas')
    createCanvas = canvasLib.createCanvas
  } catch (firstError) {
    try {
      const canvasLib = await import('canvas')
      createCanvas = canvasLib.createCanvas
    } catch (secondError) {
      createCanvas = null
    }
  }

  return {
    pdfjs,
    createCanvas,
    pdfOptions: {
      cMapUrl: `${path.join(pdfjsRoot, 'cmaps')}${path.sep}`,
      cMapPacked: true,
      standardFontDataUrl: `${path.join(
        pdfjsRoot,
        'standard_fonts'
      )}${path.sep}`,
      wasmUrl: `${path.join(pdfjsRoot, 'wasm')}${path.sep}`,
      useWorkerFetch: false,
    },
  }
}

function resolveTesseractLanguage(datasetLanguage) {
  return OCR_LANGUAGE_BY_DATASET_LANGUAGE[datasetLanguage] ?? 'eng'
}

async function getWorkerForLanguage(datasetLanguage, workerCache) {
  const ocrLanguage = resolveTesseractLanguage(datasetLanguage)
  if (workerCache.has(ocrLanguage)) {
    return workerCache.get(ocrLanguage)
  }

  console.log(`Loading OCR worker for ${datasetLanguage} (${ocrLanguage})`)
  await fs.mkdir(TESSERACT_ARTIFACTS_PATH, { recursive: true })
  const worker = await createWorker(ocrLanguage, 1, {
    cachePath: TESSERACT_ARTIFACTS_PATH,
  })
  workerCache.set(ocrLanguage, worker)
  return worker
}

async function terminateWorkers(workerCache) {
  for (const worker of workerCache.values()) {
    await worker.terminate()
  }
}

function normalizedTextItems(items) {
  return items
    .filter(
      (item) => typeof item?.str === 'string' && item.str.trim().length > 0
    )
    .map((item) => ({
      text: item.str.replace(/\s+/g, ' ').trim(),
      x: Number(item.transform?.[4] ?? 0),
      y: Number(item.transform?.[5] ?? 0),
      width: Number(item.width ?? 0),
    }))
    .sort((a, b) => {
      if (Math.abs(a.y - b.y) > 2.5) {
        return b.y - a.y
      }
      return a.x - b.x
    })
}

function joinLineText(items) {
  const sorted = [...items].sort((a, b) => a.x - b.x)
  let line = ''
  let lastRight = null

  for (const item of sorted) {
    if (!item.text) continue

    if (!line) {
      line = item.text
      lastRight = item.x + item.width
      continue
    }

    const gap = lastRight == null ? 0 : item.x - lastRight
    const noSpaceBefore = /^[,.;:!?%)}\]]/.test(item.text)
    const noSpaceAfter = /[(\[{/]$/.test(line)
    const separator = noSpaceBefore || noSpaceAfter || gap < 0.75 ? '' : ' '
    line += `${separator}${item.text}`
    lastRight = Math.max(lastRight ?? item.x, item.x + item.width)
  }

  return cleanText(line)
}

function textFromPdfItems(items) {
  const normalizedItems = normalizedTextItems(items)
  if (normalizedItems.length === 0) return ''

  const lines = []
  for (const item of normalizedItems) {
    const currentLine = lines.at(-1)
    if (!currentLine || Math.abs(currentLine.y - item.y) > 2.5) {
      lines.push({ y: item.y, items: [item] })
      continue
    }
    currentLine.items.push(item)
  }

  return lines
    .map((line) => joinLineText(line.items))
    .filter(Boolean)
    .join('\n')
}

async function renderPdfPageToPng(page, createCanvas) {
  const viewport = page.getViewport({ scale: 2.0 })
  const canvas = createCanvas(
    Math.ceil(viewport.width),
    Math.ceil(viewport.height)
  )
  const context = canvas.getContext('2d')
  await page.render({ canvasContext: context, canvas, viewport }).promise
  return canvas.toBuffer('image/png')
}

async function recognizeTextFromPdf(
  inputPath,
  datasetLanguage,
  pdfSupport,
  options,
  workerCache
) {
  const pdfBuffer = await fs.readFile(inputPath)
  const loadingTask = pdfSupport.pdfjs.getDocument({
    data: new Uint8Array(pdfBuffer),
    ...pdfSupport.pdfOptions,
  })
  const document = await loadingTask.promise
  const pageTexts = []

  try {
    for (let pageIndex = 1; pageIndex <= document.numPages; pageIndex += 1) {
      const page = await document.getPage(pageIndex)

      try {
        const textContent = await page.getTextContent()
        let pageText = cleanText(textFromPdfItems(textContent.items))

        if (
          !pageText &&
          options.ocrFallback &&
          typeof pdfSupport.createCanvas === 'function'
        ) {
          const worker = await getWorkerForLanguage(
            datasetLanguage,
            workerCache
          )
          const pageImage = await renderPdfPageToPng(
            page,
            pdfSupport.createCanvas
          )
          const result = await worker.recognize(pageImage)
          pageText = cleanText(result?.data?.text ?? '')
        }

        pageTexts.push(pageText)
      } finally {
        await page.cleanup()
      }
    }
  } finally {
    await document.destroy()
  }

  return cleanText(pageTexts.filter(Boolean).join('\n\n'))
}

function chunk(items, chunkCount) {
  const groups = Array.from({ length: chunkCount }, () => [])
  for (let index = 0; index < items.length; index += 1) {
    groups[index % chunkCount].push(items[index])
  }
  return groups.filter((group) => group.length > 0)
}

function datasetShardKey(filePath, inputRoot, options) {
  const relativeInputPath = path.relative(inputRoot, filePath)
  const parts = pathParts(relativeInputPath)
  const datasetLanguage = inferDatasetLanguage(
    relativeInputPath,
    inputRoot,
    options
  )
  const datasetName = parts[1] ?? 'root'
  return `${datasetLanguage}/${datasetName}`
}

function shardFiles(files, inputRoot, options, workerCount) {
  if (workerCount <= 1) {
    return [files]
  }

  const filesByShard = new Map()
  for (const file of files) {
    const shardKey = datasetShardKey(file, inputRoot, options)
    const shard = filesByShard.get(shardKey) ?? []
    shard.push(file)
    filesByShard.set(shardKey, shard)
  }

  const buckets = Array.from({ length: workerCount }, () => [])
  const loads = Array.from({ length: workerCount }, () => 0)
  const shards = [...filesByShard.values()].sort((left, right) => {
    return right.length - left.length
  })

  for (const shard of shards) {
    let targetIndex = 0
    for (let index = 1; index < loads.length; index += 1) {
      if (loads[index] < loads[targetIndex]) {
        targetIndex = index
      }
    }
    buckets[targetIndex].push(...shard)
    loads[targetIndex] += shard.length
  }

  return buckets.filter((bucket) => bucket.length > 0)
}

function runFileWorker(payload) {
  return new Promise((resolve, reject) => {
    const worker = new Worker(new URL(import.meta.url), {
      workerData: {
        mode: 'process-files',
        ...payload,
      },
    })
    worker.once('message', resolve)
    worker.once('error', reject)
    worker.once('exit', (code) => {
      if (code !== 0) {
        reject(new Error(`PDF OCR worker exited with code ${code}`))
      }
    })
  })
}

async function processFiles(files, inputRoot, outputRoot, options) {
  const pdfSupport = await resolvePdfSupport()
  const workerCache = new Map()
  const failures = []
  let successCount = 0
  let skippedCount = 0

  if (options.ocrFallback && !pdfSupport.createCanvas) {
    console.log(
      'Note: scanned-PDF OCR fallback requires "@napi-rs/canvas" or "canvas". Text-layer extraction still works.'
    )
  }

  try {
    for (const inputPath of files) {
      const ext = path.extname(inputPath).toLowerCase()
      const relativeInputPath = path.relative(inputRoot, inputPath)
      const datasetLanguage = inferDatasetLanguage(
        relativeInputPath,
        inputRoot,
        options
      )
      const normalizedRelativeInputPath = ensureLanguagePrefixedRelativePath(
        relativeInputPath,
        datasetLanguage
      )

      try {
        if (ext === '.zip') {
          const tmpExtractDir = path.join(
            os.tmpdir(),
            `receipt-pdf-text-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`
          )

          try {
            const extracted = await extractZipWithPython(
              inputPath,
              tmpExtractDir
            )

            for (const extractedPath of extracted) {
              const relativeInsideZip = path.relative(
                tmpExtractDir,
                extractedPath
              )
              const virtualRelative = path.join(
                stripExtension(normalizedRelativeInputPath),
                relativeInsideZip
              )
              const text = await recognizeTextFromPdf(
                extractedPath,
                datasetLanguage,
                pdfSupport,
                options,
                workerCache
              )

              if (!text) {
                skippedCount += 1
                failures.push(
                  `${virtualRelative}: no text layer found and OCR fallback produced no text`
                )
                continue
              }

              const destination = outputTextPathFromRelative(
                virtualRelative,
                outputRoot
              )
              await ensureDirectory(destination)
              await fs.writeFile(destination, `${text}\n`)
              successCount += 1
            }
          } finally {
            await fs.rm(tmpExtractDir, { recursive: true, force: true })
          }

          continue
        }

        const text = await recognizeTextFromPdf(
          inputPath,
          datasetLanguage,
          pdfSupport,
          options,
          workerCache
        )

        if (!text) {
          skippedCount += 1
          failures.push(
            `${normalizedRelativeInputPath}: no text layer found and OCR fallback produced no text`
          )
          continue
        }

        const destination = outputTextPathFromRelative(
          normalizedRelativeInputPath,
          outputRoot
        )
        await ensureDirectory(destination)
        await fs.writeFile(destination, `${text}\n`)
        successCount += 1
      } catch (error) {
        skippedCount += 1
        failures.push(
          `${normalizedRelativeInputPath}: ${error?.message ?? String(error)}`
        )
      }
    }
  } finally {
    await terminateWorkers(workerCache)
  }

  return {
    successCount,
    skippedCount,
    failures,
  }
}

async function run() {
  const options = parseArgs(process.argv.slice(2))
  if (options.help) {
    printHelp()
    return
  }

  const inputRoot = path.resolve(process.cwd(), options.input)
  const outputRoot = path.resolve(process.cwd(), options.output)
  const files = await findInputFiles(inputRoot, options)

  if (files.length === 0) {
    console.log(`No supported files found in ${inputRoot}`)
    return
  }

  const groups = shardFiles(
    files,
    inputRoot,
    options,
    Math.min(options.workers, files.length)
  )
  const results =
    groups.length === 1
      ? [await processFiles(groups[0], inputRoot, outputRoot, options)]
      : await Promise.all(
          groups.map((group) =>
            runFileWorker({
              files: group,
              inputRoot,
              outputRoot,
              options,
            })
          )
        )

  const successCount = results.reduce(
    (sum, result) => sum + result.successCount,
    0
  )
  const skippedCount = results.reduce(
    (sum, result) => sum + result.skippedCount,
    0
  )
  const failures = results.flatMap((result) => result.failures)

  console.log(`Done. PDF text outputs written to ${outputRoot}`)
  console.log(`Files processed: ${successCount}`)
  console.log(`Files skipped: ${skippedCount}`)

  if (failures.length > 0) {
    console.log('')
    console.log('Failures:')
    for (const issue of failures) {
      console.log(` - ${issue}`)
    }
    if (options.failOnError) {
      process.exitCode = 1
    }
  }
}

async function workerMain() {
  if (workerData?.mode !== 'process-files') {
    throw new Error('Unsupported worker mode')
  }

  const result = await processFiles(
    workerData.files,
    workerData.inputRoot,
    workerData.outputRoot,
    workerData.options
  )
  parentPort?.postMessage(result)
}

if (isMainThread) {
  run().catch((error) => {
    console.error(error)
    process.exit(1)
  })
} else {
  workerMain().catch((error) => {
    throw error
  })
}
