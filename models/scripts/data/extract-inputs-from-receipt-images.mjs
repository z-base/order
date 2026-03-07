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

const IMAGE_EXTS = new Set([
  '.png',
  '.jpg',
  '.jpeg',
  '.webp',
  '.bmp',
  '.tif',
  '.tiff',
  '.gif',
])
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
  input: 'models/data/downloaded_images',
  output: 'models/samples/inputs',
  includePdfs: false,
  includeZips: true,
  defaultLanguage: null,
  languageFilters: null,
  failOnError: false,
  workers: Math.max(1, Math.min(os.availableParallelism(), 2)),
}

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
    if (arg === '--with-pdf') {
      parsed.includePdfs = true
      continue
    }
    if (arg === '--no-pdf') {
      parsed.includePdfs = false
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
    if (arg === '--fail-on-error') {
      parsed.failOnError = true
      continue
    }
  }

  return parsed
}

function printHelp() {
  console.log(`Usage:
node models/scripts/data/extract-inputs-from-receipt-images.mjs [options]

Options:
  --help                 Show this help text
  --input <path>         Download root (default: models/data/downloaded_images)
  --output <path>        OCR output root (default: models/samples/inputs)
  --language <code>      Force one dataset language for the whole run
  --languages <codes>    Restrict to top-level dataset languages (comma-separated)
  --workers <n>          Worker-thread concurrency across file shards
  --with-pdf             Also OCR PDFs from the image tree
  --no-pdf               Skip PDFs (default)
  --with-zip             Extract and OCR supported files from zip archives (default)
  --no-zip               Skip zip archives
  --fail-on-error        Exit non-zero if any file fails

Notes:
  Relative paths are preserved, so downloaded_images/{lang}/... becomes samples/inputs/{lang}/...
  OCR workers are selected from the top-level language directory when available.
  For PDF datasets, use models/scripts/data/extract-inputs-from-receipt-pdfs.mjs.
`)
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

function outputPdfPagePathFromRelative(
  relativeInputPath,
  pageIndex,
  outputRoot
) {
  const parsed = path.parse(relativeInputPath)
  const directory = path.join(outputRoot, parsed.dir, parsed.name)
  return path.join(directory, `page-${String(pageIndex).padStart(3, '0')}.txt`)
}

async function ensureDirectory(filePath) {
  await fs.mkdir(path.dirname(filePath), { recursive: true })
}

function cleanText(value) {
  return value.replace(/\r\n/g, '\n').trim()
}

function pathParts(value) {
  return value.split(path.sep).filter(Boolean)
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
      if (IMAGE_EXTS.has(ext)) {
        files.push(candidate)
        continue
      }
      if (options.includePdfs && PDF_EXTS.has(ext)) {
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

async function resolvePdfRenderer() {
  try {
    const pdfLib = await import('pdfjs-dist/legacy/build/pdf.mjs')
    let canvasLib = null

    try {
      canvasLib = await import('@napi-rs/canvas')
    } catch (firstError) {
      try {
        canvasLib = await import('canvas')
      } catch (secondError) {
        canvasLib = null
      }
    }

    if (!canvasLib?.createCanvas) {
      return null
    }

    const pdfjsRoot = path.resolve(process.cwd(), 'node_modules/pdfjs-dist')

    return {
      pdfjs: pdfLib,
      createCanvas: canvasLib.createCanvas,
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
  } catch (error) {
    return null
  }
}

async function resolveImageProcessor() {
  try {
    const sharpModule = await import('sharp')
    const sharp = sharpModule.default ?? sharpModule
    return sharp
  } catch (error) {
    return null
  }
}

async function renderPdfPages(inputPath, pdfRenderer) {
  if (!pdfRenderer) return []

  const pdfBuffer = await fs.readFile(inputPath)
  const document = await pdfRenderer.pdfjs.getDocument({
    data: new Uint8Array(pdfBuffer),
    ...pdfRenderer.pdfOptions,
  }).promise
  const pageCount = document.numPages
  const renderedPages = []

  for (let pageIndex = 1; pageIndex <= pageCount; pageIndex += 1) {
    const page = await document.getPage(pageIndex)
    const viewport = page.getViewport({ scale: 2.0 })
    const canvas = pdfRenderer.createCanvas(
      Math.ceil(viewport.width),
      Math.ceil(viewport.height)
    )
    const context = canvas.getContext('2d')
    await page.render({ canvasContext: context, canvas, viewport }).promise
    renderedPages.push(canvas.toBuffer('image/png'))
    await page.cleanup()
  }

  await document.destroy()
  return renderedPages
}

async function extractZipWithPython(zipPath, extractTo, includePdfs) {
  const allowedExts = includePdfs
    ? [...IMAGE_EXTS, ...PDF_EXTS]
    : [...IMAGE_EXTS]
  const script = `
import json
import pathlib
import sys
import zipfile

zip_path = pathlib.Path(sys.argv[1]).resolve()
output_dir = pathlib.Path(sys.argv[2]).resolve()
allowed_exts = set(json.loads(sys.argv[3]))
written = []

output_dir.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(zip_path, 'r') as archive:
    for name in sorted(archive.namelist()):
        lower = name.lower()
        if lower.endswith('/'):
            continue
        suffix = pathlib.Path(lower).suffix
        if suffix not in allowed_exts:
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

  const args = ['-c', script, zipPath, extractTo, JSON.stringify(allowedExts)]
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

async function normalizeImageForOcr(imagePath, imageProcessor) {
  if (!imageProcessor) return imagePath

  const metadata = await imageProcessor(imagePath, {
    failOn: 'none',
  }).metadata()
  const width = Number(metadata.width || 0)
  const height = Number(metadata.height || 0)
  if (width <= 0 || height <= 0) return imagePath

  const minEdge = 32
  const scale = Math.max(minEdge / width, minEdge / height, 1)
  const resizedWidth = Math.max(minEdge, Math.ceil(width * scale))
  const resizedHeight = Math.max(minEdge, Math.ceil(height * scale))

  let pipeline = imageProcessor(imagePath, { failOn: 'none' })
  if (resizedWidth !== width || resizedHeight !== height) {
    pipeline = pipeline.resize(resizedWidth, resizedHeight, {
      kernel: imageProcessor.kernel.nearest,
    })
  }

  return pipeline.png().toBuffer()
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
  const worker = await createWorker(ocrLanguage)
  workerCache.set(ocrLanguage, worker)
  return worker
}

async function recognizeText(worker, input) {
  const result = await worker.recognize(input)
  return cleanText(result?.data?.text ?? '')
}

async function terminateWorkers(workerCache) {
  for (const worker of workerCache.values()) {
    await worker.terminate()
  }
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
        reject(new Error(`Image OCR worker exited with code ${code}`))
      }
    })
  })
}

async function processFiles(files, inputRoot, outputRoot, options) {
  const pdfRenderer = options.includePdfs ? await resolvePdfRenderer() : null
  const imageProcessor = await resolveImageProcessor()
  const workerCache = new Map()
  const failures = []
  let successCount = 0
  let skippedCount = 0

  if (!imageProcessor) {
    console.log(
      'Note: optional package "sharp" is not installed, so tiny-image normalization is disabled.'
    )
  }

  if (options.includePdfs && !pdfRenderer) {
    console.log(
      'Note: PDF OCR requires "pdfjs-dist" plus either "@napi-rs/canvas" or "canvas". PDF files will be skipped.'
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
        const worker = await getWorkerForLanguage(datasetLanguage, workerCache)

        if (ext === '.zip') {
          const tmpExtractDir = path.join(
            os.tmpdir(),
            `receipt-ocr-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`
          )
          try {
            const extracted = await extractZipWithPython(
              inputPath,
              tmpExtractDir,
              options.includePdfs
            )

            for (const extractedPath of extracted) {
              const extractedExt = path.extname(extractedPath).toLowerCase()
              const relativeInsideZip = path.relative(
                tmpExtractDir,
                extractedPath
              )
              const virtualRelative = path.join(
                stripExtension(normalizedRelativeInputPath),
                relativeInsideZip
              )

              if (PDF_EXTS.has(extractedExt)) {
                if (!pdfRenderer) {
                  skippedCount += 1
                  failures.push(
                    `${virtualRelative}: PDF renderer unavailable (install pdfjs-dist + @napi-rs/canvas or canvas)`
                  )
                  continue
                }

                const pages = await renderPdfPages(extractedPath, pdfRenderer)
                for (
                  let pageIndex = 0;
                  pageIndex < pages.length;
                  pageIndex += 1
                ) {
                  const text = await recognizeText(worker, pages[pageIndex])
                  const destination = outputPdfPagePathFromRelative(
                    virtualRelative,
                    pageIndex + 1,
                    outputRoot
                  )
                  await ensureDirectory(destination)
                  await fs.writeFile(destination, `${text}\n`)
                }

                successCount += 1
                continue
              }

              const preparedImage = await normalizeImageForOcr(
                extractedPath,
                imageProcessor
              )
              const text = await recognizeText(worker, preparedImage)
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

        if (ext === '.pdf') {
          if (!pdfRenderer) {
            skippedCount += 1
            failures.push(
              `${normalizedRelativeInputPath}: PDF renderer unavailable (install pdfjs-dist + @napi-rs/canvas or canvas)`
            )
            continue
          }

          const pages = await renderPdfPages(inputPath, pdfRenderer)
          if (pages.length === 0) {
            skippedCount += 1
            continue
          }

          for (let i = 0; i < pages.length; i += 1) {
            const text = await recognizeText(worker, pages[i])
            const destination = outputPdfPagePathFromRelative(
              normalizedRelativeInputPath,
              i + 1,
              outputRoot
            )
            await ensureDirectory(destination)
            await fs.writeFile(destination, `${text}\n`)
          }

          successCount += 1
          continue
        }

        const preparedImage = await normalizeImageForOcr(
          inputPath,
          imageProcessor
        )
        const text = await recognizeText(worker, preparedImage)
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

  console.log(`Done. OCR outputs written to ${outputRoot}`)
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
