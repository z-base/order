#!/usr/bin/env node
import { spawn } from 'node:child_process'
import fs from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import process from 'node:process'
import { createWorker } from 'tesseract.js'

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

const DEFAULT_OPTIONS = {
  input: 'models/data/source_images',
  output: 'models/data/extracted_texts',
  includePdfs: false,
  includeZips: true,
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
  }

  return parsed
}

function printHelp() {
  console.log(`Usage:
node models/scripts/recognize-text-from-receipt-images.mjs [options]

Options:
  --help                 Show this help text
  --input <path>         Source image root (default: models/data/source_images)
  --output <path>        OCR output root (default: models/data/extracted_texts)
  --with-pdf             Render PDFs (if optional PDF renderer packages are installed)
  --no-pdf               Skip PDFs (default)
  --with-zip             Extract and OCR supported files from zip archives (default)
  --no-zip               Skip zip archives
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
    const canvasLib = await import('canvas')
    return {
      pdfjs: pdfLib,
      createCanvas: canvasLib.createCanvas,
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
  const document = await pdfRenderer.pdfjs.getDocument({ data: pdfBuffer })
    .promise
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

  const worker = await createWorker()
  const pdfRenderer = options.includePdfs ? await resolvePdfRenderer() : null
  const imageProcessor = await resolveImageProcessor()
  const failures = []
  let successCount = 0
  let skippedCount = 0

  if (!imageProcessor) {
    console.log(
      'Note: optional package "sharp" is not installed, so tiny-image normalization is disabled.'
    )
  }

  try {
    for (const inputPath of files) {
      const ext = path.extname(inputPath).toLowerCase()
      const relativeInputPath = path.relative(inputRoot, inputPath)

      try {
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
                stripExtension(relativeInputPath),
                relativeInsideZip
              )

              if (PDF_EXTS.has(extractedExt)) {
                if (!pdfRenderer) {
                  skippedCount += 1
                  failures.push(
                    `${virtualRelative}: PDF renderer unavailable (install pdfjs-dist + canvas)`
                  )
                  continue
                }

                const pages = await renderPdfPages(extractedPath, pdfRenderer)
                for (
                  let pageIndex = 0;
                  pageIndex < pages.length;
                  pageIndex += 1
                ) {
                  const result = await worker.recognize(pages[pageIndex])
                  const text = cleanText(result?.data?.text ?? '')
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
              const result = await worker.recognize(preparedImage)
              const text = cleanText(result?.data?.text ?? '')
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
              `${relativeInputPath}: PDF renderer unavailable (install pdfjs-dist + canvas)`
            )
            continue
          }

          const pages = await renderPdfPages(inputPath, pdfRenderer)
          if (pages.length === 0) {
            skippedCount += 1
            continue
          }

          for (let i = 0; i < pages.length; i += 1) {
            const result = await worker.recognize(pages[i])
            const text = cleanText(result?.data?.text ?? '')
            const destination = outputPdfPagePathFromRelative(
              relativeInputPath,
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
        const result = await worker.recognize(preparedImage)
        const text = cleanText(result?.data?.text ?? '')
        const destination = outputTextPathFromRelative(
          relativeInputPath,
          outputRoot
        )
        await ensureDirectory(destination)
        await fs.writeFile(destination, `${text}\n`)
        successCount += 1
      } catch (error) {
        skippedCount += 1
        failures.push(
          `${relativeInputPath}: ${error?.message ?? String(error)}`
        )
      }
    }
  } finally {
    await worker.terminate()
  }

  console.log(`Done. OCR outputs written to ${outputRoot}`)
  console.log(`Files processed: ${successCount}`)
  console.log(`Files skipped: ${skippedCount}`)

  if (failures.length > 0) {
    console.log('')
    console.log('Failures:')
    for (const issue of failures) {
      console.log(` - ${issue}`)
    }
    process.exitCode = 1
  }
}

run().catch((error) => {
  console.error(error)
  process.exit(1)
})
