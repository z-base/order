#!/usr/bin/env node
import fs from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import process from 'node:process'
import { spawn } from 'node:child_process'

const DEFAULT_OPTIONS = {
  languages: null,
  workers: Math.max(1, Math.min(os.availableParallelism(), 8)),
}

function parseArgs(argv) {
  const parsed = { ...DEFAULT_OPTIONS }
  const npmConfigWorkers = Number(process.env.npm_config_workers)

  if (Number.isFinite(npmConfigWorkers) && npmConfigWorkers > 0) {
    parsed.workers = Math.max(1, Math.floor(npmConfigWorkers))
  }

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index]

    if (arg === '--help' || arg === '-h') {
      parsed.help = true
      continue
    }
    if (arg === '--languages') {
      const raw = argv[index + 1]
      if (!raw) {
        throw new Error('--languages requires a comma-separated list')
      }
      parsed.languages = raw
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
    const positionalWorkers = Number(arg)
    if (
      !arg.startsWith('-') &&
      Number.isFinite(positionalWorkers) &&
      positionalWorkers > 0
    ) {
      parsed.workers = Math.max(1, Math.floor(positionalWorkers))
    }
  }

  return parsed
}

function printHelp() {
  console.log(`Usage:
node models/scripts/data/extract-inputs.mjs [options]

Options:
  --help                 Show this help text
  --languages <codes>    Restrict to languages (comma-separated)
  --workers <n>          Worker setting passed to download and extraction scripts

What it does:
  1. Downloads image, PDF, and text sources
  2. Extracts OCR/text into models/samples/inputs/{lang}/...
  3. Prints final per-language input counts
`)
}

function spawnCommand(scriptPath, args) {
  return new Promise((resolve, reject) => {
    const child = spawn(process.execPath, [scriptPath, ...args], {
      stdio: 'inherit',
    })
    child.on('error', reject)
    child.on('close', (code) => {
      if (code === 0) {
        resolve()
        return
      }
      reject(new Error(`${path.basename(scriptPath)} exited with code ${code}`))
    })
  })
}

async function countInputsByLanguage(rootPath) {
  const absoluteRoot = path.resolve(process.cwd(), rootPath)
  const entries = await fs
    .readdir(absoluteRoot, { withFileTypes: true })
    .catch(() => [])
  const counts = {}

  for (const entry of entries) {
    if (!entry.isDirectory()) continue
    const languageRoot = path.join(absoluteRoot, entry.name)
    const stack = [languageRoot]
    let count = 0

    while (stack.length > 0) {
      const current = stack.pop()
      const items = await fs.readdir(current, { withFileTypes: true })
      for (const item of items) {
        const candidate = path.join(current, item.name)
        if (item.isDirectory()) {
          stack.push(candidate)
          continue
        }
        if (item.isFile() && path.extname(item.name).toLowerCase() === '.txt') {
          count += 1
        }
      }
    }

    counts[entry.name] = count
  }

  return counts
}

async function main() {
  const options = parseArgs(process.argv.slice(2))
  if (options.help) {
    printHelp()
    return
  }

  const sharedArgs = options.languages ? ['--languages', options.languages] : []
  const downloadWorkerArgs = [
    ...sharedArgs,
    '--workers',
    String(options.workers),
  ]
  const textExtractWorkerArgs = [
    ...sharedArgs,
    '--workers',
    String(options.workers),
  ]
  const pdfExtractWorkerArgs = [
    ...sharedArgs,
    '--workers',
    String(Math.max(1, Math.min(options.workers, 2))),
  ]
  const imageExtractWorkerArgs = [
    ...sharedArgs,
    '--workers',
    String(Math.max(1, Math.min(options.workers, 2))),
  ]

  console.log(`Download workers: ${options.workers}`)
  console.log(
    `Extract workers: texts=${textExtractWorkerArgs.at(-1)} pdfs=${pdfExtractWorkerArgs.at(-1)} images=${imageExtractWorkerArgs.at(-1)}`
  )

  const downloadPhase = [
    spawnCommand(
      path.resolve(
        process.cwd(),
        'models/scripts/data/download-receipt-images.mjs'
      ),
      downloadWorkerArgs
    ),
    spawnCommand(
      path.resolve(
        process.cwd(),
        'models/scripts/data/download-receipt-pdfs.mjs'
      ),
      downloadWorkerArgs
    ),
    spawnCommand(
      path.resolve(
        process.cwd(),
        'models/scripts/data/download-receipt-texts.mjs'
      ),
      downloadWorkerArgs
    ),
  ]

  await Promise.all(downloadPhase)

  await spawnCommand(
    path.resolve(
      process.cwd(),
      'models/scripts/data/extract-inputs-from-receipt-texts.mjs'
    ),
    textExtractWorkerArgs
  )
  await spawnCommand(
    path.resolve(
      process.cwd(),
      'models/scripts/data/extract-inputs-from-receipt-pdfs.mjs'
    ),
    pdfExtractWorkerArgs
  )
  await spawnCommand(
    path.resolve(
      process.cwd(),
      'models/scripts/data/extract-inputs-from-receipt-images.mjs'
    ),
    imageExtractWorkerArgs
  )

  const counts = await countInputsByLanguage('models/samples/inputs')
  console.log('')
  console.log('Input counts by language:')
  for (const language of Object.keys(counts).sort()) {
    console.log(` - ${language}: ${counts[language]}`)
  }
}

main().catch((error) => {
  console.error(error)
  process.exit(1)
})
