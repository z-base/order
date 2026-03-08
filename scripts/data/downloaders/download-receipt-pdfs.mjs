#!/usr/bin/env node
import path from 'node:path'
import process from 'node:process'
import { spawn } from 'node:child_process'

if (process.argv.includes('--help') || process.argv.includes('-h')) {
  console.log(`Usage:
node models/scripts/data/download-receipt-pdfs.mjs [options]

Options:
  --help                 Show this help text
  --source <slug/id>     Restrict download to one source (repeatable)
  --languages <codes>    Restrict to sources by language (comma-separated)
  --workers <n>          Worker-thread concurrency across sources
  --max-per-source <n>   Limit files per source (default: 80)
  --full                 Remove per-source limit

Defaults:
  manifests: models/data/pdf_sources
  output: models/data/downloaded_pdfs
  default-format: pdf

Notes:
  Sources are loaded from JSON files in models/data/pdf_sources/{lang}.json.
  Downloads are written to models/data/downloaded_pdfs/{lang}/{dataset}/...
`)
  process.exit(0)
}

const scriptPath = path.resolve(
  process.cwd(),
  'models/scripts/data/download-receipt-assets.mjs'
)

const defaultArgs = [
  scriptPath,
  '--manifests',
  'models/data/pdf_sources',
  '--output',
  'models/data/downloaded_pdfs',
  '--default-format',
  'pdf',
]

const child = spawn(
  process.execPath,
  [...defaultArgs, ...process.argv.slice(2)],
  {
    stdio: 'inherit',
  }
)

child.on('exit', (code, signal) => {
  if (signal) {
    process.kill(process.pid, signal)
    return
  }

  process.exit(code ?? 0)
})

child.on('error', (error) => {
  console.error(error)
  process.exit(1)
})
