#!/usr/bin/env node
import path from 'node:path'
import process from 'node:process'
import { spawn } from 'node:child_process'

function parseArgs(argv) {
  const parsed = {
    language: '',
    workers: null,
    help: false,
  }

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index]

    if (arg === '--help' || arg === '-h') {
      parsed.help = true
      continue
    }
    if ((arg === '--language' || arg === '--lang') && argv[index + 1]) {
      parsed.language = argv[index + 1]
      index += 1
      continue
    }
    if (arg === '--workers' && argv[index + 1]) {
      const value = Number(argv[index + 1])
      if (!Number.isFinite(value) || value < 1) {
        throw new Error(`Invalid --workers value: ${argv[index + 1]}`)
      }
      parsed.workers = Math.floor(value)
      index += 1
      continue
    }
    if (!arg.startsWith('-') && !parsed.language) {
      parsed.language = arg
      continue
    }

    throw new Error(
      `Unknown option: ${arg}. Use the underlying scripts for advanced flags.`
    )
  }

  return parsed
}

function printHelp() {
  console.log(`Usage:
node models/scripts/data/run-pipeline.mjs --language <lang> [options]

Options:
  --help                 Show this help text
  --language, --lang     Language code, for example: eng
  --workers <n>          Worker count forwarded to extract-inputs.mjs

What it does:
  1. Downloads and extracts raw receipt inputs for one language
  2. Builds tokenizers/{lang}/corpus.jsonl
  3. Trains tokenizers/{lang}/tokenizer.{model,vocab}
  4. Builds datasets/{lang}/{train,validation}.jsonl

Use the underlying scripts in models/scripts/data/ for advanced options.
`)
}

function runCommand(command, args) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      stdio: 'inherit',
    })
    child.on('error', reject)
    child.on('close', (code) => {
      if (code === 0) {
        resolve()
        return
      }
      reject(new Error(`${path.basename(args[0] ?? command)} exited with code ${code}`))
    })
  })
}

async function main() {
  const options = parseArgs(process.argv.slice(2))
  if (options.help) {
    printHelp()
    return
  }
  if (!options.language) {
    throw new Error('Missing required --language value')
  }

  const scriptRoot = path.resolve(process.cwd(), 'models', 'scripts')
  const extractArgs = ['--languages', options.language]
  if (options.workers !== null) {
    extractArgs.push('--workers', String(options.workers))
  }

  await runCommand(process.execPath, [
    path.join(scriptRoot, 'data', 'extract-inputs.mjs'),
    ...extractArgs,
  ])
  await runCommand(process.execPath, [
    path.join(scriptRoot, 'data', 'transform-samples-to-corpus.mjs'),
  ])
  await runCommand('python', [
    path.join(scriptRoot, 'data', 'train_tokenizer_vocabs.py'),
    '--language',
    options.language,
  ])
  await runCommand(process.execPath, [
    path.join(scriptRoot, 'data', 'transform-corpus-to-tokenized-dataset.mjs'),
    '--language',
    options.language,
  ])
}

main().catch((error) => {
  console.error(error)
  process.exit(1)
})
