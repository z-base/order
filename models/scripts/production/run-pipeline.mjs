#!/usr/bin/env node
import path from 'node:path'
import process from 'node:process'
import { spawn } from 'node:child_process'

function requireValue(argv, index, flag) {
  const value = argv[index + 1]
  if (!value || value.startsWith('-')) {
    throw new Error(`${flag} requires a value`)
  }
  return value
}

function parseArgs(argv) {
  const parsed = {
    language: '',
    help: false,
    exportArgs: [],
    quantizeArgs: [],
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
    if (!arg.startsWith('-') && !parsed.language) {
      parsed.language = arg
      continue
    }
    if (arg === '--force') {
      parsed.quantizeArgs.push(arg)
      continue
    }
    if (
      arg === '--opset-version' ||
      arg === '--num-heads' ||
      arg === '--bos-id' ||
      arg === '--eos-id'
    ) {
      const value = requireValue(argv, index, arg)
      parsed.exportArgs.push(arg, value)
      index += 1
      continue
    }
    if (
      arg === '--backend' ||
      arg === '--block-size' ||
      arg === '--int4-accuracy-level' ||
      arg === '--bnb4-quant-type'
    ) {
      const value = requireValue(argv, index, arg)
      parsed.quantizeArgs.push(arg, value)
      index += 1
      continue
    }

    throw new Error(
      `Unknown option: ${arg}. Use the underlying production scripts for advanced flags.`
    )
  }

  return parsed
}

function printHelp() {
  console.log(`Usage:
node models/scripts/production/run-pipeline.mjs --language <lang> [options]

Options:
  --help                      Show this help text
  --language, --lang          Language code, for example: eng
  --force                     Forwarded to quantization
  --opset-version <n>         Forwarded to ONNX export
  --num-heads <n>             Forwarded to ONNX export
  --bos-id <n>                Forwarded to ONNX export
  --eos-id <n>                Forwarded to ONNX export
  --backend <auto|int4|bnb4>  Forwarded to quantization
  --block-size <n>            Forwarded to quantization
  --int4-accuracy-level <n>   Forwarded to quantization
  --bnb4-quant-type <type>    Forwarded to quantization

What it does:
  1. Exports best_models/{lang}/best.pt into models/onnx_builds/{lang}/
  2. Quantizes the exported ONNX build into models/quantized_models/{lang}/

Use the underlying scripts in models/scripts/production/ for advanced flags.
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

  await runCommand('python', [
    path.join(scriptRoot, 'production', 'export_best_model_to_onnx.py'),
    '--language',
    options.language,
    ...options.exportArgs,
  ])
  await runCommand('python', [
    path.join(scriptRoot, 'production', 'quantize_onnx_models_to_4bit.py'),
    '--language',
    options.language,
    ...options.quantizeArgs,
  ])
}

main().catch((error) => {
  console.error(error)
  process.exit(1)
})
