import fg from 'fast-glob'
import { access } from 'node:fs/promises'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const scriptDir = dirname(fileURLToPath(import.meta.url))
const outputsDir = resolve(scriptDir, '..', 'samples', 'outputs')

try {
  await access(outputsDir)
} catch {
  console.log(0)
  process.exit(0)
}

const outputFiles = await fg('**/*.jsonld', {
  cwd: outputsDir,
  onlyFiles: true,
})

console.log(outputFiles.length)
