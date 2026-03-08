import fg from 'fast-glob'
import { access } from 'node:fs/promises'
import { resolve } from 'node:path'

const outputsDir = resolve(process.cwd(), 'models', 'samples', 'outputs')

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
