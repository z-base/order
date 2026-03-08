import { parseArgs } from 'util'
import unzipper from 'unzipper'

export async function unzip(path) {
  return await unzipper.Open.file(path)
}

async function cliRun() {
  const { values } = parseArgs({
    options: {
      path: { type: 'string', short: 'p' },
    },
    allowPositionals: false,
  })

  const { path } = values

  if (!path)
    throw new Error(
      'You must provide a file path using either --path ./path/here or -p ./path/here.'
    )
  const result = await unzip(path)
  console.info('[INFO]', result)
}

if (process.argv) {
  cliRun().catch((error) => {
    console.error('[ERROR]', error)
    process.exit(1)
  })
}
