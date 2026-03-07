import { mkdir, readFile, writeFile } from 'node:fs/promises'
import { dirname } from 'node:path'

const js = String.raw

const modelPath = './models/quantized_models/eng/model.int4.onnx'
const modelDataPath = './models/quantized_models/eng/model.int4.onnx.data'
const tokenizerModelPath = './models/quantized_models/eng/tokenizer.model'

const outPath = './src/Model/class.ts'

function toUint8ArraySource(bytes) {
  return `new Uint8Array([${Array.from(bytes).join(',')}])`
}

const [model, modelData, tokenizerModel] = await Promise.all([
  readFile(modelPath),
  readFile(modelDataPath),
  readFile(tokenizerModelPath),
])

const ts = js`
import * as ort from 'onnxruntime-web'
import { SentencePieceProcessor } from '@agnai/sentencepiece-js'

export async function createInferenceSession(): Promise<ort.InferenceSession> {
  return ort.InferenceSession.create(${toUint8ArraySource(model)}, {
    externalData: [
      {
        path: 'model.int4.onnx.data',
        data: ${toUint8ArraySource(modelData)},
      },
    ],
  })
}

export async function createTokenProcessor():Promise<SentencePieceProcessor> {
  const tokenProcessor = new SentencePieceProcessor()
  await tokenProcessor.load(${toUint8ArraySource(tokenizerModel)})
  return tokenProcessor
}
`.trimStart()

await mkdir(dirname(outPath), { recursive: true })
await writeFile(outPath, ts, 'utf8')
