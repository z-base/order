import { createWorker } from 'tesseract.js'

const ocrWorker = await createWorker(['fin', 'swe', 'eng'])

ocrWorker.readText()
