import { unpack } from '@nodearchive/nodearchive'
import fs from 'fs/promises'
import dotenv from 'dotenv'
import { parseArgs } from 'util'
import { env } from 'process'

console.log(env)

const { supported_languages } = dotenv.parse(await fs.readFile('./.env'))

const normalized_languages = supported_languages
  .split(',')
  .map((lang) => lang.trim())

const dir = await fs.readdir('./models/.data')

console.log(dir)
