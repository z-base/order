#!/usr/bin/env node
import { createWriteStream } from 'node:fs'
import fs from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import process from 'node:process'
import { spawn } from 'node:child_process'
import {
  Worker,
  isMainThread,
  parentPort,
  workerData,
} from 'node:worker_threads'

const TEXT_EXTS = [
  '.txt',
  '.json',
  '.jsonl',
  '.json5',
  '.yaml',
  '.yml',
  '.xml',
  '.csv',
  '.tsv',
  '.toml',
  '.ini',
  '.cfg',
  '.conf',
  '.log',
  '.md',
  '.rst',
  '.html',
  '.htm',
  '.xhtml',
  '.receipt',
]

const DEFAULT_OPTIONS = {
  manifests: 'models/data/text_sources',
  output: 'models/data/downloaded_texts',
  workers: Math.max(1, Math.min(os.availableParallelism(), 8)),
  full: true,
  maxPerSource: null,
  languageFilters: null,
  sourceFilters: null,
}

const MERCHANT_NAMES = [
  'Northwind Market',
  'Harbor Supply',
  'Maple Street Grocer',
  'Aster Pharmacy',
  'Cedar Hardware',
  'Bluebird Cafe',
  'Pine Electronics',
  'Atlas Office Depot',
  'Grand Stationery',
  'Summit Cleaners',
  'Riverstone Books',
  'Orchard Family Mart',
  'Brightline Bistro',
  'Corner Pantry',
  'Metro Home Goods',
  'Beacon Pet Shop',
  'Willow Bakery',
  'Oakfield Deli',
  'Union Travel Desk',
  'Lighthouse Auto Care',
]
const STREET_NAMES = [
  'Main St',
  'Market St',
  'Oak Ave',
  'Pine Rd',
  'Maple Dr',
  'Harbor Way',
  'Sunset Blvd',
  'Cedar Lane',
  'River Rd',
  'Station Ave',
]
const CITIES = [
  'Seattle',
  'Austin',
  'Portland',
  'Denver',
  'Boston',
  'Nashville',
  'Phoenix',
  'Madison',
  'Atlanta',
  'Chicago',
]
const STATES = ['WA', 'TX', 'OR', 'CO', 'MA', 'TN', 'AZ', 'WI', 'GA', 'IL']
const CURRENCIES = ['USD', 'USD', 'USD', 'USD', 'CAD', 'GBP']
const PAYMENT_METHODS = ['Cash', 'Card', 'Visa', 'Mastercard', 'Debit', 'Amex']
const CASHIER_NAMES = [
  'Alex',
  'Morgan',
  'Jordan',
  'Taylor',
  'Casey',
  'Avery',
  'Jamie',
  'Parker',
]
const ITEM_NAMES = [
  'Organic Apples',
  'Notebook Set',
  'USB-C Cable',
  'Desk Lamp',
  'Paper Towels',
  'Sparkling Water',
  'Ground Coffee',
  'Laundry Detergent',
  'Adhesive Tape',
  'AA Batteries',
  'Thermal Paper',
  'Ink Cartridge',
  'Travel Mug',
  'Olive Oil',
  'Sandwich Combo',
  'Blue Pen Pack',
  'Safety Gloves',
  'Cleaning Spray',
  'Extension Cord',
  'Printer Labels',
]

function parseArgs(argv) {
  const parsed = { ...DEFAULT_OPTIONS }

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index]

    if (arg === '--help' || arg === '-h') {
      parsed.help = true
      continue
    }
    if (arg === '--manifests') {
      parsed.manifests = argv[index + 1] ?? parsed.manifests
      index += 1
      continue
    }
    if (arg === '--output') {
      parsed.output = argv[index + 1] ?? parsed.output
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
    if (arg === '--source') {
      parsed.sourceFilters = parsed.sourceFilters ?? []
      parsed.sourceFilters.push((argv[index + 1] ?? '').trim().toLowerCase())
      index += 1
      continue
    }
    if (arg === '--languages') {
      const raw = argv[index + 1]
      if (!raw) {
        throw new Error('--languages requires a comma-separated list')
      }
      parsed.languageFilters = raw
        .split(',')
        .map((value) => value.trim().toLowerCase())
        .filter(Boolean)
      index += 1
      continue
    }
    if (arg === '--max-per-source') {
      const value = Number(argv[index + 1])
      if (!Number.isFinite(value) || value < 1) {
        throw new Error(`Invalid --max-per-source value: ${argv[index + 1]}`)
      }
      parsed.maxPerSource = Math.floor(value)
      parsed.full = false
      index += 1
      continue
    }
    if (arg === '--full') {
      parsed.full = true
      parsed.maxPerSource = null
    }
  }

  return parsed
}

function printHelp() {
  console.log(`Usage:
node models/scripts/data/download-receipt-texts.mjs [options]

Options:
  --help                 Show this help text
  --manifests <path>     Source manifest directory (default: models/data/text_sources)
  --output <path>        Destination root (default: models/data/downloaded_texts)
  --workers <n>          Worker-thread concurrency for generated sources
  --source <slug/id>     Restrict download to one source (repeatable)
  --languages <codes>    Restrict to sources by language (comma-separated)
  --max-per-source <n>   Limit generated samples per source
  --full                 Use the full declared source size (default)

Notes:
  Text manifests are loaded from models/data/text_sources/{lang}.json.
  Downloadable text sources are fetched through models/scripts/data/download-receipt-assets.mjs.
  Generator sources are sharded across worker threads and written as JSONL.
`)
}

function slugify(value) {
  return String(value || '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
}

function normalizeTextExtensions(rawExts) {
  const values =
    Array.isArray(rawExts) && rawExts.length > 0 ? rawExts : TEXT_EXTS
  return [...new Set(values.map((value) => String(value).toLowerCase()))]
}

function deriveLanguageCode(fileName, payload) {
  if (typeof payload?.language === 'string' && payload.language.trim()) {
    return payload.language.trim().toLowerCase()
  }
  return path.basename(fileName, path.extname(fileName)).toLowerCase()
}

function normalizeSourceDefinition(rawSource, language, manifestFile) {
  if (!rawSource || typeof rawSource !== 'object' || Array.isArray(rawSource)) {
    throw new Error(`Invalid source entry in ${manifestFile}`)
  }

  const kind = String(rawSource.kind || '').toLowerCase()
  if (!['generator', 'github', 'direct', 'huggingface'].includes(kind)) {
    throw new Error(
      `Unsupported source kind "${rawSource.kind}" in ${manifestFile}`
    )
  }

  const sourceIdentifier =
    rawSource.slug ||
    rawSource.id ||
    rawSource.repo ||
    rawSource.downloadUrl ||
    rawSource.title
  const slug = slugify(sourceIdentifier)
  if (!slug) {
    throw new Error(`Could not derive source slug in ${manifestFile}`)
  }

  if (kind === 'generator') {
    const count = Number(rawSource.count)
    if (!Number.isFinite(count) || count < 1) {
      throw new Error(
        `Generator source requires a positive count in ${manifestFile}`
      )
    }
  }

  return {
    ...rawSource,
    kind,
    slug,
    title: String(rawSource.title || sourceIdentifier),
    language: String(rawSource.language || language).toLowerCase(),
    includeExts: normalizeTextExtensions(rawSource.includeExts),
  }
}

async function loadSourceDefinitions(manifestRoot) {
  const absoluteRoot = path.resolve(process.cwd(), manifestRoot)
  const entries = await fs.readdir(absoluteRoot, { withFileTypes: true })
  const manifestFiles = entries
    .filter((entry) => entry.isFile() && path.extname(entry.name) === '.json')
    .map((entry) => path.join(absoluteRoot, entry.name))
    .sort((a, b) => a.localeCompare(b))

  if (manifestFiles.length === 0) {
    throw new Error(`No manifest files found in ${absoluteRoot}`)
  }

  const sources = []
  for (const manifestFile of manifestFiles) {
    const rawText = await fs.readFile(manifestFile, 'utf8')
    const payload = JSON.parse(rawText)
    const language = deriveLanguageCode(manifestFile, payload)
    const manifestSources = Array.isArray(payload)
      ? payload
      : Array.isArray(payload.sources)
        ? payload.sources
        : null

    if (!manifestSources) {
      throw new Error(
        `Manifest file must contain a "sources" array: ${manifestFile}`
      )
    }

    for (const source of manifestSources) {
      sources.push(normalizeSourceDefinition(source, language, manifestFile))
    }
  }

  return sources
}

function filterSources(sources, options) {
  return sources.filter((source) => {
    if (
      options.languageFilters &&
      options.languageFilters.length > 0 &&
      !options.languageFilters.includes(source.language)
    ) {
      return false
    }

    if (!options.sourceFilters || options.sourceFilters.length === 0) {
      return true
    }

    const keys = [
      source.slug,
      source.id,
      source.repo,
      source.downloadUrl,
      source.title,
    ]
      .filter(Boolean)
      .map((value) => String(value).toLowerCase())

    return options.sourceFilters.some((filterValue) =>
      keys.includes(filterValue)
    )
  })
}

async function spawnDownloader(manifestRoot, outputRoot, options) {
  const scriptPath = path.resolve(
    process.cwd(),
    'models/scripts/data/download-receipt-assets.mjs'
  )
  const args = [
    scriptPath,
    '--manifests',
    manifestRoot,
    '--output',
    outputRoot,
    '--default-format',
    'mixed',
    '--workers',
    String(options.workers),
  ]

  if (options.full) {
    args.push('--full')
  } else if (options.maxPerSource != null) {
    args.push('--max-per-source', String(options.maxPerSource))
  }

  const child = spawn(process.execPath, args, {
    stdio: 'inherit',
  })

  const exitCode = await new Promise((resolve, reject) => {
    child.on('error', reject)
    child.on('close', resolve)
  })

  if (exitCode !== 0) {
    throw new Error(`Shared downloader exited with code ${exitCode}`)
  }
}

function groupSourcesByLanguage(sources) {
  const grouped = new Map()
  for (const source of sources) {
    if (!grouped.has(source.language)) {
      grouped.set(source.language, [])
    }
    grouped.get(source.language).push(source)
  }
  return grouped
}

async function createTemporaryManifests(sources) {
  const tempRoot = await fs.mkdtemp(
    path.join(os.tmpdir(), 'receipt-text-manifests-')
  )
  const grouped = groupSourcesByLanguage(sources)

  for (const [language, group] of grouped.entries()) {
    const payload = {
      language,
      sources: group.map((source) => ({
        ...source,
        kind: source.kind,
        includeExts: source.includeExts,
      })),
    }
    await fs.writeFile(
      path.join(tempRoot, `${language}.json`),
      `${JSON.stringify(payload, null, 2)}\n`
    )
  }

  return tempRoot
}

function runWorker(workerPayload) {
  return new Promise((resolve, reject) => {
    const worker = new Worker(new URL(import.meta.url), {
      workerData: {
        mode: 'generate-source-shard',
        ...workerPayload,
      },
    })
    worker.once('message', resolve)
    worker.once('error', reject)
    worker.once('exit', (code) => {
      if (code !== 0) {
        reject(new Error(`Worker stopped with exit code ${code}`))
      }
    })
  })
}

async function runTaskPool(tasks, concurrency) {
  const results = []
  let nextIndex = 0

  async function workerLoop() {
    while (true) {
      const current = nextIndex
      nextIndex += 1
      if (current >= tasks.length) {
        return
      }
      results[current] = await tasks[current]()
    }
  }

  const runnerCount = Math.max(1, Math.min(concurrency, tasks.length))
  await Promise.all(Array.from({ length: runnerCount }, () => workerLoop()))
  return results
}

function makeRng(seed) {
  let state = seed >>> 0
  return () => {
    state = (state + 0x6d2b79f5) >>> 0
    let value = Math.imul(state ^ (state >>> 15), 1 | state)
    value ^= value + Math.imul(value ^ (value >>> 7), 61 | value)
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296
  }
}

function pick(list, rng) {
  return list[Math.floor(rng() * list.length)]
}

function randint(min, max, rng) {
  return Math.floor(rng() * (max - min + 1)) + min
}

function money(cents) {
  return (cents / 100).toFixed(2)
}

function isoDate(seed, rng) {
  const year = 2022 + (seed % 4)
  const month = String(randint(1, 12, rng)).padStart(2, '0')
  const day = String(randint(1, 28, rng)).padStart(2, '0')
  const hour = String(randint(8, 21, rng)).padStart(2, '0')
  const minute = String(randint(0, 59, rng)).padStart(2, '0')
  const second = String(randint(0, 59, rng)).padStart(2, '0')
  return `${year}-${month}-${day}T${hour}:${minute}:${second}`
}

function buildItems(rng) {
  const count = randint(1, 6, rng)
  const items = []

  for (let index = 0; index < count; index += 1) {
    const quantity = randint(1, 4, rng)
    const unitPriceCents = randint(199, 4599, rng)
    const lineTotalCents = quantity * unitPriceCents
    items.push({
      sku: `SKU-${randint(1000, 9999, rng)}`,
      name: pick(ITEM_NAMES, rng),
      quantity,
      unitPriceCents,
      lineTotalCents,
    })
  }

  return items
}

function buildOrder(recordIndex) {
  const rng = makeRng(recordIndex + 17)
  const items = buildItems(rng)
  const subtotalCents = items.reduce(
    (sum, item) => sum + item.lineTotalCents,
    0
  )
  const discountCents = rng() > 0.7 ? randint(25, 500, rng) : 0
  const taxCents = Math.round(
    (subtotalCents - discountCents) * (0.04 + rng() * 0.08)
  )
  const tipCents = rng() > 0.8 ? randint(100, 1200, rng) : 0
  const totalCents = subtotalCents - discountCents + taxCents + tipCents
  const receiptNumber = `R-${String(recordIndex).padStart(7, '0')}`

  return {
    receiptNumber,
    orderDate: isoDate(recordIndex, rng),
    merchantName: pick(MERCHANT_NAMES, rng),
    merchantAddress: `${randint(10, 9999, rng)} ${pick(STREET_NAMES, rng)}`,
    merchantCity: pick(CITIES, rng),
    merchantRegion: pick(STATES, rng),
    merchantPostalCode: String(randint(10000, 99999, rng)),
    currency: pick(CURRENCIES, rng),
    paymentMethod: pick(PAYMENT_METHODS, rng),
    cashier: pick(CASHIER_NAMES, rng),
    discountCents,
    subtotalCents,
    taxCents,
    tipCents,
    totalCents,
    items,
  }
}

function renderPlain(order) {
  const lines = []
  lines.push(order.merchantName.toUpperCase())
  lines.push(
    `${order.merchantAddress}, ${order.merchantCity}, ${order.merchantRegion} ${order.merchantPostalCode}`
  )
  lines.push(`Receipt No: ${order.receiptNumber}`)
  lines.push(`Date: ${order.orderDate}`)
  lines.push(`Cashier: ${order.cashier}`)
  lines.push(`Payment: ${order.paymentMethod}`)
  lines.push('')
  lines.push('ITEM | QTY | UNIT | TOTAL')
  for (const item of order.items) {
    lines.push(
      `${item.name} | ${item.quantity} | ${money(item.unitPriceCents)} | ${money(item.lineTotalCents)}`
    )
  }
  lines.push('')
  lines.push(`Subtotal: ${money(order.subtotalCents)} ${order.currency}`)
  if (order.discountCents > 0) {
    lines.push(`Discount: -${money(order.discountCents)} ${order.currency}`)
  }
  lines.push(`Tax: ${money(order.taxCents)} ${order.currency}`)
  if (order.tipCents > 0) {
    lines.push(`Tip: ${money(order.tipCents)} ${order.currency}`)
  }
  lines.push(`Total: ${money(order.totalCents)} ${order.currency}`)
  return lines.join('\n')
}

function renderKeyValue(order) {
  const itemLines = order.items
    .map(
      (item) =>
        `- name=${item.name}; sku=${item.sku}; qty=${item.quantity}; unit=${money(item.unitPriceCents)}; total=${money(item.lineTotalCents)}`
    )
    .join('\n')

  return [
    `merchant_name=${order.merchantName}`,
    `merchant_address=${order.merchantAddress}`,
    `merchant_city=${order.merchantCity}`,
    `merchant_region=${order.merchantRegion}`,
    `postal_code=${order.merchantPostalCode}`,
    `receipt_number=${order.receiptNumber}`,
    `order_date=${order.orderDate}`,
    `cashier=${order.cashier}`,
    `payment_method=${order.paymentMethod}`,
    `currency=${order.currency}`,
    'items=',
    itemLines,
    `subtotal=${money(order.subtotalCents)}`,
    `discount=${money(order.discountCents)}`,
    `tax=${money(order.taxCents)}`,
    `tip=${money(order.tipCents)}`,
    `total=${money(order.totalCents)}`,
  ].join('\n')
}

function renderCsv(order) {
  const header = 'type,name,sku,quantity,unit_price,line_total,currency'
  const itemRows = order.items.map((item) =>
    [
      'item',
      `"${item.name}"`,
      item.sku,
      item.quantity,
      money(item.unitPriceCents),
      money(item.lineTotalCents),
      order.currency,
    ].join(',')
  )
  const summaryRows = [
    `summary,subtotal,,,"${money(order.subtotalCents)}",,${order.currency}`,
    `summary,discount,,,"${money(order.discountCents)}",,${order.currency}`,
    `summary,tax,,,"${money(order.taxCents)}",,${order.currency}`,
    `summary,tip,,,"${money(order.tipCents)}",,${order.currency}`,
    `summary,total,,,"${money(order.totalCents)}",,${order.currency}`,
  ]

  return [
    `merchant,${JSON.stringify(order.merchantName)}`,
    `receipt_number,${order.receiptNumber}`,
    `order_date,${order.orderDate}`,
    `payment_method,${order.paymentMethod}`,
    header,
    ...itemRows,
    ...summaryRows,
  ].join('\n')
}

function renderJson(order) {
  return JSON.stringify(
    {
      merchant: {
        name: order.merchantName,
        address: {
          street: order.merchantAddress,
          city: order.merchantCity,
          region: order.merchantRegion,
          postalCode: order.merchantPostalCode,
        },
      },
      receiptNumber: order.receiptNumber,
      purchasedAt: order.orderDate,
      paymentMethod: order.paymentMethod,
      currency: order.currency,
      cashier: order.cashier,
      items: order.items.map((item) => ({
        name: item.name,
        sku: item.sku,
        quantity: item.quantity,
        unitPrice: money(item.unitPriceCents),
        lineTotal: money(item.lineTotalCents),
      })),
      totals: {
        subtotal: money(order.subtotalCents),
        discount: money(order.discountCents),
        tax: money(order.taxCents),
        tip: money(order.tipCents),
        total: money(order.totalCents),
      },
    },
    null,
    2
  )
}

function renderYaml(order) {
  const itemLines = order.items
    .map(
      (item) => `  - name: ${JSON.stringify(item.name)}
    sku: ${item.sku}
    quantity: ${item.quantity}
    unit_price: "${money(item.unitPriceCents)}"
    line_total: "${money(item.lineTotalCents)}"`
    )
    .join('\n')

  return `merchant:
  name: ${JSON.stringify(order.merchantName)}
  street: ${JSON.stringify(order.merchantAddress)}
  city: ${JSON.stringify(order.merchantCity)}
  region: ${order.merchantRegion}
  postal_code: ${order.merchantPostalCode}
receipt_number: ${order.receiptNumber}
order_date: ${order.orderDate}
payment_method: ${JSON.stringify(order.paymentMethod)}
cashier: ${JSON.stringify(order.cashier)}
currency: ${order.currency}
items:
${itemLines}
totals:
  subtotal: "${money(order.subtotalCents)}"
  discount: "${money(order.discountCents)}"
  tax: "${money(order.taxCents)}"
  tip: "${money(order.tipCents)}"
  total: "${money(order.totalCents)}"`
}

function renderXml(order) {
  const itemLines = order.items
    .map(
      (item) => `  <item sku="${item.sku}">
    <name>${item.name}</name>
    <quantity>${item.quantity}</quantity>
    <unitPrice>${money(item.unitPriceCents)}</unitPrice>
    <lineTotal>${money(item.lineTotalCents)}</lineTotal>
  </item>`
    )
    .join('\n')

  return `<receipt>
  <merchant>
    <name>${order.merchantName}</name>
    <street>${order.merchantAddress}</street>
    <city>${order.merchantCity}</city>
    <region>${order.merchantRegion}</region>
    <postalCode>${order.merchantPostalCode}</postalCode>
  </merchant>
  <receiptNumber>${order.receiptNumber}</receiptNumber>
  <orderDate>${order.orderDate}</orderDate>
  <paymentMethod>${order.paymentMethod}</paymentMethod>
  <cashier>${order.cashier}</cashier>
  <currency>${order.currency}</currency>
  <items>
${itemLines}
  </items>
  <totals subtotal="${money(order.subtotalCents)}" discount="${money(order.discountCents)}" tax="${money(order.taxCents)}" tip="${money(order.tipCents)}" total="${money(order.totalCents)}" />
</receipt>`
}

function renderJsonl(order) {
  return order.items
    .map((item) =>
      JSON.stringify({
        receipt_number: order.receiptNumber,
        merchant: order.merchantName,
        order_date: order.orderDate,
        currency: order.currency,
        payment_method: order.paymentMethod,
        item_name: item.name,
        sku: item.sku,
        quantity: item.quantity,
        unit_price: money(item.unitPriceCents),
        line_total: money(item.lineTotalCents),
        total: money(order.totalCents),
      })
    )
    .join('\n')
}

function renderHtml(order) {
  const itemRows = order.items
    .map(
      (item) =>
        `<tr><td>${item.name}</td><td>${item.quantity}</td><td>${money(item.unitPriceCents)}</td><td>${money(item.lineTotalCents)}</td></tr>`
    )
    .join('')

  return `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>Receipt ${order.receiptNumber}</title>
  </head>
  <body>
    <article class="receipt">
      <h1>${order.merchantName}</h1>
      <p>${order.merchantAddress}, ${order.merchantCity}, ${order.merchantRegion} ${order.merchantPostalCode}</p>
      <p>Order: ${order.receiptNumber}<br />Date: ${order.orderDate}<br />Payment: ${order.paymentMethod}</p>
      <table>
        <thead><tr><th>Item</th><th>Qty</th><th>Unit</th><th>Total</th></tr></thead>
        <tbody>${itemRows}</tbody>
      </table>
      <footer>
        <p>Subtotal: ${money(order.subtotalCents)} ${order.currency}</p>
        <p>Tax: ${money(order.taxCents)} ${order.currency}</p>
        <p>Total: ${money(order.totalCents)} ${order.currency}</p>
      </footer>
    </article>
  </body>
</html>`
}

function renderReceiptDocument(order) {
  const itemLines = order.items
    .map(
      (item) =>
        `${item.name}|${item.quantity}|${money(item.unitPriceCents)}|${money(item.lineTotalCents)}`
    )
    .join('\n')

  return `{width:48}
^${order.merchantName}
${order.merchantAddress}
${order.merchantCity}, ${order.merchantRegion} ${order.merchantPostalCode}
Receipt: ${order.receiptNumber}
Date: ${order.orderDate}
Payment: ${order.paymentMethod}
Cashier: ${order.cashier}
-
Item|Qty|Unit|Total
${itemLines}
-
Subtotal|${money(order.subtotalCents)} ${order.currency}
Tax|${money(order.taxCents)} ${order.currency}
Total|${money(order.totalCents)} ${order.currency}`
}

function applyNoise(content, format, recordIndex) {
  const rng = makeRng(recordIndex + 991)
  let value = content

  if (rng() < 0.55) {
    value = value.replace(/ /g, (match) => (rng() < 0.12 ? '  ' : match))
  }
  if (rng() < 0.45) {
    value = value.replace(/\n/g, (match) => (rng() < 0.08 ? '\n\n' : match))
  }
  if (rng() < 0.35) {
    value = value.replace(/[O0]/g, (char) => {
      if (rng() >= 0.04) return char
      return char === 'O' ? '0' : 'O'
    })
  }
  if (rng() < 0.35) {
    value = value.replace(/[Il1]/g, (char) => {
      if (rng() >= 0.04) return char
      if (char === 'I') return '1'
      if (char === 'l') return '1'
      return 'I'
    })
  }
  if (rng() < 0.2 && format !== 'html' && format !== 'xml') {
    value = value.replace(/[:;]/g, (char) => (rng() < 0.1 ? ' ' : char))
  }
  if (rng() < 0.2) {
    value = value.replace(/\bTotal\b/gi, () =>
      pick(['TOTAL', 'T0TAL', 'Total', 'TotaI'], rng)
    )
  }
  if (rng() < 0.15) {
    value = value.replace(/\bReceipt\b/gi, () =>
      pick(['RECEIPT', 'Receipt', 'RECElPT', 'RECElP7'], rng)
    )
  }
  if (rng() < 0.1) {
    value += '\n' + pick(['* * *', '#', '---', '///'], rng)
  }

  return value
}

function formatForIndex(index) {
  const formats = [
    'plain',
    'keyvalue',
    'csv',
    'html',
    'json',
    'yaml',
    'xml',
    'jsonl',
    'receipt',
  ]
  return formats[index % formats.length]
}

function renderContent(order, format) {
  switch (format) {
    case 'plain':
      return renderPlain(order)
    case 'keyvalue':
      return renderKeyValue(order)
    case 'csv':
      return renderCsv(order)
    case 'json':
      return renderJson(order)
    case 'yaml':
      return renderYaml(order)
    case 'xml':
      return renderXml(order)
    case 'jsonl':
      return renderJsonl(order)
    case 'html':
      return renderHtml(order)
    case 'receipt':
      return renderReceiptDocument(order)
    default:
      return renderPlain(order)
  }
}

async function runGeneratorWorker() {
  const { source, startIndex, count, outputFile } = workerData
  await fs.mkdir(path.dirname(outputFile), { recursive: true })

  const stream = createWriteStream(outputFile, { encoding: 'utf8' })
  let generated = 0

  await new Promise((resolve, reject) => {
    stream.once('error', reject)

    for (let offset = 0; offset < count; offset += 1) {
      const recordIndex = startIndex + offset
      const order = buildOrder(recordIndex)
      const format = formatForIndex(recordIndex)
      const recordId = `${source.language}-${source.slug}-${String(recordIndex).padStart(7, '0')}`
      const line = JSON.stringify({
        recordId,
        language: source.language,
        sourceSlug: source.slug,
        sourceTitle: source.title,
        license: source.licenseExpectation,
        format,
        content: applyNoise(renderContent(order, format), format, recordIndex),
      })

      stream.write(`${line}\n`)
      generated += 1
    }

    stream.end(resolve)
  })

  parentPort?.postMessage({
    outputFile,
    generated,
  })
}

async function generateSource(source, outputRoot, options) {
  const sourceRoot = path.join(
    outputRoot,
    source.language,
    source.slug,
    'jsonl'
  )
  await fs.mkdir(sourceRoot, { recursive: true })

  const totalCount = options.full
    ? Number(source.count)
    : Math.min(
        Number(source.count),
        Number(options.maxPerSource ?? source.count)
      )
  const shardCount = Math.max(
    1,
    Math.min(
      Number(source.shards || options.workers),
      totalCount,
      options.workers
    )
  )
  const perShard = Math.ceil(totalCount / shardCount)

  const tasks = []
  for (let shardIndex = 0; shardIndex < shardCount; shardIndex += 1) {
    const startIndex = shardIndex * perShard
    const count = Math.min(perShard, totalCount - startIndex)
    if (count <= 0) continue

    const outputFile = path.join(
      sourceRoot,
      `shard-${String(shardIndex + 1).padStart(4, '0')}.jsonl`
    )

    tasks.push(() =>
      runWorker({
        source,
        startIndex,
        count,
        outputFile,
      })
    )
  }

  const results = await runTaskPool(tasks, options.workers)
  return {
    source: source.slug,
    language: source.language,
    title: source.title,
    license: source.licenseExpectation ?? 'Unknown',
    generatedFiles: results.map((result) =>
      path.relative(outputRoot, result.outputFile).split(path.sep).join('/')
    ),
    generatedCount: results.reduce((sum, result) => sum + result.generated, 0),
  }
}

async function writeManifest(outputRoot, generatedSources) {
  const manifestPath = path.join(outputRoot, 'receipt-text-manifest.json')
  const payload = {
    generatedAt: new Date().toISOString(),
    outputDirectory: path.relative(process.cwd(), outputRoot),
    sources: generatedSources,
  }
  await fs.writeFile(manifestPath, `${JSON.stringify(payload, null, 2)}\n`)
}

async function main() {
  const options = parseArgs(process.argv.slice(2))
  if (options.help) {
    printHelp()
    return
  }

  const allSources = await loadSourceDefinitions(options.manifests)
  const selectedSources = filterSources(allSources, options)
  if (selectedSources.length === 0) {
    throw new Error('No text sources matched the provided filters')
  }

  const outputRoot = path.resolve(process.cwd(), options.output)
  await fs.mkdir(outputRoot, { recursive: true })

  const downloadableSources = selectedSources.filter(
    (source) => source.kind !== 'generator'
  )
  const generatorSources = selectedSources.filter(
    (source) => source.kind === 'generator'
  )

  if (downloadableSources.length > 0) {
    const tempManifests = await createTemporaryManifests(downloadableSources)
    try {
      await spawnDownloader(tempManifests, outputRoot, options)
    } finally {
      await fs.rm(tempManifests, { recursive: true, force: true })
    }
  }

  const generatedSources = []
  for (const source of generatorSources) {
    console.log(`Generating ${source.language}/${source.slug}`)
    generatedSources.push(await generateSource(source, outputRoot, options))
  }

  await writeManifest(outputRoot, generatedSources)
  console.log(`Done. Text sources written to ${outputRoot}`)
}

if (isMainThread) {
  main().catch((error) => {
    console.error(error)
    process.exit(1)
  })
} else {
  runGeneratorWorker().catch((error) => {
    throw error
  })
}
