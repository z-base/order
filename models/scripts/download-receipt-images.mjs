#!/usr/bin/env node
import { createHash } from 'node:crypto'
import { createReadStream, createWriteStream } from 'node:fs'
import fs from 'node:fs/promises'
import path from 'node:path'
import { Readable } from 'node:stream'
import { pipeline } from 'node:stream/promises'
import { spawn } from 'node:child_process'
import process from 'node:process'

const IMAGE_EXTS = new Set([
  '.jpg',
  '.jpeg',
  '.png',
  '.gif',
  '.webp',
  '.bmp',
  '.tif',
  '.tiff',
])
const PDF_EXTS = new Set(['.pdf'])
const PARQUET_EXTS = new Set(['.parquet'])
const DEFAULT_RECEIPT_EXTS = [
  ...IMAGE_EXTS,
  ...PDF_EXTS,
  ...PARQUET_EXTS,
  '.zip',
]
const PERMISSIVE_LICENSES = new Set([
  'MIT',
  'Apache-2.0',
  'BSD-2-Clause',
  'BSD-3-Clause',
  'ISC',
  'Unlicense',
  'CC0-1.0',
  'CC-BY-4.0',
])

function sourceSlugFromId(datasetId) {
  return datasetId
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
}

const HUGGINGFACE_RECEIPT_CANDIDATES = [
  'CC1984/mall_receipt_extraction_dataset',
  'ilhamxx/Receipt_dataset',
  'ilhamxx/data_receipt',
  'Krzyzak300/receipt-detection',
  'mahmoud2019/ReceiptQA',
  'Saran-R12/Receipts',
  'wasanx/receipt-4k',
  'HumynLabs/Korean_Receipts_Dataset',
  'Voxel51/scanned_receipts',
  'Voxel51/consolidated_receipt_dataset',
  'bitmind/Fake-receipts',
  'Hemgg/invoices-and-receipts_ocr_v1',
  'Hemgg/consolidated_receipt_dataset',
  'Hemgg/invoices-and-receipts_ocr_v2',
  'Hemgg/mall_receipt_extraction_dataset',
  'Hemgg/ds_receipts_v2_train',
  'deepthink8/dataset_receipt',
  'deepthink8/receipt-4k',
  'deepthink8/he-synth-receipt-clean-5k',
  'deepthink8/invoice-receipt-cheque-bankstatement-dataset',
  'SvetaLana25/dek-receipt-cord',
  'cdek-ocr/receipt-ocr-ru',
  'Lakshmiperumal/scanned_receipts',
  'chickencaesar-yang/NTIRE_2026_Financial_Receipt',
  'Shiva282002/invoices-and-receipts_ocr_v2',
]

const SOURCE_DEFINITIONS = HUGGINGFACE_RECEIPT_CANDIDATES.map((id) => ({
  id,
  kind: 'huggingface',
  title: id,
  slug: sourceSlugFromId(id),
  description: 'Receipt dataset candidate from Hugging Face search.',
  languages: [],
  includeExts: DEFAULT_RECEIPT_EXTS,
  includePrefix: ['.'],
  extractArchives: true,
  licenseExpectation: null,
  licenseHint: `https://huggingface.co/datasets/${id}`,
}))

const defaultOptions = {
  output: 'models/data/source_images',
  full: false,
  maxPerSource: 80,
  languageFilters: null,
}

function parseArgs(argv) {
  const parsed = { ...defaultOptions, sourceFilters: null }

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i]

    if (arg === '--help' || arg === '-h') {
      parsed.help = true
      continue
    }

    if (arg === '--output') {
      parsed.output = argv[i + 1] ?? ''
      i += 1
      continue
    }

    if (arg === '--full') {
      parsed.full = true
      parsed.maxPerSource = null
      continue
    }

    if (arg === '--max-per-source') {
      const parsedValue = Number(argv[i + 1])
      if (!Number.isFinite(parsedValue) || parsedValue < 1) {
        throw new Error(`Invalid --max-per-source value: ${argv[i + 1]}`)
      }
      parsed.maxPerSource = parsedValue
      i += 1
      continue
    }

    if (arg === '--source') {
      parsed.sourceFilters = parsed.sourceFilters ?? []
      parsed.sourceFilters.push(argv[i + 1] ?? '')
      i += 1
      continue
    }

    if (arg === '--languages') {
      const raw = argv[i + 1]
      if (!raw) {
        throw new Error('--languages requires a comma-separated list')
      }
      parsed.languageFilters = raw
        .split(',')
        .map((value) => value.trim().toLowerCase())
        .filter(Boolean)
      i += 1
      continue
    }
  }

  return parsed
}

function printHelp() {
  console.log(`Usage:
node scripts/download-receipts.mjs [options]

Options:
  --help                 Show this help text
  --output <path>        Destination directory (default: data/images)
  --source <slug/id>     Restrict download to one source (repeatable)
  --languages <codes>    Restrict to sources by language (comma-separated)
  --max-per-source <n>   Limit files per source (default: 80)
  --full                 Remove per-source limit

Note:
  Only datasets with permissive licenses are downloaded (MIT, Apache-2.0, BSD, ISC, Unlicense, CC0, CC-BY-4.0).
`)
}

function normalizeLicense(rawLicense) {
  if (!rawLicense) return 'Unknown'
  if (Array.isArray(rawLicense)) {
    return normalizeLicense(rawLicense.join(' '))
  }
  if (typeof rawLicense === 'object') {
    const candidate =
      rawLicense.spdx ||
      rawLicense.identifier ||
      rawLicense.name ||
      rawLicense.type
    if (typeof candidate !== 'string') return 'Unknown'
    return normalizeLicense(candidate)
  }
  if (typeof rawLicense !== 'string') return 'Unknown'
  const normalized = rawLicense.trim().toLowerCase()
  if (normalized === 'unknown') return 'Unknown'
  if (normalized.includes('mit')) return 'MIT'
  if (normalized.includes('apache-2.0') || normalized.includes('apache 2.0'))
    return 'Apache-2.0'
  if (normalized.includes('bsd-2')) return 'BSD-2-Clause'
  if (normalized.includes('bsd-3')) return 'BSD-3-Clause'
  if (normalized === 'isc') return 'ISC'
  if (normalized.includes('unlicense')) return 'Unlicense'
  if (normalized === 'cc0-1.0' || normalized === 'cc0') return 'CC0-1.0'
  if (normalized === 'cc-by-4.0' || normalized === 'cc by 4.0')
    return 'CC-BY-4.0'
  return rawLicense
}

function isPermissiveLicense(licenseValue) {
  const normalized = normalizeLicense(licenseValue)
  return PERMISSIVE_LICENSES.has(normalized)
}

function shouldIncludePath(filePath, source) {
  const target = filePath.replace(/\\/g, '/')
  const lowerTarget = target.toLowerCase()
  const ext = path.extname(lowerTarget)
  const extOk = source.includeExts.includes(ext)
  if (!extOk) return false
  const includesPrefix = source.includePrefix.some((prefix) => {
    if (prefix === '.') {
      return true
    }
    const normalizedPrefix = prefix.replace(/\\/g, '/').toLowerCase()
    return lowerTarget.startsWith(normalizedPrefix)
  })
  return includesPrefix
}

function slugify(value) {
  return value
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
}

function encodeHFRepoId(datasetId) {
  const [owner, repo] = datasetId.split('/')
  return `${encodeURIComponent(owner)}/${encodeURIComponent(repo)}`
}

function encodeHFPath(filePath) {
  return filePath
    .split('/')
    .map((segment) => encodeURIComponent(segment))
    .join('/')
}

function encodeGitHubRepo(repo) {
  const [owner, repoName] = repo.split('/')
  return `${encodeURIComponent(owner)}/${encodeURIComponent(repoName)}`
}

function safeTargetPath(filePath) {
  return filePath
    .split('/')
    .map((segment) => segment.replace(/[^\w.\- ()\[\]()+]/g, '_'))
    .join(path.sep)
}

function evidenceFilePath(outputRoot, source) {
  return path.join(outputRoot, 'licenses', `${source.slug}.json`)
}

async function writeJsonFile(filePath, value) {
  await fs.mkdir(path.dirname(filePath), { recursive: true })
  await fs.writeFile(filePath, `${JSON.stringify(value, null, 2)}\n`)
}

async function writeTextFile(filePath, value) {
  await fs.mkdir(path.dirname(filePath), { recursive: true })
  await fs.writeFile(filePath, `${value}\n`)
}

async function downloadFile(url, destination) {
  const response = await fetch(url, {
    headers: { 'User-Agent': 'receipt-downloader-script' },
  })
  if (!response.ok) {
    throw new Error(
      `Download failed: ${url} (${response.status} ${response.statusText})`
    )
  }
  if (!response.body) {
    throw new Error(`No body stream from: ${url}`)
  }
  await fs.mkdir(path.dirname(destination), { recursive: true })
  const stream = createWriteStream(destination)
  await pipeline(Readable.fromWeb(response.body), stream)
  return {
    size: Number(response.headers.get('content-length') ?? '0'),
    mediaType:
      response.headers.get('content-type') ?? 'application/octet-stream',
  }
}

async function computeSha256(filePath) {
  return new Promise((resolve, reject) => {
    const hasher = createHash('sha256')
    const input = createReadStream(filePath)
    input.on('error', reject)
    input.on('data', (chunk) => hasher.update(chunk))
    input.on('end', () => resolve(hasher.digest('hex')))
  })
}

async function collectFilesRecursively(directory, extFilter) {
  const collected = []
  const normalizedFilter = new Set(
    (extFilter || []).map((ext) => ext.toLowerCase())
  )
  const stack = [directory]
  while (stack.length > 0) {
    const current = stack.pop()
    const entries = await fs.readdir(current, { withFileTypes: true })
    for (const entry of entries) {
      const candidate = path.join(current, entry.name)
      if (entry.isDirectory()) {
        stack.push(candidate)
        continue
      }
      if (!entry.isFile()) continue
      const ext = path.extname(entry.name).toLowerCase()
      if (!normalizedFilter.has(ext)) continue
      collected.push(candidate)
    }
  }
  return collected
}

async function extractZip(pathToZip, extractTo, allowedExts, maxFiles) {
  const allowed = JSON.stringify(allowedExts || [])
  const script = `
import zipfile
import pathlib
import sys
import json

zip_path = pathlib.Path(sys.argv[1])
output_dir = pathlib.Path(sys.argv[2])
allowed_exts = set(json.loads(sys.argv[3]))
max_files = int(sys.argv[4]) if len(sys.argv) > 4 else -1
written = []

with zipfile.ZipFile(zip_path, 'r') as archive:
    names = archive.namelist()
    for name in sorted(names):
        if max_files >= 0 and len(written) >= max_files:
            break
        lower = name.lower()
        if name.endswith('/'):
            continue
        if '__MACOSX/' in name:
            continue
        if any(lower.endswith(ext) for ext in allowed_exts):
            target = output_dir / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(name) as source, target.open('wb') as destination:
                destination.write(source.read())
            written.append(str(target))
            if max_files >= 0 and len(written) >= max_files:
                break

print(json.dumps(written))
`
  const args = [
    '-c',
    script,
    pathToZip,
    extractTo,
    allowed,
    String(maxFiles ?? -1),
  ]
  const child = spawn('python', args, { stdio: ['ignore', 'pipe', 'pipe'] })
  let stdout = ''
  let stderr = ''
  child.stdout.on('data', (chunk) => {
    stdout += chunk.toString()
  })
  child.stderr.on('data', (chunk) => {
    stderr += chunk.toString()
  })
  const exitCode = await new Promise((resolve, reject) => {
    child.on('error', reject)
    child.on('close', resolve)
  })
  if (exitCode !== 0) {
    throw new Error(`zip extraction failed for ${pathToZip}: ${stderr}`)
  }
  const parsed = JSON.parse(stdout || '[]')
  return parsed
}

async function getHuggingFaceInfo(source, outputRoot) {
  const encoded = encodeHFRepoId(source.id)
  const infoUrl = `https://huggingface.co/api/datasets/${encoded}`
  const treeUrl = `https://huggingface.co/api/datasets/${encoded}/tree/main?recursive=true`
  const readmeUrl = `https://huggingface.co/datasets/${encoded}/resolve/main/README.md`
  const licenseCandidates = [
    'LICENSE',
    'license',
    'LICENSE.md',
    'license.md',
    'COPYING',
  ]

  const [info, tree] = await Promise.all([
    fetch(infoUrl, {
      headers: { 'User-Agent': 'receipt-downloader-script' },
    }).then((response) => {
      if (!response.ok) {
        throw new Error(`Cannot fetch dataset info for ${source.id}`)
      }
      return response.json()
    }),
    fetch(treeUrl, {
      headers: { 'User-Agent': 'receipt-downloader-script' },
    }).then((response) => {
      if (!response.ok) {
        throw new Error(`Cannot fetch dataset tree for ${source.id}`)
      }
      return response.json()
    }),
  ])

  const readme = await fetch(readmeUrl, {
    headers: { 'User-Agent': 'receipt-downloader-script' },
  }).then((response) => {
    if (!response.ok) return ''
    return response.text()
  })

  const evidence = {
    source: `https://huggingface.co/datasets/${source.id}`,
    licenseFromApi:
      info?.cardData?.license ??
      info?.tags
        ?.find((tag) => tag.startsWith('license:'))
        ?.split(':')
        .pop(),
    licenseFromReadme: null,
  }
  const readmeMatch = readme.match(/^license:\s*(.+)$/m)
  if (readmeMatch) {
    evidence.licenseFromReadme = readmeMatch[1].trim()
  }

  const licenseText = await Promise.all(
    licenseCandidates.map(async (candidate) => {
      const url = `https://huggingface.co/datasets/${encoded}/resolve/main/${encodeHFPath(candidate)}`
      const response = await fetch(url, {
        headers: { 'User-Agent': 'receipt-downloader-script' },
      })
      if (!response.ok) return null
      return response.text()
    })
  ).then((texts) =>
    texts.find((text) => typeof text === 'string' && text.length > 0)
  )

  const resolvedLicense = normalizeLicense(
    evidence.licenseFromReadme ??
      evidence.licenseFromApi ??
      source.licenseExpectation ??
      'Unknown'
  )
  const evidencePath = evidenceFilePath(outputRoot, source)
  await writeJsonFile(evidencePath, {
    source: source.id,
    title: source.title,
    evidence,
    resolvedLicense,
    licenseText: licenseText
      ? `Saved from dataset root: ${licenseCandidates.find(Boolean)}`
      : null,
  })

  const candidates = (Array.isArray(tree) ? tree : []).filter((entry) => {
    if (entry.type !== 'file') return false
    return shouldIncludePath(entry.path, source)
  })
  return {
    kind: 'huggingface',
    source,
    entries: candidates,
    evidence,
    resolvedLicense,
    evidencePath,
  }
}

async function getGitHubInfo(source, outputRoot) {
  const [owner, repo] = source.repo.split('/')
  const encoded = encodeGitHubRepo(source.repo)
  const url = `https://api.github.com/repos/${encoded}/git/trees/${encodeURIComponent(source.branch ?? 'master')}?recursive=1`
  const licenseUrl = `https://raw.githubusercontent.com/${encoded}/${encodeURIComponent(source.branch ?? 'master')}/LICENSE`
  const noticeUrl = `https://raw.githubusercontent.com/${encoded}/${encodeURIComponent(source.branch ?? 'master')}/NOTICE`
  const treeResponse = await fetch(url, {
    headers: { 'User-Agent': 'receipt-downloader-script' },
  })
  if (!treeResponse.ok) {
    throw new Error(`Cannot fetch repo tree for ${source.repo}`)
  }
  const treePayload = await treeResponse.json()
  const readmeResponse = await fetch(
    `https://raw.githubusercontent.com/${encoded}/${encodeURIComponent(source.branch ?? 'master')}/README.md`,
    {
      headers: { 'User-Agent': 'receipt-downloader-script' },
    }
  )
  const readme = readmeResponse.ok ? await readmeResponse.text() : ''
  const readmeLicense = readme.match(/^license:\s*(.+)$/m)?.[1]?.trim()

  const licenseResponse = await fetch(licenseUrl, {
    headers: { 'User-Agent': 'receipt-downloader-script' },
  })
  const licenseText = licenseResponse.ok ? await licenseResponse.text() : null
  const noticeResponse = await fetch(noticeUrl, {
    headers: { 'User-Agent': 'receipt-downloader-script' },
  })
  const noticeText = noticeResponse.ok ? await noticeResponse.text() : null

  const evidence = {
    source: `https://github.com/${source.repo}`,
    licenseFromReadme: readmeLicense,
    licenseFiles: {
      license: licenseResponse.ok ? 'LICENSE' : null,
      notice: noticeResponse.ok ? 'NOTICE' : null,
    },
  }

  const evidencePath = evidenceFilePath(outputRoot, source)
  await writeJsonFile(evidencePath, {
    source: source.repo,
    title: source.title,
    evidence,
    resolvedLicense: normalizeLicense(readmeLicense ?? 'MIT'),
    licenseText: licenseText
      ? 'Captured from LICENSE and NOTICE where present.'
      : 'No LICENSE file found; repository README/license metadata is treated as the source.',
  })

  const entries = (treePayload.tree ?? []).filter((entry) => {
    if (entry.type !== 'blob') return false
    return shouldIncludePath(entry.path, source)
  })
  const resolvedLicense = normalizeLicense(
    licenseText ? 'MIT' : (readmeLicense ?? 'MIT')
  )
  return {
    kind: 'github',
    source,
    entries,
    evidence,
    resolvedLicense,
    evidencePath,
  }
}

function buildSourceManifest(source, summary) {
  return {
    sourceId: source.id || source.repo,
    title: source.title,
    sourceType: source.kind,
    license: summary.resolvedLicense || source.licenseExpectation || 'Unknown',
    evidencePath: path
      .relative(summary.outputRoot, summary.evidencePath)
      .split(path.sep)
      .join('/'),
    languages: source.languages,
    totalAvailable: summary.entries.length,
    downloadedFiles: summary.downloadedFiles,
    skipped: summary.skipped,
    notes: source.description,
  }
}

function hasLanguageMatch(source, requestedLanguages) {
  if (!requestedLanguages || requestedLanguages.length === 0) return true
  if (!source.languages || source.languages.length === 0) return false
  const normalized = source.languages.map((language) => language.toLowerCase())
  return requestedLanguages.some((requested) => normalized.includes(requested))
}

async function processHuggingFaceSource(
  source,
  options,
  outputRoot,
  perSourceLimit
) {
  const summary = await getHuggingFaceInfo(source, outputRoot)
  const sourceRoot = path.join(outputRoot, source.slug)
  const files = []
  const skipped = []
  if (!isPermissiveLicense(summary.resolvedLicense)) {
    skipped.push(
      `Skipped: non-permissive or unknown license (${summary.resolvedLicense}).`
    )
    return buildSourceManifest(source, {
      outputRoot,
      evidencePath: summary.evidencePath,
      resolvedLicense: summary.resolvedLicense,
      entries: summary.entries,
      downloadedFiles: files,
      skipped,
    })
  }
  const maxFiles = perSourceLimit
  let remaining = maxFiles
  const resolvedEntries = summary.entries.sort((a, b) =>
    a.path.localeCompare(b.path)
  )
  const effectiveEntries =
    remaining == null ? resolvedEntries : resolvedEntries.slice(0, remaining)

  for (const entry of effectiveEntries) {
    const encodedId = encodeHFRepoId(source.id)
    const ext = path.extname(entry.path).toLowerCase()
    const sourcePath = entry.path
    if (!entry.path) continue

    if (ext === '.zip' && source.extractArchives) {
      if (remaining != null && remaining <= 0) break
      const zipUrl = `https://huggingface.co/datasets/${encodedId}/resolve/main/${encodeHFPath(sourcePath)}`
      const zipTarget = path.join(
        sourceRoot,
        'archives',
        safeTargetPath(sourcePath)
      )
      await downloadFile(zipUrl, zipTarget)
      const extractedDir = path.join(sourceRoot, 'images', slugify(sourcePath))
      const extracted = await extractZip(
        zipTarget,
        extractedDir,
        Array.from(IMAGE_EXTS),
        remaining == null ? -1 : remaining
      )
      const absoluteExtracted = await collectFilesRecursively(
        extractedDir,
        Array.from(IMAGE_EXTS)
      )
      for (const item of absoluteExtracted.slice(
        0,
        remaining == null ? undefined : remaining
      )) {
        const sha256 = await computeSha256(item)
        files.push({
          localPath: path.relative(outputRoot, item).split(path.sep).join('/'),
          sourcePath,
          source: `https://huggingface.co/datasets/${source.id}/resolve/main/${sourcePath}`,
          category: 'image',
          extension: path.extname(item).toLowerCase(),
          sha256,
          size: Number((await fs.stat(item)).size),
        })
      }
      await fs.unlink(zipTarget).catch(() => {})
      if (remaining != null)
        remaining -= absoluteExtracted.slice(0, remaining).length
      continue
    }

    const targetKind = PDF_EXTS.has(ext)
      ? 'pdfs'
      : PARQUET_EXTS.has(ext)
        ? 'parquet'
        : 'images'
    const targetDir = path.join(sourceRoot, targetKind)
    const targetPath = path.join(targetDir, safeTargetPath(sourcePath))
    if (remaining != null && remaining <= 0) break
    const meta = await downloadFile(
      `https://huggingface.co/datasets/${encodedId}/resolve/main/${encodeHFPath(sourcePath)}`,
      targetPath
    )
    const sha256 = await computeSha256(targetPath)
    files.push({
      localPath: path
        .relative(outputRoot, targetPath)
        .split(path.sep)
        .join('/'),
      sourcePath,
      source: `https://huggingface.co/datasets/${source.id}/resolve/main/${sourcePath}`,
      category: targetKind === 'images' ? 'image' : targetKind,
      extension: ext,
      sha256,
      size: meta.size || Number((await fs.stat(targetPath)).size),
      mediaType: meta.mediaType,
    })
    if (remaining != null) remaining -= 1
  }

  if (effectiveEntries.length < summary.entries.length) {
    skipped.push(
      `Limit reached (${effectiveEntries.length} of ${summary.entries.length} files).`
    )
  }

  return buildSourceManifest(source, {
    outputRoot,
    evidencePath: summary.evidencePath,
    resolvedLicense: summary.resolvedLicense,
    entries: summary.entries,
    downloadedFiles: files,
    skipped,
  })
}

async function processGitHubSource(
  source,
  options,
  outputRoot,
  perSourceLimit
) {
  const summary = await getGitHubInfo(source, outputRoot)
  const sourceRoot = path.join(outputRoot, source.slug)
  const files = []
  const skipped = []
  if (!isPermissiveLicense(summary.resolvedLicense)) {
    skipped.push(
      `Skipped: non-permissive or unknown license (${summary.resolvedLicense}).`
    )
    return buildSourceManifest(source, {
      outputRoot,
      evidencePath: summary.evidencePath,
      resolvedLicense: summary.resolvedLicense,
      entries: summary.entries,
      downloadedFiles: files,
      skipped,
    })
  }
  const maxFiles = perSourceLimit
  let remaining = maxFiles
  const effectiveEntries = summary.entries
    .sort((a, b) => a.path.localeCompare(b.path))
    .slice(0, remaining == null ? undefined : remaining)

  const encodedRepo = encodeGitHubRepo(source.repo)
  const branch = source.branch ?? 'master'

  for (const entry of effectiveEntries) {
    if (remaining != null && remaining <= 0) break
    const sourcePath = entry.path
    const ext = path.extname(sourcePath).toLowerCase()
    const targetKind = PDF_EXTS.has(ext)
      ? 'pdfs'
      : PARQUET_EXTS.has(ext)
        ? 'parquet'
        : 'images'
    const targetPath = path.join(
      sourceRoot,
      targetKind,
      safeTargetPath(sourcePath)
    )
    const fileUrl = `https://raw.githubusercontent.com/${encodedRepo}/${branch}/${encodeHFPath(sourcePath)}`
    await downloadFile(fileUrl, targetPath)
    const sha256 = await computeSha256(targetPath)
    files.push({
      localPath: path
        .relative(outputRoot, targetPath)
        .split(path.sep)
        .join('/'),
      sourcePath,
      source: `https://raw.githubusercontent.com/${encodedRepo}/${branch}/${sourcePath}`,
      category: targetKind === 'images' ? 'image' : targetKind,
      extension: ext,
      sha256,
      size: Number((await fs.stat(targetPath)).size),
    })
    if (remaining != null) remaining -= 1
  }

  if (effectiveEntries.length < summary.entries.length) {
    skipped.push(
      `Limit reached (${effectiveEntries.length} of ${summary.entries.length} files).`
    )
  }

  return buildSourceManifest(source, {
    outputRoot,
    evidencePath: summary.evidencePath,
    resolvedLicense: summary.resolvedLicense,
    entries: summary.entries,
    downloadedFiles: files,
    skipped,
  })
}

async function generateMarkdownReport(manifest, manifestPath) {
  const lines = []
  lines.push('# Receipt Download Report')
  lines.push(`Generated: ${manifest.generatedAt}`)
  lines.push('')
  lines.push(`Output: ${manifest.outputDirectory}`)
  lines.push(`Scope: ${manifest.scope}`)
  lines.push('')
  lines.push('| Source | License | Files | Languages | Notes | Evidence |')
  lines.push('|---|---|---:|---|---|---|')
  for (const source of manifest.sources) {
    lines.push(
      `| ${source.title} | ${source.license} | ${source.downloadedFiles.length} | ${source.languages.join(', ')} | ${source.notes} | ${source.evidencePath} |`
    )
  }
  lines.push('')
  await writeTextFile(manifestPath, lines.join('\n'))
}

async function main() {
  const args = parseArgs(process.argv.slice(2))
  if (args.help) {
    printHelp()
    return
  }

  const selectedSources = SOURCE_DEFINITIONS.filter((source) => {
    if (args.sourceFilters && args.sourceFilters.length > 0) {
      const matchesSource = args.sourceFilters.some((filterValue) => {
        const target = filterValue.toLowerCase()
        return (
          (source.id || '').toLowerCase() === target ||
          source.slug === target ||
          source.title.toLowerCase() === target
        )
      })
      if (!matchesSource) return false
    }

    if (args.languageFilters && args.languageFilters.length > 0) {
      return hasLanguageMatch(source, args.languageFilters)
    }

    return true
  })
  if (selectedSources.length === 0) {
    if (args.sourceFilters && args.sourceFilters.length > 0) {
      throw new Error('No sources matched --source filter')
    }
    const availableLanguages = [
      ...new Set(
        SOURCE_DEFINITIONS.flatMap((source) => source.languages ?? [])
      ),
    ]
      .sort()
      .join(', ')
    throw new Error(
      `No sources matched --languages filter. Available languages: ${availableLanguages}`
    )
  }

  const outputRoot = path.resolve(process.cwd(), args.output)
  await fs.mkdir(outputRoot, { recursive: true })
  const manifest = {
    generatedAt: new Date().toISOString(),
    outputDirectory: path.relative(process.cwd(), outputRoot),
    scope: args.full ? 'full' : `limited to ${args.maxPerSource}`,
    options: {
      output: args.output,
      maxPerSource: args.maxPerSource,
      full: args.full,
      sourceFilters: args.sourceFilters,
      languageFilters: args.languageFilters,
    },
    sources: [],
  }

  for (const source of selectedSources) {
    console.log(`Downloading from ${source.title}`)
    const perSourceLimit = args.full ? null : args.maxPerSource
    let sourceSummary
    if (source.kind === 'huggingface') {
      sourceSummary = await processHuggingFaceSource(
        source,
        args,
        outputRoot,
        perSourceLimit
      )
    } else {
      sourceSummary = await processGitHubSource(
        source,
        args,
        outputRoot,
        perSourceLimit
      )
    }
    manifest.sources.push(sourceSummary)
  }

  const manifestPath = path.join(outputRoot, 'receipt-license-manifest.json')
  await writeJsonFile(manifestPath, manifest)
  const reportPath = path.join(outputRoot, 'receipt-license-report.md')
  await generateMarkdownReport(manifest, reportPath)
  console.log(`Done. Manifest written to ${manifestPath}`)
  console.log(`Report written to ${reportPath}`)
}

main().catch((error) => {
  console.error(error)
  process.exit(1)
})
