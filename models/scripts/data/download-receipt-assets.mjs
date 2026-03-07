#!/usr/bin/env node
import { createHash } from 'node:crypto'
import { createReadStream, createWriteStream } from 'node:fs'
import fs from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { spawn } from 'node:child_process'
import process from 'node:process'
import { Readable } from 'node:stream'
import { pipeline } from 'node:stream/promises'
import {
  Worker,
  isMainThread,
  parentPort,
  workerData,
} from 'node:worker_threads'

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
const ARCHIVE_EXTS = new Set(['.zip'])
const DEFAULT_SOURCE_EXTS_BY_FORMAT = {
  image: [...IMAGE_EXTS, ...ARCHIVE_EXTS],
  pdf: [...PDF_EXTS, ...ARCHIVE_EXTS],
  mixed: [...IMAGE_EXTS, ...PDF_EXTS, ...PARQUET_EXTS, ...ARCHIVE_EXTS],
}
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

const DEFAULT_OPTIONS = {
  manifests: 'models/data/image_sources',
  output: 'models/data/downloaded_images',
  defaultFormat: 'image',
  full: false,
  maxPerSource: 80,
  languageFilters: null,
  workers: Math.max(1, Math.min(os.availableParallelism(), 8)),
}

function slugify(value) {
  return String(value || '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
}

function sourceSlugFromId(value) {
  return slugify(value)
}

function parseArgs(argv) {
  const parsed = { ...DEFAULT_OPTIONS, sourceFilters: null }

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i]

    if (arg === '--help' || arg === '-h') {
      parsed.help = true
      continue
    }

    if (arg === '--manifests') {
      parsed.manifests = argv[i + 1] ?? ''
      i += 1
      continue
    }

    if (arg === '--output') {
      parsed.output = argv[i + 1] ?? ''
      i += 1
      continue
    }

    if (arg === '--default-format') {
      const value = (argv[i + 1] ?? '').trim().toLowerCase()
      if (!['image', 'pdf', 'mixed'].includes(value)) {
        throw new Error(
          `Invalid --default-format value: ${argv[i + 1]}. Expected image, pdf, or mixed.`
        )
      }
      parsed.defaultFormat = value
      i += 1
      continue
    }

    if (arg === '--full') {
      parsed.full = true
      parsed.maxPerSource = null
      continue
    }

    if (arg === '--workers') {
      const parsedValue = Number(argv[i + 1])
      if (!Number.isFinite(parsedValue) || parsedValue < 1) {
        throw new Error(`Invalid --workers value: ${argv[i + 1]}`)
      }
      parsed.workers = Math.max(1, Math.floor(parsedValue))
      i += 1
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
node models/scripts/data/download-receipt-assets.mjs [options]

Options:
  --help                 Show this help text
  --manifests <path>     Source manifest directory (default: models/data/image_sources)
  --output <path>        Destination root (default: models/data/downloaded_images)
  --default-format <t>   Fallback file family: image, pdf, or mixed (default: image)
  --source <slug/id>     Restrict download to one source (repeatable)
  --languages <codes>    Restrict to sources by language (comma-separated)
  --workers <n>          Worker-thread concurrency across sources
  --max-per-source <n>   Limit files per source (default: 80)
  --full                 Remove per-source limit

Notes:
  Sources are loaded from JSON files in models/data/image_sources/{lang}.json.
  Downloads are written to models/data/downloaded_images/{lang}/{dataset}/...
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
  return PERMISSIVE_LICENSES.has(normalizeLicense(licenseValue))
}

function normalizeExtensions(rawExts, defaultFormat) {
  const fallback =
    DEFAULT_SOURCE_EXTS_BY_FORMAT[defaultFormat] ??
    DEFAULT_SOURCE_EXTS_BY_FORMAT.image
  const values =
    Array.isArray(rawExts) && rawExts.length > 0 ? rawExts : fallback
  return [...new Set(values.map((value) => String(value).toLowerCase()))]
}

function normalizePrefixes(rawPrefixes) {
  const values =
    Array.isArray(rawPrefixes) && rawPrefixes.length > 0 ? rawPrefixes : ['.']
  return values.map((value) => String(value))
}

function deriveLanguageCode(fileName, payload) {
  if (typeof payload?.language === 'string' && payload.language.trim()) {
    return payload.language.trim().toLowerCase()
  }
  return path.basename(fileName, path.extname(fileName)).toLowerCase()
}

function normalizeSourceDefinition(
  rawSource,
  language,
  manifestFile,
  defaultFormat
) {
  if (!rawSource || typeof rawSource !== 'object' || Array.isArray(rawSource)) {
    throw new Error(`Invalid source entry in ${manifestFile}`)
  }

  const kind = String(rawSource.kind || '').toLowerCase()
  if (!['huggingface', 'github', 'direct'].includes(kind)) {
    throw new Error(
      `Unsupported source kind "${rawSource.kind}" in ${manifestFile}`
    )
  }

  if (kind === 'huggingface' && typeof rawSource.id !== 'string') {
    throw new Error(`Hugging Face source requires "id" in ${manifestFile}`)
  }
  if (kind === 'github' && typeof rawSource.repo !== 'string') {
    throw new Error(`GitHub source requires "repo" in ${manifestFile}`)
  }
  if (kind === 'direct' && typeof rawSource.downloadUrl !== 'string') {
    throw new Error(`Direct source requires "downloadUrl" in ${manifestFile}`)
  }

  const sourceIdentifier =
    rawSource.id || rawSource.repo || rawSource.downloadUrl || rawSource.title
  const slug = slugify(rawSource.slug || sourceSlugFromId(sourceIdentifier))
  if (!slug) {
    throw new Error(`Could not derive source slug in ${manifestFile}`)
  }

  const title = String(rawSource.title || sourceIdentifier)
  const sourceLanguage = String(rawSource.language || language).toLowerCase()

  return {
    ...rawSource,
    kind,
    title,
    slug,
    language: sourceLanguage,
    languages: [sourceLanguage],
    includeExts: normalizeExtensions(rawSource.includeExts, defaultFormat),
    includePrefix: normalizePrefixes(rawSource.includePrefix),
    extractArchives: rawSource.extractArchives ?? true,
    licenseExpectation: rawSource.licenseExpectation ?? null,
  }
}

async function loadSourceDefinitions(manifestRoot, defaultFormat) {
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
    if (!rawText.trim()) {
      throw new Error(`Manifest file is empty: ${manifestFile}`)
    }

    const payload = JSON.parse(rawText)
    const manifestLanguage = deriveLanguageCode(manifestFile, payload)
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

    for (const rawSource of manifestSources) {
      sources.push(
        normalizeSourceDefinition(
          rawSource,
          manifestLanguage,
          manifestFile,
          defaultFormat
        )
      )
    }
  }

  return sources
}

function shouldIncludePath(filePath, source) {
  const target = filePath.replace(/\\/g, '/')
  const lowerTarget = target.toLowerCase()
  const ext = path.extname(lowerTarget)
  if (!source.includeExts.includes(ext)) return false

  return source.includePrefix.some((prefix) => {
    if (prefix === '.') return true
    const normalizedPrefix = prefix.replace(/\\/g, '/').toLowerCase()
    return lowerTarget.startsWith(normalizedPrefix)
  })
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

function sourceRootPath(outputRoot, source) {
  return path.join(outputRoot, source.language, source.slug)
}

function evidenceFilePath(outputRoot, source) {
  return path.join(
    outputRoot,
    'licenses',
    source.language,
    `${source.slug}.json`
  )
}

function targetKindFromExtension(ext) {
  if (PDF_EXTS.has(ext)) return 'pdfs'
  if (PARQUET_EXTS.has(ext)) return 'parquet'
  if (IMAGE_EXTS.has(ext)) return 'images'
  return 'files'
}

function extractRootNameFromExtensions(exts) {
  const kinds = new Set(
    exts
      .filter((ext) => !ARCHIVE_EXTS.has(ext))
      .map((ext) => targetKindFromExtension(ext))
  )

  if (kinds.size === 1) {
    return [...kinds][0]
  }

  return 'files'
}

function filterArchiveExtractExtensions(source) {
  return source.includeExts.filter((ext) => !ARCHIVE_EXTS.has(ext))
}

function relativeOutputPath(outputRoot, filePath) {
  return path.relative(outputRoot, filePath).split(path.sep).join('/')
}

function sourceKey(source) {
  return String(source.id || source.repo || source.downloadUrl || source.title)
}

function pathNameFromUrl(url) {
  const parsed = new URL(url)
  const pathname = decodeURIComponent(parsed.pathname)
  const baseName = path.basename(pathname)
  return baseName || 'download'
}

async function writeJsonFile(filePath, value) {
  await fs.mkdir(path.dirname(filePath), { recursive: true })
  await fs.writeFile(filePath, `${JSON.stringify(value, null, 2)}\n`)
}

async function writeTextFile(filePath, value) {
  await fs.mkdir(path.dirname(filePath), { recursive: true })
  await fs.writeFile(filePath, `${value}\n`)
}

async function downloadFile(url, destination, options = {}) {
  if (!options.force) {
    try {
      const existing = await fs.stat(destination)
      if (existing.isFile() && existing.size > 0) {
        return {
          size: Number(existing.size),
          mediaType: 'application/octet-stream',
          reusedExisting: true,
        }
      }
    } catch (error) {
      if (error?.code !== 'ENOENT') {
        throw error
      }
    }
  }

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
  const tempPath = `${destination}.part-${process.pid}-${Date.now()}-${Math.random()
    .toString(36)
    .slice(2, 10)}`
  const stream = createWriteStream(tempPath)

  try {
    await pipeline(Readable.fromWeb(response.body), stream)
    await fs.rename(tempPath, destination)
  } catch (error) {
    await fs.rm(tempPath, { force: true }).catch(() => {})

    try {
      const existing = await fs.stat(destination)
      if (existing.isFile() && existing.size > 0) {
        return {
          size: Number(existing.size),
          mediaType:
            response.headers.get('content-type') ?? 'application/octet-stream',
          reusedExisting: true,
        }
      }
    } catch (statError) {
      if (statError?.code !== 'ENOENT') {
        throw statError
      }
    }

    throw error
  }

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

  return collected.sort((a, b) => a.localeCompare(b))
}

async function extractZip(pathToZip, extractTo, allowedExts, maxFiles) {
  const allowed = JSON.stringify(allowedExts || [])
  const script = `
import json
import pathlib
import sys
import zipfile

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

  return JSON.parse(stdout || '[]')
}

async function moveExtractedFilesIntoSourceTree(
  extractedDir,
  sourceRoot,
  outputRoot,
  sourcePath,
  sourceUrl,
  allowedExts,
  remaining
) {
  const extractedFiles = await collectFilesRecursively(
    extractedDir,
    allowedExts
  )
  const selectedFiles =
    remaining == null ? extractedFiles : extractedFiles.slice(0, remaining)
  const records = []

  for (const extractedFile of selectedFiles) {
    const ext = path.extname(extractedFile).toLowerCase()
    const relativeInsideArchive = path
      .relative(extractedDir, extractedFile)
      .split(path.sep)
      .join('/')
    const targetKind = targetKindFromExtension(ext)
    const targetPath = path.join(
      sourceRoot,
      targetKind,
      safeTargetPath(relativeInsideArchive)
    )
    await fs.mkdir(path.dirname(targetPath), { recursive: true })
    try {
      const existing = await fs.stat(targetPath)
      if (!(existing.isFile() && existing.size > 0)) {
        await fs.cp(extractedFile, targetPath, { force: true })
      }
    } catch (error) {
      if (error?.code !== 'ENOENT') {
        throw error
      }
      await fs.cp(extractedFile, targetPath, { force: true })
    }
    const sha256 = await computeSha256(targetPath)
    const stats = await fs.stat(targetPath)

    records.push({
      localPath: relativeOutputPath(outputRoot, targetPath),
      sourcePath: `${sourcePath}::${relativeInsideArchive}`,
      source: sourceUrl,
      category: targetKind === 'images' ? 'image' : targetKind,
      extension: ext,
      sha256,
      size: Number(stats.size),
    })
  }

  return {
    records,
    count: selectedFiles.length,
  }
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
    language: source.language,
    evidence,
    resolvedLicense,
    licenseText: licenseText ? 'Saved from dataset root.' : null,
  })

  const entries = (Array.isArray(tree) ? tree : []).filter((entry) => {
    if (entry.type !== 'file') return false
    return shouldIncludePath(entry.path, source)
  })

  return {
    kind: 'huggingface',
    source,
    entries,
    resolvedLicense,
    evidencePath,
  }
}

async function getGitHubInfo(source, outputRoot) {
  const encoded = encodeGitHubRepo(source.repo)
  const branch = source.branch ?? 'master'
  const treeUrl = `https://api.github.com/repos/${encoded}/git/trees/${encodeURIComponent(branch)}?recursive=1`
  const licenseUrl = `https://raw.githubusercontent.com/${encoded}/${encodeURIComponent(branch)}/LICENSE`
  const noticeUrl = `https://raw.githubusercontent.com/${encoded}/${encodeURIComponent(branch)}/NOTICE`
  const readmeUrl = `https://raw.githubusercontent.com/${encoded}/${encodeURIComponent(branch)}/README.md`

  const treeResponse = await fetch(treeUrl, {
    headers: { 'User-Agent': 'receipt-downloader-script' },
  })
  if (!treeResponse.ok) {
    throw new Error(`Cannot fetch repo tree for ${source.repo}`)
  }
  const treePayload = await treeResponse.json()

  const readmeResponse = await fetch(readmeUrl, {
    headers: { 'User-Agent': 'receipt-downloader-script' },
  })
  const readme = readmeResponse.ok ? await readmeResponse.text() : ''
  const readmeLicense = readme.match(/^license:\s*(.+)$/m)?.[1]?.trim()

  const licenseResponse = await fetch(licenseUrl, {
    headers: { 'User-Agent': 'receipt-downloader-script' },
  })
  const noticeResponse = await fetch(noticeUrl, {
    headers: { 'User-Agent': 'receipt-downloader-script' },
  })
  const licenseText = licenseResponse.ok ? await licenseResponse.text() : null
  const noticeText = noticeResponse.ok ? await noticeResponse.text() : null

  const resolvedLicense = normalizeLicense(
    readmeLicense ?? source.licenseExpectation ?? 'Unknown'
  )
  const evidencePath = evidenceFilePath(outputRoot, source)
  await writeJsonFile(evidencePath, {
    source: source.repo,
    title: source.title,
    language: source.language,
    evidence: {
      source: `https://github.com/${source.repo}`,
      licenseFromReadme: readmeLicense,
      licenseFiles: {
        license: licenseResponse.ok ? 'LICENSE' : null,
        notice: noticeResponse.ok ? 'NOTICE' : null,
      },
    },
    resolvedLicense,
    licenseText:
      licenseText || noticeText
        ? 'Captured from repository metadata and license files.'
        : null,
  })

  const entries = (treePayload.tree ?? []).filter((entry) => {
    if (entry.type !== 'blob') return false
    return shouldIncludePath(entry.path, source)
  })

  return {
    kind: 'github',
    source,
    entries,
    resolvedLicense,
    evidencePath,
  }
}

async function getDirectInfo(source, outputRoot) {
  const resolvedLicense = normalizeLicense(
    source.licenseExpectation ?? 'Unknown'
  )
  const evidencePath = evidenceFilePath(outputRoot, source)
  await writeJsonFile(evidencePath, {
    source: source.downloadUrl,
    title: source.title,
    language: source.language,
    evidence: {
      source: source.sourceUrl ?? source.downloadUrl,
      downloadUrl: source.downloadUrl,
      licenseReference: source.licenseReference ?? source.sourceUrl ?? null,
    },
    resolvedLicense,
    licenseText: null,
  })

  return {
    kind: 'direct',
    source,
    entries: [
      {
        path: source.fileName || pathNameFromUrl(source.downloadUrl),
        downloadUrl: source.downloadUrl,
      },
    ],
    resolvedLicense,
    evidencePath,
  }
}

function buildSourceManifest(source, summary) {
  return {
    sourceId: sourceKey(source),
    title: source.title,
    sourceType: source.kind,
    language: source.language,
    languages: [source.language],
    license: summary.resolvedLicense || source.licenseExpectation || 'Unknown',
    evidencePath: relativeOutputPath(summary.outputRoot, summary.evidencePath),
    totalAvailable: summary.entries.length,
    downloadedFiles: summary.downloadedFiles,
    skipped: summary.skipped,
    notes: source.description ?? '',
  }
}

function hasLanguageMatch(source, requestedLanguages) {
  if (!requestedLanguages || requestedLanguages.length === 0) return true
  return requestedLanguages.includes(source.language.toLowerCase())
}

async function processArchiveEntry({
  archiveUrl,
  archiveName,
  sourceRoot,
  sourcePath,
  outputRoot,
  source,
  remaining,
}) {
  const archiveTarget = path.join(
    sourceRoot,
    'archives',
    safeTargetPath(archiveName)
  )
  await downloadFile(archiveUrl, archiveTarget)

  const extractRootName = extractRootNameFromExtensions(source.includeExts)
  const extractedDir = path.join(
    sourceRoot,
    extractRootName,
    slugify(sourcePath)
  )
  await fs.rm(extractedDir, { recursive: true, force: true }).catch(() => {})

  const allowedExtractExts = filterArchiveExtractExtensions(source)
  try {
    await extractZip(
      archiveTarget,
      extractedDir,
      allowedExtractExts,
      remaining == null ? -1 : remaining
    )
  } catch (error) {
    const message = error?.message ?? String(error)
    const isBadArchive =
      message.includes('BadZipFile') || message.includes('not a zip file')

    if (!isBadArchive) {
      throw error
    }

    await fs.rm(archiveTarget, { force: true }).catch(() => {})
    await fs.rm(extractedDir, { recursive: true, force: true }).catch(() => {})
    await downloadFile(archiveUrl, archiveTarget, { force: true })
    await extractZip(
      archiveTarget,
      extractedDir,
      allowedExtractExts,
      remaining == null ? -1 : remaining
    )
  }

  const extracted = await moveExtractedFilesIntoSourceTree(
    extractedDir,
    sourceRoot,
    outputRoot,
    sourcePath,
    archiveUrl,
    allowedExtractExts,
    remaining
  )

  await fs.unlink(archiveTarget).catch(() => {})
  await fs.rm(extractedDir, { recursive: true, force: true }).catch(() => {})
  return extracted
}

async function processHuggingFaceSource(source, outputRoot, perSourceLimit) {
  const summary = await getHuggingFaceInfo(source, outputRoot)
  const sourceRoot = sourceRootPath(outputRoot, source)
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

  let remaining = perSourceLimit
  const resolvedEntries = summary.entries.sort((a, b) =>
    a.path.localeCompare(b.path)
  )
  const effectiveEntries =
    remaining == null ? resolvedEntries : resolvedEntries.slice(0, remaining)

  for (const entry of effectiveEntries) {
    if (!entry.path) continue
    if (remaining != null && remaining <= 0) break

    const sourcePath = entry.path
    const ext = path.extname(sourcePath).toLowerCase()
    const encodedId = encodeHFRepoId(source.id)
    const sourceUrl = `https://huggingface.co/datasets/${source.id}/resolve/main/${sourcePath}`

    if (ARCHIVE_EXTS.has(ext) && source.extractArchives) {
      const extracted = await processArchiveEntry({
        archiveUrl: sourceUrl,
        archiveName: sourcePath,
        sourceRoot,
        sourcePath,
        outputRoot,
        source,
        remaining,
      })
      files.push(...extracted.records)
      if (remaining != null) {
        remaining -= extracted.count
      }
      continue
    }

    const targetKind = targetKindFromExtension(ext)
    const targetPath = path.join(
      sourceRoot,
      targetKind,
      safeTargetPath(sourcePath)
    )
    const meta = await downloadFile(
      `https://huggingface.co/datasets/${encodedId}/resolve/main/${encodeHFPath(sourcePath)}`,
      targetPath
    )
    const sha256 = await computeSha256(targetPath)

    files.push({
      localPath: relativeOutputPath(outputRoot, targetPath),
      sourcePath,
      source: sourceUrl,
      category: targetKind === 'images' ? 'image' : targetKind,
      extension: ext,
      sha256,
      size: meta.size || Number((await fs.stat(targetPath)).size),
      mediaType: meta.mediaType,
    })

    if (remaining != null) {
      remaining -= 1
    }
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

async function processGitHubSource(source, outputRoot, perSourceLimit) {
  const summary = await getGitHubInfo(source, outputRoot)
  const sourceRoot = sourceRootPath(outputRoot, source)
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

  let remaining = perSourceLimit
  const effectiveEntries = summary.entries
    .sort((a, b) => a.path.localeCompare(b.path))
    .slice(0, remaining == null ? undefined : remaining)

  const encodedRepo = encodeGitHubRepo(source.repo)
  const branch = source.branch ?? 'master'

  for (const entry of effectiveEntries) {
    if (remaining != null && remaining <= 0) break

    const sourcePath = entry.path
    const ext = path.extname(sourcePath).toLowerCase()
    const fileUrl = `https://raw.githubusercontent.com/${encodedRepo}/${branch}/${encodeHFPath(sourcePath)}`

    if (ARCHIVE_EXTS.has(ext) && source.extractArchives) {
      const extracted = await processArchiveEntry({
        archiveUrl: fileUrl,
        archiveName: sourcePath,
        sourceRoot,
        sourcePath,
        outputRoot,
        source,
        remaining,
      })
      files.push(...extracted.records)
      if (remaining != null) {
        remaining -= extracted.count
      }
      continue
    }

    const targetKind = targetKindFromExtension(ext)
    const targetPath = path.join(
      sourceRoot,
      targetKind,
      safeTargetPath(sourcePath)
    )
    const meta = await downloadFile(fileUrl, targetPath)
    const sha256 = await computeSha256(targetPath)

    files.push({
      localPath: relativeOutputPath(outputRoot, targetPath),
      sourcePath,
      source: fileUrl,
      category: targetKind === 'images' ? 'image' : targetKind,
      extension: ext,
      sha256,
      size: meta.size || Number((await fs.stat(targetPath)).size),
      mediaType: meta.mediaType,
    })

    if (remaining != null) {
      remaining -= 1
    }
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

async function processDirectSource(source, outputRoot, perSourceLimit) {
  const summary = await getDirectInfo(source, outputRoot)
  const sourceRoot = sourceRootPath(outputRoot, source)
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

  const entry = summary.entries[0]
  const sourcePath = entry.path
  const ext = path.extname(sourcePath).toLowerCase()

  if (ARCHIVE_EXTS.has(ext) && source.extractArchives) {
    const extracted = await processArchiveEntry({
      archiveUrl: entry.downloadUrl,
      archiveName: sourcePath,
      sourceRoot,
      sourcePath,
      outputRoot,
      source,
      remaining: perSourceLimit,
    })
    files.push(...extracted.records)
  } else {
    const targetKind = targetKindFromExtension(ext)
    const targetPath = path.join(
      sourceRoot,
      targetKind,
      safeTargetPath(sourcePath)
    )
    const meta = await downloadFile(entry.downloadUrl, targetPath)
    const sha256 = await computeSha256(targetPath)

    files.push({
      localPath: relativeOutputPath(outputRoot, targetPath),
      sourcePath,
      source: entry.downloadUrl,
      category: targetKind === 'images' ? 'image' : targetKind,
      extension: ext,
      sha256,
      size: meta.size || Number((await fs.stat(targetPath)).size),
      mediaType: meta.mediaType,
    })
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
  lines.push('| Language | Source | License | Files | Notes | Evidence |')
  lines.push('|---|---|---|---:|---|---|')

  for (const source of manifest.sources) {
    lines.push(
      `| ${source.language} | ${source.title} | ${source.license} | ${source.downloadedFiles.length} | ${source.notes} | ${source.evidencePath} |`
    )
  }

  lines.push('')
  await writeTextFile(manifestPath, lines.join('\n'))
}

function runTaskPool(tasks, concurrency) {
  const results = new Array(tasks.length)
  let nextIndex = 0

  async function workerLoop() {
    while (true) {
      const currentIndex = nextIndex
      nextIndex += 1
      if (currentIndex >= tasks.length) {
        return
      }
      results[currentIndex] = await tasks[currentIndex]()
    }
  }

  const loopCount = Math.max(1, Math.min(concurrency, tasks.length))
  return Promise.all(
    Array.from({ length: loopCount }, () => workerLoop())
  ).then(() => results)
}

function runSourceWorker(task) {
  return new Promise((resolve, reject) => {
    const worker = new Worker(new URL(import.meta.url), {
      workerData: {
        mode: 'process-source',
        task,
      },
    })

    worker.once('message', resolve)
    worker.once('error', reject)
    worker.once('exit', (code) => {
      if (code !== 0) {
        reject(new Error(`Source worker exited with code ${code}`))
      }
    })
  })
}

async function processSourceTask(task) {
  const { source, outputRoot, perSourceLimit } = task

  if (source.kind === 'huggingface') {
    return processHuggingFaceSource(source, outputRoot, perSourceLimit)
  }
  if (source.kind === 'github') {
    return processGitHubSource(source, outputRoot, perSourceLimit)
  }
  return processDirectSource(source, outputRoot, perSourceLimit)
}

async function main() {
  const args = parseArgs(process.argv.slice(2))
  if (args.help) {
    printHelp()
    return
  }

  const loadedSources = await loadSourceDefinitions(
    args.manifests,
    args.defaultFormat
  )
  const selectedSources = loadedSources.filter((source) => {
    if (args.sourceFilters && args.sourceFilters.length > 0) {
      const matchesSource = args.sourceFilters.some((filterValue) => {
        const target = filterValue.toLowerCase()
        return (
          sourceKey(source).toLowerCase() === target ||
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
      ...new Set(loadedSources.map((source) => source.language)),
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
      manifests: args.manifests,
      output: args.output,
      defaultFormat: args.defaultFormat,
      maxPerSource: args.maxPerSource,
      full: args.full,
      workers: args.workers,
      sourceFilters: args.sourceFilters,
      languageFilters: args.languageFilters,
    },
    sources: [],
  }

  const perSourceLimit = args.full ? null : args.maxPerSource
  const sourceTasks = selectedSources.map((source) => ({
    source,
    outputRoot,
    perSourceLimit,
  }))

  for (const source of selectedSources) {
    console.log(`Queueing ${source.language}/${source.slug}`)
  }

  manifest.sources = await runTaskPool(
    sourceTasks.map((task) => () => runSourceWorker(task)),
    args.workers
  )

  const manifestPath = path.join(outputRoot, 'receipt-license-manifest.json')
  await writeJsonFile(manifestPath, manifest)

  const reportPath = path.join(outputRoot, 'receipt-license-report.md')
  await generateMarkdownReport(manifest, reportPath)

  console.log(`Done. Manifest written to ${manifestPath}`)
  console.log(`Report written to ${reportPath}`)
}

async function workerMain() {
  if (workerData?.mode !== 'process-source') {
    throw new Error('Unsupported worker mode')
  }

  const result = await processSourceTask(workerData.task)
  parentPort?.postMessage(result)
}

if (isMainThread) {
  main().catch((error) => {
    console.error(error)
    process.exit(1)
  })
} else {
  workerMain().catch((error) => {
    throw error
  })
}
