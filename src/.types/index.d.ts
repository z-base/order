declare module '@agnai/sentencepiece-js' {
  export interface SentencePieceStatus {
    delete(): void
  }

  export interface SentencePieceVector<T> {
    size(): number
    get(index: number): T
    delete(): void
  }

  export interface SentencePieceStringViewHandle {
    delete(): void
  }

  export interface SentencePieceStringView {
    getView(): SentencePieceStringViewHandle
    delete(): void
  }

  export interface SentencePieceProcessorBinding {
    Load(model: SentencePieceStringViewHandle): SentencePieceStatus
    EncodeAsIds(text: SentencePieceStringViewHandle): SentencePieceVector<number>
    EncodeAsPieces(text: SentencePieceStringViewHandle): SentencePieceVector<string>
    DecodeIds(ids: SentencePieceVector<number>): string
    LoadVocabulary(vocab: SentencePieceStringViewHandle, threshold: number): SentencePieceStatus
  }

  export interface SentencePieceModule {
    FS: {
      writeFile(path: string, data: string | Uint8Array | ArrayBufferLike): void
    }
    StringView: new (value: string) => SentencePieceStringView
    SentencePieceProcessor: new () => SentencePieceProcessorBinding
    vecFromJSArray(values: readonly number[]): SentencePieceVector<number>
  }

  export class SentencePieceProcessor {
    sentencepiece?: SentencePieceModule
    processor?: SentencePieceProcessorBinding

    constructor()
    load(url: string): Promise<void>
    encodeIds(text: string): number[]
    encodePieces(text: string): string[]
    decodeIds(ids: readonly number[]): string
    loadVocabulary(url: string): void
  }

  export function cleanText(text: string): string

  const defaultExport: {
    SentencePieceProcessor: typeof SentencePieceProcessor
    cleanText: typeof cleanText
  }

  export default defaultExport
}
