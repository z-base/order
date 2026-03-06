import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="corpus.txt",
    model_prefix="tokenizer",
    vocab_size=8000,
    model_type="unigram",  # or "bpe"
    character_coverage=1.0
)