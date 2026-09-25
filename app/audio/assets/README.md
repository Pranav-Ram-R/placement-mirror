# Whisper assets

Written by `tools/make_whisper_assets.py` in the eval venv. The app reads these files so
it does not need transformers or tokenizers at runtime.

Source checkpoint: `openai/whisper-tiny`, Hugging Face revision `169d4a4341b33bc18d8881c4b69c2e104e1cc0af`, the checkpoint the
AI Hub whisper_tiny models are exported from (qai_hub_models whisper_tiny
`WHISPER_VERSION`). Read with transformers 4.56.2. Whisper is MIT licensed.

| file | content |
|---|---|
| `mel_filters.npy` | `WhisperFeatureExtractor.mel_filters` transposed to (80, 201), float64. n_fft 400, hop 160, 16000 Hz, 30 s window |
| `vocab_bytes.json` | list indexed by token id: the token's bytes as hex (GPT-2 byte to unicode table reversed), `null` for special tokens (id 50257 and up) |
| `special_tokens.json` | special token ids, `suppress_tokens` and `begin_suppress_tokens` from the generation config |
| `prompt_ids.json` | `decoder_prefix` ['<|startoftranscript|>', '<|en|>', '<|transcribe|>', '<|notimestamps|>'], and the optional filler prompt from Task B as `WhisperProcessor.get_prompt_ids` ids |
