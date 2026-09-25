# Placement Mirror

A mock interview coach for Snapdragon X laptops that gives live feedback on delivery and runs fully on-device.

Work in progress.

## Development setup

The project uses three Python environments. They cannot be merged: qai-hub-models 0.63.0
requires onnxruntime 1.22.1 or older, and onnxruntime-qnn 2.6.0 requires onnxruntime
1.24.2 or newer. All three use Python 3.11. Commands are for PowerShell from the repo root.

| venv | requirements | used for | platforms |
|---|---|---|---|
| `.venv-app` | `requirements.txt` | the app, `app.runtime.runner --check`, tests | Windows ARM64 and x64 |
| `.venv-aihub` | `requirements-aihub.txt` | `aihub/` scripts: submit and collect AI Hub jobs, fetch models | Windows x64 |
| `.venv-eval` | `eval/requirements.txt` | `eval/` scripts: filler and eye contact evaluation | Windows x64 |

### App

On a Snapdragon X laptop use a native ARM64 Python 3.11. `requirements.txt` is checked on
Windows ARM64 in CI (`.github/workflows/arm64-deps.yml`).

```powershell
py -3.11 -m venv .venv-app
.venv-app\Scripts\python -m pip install -r requirements.txt
.venv-app\Scripts\python -m app.runtime.runner --check
```

The runner check needs the model files, which come from the AI Hub environment below.
For tests, also install pytest in this venv and run `.venv-app\Scripts\python -m pytest`.

To run the app, start the server and open http://127.0.0.1:8000 in Edge:

```powershell
.venv-app\Scripts\python -m app.server
```

Press Start camera, then Calibrate while sitting as in an interview and looking at the
camera. Audio comes from the default microphone. With `--stats-dir <dir>` each session's
video stage p50 and p95 and per segment audio timings are saved there (never the
transcript). `--audio-file clip.wav` plays a 16 kHz mono recording instead of the
microphone and `--no-audio` runs video only.

The Whisper parity tests (`tests/test_mel_parity.py`, `tests/test_decode_parity.py`) need
transformers and torch, so they skip in `.venv-app`. Run them where the eval requirements
and pytest are installed.

### AI Hub tools

Needs an AI Hub API token, configured once with `qai-hub configure --api_token <token>`.
The token is stored in `~/.qai_hub/client.ini` and never goes in the repo.

```powershell
py -3.11 -m venv .venv-aihub
.venv-aihub\Scripts\python -m pip install -r requirements-aihub.txt
.venv-aihub\Scripts\python -m aihub.fetch_models
```

`aihub.fetch_models` downloads the compiled models into `models/` and writes
`models/manifest.json`. Only the manifest is committed.

### Evaluation

x64 only. Versions are pinned because evaluation results depend on them.

```powershell
py -3.11 -m venv .venv-eval
.venv-eval\Scripts\python -m pip install -r eval\requirements.txt
```
