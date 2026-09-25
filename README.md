# Placement Mirror

A mock interview coach for Snapdragon X laptops that gives live feedback on delivery and runs fully on-device.

Work in progress.

## Development setup

The project uses three Python environments. They cannot be merged: qai-hub-models 0.63.0
requires onnxruntime 1.22.1 or older, and onnxruntime-qnn 2.6.0 requires onnxruntime
1.24.2 or newer. All three use Python 3.11. Commands are for PowerShell from the repo root.

| venv | requirements | used for | platforms |
|---|---|---|---|
| `.venv-app` | `requirements.txt` | the app, `app.runtime.runner --check`, tests except the Whisper parity tests | Windows ARM64 and x64 |
| `.venv-aihub` | `requirements-aihub.txt` | `aihub/` scripts: submit and collect AI Hub jobs, fetch models | Windows x64 |
| `.venv-eval` | `eval/requirements.txt` | `eval/` scripts, `tools/make_whisper_assets.py`, the Whisper parity tests | Windows x64 |

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

Press Start camera, choose a question, press Calibrate while sitting as in an interview and
looking at the camera, then Start answer and Stop answer. An answer also stops by itself at
the question's suggested time plus 60 s. Audio comes from the default microphone and runs
only during the answer. Each answer is saved as `timeline.json` and `report.json` under
`%LOCALAPPDATA%\PlacementMirror\sessions\<session id>\` (metrics and transcript text
only, never audio or video). `--data-dir <dir>` moves that folder.

After an answer, Open the report shows facing the camera, face not visible, posture and words
per minute per 10 s window as charts with data tables, the transcript with fillers highlighted and long
pauses inline, and three things to work on. Each metric is compared with a coaching
guideline in `app/config.py` (`ReportConfig`). The guidelines are coaching targets, not
measurements, and are provisional until reviewed. The History page lists every saved answer
and the trend of facing the camera, words per minute, fillers per minute and long pauses.
All pages and charts are served by the app. Nothing is loaded from the network.

To check that nothing leaves localhost, run the server under the offline check, which logs
every network attempt of the Python process and refuses any that is not loopback. Start
Edge with `--log-net-log=<file>` to log the browser side too, then summarize both:

```powershell
.venv-app\Scripts\python tools\offline_check.py serve net.jsonl
.venv-app\Scripts\python tools\offline_check.py summarize net.jsonl --net-log edge-netlog.json
```

Replay mode, for development: `--replay <video> <wav>` makes the browser play the video
file instead of the camera and the server play the WAV (16 kHz mono 16 bit) instead of the
microphone. Both start from the beginning on Start answer, and the answer stops when the
video ends. The video must be a format Edge plays (WebM VP8 or VP9, or MP4 H.264). With `--stats-dir <dir>` each session's
video stage p50 and p95 and per segment audio timings are saved there (never the
transcript). `--audio-file clip.wav` plays a 16 kHz mono recording instead of the
microphone and `--no-audio` runs video only.

The Whisper parity tests (`tests/test_mel_parity.py`, `tests/test_decode_parity.py`) need
transformers and torch. They skip with a message in `.venv-app` and run in `.venv-eval`
(see Evaluation below). transformers and tokenizers are never app dependencies.

### AI Hub tools

Needs an AI Hub API token, configured once with `qai-hub configure --api_token <token>`.
The token is stored in `~/.qai_hub/client.ini` and never goes in the repo.

```powershell
py -3.11 -m venv .venv-aihub
.venv-aihub\Scripts\python -m pip install -r requirements-aihub.txt
.venv-aihub\Scripts\python -m aihub.fetch_models
```

`aihub.fetch_models` downloads the compiled models into `models/` and writes
`models/manifest.json`. Only the manifest is committed. torch is pinned to 2.10.0, the
version the AI Hub models were exported with.

### Evaluation

x64 only. Versions are pinned because evaluation results depend on them.

```powershell
py -3.11 -m venv .venv-eval
.venv-eval\Scripts\python -m pip install -r eval\requirements.txt
.venv-eval\Scripts\python -m pytest tests/test_mel_parity.py tests/test_decode_parity.py
```

The parity tests also need the models in `models/` and the qai_hub_models Whisper sample
audio (`jfk.npz`) in the local asset cache.
