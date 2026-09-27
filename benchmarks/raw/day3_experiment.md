# Day 3: whisper_tiny runtime format x precision on the NPU

Goal: isolate the effect of runtime format and precision on whisper_tiny latency on the
NPU, one variable at a time. Numbers only, no comparisons yet.

Source for every number below: AI Hub hosted Snapdragon X Elite ("Snapdragon X Elite CRD"),
Measurement records in benchmarks/raw/day3_aihub_results.json, raw per-iteration times in
benchmarks/raw/samples/<profile job>.json. Jobs: benchmarks/raw/jobs_day3.json. Script:
aihub/submit_day3.py.

## Design

All cells use the same source models (encoder mnwvkg23q, decoder mq33z49rq, the Day 0
qai_hub_models 0.63.0 export) and the same input specs. Every cell, existing ones included,
was profiled again in one batch with the same options:

`--compute_unit npu --qairt_version 2.50 --max_profiler_iterations 100`

| Cell | Compile options (encoder shown, decoder uses its own output names and graph name) | Compile jobs (encoder, decoder) |
|---|---|---|
| onnx_fp32 | `--target_runtime onnx --output_names ...` | j5688od7g, jp3zzowz5 (Day 1) |
| precompiled_fp16 | `--target_runtime precompiled_qnn_onnx --qairt_version latest --output_names ... --qnn_options context_enable_graphs=hf_whisper_encoder --quantize_full_type float16 --quantize_io` | jp1nnvn2g, jp4yy9yvp (Day 0, latest resolved to QAIRT 2.50.0.260828221209) |
| onnx_fp16 | onnx_fp32 options plus `--quantize_full_type float16 --quantize_io` | not created, AI Hub rejected both at submission |
| precompiled_fp32 | precompiled_fp16 options without `--quantize_full_type float16 --quantize_io`, `--qairt_version 2.50` | jp4y2q3qp, jpxlzvxjp |

## Flags checked against the docs

Quoted from https://app.aihub.qualcomm.com/docs/hub/api.html (fetched 2026-09-27). Each
job entry in jobs_day3.json also carries the quotes for the flags it uses.

| Flag | Docs |
|---|---|
| `--target_runtime` | "onnx : ONNX Runtime ( .onnx ) precompiled_qnn_onnx : ONNX Runtime model with an embedded Qualcomm AI Engine Direct context binary." |
| `--output_names` | "Overrides the default output names. [...] When used, a name must be specified for each model output." |
| `--qairt_version` | "Specifies the version of Qualcomm AI Runtime to use in the job. This is applicable to: [...] Compile jobs targeting ONNX Runtime with an embedded Qualcomm AI Engine Direct context binary (i.e., precompiled_qnn_onnx ). All link, inference, and profile jobs." |
| `--qnn_options context_enable_graphs=<name>` | "The name of the graphs to compile into a model file or load at runtime." |
| `--quantize_full_type` | "Quantizes an unquantized model to the specified type. [...] Options: int8 [...] int16 [...] w8a16 [...] w4a8 [...] w4a16 [...] Requirements: This option cannot be used if the input is an AIMET model, or if the target runtime is ONNX." |
| `--quantize_io` | "Quantize the input and outputs when quantizing a model. [...] Default: Inputs and output are not quantized." |
| `--compute_unit npu` | "When used in a profile or inference job, this targets the specified compute unit(s). Implicit fall back to CPU will be determined by the target_runtime ; TfLite and ONNX Runtime always include CPU fallback." |
| `--max_profiler_iterations` | "Specifies the maximum number of profile iterations. [...] Default: 100" |

Docs that bear on what the precision variable changes (options not set here):
- ONNX Runtime QNN EP `qnn_enable_htp_fp16_precision`: "Enable a fp32 model to be inferenced
  with fp16 precision. [...] Default is 1 (enable)."
- QNN `default_graph_htp_precision`: "If no precision value is set, the QNN HTP backend
  assumes that the client expects to run a quantized network. When the precision value is
  set to FLOAT16 , the QNN HTP backend will convert user provided float32 inputs to float16
  and execute the graph with float16 math."

Decisions by the author on 2026-09-27, after these doc conflicts were raised: submit
onnx_fp16 as specified and record what AI Hub does, submit precompiled_fp32 as specified,
and profile all cells again in one batch.

## Results

| Cell | Component | Profile job | Model I/O precision | Exec precision (profile log) | Ops on | Iterations | p50 (us) | p95 (us) | Peak memory, upper end (bytes) |
|---|---|---|---|---|---|---|---|---|---|
| onnx_fp32 | encoder | jpe7n9y75 | float32 | float16 | NPU 274 | 100 | 23083 | 23704 | 69849088 |
| onnx_fp32 | decoder | jgzl0enz5 | float32 | float16 | NPU 472 | 100 | 3454 | 4037 | 77717504 |
| precompiled_fp16 | encoder | jp1nm06kg | float16 | float16 | NPU 294 | 100 | 25008 | 25480 | 35098624 |
| precompiled_fp16 | decoder | j57e8z9qp | float16 | float16 | NPU 509 | 100 | 2375 | 2888 | 87826432 |
| onnx_fp16 | encoder | none | | | | | | | |
| onnx_fp16 | decoder | none | | | | | | | |
| precompiled_fp32 | encoder | jp0mxyj2g | float32 | float16 | NPU 294 | 100 | 23739 | 24641 | 69844992 |
| precompiled_fp32 | decoder | j5qld8y7p | float32 | float16 | NPU 509 | 100 | 3586 | 4325 | 77512704 |

- p50 and p95: numpy.percentile(samples, q, method='linear') over the per-iteration times
  (execution_summary.all_inference_times). Peak memory: upper end of
  execution_summary.inference_memory_peak_range.
- Model I/O precision: from the compile options and the target model's float I/O dtypes.
  Exec precision: every profile log has the line "enable_htp_fp16_precision = 1", so all
  six profiles ran float16 math on the HTP, the float32 cells included.
- onnx_fp16: both compiles were rejected at submission with "UserError: The
  --quantize_full_type option is not supported for target_runtime='ONNX'.", as the docs
  say. No model, no profile.
- Profile runtime versions from the logs: ONNX Runtime 1.27.1, QNN 2.50.0.
