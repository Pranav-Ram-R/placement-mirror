# whisper_tiny on the NPU: runtime format and precision

Question: which compiled form of whisper_tiny (encoder and decoder) should the app ship
for the Hexagon NPU. Two variables: runtime format (plain ONNX compiled on device, or a
precompiled QNN context binary inside ONNX) and precision of the stored model (float32
or float16).

Every timing and memory number here is from AI Hub hosted Snapdragon X Elite ("Snapdragon
X Elite CRD"), QAIRT 2.50, ONNX Runtime 1.27.1, profile options `--compute_unit npu
--qairt_version 2.50 --max_profiler_iterations 100`, one profile job per cell. Nothing
here is validated on a physical device yet.

Files:
- design, flags and doc quotes: benchmarks/raw/day3_experiment.md, jobs in
  benchmarks/raw/jobs_day3.json
- Measurements: benchmarks/raw/day3_aihub_results.json, per-iteration times in
  benchmarks/raw/samples/
- per-layer Measurements: benchmarks/raw/day3_decoder_layers.json
  (aihub/collect_day3_details.py)
- Derived records: benchmarks/raw/day3_derived.json (benchmarks/analyze_day3.py)
- decoder bytes per call: benchmarks/raw/day3_decoder_io_bytes.json, from
  benchmarks/raw/day3_decoder_signatures.json

## Results: the 2x2 table

p50 and p95 in microseconds over 100 iterations. Peak memory is the upper end of
inference_memory_peak_range, in bytes. Every profile log has "enable_htp_fp16_precision =
1", so all cells ran float16 math on the HTP. The precision column is the stored model's
I/O precision.

Encoder:

| Precision | onnx | precompiled_qnn_onnx |
|---|---|---|
| float32 | jpe7n9y75: p50 23083, p95 23704, peak 69849088 | jp0mxyj2g: p50 23739, p95 24641, peak 69844992 |
| float16 | rejected at compile submission | jp1nm06kg: p50 25008.5, p95 25480, peak 35098624 |

Decoder (one call, one token):

| Precision | onnx | precompiled_qnn_onnx |
|---|---|---|
| float32 | jgzl0enz5: p50 3453.5, p95 4037, peak 77717504 | j5qld8y7p: p50 3586.5, p95 4325, peak 77512704 |
| float16 | rejected at compile submission | j57e8z9qp: p50 2375, p95 2888, peak 87826432 |

p95 values are rounded to the microsecond here. The records hold the unrounded values.

### The rejected cell (onnx float16)

Both compile jobs (`--target_runtime onnx --quantize_full_type float16 --quantize_io`)
were rejected at submission with:

"UserError: The --quantize_full_type option is not supported for target_runtime='ONNX'."

The AI Hub docs for `--quantize_full_type` say: "This option cannot be used if the input
is an AIMET model, or if the target runtime is ONNX." The cell has no model and no
profile.

## Derived comparisons

From the p50 Measurements. Each is a Derived record in benchmarks/raw/day3_derived.json
with its formula and inputs.

| Comparison | Component | Formula | Difference (us) | Relative |
|---|---|---|---|---|
| Format effect at float32 | encoder | p50(precompiled fp32, jp0mxyj2g) - p50(onnx fp32, jpe7n9y75) | 656 | 2.84 % |
| Format effect at float32 | decoder | p50(precompiled fp32, j5qld8y7p) - p50(onnx fp32, jgzl0enz5) | 133 | 3.85 % |
| Storage effect within precompiled | encoder | p50(precompiled fp16, jp1nm06kg) - p50(precompiled fp32, jp0mxyj2g) | 1269.5 | 5.35 % |
| Storage effect within precompiled | decoder | p50(precompiled fp16, j57e8z9qp) - p50(precompiled fp32, j5qld8y7p) | -1211.5 | -33.78 % |

Relative = difference / p50 of the second cell * 100. A positive number means the first
cell is slower. Each cell is one profile job, so the spread between repeated jobs is
not known. These differences have no run-to-run error bar.

## Per-layer finding (decoder, precompiled fp32 against fp16)

Per-layer timing is available for the precompiled models. Each profile's
execution_detail lists 509 layers, all on the NPU, with an execution_time. What these
numbers are:

- qai_hub 0.55.0 (public_api_pb2.pyi, LayerDetail.execution_time): "If available, the
  minimum execution time for this layer." The unit is not stated there.
  estimated_inference_time is documented as "Time spent in inference, in microseconds."
  The tables below label the layer unit accordingly.
- The profile job logs show the layer times come from a separate task, "performing
  inference by layer", which "Successfully ran model for 2 iterations" with the QNN EP
  option `profiling_level = optrace`. The 100-iteration timed task has no profiling_level
  line. ONNX Runtime QNN EP docs, profiling_level: "'off' - default. 'basic' 'detailed'
  'optrace' - Requires QAIRT 2.39 or later".
- So the layer times come from an instrumented 2-iteration run, not from the timed run.
  Their sums (Derived: 4558 for j5qld8y7p, 2933 for j57e8z9qp) are larger than the
  timed p50 values (3586.5 and 2375). They show where the difference sits, not its size
  in the timed run.

Top 10 layers by execution_time (unit: us per the note above):

| Rank | j5qld8y7p (fp32 I/O) | Time | j57e8z9qp (fp16 I/O) | Time |
|---|---|---|---|---|
| 1 | node_Conv_391 | 1039 | node_Conv_391 | 1059 |
| 2 | Output | 350 | node_add_1 | 42 |
| 3 | Input | 261 | node_layer_norm | 39 |
| 4 | node_Split_55_to_slice_0 | 185 | node_Split_55_to_slice_0 | 35 |
| 5 | node_Split_237_to_slice_0 | 175 | node_Split_155_to_slice_1 | 24 |
| 6 | node_Split_337_to_slice_0 | 174 | node_conv2d_55 | 22 |
| 7 | node_Split_155_to_slice_0 | 173 | node_layer_norm_9 | 22 |
| 8 | node_Split_246_to_slice_0 | 147 | node_slice_7 | 22 |
| 9 | node_Split_67_to_slice_0 | 144 | node_slice_1 | 20 |
| 10 | node_Split_167_to_slice_0 | 144 | node_Split_155_to_slice_3 | 20 |

node_Conv_391 produces the logits (the output projection, from the source graph). It
takes about the same time in both.

Where the difference sits (Derived records, fp32 layer sum minus fp16 layer sum over the
same layers):

| Layer group | Layers | fp32 sum | fp16 sum | Difference |
|---|---|---|---|---|
| Input and Output | 2 | 611 | 19 | 592 |
| First slice of the Split on each cross attention cache input | 8 | 1286 | 83 | 1203 |
| logits_0231 | 1 | 37 | 0 | 37 |
| All other layers | 498 | 2624 | 2831 | -207 |

The eight Split layers were mapped to their inputs by name through the source graph. In
the onnx float32 decoder (compile job jp3zzowz5), node_Split_55, 67, 155, 167, 237, 246,
337 and 349 take k_cache_cross_0 to 3 and v_cache_cross_0 to 3 as input. The precompiled
graphs use the same layer names.

Finding: in the layer profile, the fp32 decoder's extra time sits at the graph input and
output layers and at the first layer that reads each of the eight cross attention cache
inputs. The rest of the graph is not slower. This matches the QNN docs for
default_graph_htp_precision: "the QNN HTP backend will convert user provided float32
inputs to float16 and execute the graph with float16 math". The profile does not name a
conversion step, so the link to input conversion is an interpretation, not something
the profile shows directly.

## Bytes passed per decoder call

Derived from the tensor shapes and dtypes of the two precompiled decoders (graph inputs
and outputs of their ONNX wrappers). This is not a measurement. bytes = product(shape) *
bytes per element (float32 4, float16 2, int32 4).

| Tensors | Shapes | fp32 bytes | fp16 bytes |
|---|---|---|---|
| input_ids, position_ids (int32) | [1,1], [1] | 8 | 8 |
| attention_mask | [1,1,1,200] | 800 | 400 |
| self attention caches in (8) | [6,1,64,199] and [6,1,199,64] | 2445312 | 1222656 |
| cross attention caches in (8) | [6,1,64,1500] and [6,1,1500,64] | 18432000 | 9216000 |
| Passed in | | 20878120 | 10439064 |
| self attention caches out (8) | as above | 2445312 | 1222656 |
| logits | [1,51865,1,1] | 207460 | 103730 |
| Passed out | | 2652772 | 1326386 |
| Total per call | | 23530892 | 11765450 |

The app passes every input on every decoder call, the cross attention caches included.

## Decision

Shipping configuration, decided by the author on 2026-09-28:

- encoder: precompiled_qnn_onnx float32 I/O (compile job jp4y2q3qp, profiled as jp0mxyj2g)
- decoder: precompiled_qnn_onnx float16 I/O (compile job jp4yy9yvp, profiled as j57e8z9qp)
- the onnx float32 models stay as the fallback step (onnx on the QNN NPU with an EP
  context cache, then onnx on CPU), as app/runtime/runner.py loads them

In these runs the onnx float32 encoder had the lower p50 (23083 against 23739,
Derived 656 us). The decision still takes the precompiled encoder.

## What is not explained

- The fp16 encoder is slower than the fp32 encoder within precompiled (Derived 1269.5
  us) even though its I/O is half the size. The decoder shows the opposite. The encoder
  layer profiles (jp0mxyj2g, jp1nm06kg) exist but were not analyzed here.
- The fp16 decoder has the higher peak memory (87826432 against 77512704 bytes) even
  though its I/O is half the size (Derived above: 11765450 against 23530892 bytes per
  call).
- The precompiled fp32 cells are slower than the onnx fp32 cells for both components
  (Derived 656 us and 133 us). Both formats ran the same QAIRT and ONNX Runtime with
  float16 math. The cause is not known.
- The layer times come from a different, instrumented run and do not add up to the
  timed p50. The size of the input conversion cost in the timed run is not measured.
- None of this is validated on a physical Snapdragon device. The app's own timings on
  the target laptop may differ.
