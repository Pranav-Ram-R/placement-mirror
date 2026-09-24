"""Inspect the AI Hub Whisper target models downloaded under models/.

Prints every input and output (name, shape, dtype) of each encoder and decoder, then
answers two questions from the decoder signatures only:
1. Can the decoder accept an arbitrary prefix of prompt tokens?
2. Are cross-attention weights exposed as outputs?

Precompiled QNN models (precompiled_qnn_onnx) are ONNX wrappers around an EPContext node
that holds the QNN context binary. Their graph inputs and outputs are readable with the
onnx package on x86, but the model can only execute through the QNN EP on Snapdragon.

Writes eval/fillers/results/aihub_whisper_signatures.json.

Usage: python eval/fillers/inspect_aihub_whisper.py [--models-dir models]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import onnx

REPO = Path(__file__).resolve().parents[2]


def tensor_info(value_info) -> dict:
    t = value_info.type.tensor_type
    dims = [d.dim_value if d.HasField("dim_value") else (d.dim_param or "?") for d in t.shape.dim]
    return {"name": value_info.name, "shape": dims, "dtype": onnx.TensorProto.DataType.Name(t.elem_type).lower()}


def inspect(path: Path) -> dict:
    model = onnx.load(str(path), load_external_data=False)
    ops = sorted({n.op_type for n in model.graph.node})
    initializer_names = {i.name for i in model.graph.initializer}
    return {
        "path": path.relative_to(REPO).as_posix(),
        "component": "decoder" if "decoder" in path.as_posix().lower() else "encoder",
        "precompiled_qnn_context": "EPContext" in ops,
        "node_count": len(model.graph.node),
        "op_types_sample": ops[:12],
        "inputs": [tensor_info(v) for v in model.graph.input if v.name not in initializer_names],
        "outputs": [tensor_info(v) for v in model.graph.output],
    }


def answer(decoder: dict) -> dict:
    ins = {t["name"]: t for t in decoder["inputs"]}
    outs = [t["name"] for t in decoder["outputs"]]
    ids = ins.get("input_ids")
    mask = ins.get("attention_mask")
    self_cache = sorted(n for n in ins if "cache_self" in n)
    context = mask["shape"][-1] if mask else None
    one_token = bool(ids) and ids["shape"][-1] == 1
    if one_token and self_cache:
        prefix = (f"Not in one call. input_ids has shape {ids['shape']}, so each call takes one token. "
                  f"A prompt prefix can only be fed token by token through the self-attention KV cache inputs "
                  f"({len(self_cache)} tensors such as {self_cache[0]} {ins[self_cache[0]]['shape']}), one decoder "
                  f"call per prompt token. attention_mask has {context} positions, so prompt tokens, start tokens "
                  f"and generated tokens together must fit in {context}. Inferred from the signature, not run.")
    else:
        prefix = f"input_ids shape {ids['shape'] if ids else 'missing'}. See the printed signature."
    attn_like = [n for n in outs if ("attn" in n.lower() or "attention" in n.lower() or "cross" in n.lower())
                 and "cache" not in n.lower()]
    cross = ("Yes: " + ", ".join(attn_like)) if attn_like else (
        "No. Outputs are " + ", ".join(outs[:3]) + (" ..." if len(outs) > 3 else "") +
        ", which are logits and self-attention KV cache only. The k_cache_cross and v_cache_cross tensors are "
        "decoder inputs computed by the encoder (cross-attention keys and values), not attention weights.")
    return {"arbitrary_prompt_prefix": prefix, "cross_attention_weights_exposed": cross}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default=str(REPO / "models"))
    ap.add_argument("--out", default=str(Path(__file__).parent / "results" / "aihub_whisper_signatures.json"))
    args = ap.parse_args()

    paths = sorted(p for p in Path(args.models_dir).rglob("*.onnx") if "whisper" in p.as_posix().lower())
    if not paths:
        raise SystemExit(f"No Whisper .onnx files under {args.models_dir}")
    report = {"models": [], "answers": {}}
    for path in paths:
        info = inspect(path)
        report["models"].append(info)
        kind = "precompiled QNN context (EPContext), signature readable, executes only via QNN EP" \
            if info["precompiled_qnn_context"] else "plain ONNX graph"
        print(f"\n=== {info['path']}\n    {info['component']}, {kind}, {info['node_count']} nodes")
        for side in ("inputs", "outputs"):
            print(f"    {side}:")
            for t in info[side]:
                print(f"      {t['name']:22} {str(t['shape']):22} {t['dtype']}")
        if info["component"] == "decoder":
            report["answers"][info["path"]] = answer(info)

    print("\n=== Answers from the decoder signatures")
    for path, a in report["answers"].items():
        print(f"\n{path}")
        print(f"  Arbitrary prompt prefix: {a['arbitrary_prompt_prefix']}")
        print(f"  Cross-attention weights exposed: {a['cross_attention_weights_exposed']}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
