"""Build the submission files from the committed results.

- docs/description/PlacementMirror_Description.docx (python-docx), A4, at most 2 pages
- docs/deck/PlacementMirror.pptx (python-pptx), 16:9, 10 slides

Every number is read from a source file, none is typed here:
- benchmarks/raw/*.json Measurement records (AI Hub hosted Snapdragon X Elite, benchmarks/TABLE.md)
- Derived records: benchmarks/raw/budgets_derived.json, benchmarks/raw/day3_derived.json
- benchmarks/raw/day3_decoder_io_bytes.json (Derived from tensor shapes, not measured)
- eval/eye_contact/results/<name>_results.json, eval/fillers/results/filler_results.json and
  eval/pace/results/pace_results.json when they exist (eval/RESULTS.md is written from the same files)
- docs/evidence/offline_check.json, docs/evidence/arm64_build.json
- app/config.py for the frame rates, the report window and the prompt setting

A missing filler or pace results file gives "not yet evaluated" with no numbers.
Run in .venv-eval (python-docx, python-pptx). tools/check_docs.py checks the output.

Usage: python tools/build_docs.py [--eye-contact eval/eye_contact/results/<name>_results.json]
"""

from __future__ import annotations

import argparse
import json
from decimal import ROUND_HALF_UP, Decimal
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from app.config import AUDIO, REPORT, VISION  # noqa: E402
from benchmarks.schema import Derived, load_json  # noqa: E402
from benchmarks.table import load_all  # noqa: E402

DOCX_OUT = REPO / "docs" / "description" / "PlacementMirror_Description.docx"
PPTX_OUT = REPO / "docs" / "deck" / "PlacementMirror.pptx"
ASSETS = REPO / "docs" / "deck" / "assets"
EYE_DEFAULT = REPO / "eval" / "eye_contact" / "results" / "scripted_20260929_175949_results.json"
FILLERS = REPO / "eval" / "fillers" / "results" / "filler_results.json"
PACE = REPO / "eval" / "pace" / "results" / "pace_results.json"
REPO_URL = "https://github.com/Pranav-Ram-R/placement-mirror"

AUTHOR = "Pranav Ram, IIT Gandhinagar"
TITLE = "Placement Mirror"
SUBTITLE = "Real-time on-device interview delivery coach for Snapdragon PCs"
AI_HUB_LINE = "All performance numbers were measured on Snapdragon X Elite hardware through Qualcomm AI Hub."
BUILT_LINE = ("Built without a Snapdragon laptop, using Qualcomm AI Hub hosted devices for all compile, profile "
              "and benchmark jobs, with a Windows ARM64 build verified in CI.")
ANGLE = ("Every placement season, students rehearse what to say and rarely rehearse how they say it. A mock "
         "interview with a senior happens once or twice, if at all, and recording yourself means watching it back "
         "alone. I am preparing for internship interviews myself, and I built Placement Mirror so that any student "
         "with a laptop gets a coach that watches their delivery live, for as many practice rounds as they need, "
         "without their face or voice ever leaving the device.")
SAME_CODE = "measured on recorded clips on the development machine, using the same ONNX models and processing code that ship"
EYE_CODE = ("measured on a recorded clip on the development machine, using the shipped ONNX models and the app's "
            "head pose code")
NOT_YET = "not yet evaluated"

# AI Hub profile jobs (benchmarks/TABLE.md)
JOBS = {
    "enc": "jp0mxyj2g", "dec": "j57e8z9qp",  # shipping whisper_tiny: precompiled fp32 encoder, fp16 decoder
    "face_det": "jpe776l05", "face_lm": "jgzllz465", "pose_det": "jp2rrvorg", "pose_lm": "jpyoo7885",
    "enc_cpu": "j57eeo9rp", "enc_gpu": "jp4yye3lp", "enc_onnx_npu": "jpxll0x9p",  # same onnx fp32 artifact
    "enc_onnx_fp32": "jpe7n9y75", "dec_onnx_fp32": "jgzl0enz5", "dec_pre_fp32": "j5qld8y7p",
    "enc_pre_fp16": "jp1nm06kg",
}


def ms(us: float) -> str:
    """Microseconds as milliseconds, 3 decimals, rounded half up from the decimal value (as Excel does)."""
    v = (Decimal(repr(us)) / 1000).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
    return f"{v:,}"


def facts(eye_path: Path) -> dict:
    recs = load_all()

    def meas(job: str, metric: str):
        [m] = [r for r in recs if r.job_id == job and r.metric == metric]
        return m

    p50 = {k: meas(j, "inference_time_p50").value for k, j in JOBS.items()}
    p95 = {k: meas(j, "inference_time_p95").value for k, j in JOBS.items()}
    gpu_cpu_ops = meas(JOBS["enc_gpu"], "ops_on_CPU").value
    der = {}
    for f in ("budgets_derived.json", "day3_derived.json"):
        der.update({d.name: d.value for d in load_json(REPO / "benchmarks" / "raw" / f) if isinstance(d, Derived)})
    io = json.loads((REPO / "benchmarks" / "raw" / "day3_decoder_io_bytes.json").read_text(encoding="utf-8"))["cells"]
    eye = json.loads(eye_path.read_text(encoding="utf-8"))
    off = json.loads((REPO / "docs" / "evidence" / "offline_check.json").read_text(encoding="utf-8"))
    build = json.loads((REPO / "docs" / "evidence" / "arm64_build.json").read_text(encoding="utf-8"))
    [pkg] = [a for a in build["artifacts"] if a["name"] == "PlacementMirror-win-arm64"]

    fill = None
    if FILLERS.exists():
        r = json.loads(FILLERS.read_text(encoding="utf-8"))
        cond = "disfluent_prompt" if AUDIO.use_prompt else "no_prompt"
        [row] = [s for s in r["summary"] if s["model"] == r["verdict_model"] and s["condition"] == cond]
        app_rows = [s for s in r["summary"] if s["model"] == r["verdict_model"]]
        passed = any(s["recall"] is not None and s["recall"] >= 0.70 for s in app_rows)
        fill = {"clips": row["clips"], "labeled": row["total_labeled"], "recall": row["recall"],
                "precision": row["precision"], "condition": cond, "verdict": "PASS" if passed else "FAIL"}
    pace = None
    if PACE.exists():
        r = json.loads(PACE.read_text(encoding="utf-8"))
        pace = {"clips": len(r["results"]), "errors": [x["error_pct"] for x in r["results"]]}

    c = eye["classifier"]
    return {
        "p50": p50, "p95": p95, "der": der, "io": io, "eye": eye, "c": c, "off": off, "build": build,
        "pkg_bytes": pkg["size_in_bytes"], "fill": fill, "pace": pace, "gpu_cpu_ops": gpu_cpu_ops,
        "face_fps": VISION.npu_face_max_fps, "pose_fps": VISION.npu_pose_fps,
        "window_s": REPORT.window_s, "rate_khz": AUDIO.sample_rate / 1000,
    }


def text(F: dict) -> dict:
    """Every sentence with a number, built once and used by both files."""
    d, p50, p95, c, te = F["der"], F["p50"], F["p95"], F["c"], F["c"]["test_eval"]
    t = {}
    t["ratio"] = f"{d['whisper_tiny_encoder_cpu_over_npu']:.2f}x"
    t["npu_why"] = (
        f"The camera models run on every frame while speech recognition runs on each speech segment, both for "
        f"the whole answer. For the same onnx float32 whisper_tiny encoder, the p50 is {ms(p50['enc_cpu'])} ms on "
        f"the CPU, {ms(p50['enc_gpu'])} ms on the GPU and {ms(p50['enc_onnx_npu'])} ms on the NPU, so the CPU "
        f"takes {t['ratio']} as long (Derived: CPU p50 / NPU p50). Running on the device keeps video and audio "
        f"on the laptop, and the app works with the network off.")
    t["per_frame"] = ms(d["vision_npu_steady_state_per_frame"])
    t["share"] = f"{d['vision_npu_steady_state_share_of_frame_interval']:.2f} %"
    t["interval"] = f"{1000 / F['face_fps']:.1f}"
    t["per_frame_sentence"] = (
        f"Vision per frame, steady state: face landmark p50 + pose landmark p50 x {F['pose_fps']:g} / "
        f"{F['face_fps']:g} = {t['per_frame']} ms, {t['share']} of the {t['interval']} ms frame interval at "
        f"{F['face_fps']:g} fps (Derived). The detectors run only when tracking is lost and are left out.")
    t["asr"] = f"{ms(p50['enc'])} + n x {ms(p50['dec'])} ms"
    t["asr_sentence"] = (f"Speech recognition per segment: encoder p50 + n x decoder p50 = {t['asr']}, where n is "
                         f"the number of decoder calls (Derived, n left open).")
    t["dec_rel"] = f"{d['whisper_tiny_decoder_storage_effect_within_precompiled_p50_relative']:.2f} %"
    t["enc_rel"] = f"+{d['whisper_tiny_encoder_storage_effect_within_precompiled_p50_relative']:.2f} %"
    io32, io16 = F["io"]["precompiled_fp32"]["bytes_total"], F["io"]["precompiled_fp16"]["bytes_total"]
    t["bytes_exact"] = f"{io32:,} against {io16:,} bytes"
    t["bytes_mb"] = f"{io32 / 1e6:.1f} MB vs {io16 / 1e6:.1f} MB"
    t["eye_short"] = (f"balanced accuracy {te['balanced_accuracy']:.3f} on {te['frames']} held-out frames "
                      f"({te['tp'] + te['fn']} facing, {te['tn'] + te['fp']} not facing)")
    t["eye_threshold"] = f"{c['threshold_deg']:.2f}"
    t["eye_frames"] = F["eye"]["frames_labeled"]
    f = F["fill"]
    t["fill"] = (f"recall {f['recall']:.3f} and precision {f['precision']:.3f} on {f['clips']} recorded answers "
                 f"with {f['labeled']} hand labeled fillers ({f['verdict']} against a recall of 0.70)"
                 if f else NOT_YET)
    p = F["pace"]
    t["pace"] = (f"app word count error {', '.join(f'{e:+.1f} %' for e in p['errors'])} against a manual count on "
                 f"{p['clips']} recorded answers" if p else NOT_YET)
    o = F["off"]
    t["offline"] = (f"In an offline check on the development machine, the app pages made "
                    f"{o['browser']['requests_started_by_app_pages']} requests, "
                    f"{o['browser']['app_page_requests_not_loopback']} off loopback, and the app server attempted "
                    f"{o['python_server']['not_loopback']} connections off loopback (docs/evidence/offline_check.json).")
    t["pkg_mb"] = f"{F['pkg_bytes'] / 1e6:.1f} MB"
    return t


# ---------------------------------------------------------------- description (docx)

def build_docx(F: dict, t: dict) -> None:
    from docx import Document
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Cm, Pt, RGBColor

    INK, RED, GREY = RGBColor(0x1A, 0x1A, 0x1A), RGBColor(0xD6, 0x00, 0x1C), RGBColor(0x55, 0x55, 0x55)
    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    sec.left_margin = sec.right_margin = Cm(1.8)
    sec.top_margin = sec.bottom_margin = Cm(1.5)
    normal = doc.styles["Normal"]
    normal.font.name, normal.font.size, normal.font.color.rgb = "Calibri", Pt(10.5), INK
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
    pf = normal.paragraph_format
    pf.space_before, pf.space_after, pf.line_spacing = Pt(0), Pt(3), 1.0

    def para(txt="", bold_lead=None, size=None, color=None, after=None, italic=False):
        p = doc.add_paragraph()
        if bold_lead:
            r = p.add_run(bold_lead)
            r.bold = True
        r = p.add_run(txt)
        r.italic = italic
        if size:
            for run in p.runs:
                run.font.size = Pt(size)
        if color:
            for run in p.runs:
                run.font.color.rgb = color
        if after is not None:
            p.paragraph_format.space_after = Pt(after)
        return p

    def heading(txt):
        p = doc.add_paragraph()
        p.paragraph_format.space_before, p.paragraph_format.space_after = Pt(5), Pt(2)
        p.paragraph_format.keep_with_next = True
        r = p.add_run(txt)
        r.bold, r.font.size, r.font.color.rgb = True, Pt(12), RED

    def bullet(txt, lead=None):
        p = doc.add_paragraph(style="List Bullet")
        p.paragraph_format.space_after = Pt(1)
        if lead:
            p.add_run(lead).bold = True
        p.add_run(txt)

    def table(rows, widths_cm):
        tb = doc.add_table(rows=len(rows), cols=len(rows[0]))
        tb.alignment = WD_TABLE_ALIGNMENT.LEFT
        tb.style = doc.styles["Table Grid"]
        for i, row in enumerate(rows):
            for j, val in enumerate(row):
                cell = tb.cell(i, j)
                cell.width = Cm(widths_cm[j])
                cell.text = ""
                p = cell.paragraphs[0]
                p.paragraph_format.space_after = Pt(0)
                r = p.add_run(str(val))
                r.font.size = Pt(10.5)
                if i == 0:
                    r.bold = True
                    shd = OxmlElement("w:shd")
                    shd.set(qn("w:val"), "clear")
                    shd.set(qn("w:color"), "auto")
                    shd.set(qn("w:fill"), "EDEDED")
                    cell._tc.get_or_add_tcPr().append(shd)
        borders = tb._tbl.tblPr.find(qn("w:tblBorders"))
        if borders is None:
            borders = OxmlElement("w:tblBorders")
            tb._tbl.tblPr.append(borders)
        for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
            el = borders.find(qn(f"w:{side}"))
            if el is None:
                el = OxmlElement(f"w:{side}")
                borders.append(el)
            el.set(qn("w:val"), "single")
            el.set(qn("w:sz"), "4")
            el.set(qn("w:color"), "BFBFBF")
        doc.add_paragraph().paragraph_format.space_after = Pt(0)
        return tb

    p50, p95 = F["p50"], F["p95"]
    p = doc.add_paragraph()
    r = p.add_run(TITLE)
    r.bold, r.font.size, r.font.color.rgb = True, Pt(20), INK
    p.paragraph_format.space_after = Pt(0)
    para(SUBTITLE, size=12, after=0)
    para(AUTHOR, size=10.5, color=GREY, after=4)
    para(AI_HUB_LINE, bold_lead=None, after=4).runs[0].bold = True

    heading("Why I built it")
    para(ANGLE)
    heading("Problem")
    para("Delivery (eye contact, posture, pace and filler words) shapes how an interview answer lands. Feedback on "
         "it needs another person or a recording reviewed later, and cloud coaching tools send the camera and "
         "microphone stream to a server.")
    heading("What it does")
    para(f"The student picks a question (HR, software or core hardware), calibrates and answers. While they speak, "
         f"the app shows four live indicators: facing the camera, posture, words per minute and fillers. After the "
         f"answer, a report charts each metric per {F['window_s']:g} s window, shows the transcript with fillers "
         f"and long pauses marked, and lists three things to work on. A history page tracks the trend across "
         f"answers. The guidelines are the author's coaching defaults, configurable and not validated against "
         f"outcomes.")
    heading("Why it runs on the NPU")
    para(t["npu_why"])
    heading("Architecture")
    table([["Stage", "Model", "Compute unit"],
           ["Face detection (on track loss)", "mediapipe_face detector", "NPU"],
           ["Face landmarks (every frame)", "mediapipe_face landmark", "NPU"],
           [f"Posture ({F['pose_fps']:g} fps)", "mediapipe_pose detector and landmark", "NPU"],
           ["Head pose and facing", "Kabsch alignment (numpy)", "CPU"],
           ["Voice activity", "Silero VAD", "CPU"],
           ["Speech recognition", "Whisper tiny encoder and decoder", "NPU"],
           ["Pace, fillers, pauses", "rules", "CPU"]], [5.6, 7.0, 3.0])
    para("NPU placement follows the AI Hub results and is not yet validated on device.", italic=True)
    heading("Performance on Snapdragon X Elite (AI Hub hosted, shipping configuration)")
    table([["Model", "p50 ms", "p95 ms", "AI Hub job"],
           ["Whisper tiny encoder", ms(p50["enc"]), ms(p95["enc"]), JOBS["enc"]],
           ["Whisper tiny decoder (one call)", ms(p50["dec"]), ms(p95["dec"]), JOBS["dec"]],
           ["Face detector", ms(p50["face_det"]), ms(p95["face_det"]), JOBS["face_det"]],
           ["Face landmark", ms(p50["face_lm"]), ms(p95["face_lm"]), JOBS["face_lm"]],
           ["Pose landmark", ms(p50["pose_lm"]), ms(p95["pose_lm"]), JOBS["pose_lm"]]], [6.4, 2.6, 2.6, 3.0])
    bullet(t["per_frame_sentence"])
    bullet(t["asr_sentence"])
    heading("Optimization")
    para(f"Whisper tiny was profiled in a 2x2 grid: onnx or a precompiled QNN context binary, with float32 or "
         f"float16 stored I/O. onnx with float16 was rejected at compile submission. Within the precompiled format, "
         f"float16 storage changed the decoder p50 by {t['dec_rel']} and the encoder p50 by {t['enc_rel']} "
         f"(Derived). The app ships the precompiled float32 encoder and the precompiled float16 decoder, with the "
         f"onnx float32 models as the fallback. A float32 decoder call passes {t['bytes_exact']} at float16 "
         f"(Derived from tensor shapes, not measured), and the layer profile puts most of the extra float32 time "
         f"at the graph input and output and at the first layer reading each cross attention cache. Not "
         f"explained: the float16 encoder slowdown and the float16 decoder's higher peak memory.")
    heading("Accuracy evaluation")
    bullet(f"{EYE_CODE[0].upper() + EYE_CODE[1:]}, one scripted recording of one speaker ({t['eye_frames']} "
           f"labeled frames). The threshold, {t['eye_threshold']} degrees, was chosen on the first half of each "
           f"segment. On the second halves: {t['eye_short']}. Looking at the camera and at the screen overlap in "
           f"head pose, so the app treats both as facing.", lead="Eye contact: ")
    fp = SAME_CODE[0].upper() + SAME_CODE[1:]
    bullet(f"{t['fill']}." if F["fill"] else f"{NOT_YET[0].upper() + NOT_YET[1:]}.",
           lead="Filler detection: " + (f"{fp}, " if F["fill"] else ""))
    bullet(f"{t['pace']}." if F["pace"] else f"{NOT_YET[0].upper() + NOT_YET[1:]}.",
           lead="Speaking pace: " + (f"{fp}, " if F["pace"] else ""))
    heading("Deployment")
    para(f"A Windows ARM64 build is built and smoke tested in CI on a windows-11-arm runner "
         f"({t['pkg_mb']} package, docs/evidence/arm64_build.json). Each model loads as a precompiled QNN context "
         f"binary on the NPU, then onnx on the QNN execution provider with a context cache, then onnx on the CPU "
         f"with a visible notice. The app loads nothing from the network. {t['offline']}")
    para(BUILT_LINE)
    heading("Limitations and next milestone")
    para("Next milestone: validation on a physical Snapdragon laptop. Not yet validated on device: loading the AI "
         "Hub context binaries with the bundled QNN runtime, the app's own latency with Whisper and the vision "
         "models sharing the NPU, NPU float16 transcripts, and the fallback chain on real hardware.",
         bold_lead=None)
    para("Other limits: one speaker in one room and small samples. Head pose cannot separate camera from screen. "
         "The coaching guidelines are not validated against outcomes. Future work: content feedback from an "
         "on-device language model, and more speakers.")
    heading("Links")
    para(f"GitHub: {REPO_URL}")
    DOCX_OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(DOCX_OUT)


# ---------------------------------------------------------------- deck (pptx)

def build_pptx(F: dict, t: dict) -> None:
    from pptx import Presentation
    from pptx.chart.data import CategoryChartData
    from pptx.dml.color import RGBColor
    from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION
    from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
    from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
    from pptx.oxml.ns import qn
    from pptx.oxml.xmlchemy import OxmlElement
    from pptx.util import Emu, Inches, Pt

    RED, INK, GREY, LIGHT, RULE = (RGBColor(0xD6, 0x00, 0x1C), RGBColor(0x1A, 0x1A, 0x1A),
                                   RGBColor(0x6B, 0x6B, 0x6B), RGBColor(0xF2, 0xF2, 0xF2),
                                   RGBColor(0xBF, 0xBF, 0xBF))
    FONT = "Calibri"
    W, H, M = 13.333, 7.5, 0.5
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(W), Inches(H)
    blank = prs.slide_layouts[6]
    p50, p95, c = F["p50"], F["p95"], F["c"]

    def box(slide, x, y, w, h, paras, size=20, color=INK, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
        tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.vertical_anchor = anchor
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        for i, item in enumerate(paras if isinstance(paras, list) else [paras]):
            txt, kw = (item, {}) if isinstance(item, str) else item
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.alignment = align
            p.space_after = Pt(kw.get("after", 8))
            r = p.add_run()
            r.text = txt
            f = r.font
            f.name, f.size, f.bold = FONT, Pt(kw.get("size", size)), kw.get("bold", bold)
            f.color.rgb = kw.get("color", color)
        return tb

    def slide(title, notes, footer=None):
        s = prs.slides.add_slide(blank)
        box(s, M, 0.4, W - 2 * M, 1.0, title, size=32, bold=True, anchor=MSO_ANCHOR.BOTTOM)
        bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(M), Inches(1.5), Inches(0.9), Inches(0.06))
        bar.fill.solid()
        bar.fill.fore_color.rgb = RED
        bar.line.fill.background()
        if footer:
            box(s, M, H - 0.45, W - 2 * M, 0.3, footer, size=10, color=GREY)
        s.notes_slide.notes_text_frame.text = notes
        return s

    def big(s, x, y, w, number, label):
        box(s, x, y, w, 1.0, number, size=60, bold=True, color=RED)
        box(s, x, y + 1.05, w, 0.8, label, size=20, color=INK)

    def picture(s, path, x, y, w):
        pic = s.shapes.add_picture(str(path), Inches(x), Inches(y), width=Inches(w))
        pic.line.color.rgb = RULE
        pic.line.width = Pt(0.75)
        return pic

    def table(s, rows, x, y, widths, row_h=0.46, size=18, bold_cells=()):
        shape = s.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(sum(widths)),
                                   Inches(row_h * len(rows)))
        tbl = shape.table
        tbl._tbl.tblPr.find(qn("a:tableStyleId")).text = "{2D5ABB26-0587-4C30-8999-92F81FD0307C}"  # no style, no grid
        for j, wd in enumerate(widths):
            tbl.columns[j].width = Inches(wd)
        for i, row in enumerate(rows):
            tbl.rows[i].height = Inches(row_h)
            for j, val in enumerate(row):
                cell = tbl.cell(i, j)
                tcPr = cell._tc.get_or_add_tcPr()
                ln = OxmlElement("a:lnB")
                ln.set("w", "9525")
                sf = OxmlElement("a:solidFill")
                clr = OxmlElement("a:srgbClr")
                clr.set("val", "BFBFBF")
                sf.append(clr)
                ln.append(sf)
                tcPr.insert(0, ln)
                cell.fill.solid()
                cell.fill.fore_color.rgb = LIGHT if i == 0 else RGBColor(0xFF, 0xFF, 0xFF)
                cell.margin_left = cell.margin_right = Inches(0.08)
                cell.vertical_anchor = MSO_ANCHOR.MIDDLE
                tf = cell.text_frame
                tf.word_wrap = True
                tf.paragraphs[0].text = ""
                r = tf.paragraphs[0].add_run()
                r.text = str(val)
                r.font.name, r.font.size = FONT, Pt(size)
                r.font.bold = i == 0 or (i, j) in bold_cells
                r.font.color.rgb = RED if (i, j) in bold_cells else INK
        return tbl

    def node(s, x, y, w, h, title, unit):
        shp = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
        shp.fill.solid()
        shp.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        npu = unit.startswith("NPU")
        shp.line.color.rgb = RED if npu else GREY
        shp.line.width = Pt(2.25 if npu else 1.25)
        tf = shp.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = Inches(0.06)
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        for i, (txt, sz, col, b) in enumerate(((title, 16, INK, True), (unit, 14, RED if npu else GREY, npu))):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.alignment = PP_ALIGN.CENTER
            r = p.add_run()
            r.text = txt
            r.font.name, r.font.size, r.font.bold, r.font.color.rgb = FONT, Pt(sz), b, col
        return shp

    def arrow(s, x1, y1, x2, y2):
        ln = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
        ln.line.color.rgb = GREY
        ln.line.width = Pt(1.5)
        ln.line._get_or_add_ln().append(_tail())
        return ln

    def _tail():
        e = OxmlElement("a:tailEnd")
        e.set("type", "triangle")
        return e

    SHOT_W = 9.9  # 9.9 x 5.57 in, 55 % of the slide
    shot_caption = "Screenshot: development machine in CPU fallback mode, replay of a recorded answer."

    # 1 Title
    s = prs.slides.add_slide(blank)
    box(s, M, 1.7, W - 2 * M, 1.1, TITLE, size=60, bold=True)
    bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(M), Inches(2.95), Inches(1.4), Inches(0.08))
    bar.fill.solid()
    bar.fill.fore_color.rgb = RED
    bar.line.fill.background()
    box(s, M, 3.25, W - 2 * M, 0.6, SUBTITLE, size=28)
    box(s, M, 3.95, W - 2 * M, 0.5, AUTHOR, size=20, color=GREY)
    box(s, M, 5.3, W - 2 * M, 1.3, [(AI_HUB_LINE, {"bold": True}), BUILT_LINE], size=20)
    s.notes_slide.notes_text_frame.text = (f"{TITLE}. {SUBTITLE}. {AUTHOR}. {AI_HUB_LINE} {BUILT_LINE} "
                                           f"Code: {REPO_URL}")

    # 2 Problem
    s = slide("Interviewers hear how you answer, not just what you answer. Students almost never get to "
              "practice that part.", ANGLE)
    box(s, M, 2.0, 8.6, 4.2, ["Students rehearse what to say, rarely how they say it.",
                              "Mock interviews happen once or twice.",
                              "Watching your own recording alone is hard.",
                              "Cloud coaching sends your face and voice off the device."], size=28)

    # 3 What it does
    s = slide("What it does", "Pick a question from HR, software or core hardware, sit as in an interview and "
              "calibrate while looking at the camera, then answer. The indicators update live. Everything runs "
              "on the laptop. " + shot_caption, shot_caption)
    picture(s, ASSETS / "live_view.png", W - M - SHOT_W, 1.4, SHOT_W)
    box(s, M, 1.9, W - 2 * M - SHOT_W - 0.25, 4.6, ["Four live indicators while you answer:", "facing the camera, posture, words per "
                              "minute, fillers.", "Everything runs on the laptop."], size=20)

    # 4 The report
    s = slide("The report", f"After the answer the report charts facing the camera, face not visible, posture "
              f"and words per minute per {F['window_s']:g} s window, shows the transcript with fillers and long "
              f"pauses marked, and lists three things to work on. Guidelines: author's coaching defaults, "
              f"configurable, not validated against outcomes. The history page tracks the trend across answers. "
              + shot_caption, shot_caption + f" Window length: app/config.py ReportConfig.window_s = "
              f"{F['window_s']:g} s.")
    picture(s, ASSETS / "report.png", W - M - SHOT_W, 1.4, SHOT_W)
    box(s, M, 1.9, W - 2 * M - SHOT_W - 0.25, 4.6, [f"Each metric per {F['window_s']:g} s window.", "Transcript with fillers and "
                              "pauses marked.", "Three things to work on.", "History tracks the trend."], size=20)

    # 5 Why the NPU
    s = slide("Why the NPU", t["npu_why"] + f" The GPU run placed {F['gpu_cpu_ops']:g} ops on the CPU. The "
              f"shipping encoder ({ms(p50['enc'])} ms) is on the performance slide.",
              f"Source: AI Hub hosted Snapdragon X Elite, onnx float32 whisper_tiny encoder, jobs {JOBS['enc_cpu']} "
              f"(CPU), {JOBS['enc_gpu']} (GPU), {JOBS['enc_onnx_npu']} (NPU). Ratio Derived: CPU p50 / NPU p50.")
    big(s, M, 1.95, 4.4, t["ratio"], "CPU p50 / NPU p50, same whisper_tiny encoder")
    box(s, M, 3.95, 4.4, 2.5, ["Vision runs every frame while speech recognition runs per segment.",
                               "Video and audio stay on the laptop, and it works offline."], size=20)
    cd = CategoryChartData()
    cd.categories = ["CPU", "GPU", "NPU"]
    cd.add_series("p50 ms", [p50["enc_cpu"] / 1000, p50["enc_gpu"] / 1000, p50["enc_onnx_npu"] / 1000])
    gf = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(5.3), Inches(1.85), Inches(7.53), Inches(4.7), cd)
    ch = gf.chart
    ch.has_legend = False
    ch.has_title = False
    ch.font.name, ch.font.size, ch.font.color.rgb = FONT, Pt(18), INK
    plot = ch.plots[0]
    plot.gap_width = 60
    plot.has_data_labels = True
    dl = plot.data_labels
    dl.number_format, dl.number_format_is_linked = '#,##0.000" ms"', False
    dl.position = XL_LABEL_POSITION.OUTSIDE_END
    dl.font.size = Pt(18)
    for i, pt in enumerate(plot.series[0].points):
        pt.format.fill.solid()
        pt.format.fill.fore_color.rgb = RED if i == 2 else RGBColor(0xA6, 0xA6, 0xA6)
    ca = ch.category_axis
    ca.reverse_order = True
    ca.format.line.color.rgb = RULE
    ca.has_major_gridlines = False
    va = ch.value_axis
    va.has_major_gridlines = False
    va.visible = False
    scaling = va._element.find(qn("c:scaling"))
    lb = OxmlElement("c:logBase")
    lb.set("val", "10")
    scaling.insert(0, lb)
    va.minimum_scale, va.maximum_scale = 10, 10000
    box(s, 5.3, 6.55, 7.53, 0.35, "Encoder p50 in ms, log scale", size=14, color=GREY, align=PP_ALIGN.RIGHT)

    # 6 Architecture
    s = slide("Architecture", "Planned placement from the AI Hub results, not yet validated on device. The face "
              "detector runs only when the landmark track is lost, the face landmarks run on every frame, pose runs "
              f"at {F['pose_fps']:g} fps. Voice activity detection owns the timing and Whisper tiny runs once per "
              "speech segment. Head pose, posture and the speech metrics are numpy rules on the CPU.",
              f"Source: app/config.py (camera {F['face_fps']:g} fps, pose {F['pose_fps']:g} fps, audio "
              f"{F['rate_khz']:g} kHz), docs/architecture.md.")
    y1, y2, y3, bh, bw = 1.95, 3.45, 4.95, 1.05, 2.2
    xs = [M, 3.2, 5.9, 8.6]
    node(s, xs[0], y1, bw, bh, "Camera", f"{F['face_fps']:g} fps")
    node(s, xs[1], y1, bw, bh, "Face detector", "NPU, on track loss")
    node(s, xs[2], y1, bw, bh, "Face landmarks", "NPU, every frame")
    node(s, xs[3], y1, bw, bh, "Head pose, facing", "CPU")
    node(s, xs[0], y2, bw, bh, "Camera", f"{F['face_fps']:g} fps")
    node(s, xs[1], y2, bw + 2.7, bh, "Pose detector and landmarks", f"NPU, {F['pose_fps']:g} fps")
    node(s, xs[3], y2, bw, bh, "Posture", "CPU")
    node(s, xs[0], y3, bw, bh, "Microphone", f"{F['rate_khz']:g} kHz")
    node(s, xs[1], y3, bw, bh, "Silero VAD", "CPU")
    node(s, xs[2], y3, bw, bh, "Whisper tiny", "NPU, per segment")
    node(s, xs[3], y3, bw, bh, "Pace, fillers, pauses", "CPU")
    out_x = 11.3
    node(s, out_x, y1, W - M - out_x, y3 + bh - y1, "Live indicators, report, history", "Browser on localhost")
    for y in (y1, y3):
        for i in range(3):
            arrow(s, xs[i] + bw, y + bh / 2, xs[i + 1], y + bh / 2)
    arrow(s, xs[0] + bw, y2 + bh / 2, xs[1], y2 + bh / 2)
    arrow(s, xs[1] + bw + 2.7, y2 + bh / 2, xs[3], y2 + bh / 2)
    for y in (y1, y2, y3):
        arrow(s, xs[3] + bw, y + bh / 2, out_x, y + bh / 2)
    box(s, M, 6.25, W - 2 * M, 0.4, "Planned placement. Not yet validated on device.", size=20, color=GREY)

    # 7 Performance
    s = slide("Performance on Snapdragon X Elite",
              f"Shipping configuration, p50 and p95 over 100 iterations, AI Hub hosted Snapdragon X Elite. "
              f"{t['per_frame_sentence']} {t['asr_sentence']} Encoder {JOBS['enc']}, decoder {JOBS['dec']}, face "
              f"detector {JOBS['face_det']}, face landmark {JOBS['face_lm']}, pose detector {JOBS['pose_det']}, "
              f"pose landmark {JOBS['pose_lm']}.",
              "Source: AI Hub hosted Snapdragon X Elite, benchmarks/TABLE.md. Budgets Derived, "
              "benchmarks/raw/budgets_derived.json.")
    table(s, [["Model (shipping)", "p50 ms", "p95 ms"],
              ["Whisper tiny encoder", ms(p50["enc"]), ms(p95["enc"])],
              ["Whisper tiny decoder, one call", ms(p50["dec"]), ms(p95["dec"])],
              ["Face detector", ms(p50["face_det"]), ms(p95["face_det"])],
              ["Face landmark", ms(p50["face_lm"]), ms(p95["face_lm"])],
              ["Pose detector", ms(p50["pose_det"]), ms(p95["pose_det"])],
              ["Pose landmark", ms(p50["pose_lm"]), ms(p95["pose_lm"])]], M, 1.95, [4.1, 1.5, 1.5], row_h=0.58,
          size=20)
    big(s, 8.3, 1.95, 4.5, t["share"], f"of the {t['interval']} ms frame for vision on the NPU "
        f"({t['per_frame']} ms)")
    box(s, 8.3, 4.35, 4.5, 1.6, [("Speech per segment", {"bold": True, "after": 2}), t["asr"],
                                 ("n = decoder calls", {"color": GREY})], size=20)

    # 8 Optimization
    d = F["der"]
    s = slide("Optimization: runtime format and storage precision",
              f"2x2 grid of whisper_tiny encoder and decoder p50 on the NPU, one profile job per cell. onnx "
              f"float16 was rejected at compile submission: the --quantize_full_type option is not supported for "
              f"target_runtime ONNX. Storage effect within precompiled: decoder "
              f"{d['whisper_tiny_decoder_storage_effect_within_precompiled_p50_difference']:g} us ({t['dec_rel']}), "
              f"encoder +{d['whisper_tiny_encoder_storage_effect_within_precompiled_p50_difference']:g} us "
              f"({t['enc_rel']}), Derived. Bytes per decoder call: {t['bytes_exact']} (Derived from tensor shapes, "
              f"not measured). In the layer profile, from a separate 2-iteration instrumented run with an "
              f"undocumented unit, the extra float32 time sits at the graph input and output layers and at the "
              f"first layer that reads each cross attention cache. Not explained: the float16 encoder slowdown, "
              f"the float16 decoder's higher peak memory, and why precompiled float32 is slower than onnx float32.",
              "Source: AI Hub hosted Snapdragon X Elite, benchmarks/optimization.md, "
              "benchmarks/raw/day3_derived.json and day3_decoder_io_bytes.json (Derived).")
    table(s, [["p50 ms", "onnx", "precompiled"],
              ["float32 encoder", ms(p50["enc_onnx_fp32"]), ms(p50["enc"])],
              ["float16 encoder", "rejected", ms(p50["enc_pre_fp16"])],
              ["float32 decoder", ms(p50["dec_onnx_fp32"]), ms(p50["dec_pre_fp32"])],
              ["float16 decoder", "rejected", ms(p50["dec"])]], M, 1.95, [2.9, 1.8, 2.1], row_h=0.62, size=20,
          bold_cells=((1, 2), (4, 2)))
    box(s, M, 5.2, 6.8, 0.9, "Red: shipped. onnx float32 is the fallback.", size=20, color=GREY)
    big(s, 7.9, 1.95, 4.9, t["dec_rel"], "decoder p50, float16 vs float32 storage")
    box(s, 7.9, 3.95, 4.9, 2.6, [f"Encoder: {t['enc_rel']}.",
                                 f"{t['bytes_mb']} per decoder call (Derived from tensor shapes).",
                                 "Extra float32 time: cross attention cache inputs.",
                                 "Unexplained: float16 encoder slowdown, decoder peak memory."], size=20)

    # 9 Accuracy and deployment
    te = c["test_eval"]
    s = slide("Accuracy and deployment",
              f"Eye contact was {EYE_CODE}: one scripted recording of one speaker, {t['eye_frames']} labeled frames, "
              f"threshold {t['eye_threshold']} degrees chosen on the first half of every segment, {t['eye_short']} "
              f"on the second halves. Filler detection and speaking pace are {SAME_CODE}: filler detection "
              f"{t['fill']}, speaking pace {t['pace']}. Never run on a Snapdragon device yet. {BUILT_LINE} ARM64 "
              f"package {F['pkg_bytes']:,} bytes (CI run {F['build']['run_id']}). {t['offline']}",
              f"Sources: eval/RESULTS.md, docs/evidence/arm64_build.json (CI run {F['build']['run_id']}), "
              f"docs/evidence/offline_check.json.")
    box(s, M, 1.95, 5.8, 4.7, [("Accuracy", {"bold": True, "color": RED}),
                               f"Eye contact: balanced accuracy {te['balanced_accuracy']:.3f} on {te['frames']} "
                               f"held-out frames.",
                               "Fillers: " + (f"recall {F['fill']['recall']:.3f}, {F['fill']['clips']} answers."
                                              if F["fill"] else NOT_YET + "."),
                               "Pace: " + (f"{F['pace']['clips']} answers, see notes." if F["pace"]
                                           else NOT_YET + "."),
                               ("Recorded clips, development machine.", {"color": GREY})], size=24)
    o = F["off"]
    box(s, 7.0, 1.95, W - M - 7.0, 4.7, [("Deployment", {"bold": True, "color": RED}),
                                         f"ARM64 build verified in CI, {t['pkg_mb']}.",
                                         "Fallback: precompiled NPU, onnx NPU, onnx CPU.",
                                         f"Offline: {o['browser']['app_page_requests_not_loopback']} of "
                                         f"{o['browser']['requests_started_by_app_pages']} app requests off "
                                         f"loopback."], size=24)

    # 10 Limitations and next milestone
    s = slide("Limitations and next milestone",
              "Not yet validated on device: loading the AI Hub context binaries with the bundled QNN runtime, the "
              "app's own latency with Whisper and the vision models sharing the NPU, NPU float16 transcripts, and "
              "the fallback chain on real hardware. Other limits: one speaker in one room, small samples, head pose "
              "cannot separate camera from screen, coaching guidelines not validated against outcomes. Future "
              "work: content feedback from an on-device language model, more speakers.")
    box(s, M, 1.95, 6.0, 4.7, [("Next milestone: validation on a physical Snapdragon laptop",
                                {"bold": True, "color": RED}),
                               "Context binary loading", "NPU shared by Whisper and vision",
                               "NPU float16 transcripts", "Fallback chain on real hardware"], size=24)
    box(s, 7.0, 1.95, W - M - 7.0, 4.7, [("Limits", {"bold": True, "color": RED}),
                                         "One speaker, one room, small samples.",
                                         "Head pose: camera and screen overlap.",
                                         "Guidelines not validated.",
                                         "Future: on-device content feedback."], size=24)
    PPTX_OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(PPTX_OUT)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eye-contact", type=Path, default=EYE_DEFAULT)
    args = ap.parse_args()
    F = facts(args.eye_contact)
    t = text(F)
    build_docx(F, t)
    build_pptx(F, t)
    print(f"Wrote {DOCX_OUT.relative_to(REPO)} and {PPTX_OUT.relative_to(REPO)}")
    print(f"Fillers: {'results' if F['fill'] else NOT_YET}. Pace: {'results' if F['pace'] else NOT_YET}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
