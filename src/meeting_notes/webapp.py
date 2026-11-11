"""Local web demo (Gradio): pick or upload a recording, get a speaker-labelled transcript, notes and exports."""

from __future__ import annotations

import html
import json
import tempfile
from pathlib import Path

from . import __version__
from .config import Settings
from .export import to_json, to_markdown, to_srt
from .models import Notes, clock
from .pipeline import ResultCache, run

AUDIO_TYPES = (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".mp4", ".webm")
# ink-friendly speaker colours (rust, teal, olive, plum, slate, ochre)
SPEAKER_COLOURS = ["#8a3a24", "#2f6b6b", "#6b6b2f", "#6b3a6b", "#3a4f6b", "#9a6a1c"]


def samples_dir() -> Path:
    for d in (Path.cwd() / "samples", Path(__file__).resolve().parents[2] / "samples"):
        if d.is_dir():
            return d
    return Path.cwd() / "samples"


def list_samples() -> list[str]:
    return sorted(p.name for p in samples_dir().iterdir() if p.suffix.lower() in AUDIO_TYPES)


def sample_title(path: str | Path) -> str:
    truth = Path(path).with_suffix(".truth.json")
    if truth.exists():
        t = json.loads(truth.read_text())
        return f"{t['title']} · {t['language']} · {len(t['speakers'])} speakers · {clock(t['duration'])}"
    return Path(path).name


def colour(speaker: str) -> str:
    try:
        return SPEAKER_COLOURS[(int(speaker[1:]) - 1) % len(SPEAKER_COLOURS)]
    except (ValueError, IndexError):
        return SPEAKER_COLOURS[0]


def timeline_html(notes: Notes) -> str:
    t = notes.transcript
    total = max(t.duration, 0.1)
    lanes = []
    for spk in t.speakers:
        blocks = "".join(
            f"<span style='left:{100 * s.start / total:.2f}%;width:{max(0.4, 100 * (s.end - s.start) / total):.2f}%;"
            f"background:{colour(spk)}' title='{clock(s.start)}–{clock(s.end)}'></span>"
            for s in t.segments if s.speaker == spk)
        lanes.append(f"<div class='ms-lane'><b style='color:{colour(spk)}'>{spk}</b><div class='ms-track'>{blocks}"
                     "</div></div>")
    marks = "".join(f"<i style='left:{100 * a.at / total:.2f}%' title='{html.escape(a.task)}'>A{i}</i>"
                    for i, a in enumerate(notes.action_items, start=1) if a.at is not None)
    lanes.append(f"<div class='ms-lane'><b>items</b><div class='ms-track ms-marks'>{marks}</div></div>")
    ticks = "".join(f"<em style='left:{100 * x / total:.2f}%'>{clock(x)}</em>"
                    for x in range(0, int(total) + 1, 10 if total < 150 else 60))
    lanes.append(f"<div class='ms-lane'><b></b><div class='ms-track ms-ticks'>{ticks}</div></div>")
    return "<div class='ms-timeline'>" + "".join(lanes) + "</div>"


def transcript_html(notes: Notes) -> str:
    rows = []
    linked: dict[int, list[int]] = {}
    segs = notes.transcript.segments
    for i, a in enumerate(notes.action_items, start=1):
        if a.at is None or not segs:
            continue
        j = min(range(len(segs)), key=lambda k: abs(segs[k].start - a.at))
        linked.setdefault(j, []).append(i)
    for j, s in enumerate(segs):
        badges = "".join(f"<span class='ms-badge'>A{n}</span>" for n in linked.get(j, []))
        cls = "ms-line ms-linked" if j in linked else "ms-line"
        rows.append(
            f"<div class='{cls}'><span class='ms-time'>{clock(s.start)}</span>"
            f"<span class='ms-spk' style='background:{colour(s.speaker)}'>{s.speaker}</span>"
            f"<span class='ms-text'>{html.escape(s.text)}</span>{badges}</div>")
    return timeline_html(notes) + "<div class='ms-transcript'>" + "".join(rows) + "</div>"


def notes_markdown(notes: Notes) -> str:
    out = ["**Summary**", ""] + [f"- {p}" for p in notes.summary or ["(none)"]]
    out += ["", "**Decisions**", ""] + [f"- {d}" for d in notes.decisions or ["(none recorded)"]]
    return "\n".join(out)


def item_rows(notes: Notes) -> list[list]:
    return [[f"A{i}", a.task, a.owner or "—", a.due or "—", clock(a.at) if a.at is not None else ""]
            for i, a in enumerate(notes.action_items, start=1)]


def export_files(notes: Notes) -> list[str]:
    out = Path(tempfile.mkdtemp(prefix="meeting-notes-"))
    stem = Path(notes.transcript.file).stem
    files = {f"{stem}.md": to_markdown(notes), f"{stem}.json": to_json(notes), f"{stem}.srt": to_srt(notes.transcript)}
    for name, text in files.items():
        (out / name).write_text(text, encoding="utf-8")
    return [str(out / n) for n in files]


CSS = """
:root, .gradio-container { --ms-ink:#1c1a17; --ms-rust:#8a3a24; --ms-paper:#f3efe6; --ms-cream:#faf7f0; }
.gradio-container { background: var(--ms-cream) !important; font-family: Georgia, 'Times New Roman', serif !important; max-width: 1240px !important; }
.ms-header { border-bottom: 1px solid var(--ms-ink); padding: 10px 4px 12px; margin-bottom: 4px; }
.ms-brand { letter-spacing: .22em; font-size: 12px; color: var(--ms-rust); font-weight: 700; }
.ms-header h1 { font-family: Georgia, serif; font-weight: 400; font-size: 30px; margin: 2px 0 2px; color: var(--ms-ink); }
.ms-header p { margin: 0; color: #4a463f; font-style: italic; }
.ms-meta { margin-top: 8px; font-family: ui-monospace, Menlo, monospace; font-size: 12px; color: #6b665e; }
.ms-muted { color: #6b665e; font-size: 12px; }
.ms-section-title { font-size: 12px; letter-spacing: .14em; text-transform: uppercase; color: var(--ms-rust); margin: 2px 0; }
.ms-left { max-height: calc(100vh - 150px); min-height: 560px; overflow-y: auto !important; padding-right: 8px; scroll-behavior: smooth; flex-wrap: nowrap !important; }
.ms-left > * { flex-shrink: 0 !important; }
.ms-table * { font-family: ui-sans-serif, system-ui, -apple-system, 'Segoe UI', 'PingFang SC', 'Noto Sans CJK SC', sans-serif !important; font-size: 12.5px !important; }
.ms-status, .ms-notes { font-size: 14px; }
.ms-secondary, button.ms-secondary { background: #fff !important; color: var(--ms-rust) !important; border: 1px solid var(--ms-rust) !important; }
button.primary, .ms-primary button { background: var(--ms-rust) !important; border-color: var(--ms-rust) !important; color: #fff !important; }
.ms-panel { background: #fff; border: 1px solid #ddd5c6; border-radius: 10px; padding: 14px 16px; height: 720px; overflow-y: auto; }
.ms-empty { color: #6b665e; font-style: italic; padding: 40px 10px; text-align: center; }
.ms-timeline { margin: 2px 0 14px; padding-bottom: 6px; border-bottom: 1px solid #e8e2d6; }
.ms-lane { display: flex; align-items: center; gap: 8px; margin: 3px 0; }
.ms-lane > b { width: 40px; font: 600 11px ui-monospace, Menlo, monospace; color: #6b665e; text-align: right; }
.ms-track { position: relative; flex: 1; height: 14px; background: #f3efe6; border-radius: 3px; }
.ms-track span { position: absolute; top: 2px; height: 10px; border-radius: 2px; opacity: .85; }
.ms-marks { background: transparent; }
.ms-marks i { position: absolute; top: -1px; transform: translateX(-50%); font: 700 10px ui-sans-serif, system-ui, sans-serif; font-style: normal; color: #fff; background: var(--ms-ink); border-radius: 3px; padding: 1px 4px; }
.ms-ticks { background: transparent; height: 12px; }
.ms-ticks em { position: absolute; transform: translateX(-50%); font: 10px ui-monospace, Menlo, monospace; font-style: normal; color: #8b857b; }
.ms-transcript { font-family: ui-sans-serif, system-ui, -apple-system, 'Segoe UI', 'PingFang SC', 'Noto Sans CJK SC', sans-serif; font-size: 14px; line-height: 1.5; color: var(--ms-ink); }
.ms-line { display: flex; align-items: baseline; gap: 10px; padding: 6px 8px; border-left: 3px solid transparent; border-radius: 4px; }
.ms-line:nth-child(odd) { background: #fbf9f4; }
.ms-linked { border-left-color: var(--ms-rust); background: #f6ece6 !important; }
.ms-time { font: 12px ui-monospace, Menlo, monospace; color: #8b857b; min-width: 40px; }
.ms-spk { color: #fff; font: 700 11px ui-monospace, Menlo, monospace; border-radius: 4px; padding: 1px 6px; }
.ms-text { flex: 1; }
.ms-badge { font: 700 10px ui-sans-serif, system-ui, sans-serif; color: #fff; background: var(--ms-ink); border-radius: 3px; padding: 1px 5px; }
footer { display: none !important; }
"""

EMPTY = "<div class='ms-panel'><div class='ms-empty'>The transcript appears here, one line per speaker turn.</div></div>"


def build_app(settings: Settings | None = None):
    import gradio as gr

    settings = settings or Settings.from_env()
    cache = ResultCache(settings.cache_dir)

    def pick_sample(name):
        if not name:
            return gr.update(), gr.update()
        path = str(samples_dir() / name)
        return path, path

    def pick_upload(file):
        path = file if isinstance(file, str) else getattr(file, "name", None)
        return path, path, gr.update(value=None)

    def preview(path):
        if not path:
            return "Pick a synthetic sample or upload a recording.", "", [], EMPTY, None
        return (f"Loaded **{Path(path).name}** ({sample_title(path)}). Press **Make notes**.", "", [], EMPTY, None)

    def do_run(path, speakers):
        if not path:
            return "Pick a sample or upload a recording first.", "", [], EMPTY, None
        n = 0 if speakers in (None, "Auto") else int(speakers)
        key = f"speakers={n}|lang={settings.language}"
        cached = cache.get(path, key, settings)
        if cached:
            notes = Notes.model_validate(cached)
            note = f"Loaded a saved result for this file, computed earlier by `{notes.model}` and `{notes.transcript.asr_model}`."
        else:
            try:
                notes = run(path, settings, n)
            except (ValueError, RuntimeError) as exc:
                return f"⚠️ {exc}", "", [], EMPTY, None
            cache.put(path, key, settings, notes.model_dump())
            t = notes.timings
            note = (f"Done with `{notes.transcript.asr_model}` and `{notes.model}` ({settings.backend} backend) in "
                    f"{sum(t.values()):.0f}s.")
        tr = notes.transcript
        status = (f"{note} {clock(tr.duration)} of audio · language {tr.language} · {len(tr.speakers)} speakers · "
                  f"{len(tr.segments)} turns · {len(notes.action_items)} action items.")
        return (status, notes_markdown(notes), item_rows(notes),
                f"<div class='ms-panel'>{transcript_html(notes)}</div>", export_files(notes))

    with gr.Blocks(title="meeting-notes · Modelshifts") as demo:
        gr.HTML(
            "<div class='ms-header'><div class='ms-brand'>MODELSHIFTS</div><h1>meeting-notes</h1>"
            "<p>Turn a meeting recording in English, Chinese or both into a transcript, summary and action items.</p>"
            f"<div class='ms-meta'>backend: {settings.backend} · transcription: {settings.asr_name} · "
            f"notes: {settings.model_name} · v{__version__}</div></div>"
        )
        current = gr.State(None)
        with gr.Row(equal_height=False):
            with gr.Column(scale=4, min_width=400, elem_classes="ms-left", elem_id="ms-left"):
                gr.HTML("<div class='ms-section-title'>1 · Recording</div>")
                with gr.Row(equal_height=True):
                    sample = gr.Dropdown(list_samples(), value=None, label="Synthetic sample", interactive=True, scale=3)
                    upload = gr.UploadButton("Upload audio", file_types=list(AUDIO_TYPES), type="filepath", scale=1,
                                             elem_classes="ms-secondary")
                player = gr.Audio(type="filepath", show_label=False, interactive=False, buttons=[])
                speakers = gr.Dropdown(["Auto", "2", "3", "4", "5", "6"], value="Auto", label="Speakers")
                run_btn = gr.Button("Make notes", variant="primary", elem_classes="ms-primary")
                status = gr.Markdown("Pick a synthetic sample or upload a recording.", elem_classes="ms-status")
                gr.HTML("<div class='ms-section-title'>2 · Notes</div>")
                notes_md = gr.Markdown("", elem_classes="ms-notes")
                gr.HTML("<div class='ms-section-title'>3 · Action items</div>")
                items = gr.Dataframe(headers=["#", "Task", "Owner", "Due", "At"],
                                     datatype=["str", "str", "str", "str", "str"], interactive=False, wrap=True,
                                     column_widths=["8%", "44%", "15%", "19%", "14%"], elem_classes="ms-table")
                gr.HTML("<div class='ms-section-title'>4 · Export</div>")
                files = gr.File(label="Markdown · JSON · SRT", file_count="multiple", interactive=False)
            with gr.Column(scale=6, min_width=480):
                gr.HTML("<div class='ms-section-title'>Transcript · speaker timeline on top, A1… mark where an action "
                        "item was agreed</div>")
                transcript = gr.HTML(EMPTY)
        gr.HTML("<p class='ms-muted'>All sample recordings are synthetic speech made with an open text-to-speech "
                "model. Uploaded files stay on the machine running this demo.</p>")

        outs = [status, notes_md, items, transcript, files]
        sample.change(pick_sample, [sample], [current, player]).then(preview, [current], outs)
        upload.upload(pick_upload, [upload], [current, player, sample]).then(preview, [current], outs)
        run_btn.click(do_run, [current, speakers], outs)
    return demo


def theme():
    import gradio as gr

    return gr.themes.Base(
        primary_hue=gr.themes.colors.orange,
        neutral_hue=gr.themes.colors.stone,
        font=[gr.themes.Font("Georgia"), gr.themes.Font("Times New Roman"), gr.themes.Font("serif")],
        font_mono=[gr.themes.Font("ui-monospace"), gr.themes.Font("Menlo"), gr.themes.Font("monospace")],
        radius_size=gr.themes.sizes.radius_md,
    )


def launch(host: str = "127.0.0.1", port: int = 7860, share: bool = False) -> None:
    demo = build_app()
    demo.queue().launch(server_name=host, server_port=port, share=share, theme=theme(), css=CSS)
