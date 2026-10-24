"""Annotated MP4s of MiniWoB rollouts ([observation | goal] frames).

Every video gets a title card (task, seed, policy), a header with the step and
observation / goal labels, and an end frame with a green (goal reached) or red
border. When click positions are known (demo_miniwob.py passes them), each
click is drawn as a crosshair on the frame it was issued from, with the
model's action, the pixel it maps to, and the planner cost.

demo_miniwob.py writes these directly. This script converts older GIFs, where
click positions were not saved. Pillow merges identical consecutive GIF
frames (e.g. a click that changed nothing) and sums their durations, so
per-step frames are rebuilt from the durations, and success is judged by a
pixel-exact match of the final observation and the goal.

    python make_videos.py outputs/miniwob_eval_final
"""

import argparse
import re
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

STEP_MS = 700  # demo_miniwob.py writes one step per 700 ms
FPS = 10
STEP_S, TITLE_S, END_S = 1.2, 1.5, 2.5
SCALE = 2
HEADER = 90
GREEN, RED, GREY = (34, 139, 34), (200, 40, 40), (90, 90, 90)


def font(size):
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


F_BIG, F_MED, F_SMALL = font(30), font(20), font(16)


def read_steps(path):
    """GIF -> list of [obs | goal] frames, one per environment step."""
    im = Image.open(path)
    steps = []
    try:
        while True:
            n = max(1, round(im.info.get("duration", STEP_MS) / STEP_MS))
            steps += [np.asarray(im.convert("RGB"))] * n
            im.seek(im.tell() + 1)
    except EOFError:
        pass
    return steps


def matches_goal(frame, min_px=10):
    w = frame.shape[1] // 2
    diff = np.abs(frame[:, :w].astype(int) - frame[:, w:].astype(int)).max(-1)
    return (diff > 30).sum() <= min_px


def canvas(frame, header_text, sub_text, border=None):
    h, w, _ = frame.shape
    img = Image.new("RGB", (w * SCALE, h * SCALE + HEADER), "white")
    img.paste(Image.fromarray(frame).resize((w * SCALE, h * SCALE), Image.NEAREST), (0, HEADER))
    d = ImageDraw.Draw(img)
    d.text((10, 6), header_text, fill="black", font=F_MED)
    d.text((10, 34), sub_text, fill=border if border and border != GREY else "black", font=F_MED)
    d.text((w * SCALE // 4 - 45, 64), "observation", fill=GREY, font=F_SMALL)
    d.text((3 * w * SCALE // 4 - 18, 64), "goal", fill=GREY, font=F_SMALL)
    d.line([(w * SCALE // 2, HEADER), (w * SCALE // 2, img.height)], fill=GREY, width=2)
    if border:
        d.rectangle([0, 0, img.width - 1, img.height - 1], outline=border, width=8)
    return np.asarray(img)


def card(shape, lines, color):
    img = Image.new("RGB", (shape[1], shape[0]), color)
    d = ImageDraw.Draw(img)
    y = shape[0] // 2 - 30 * len(lines) // 2 - 10
    for text, f in lines:
        tw = d.textlength(text, font=f)
        d.text(((shape[1] - tw) // 2, y), text, fill="white", font=f)
        y += f.size + 14
    return np.asarray(img)


def draw_clicks(img, clicks, current):
    """Crosshair at `current` (frame px), grey dots at earlier clicks."""
    d = ImageDraw.Draw(img)
    to_canvas = lambda xy: (xy[0] * SCALE, HEADER + xy[1] * SCALE)
    for xy in clicks:
        x, y = to_canvas(xy)
        d.ellipse([x - 4, y - 4, x + 4, y + 4], fill=GREY)
    if current is not None:
        x, y = to_canvas(current)
        d.ellipse([x - 12, y - 12, x + 12, y + 12], outline=RED, width=3)
        d.line([(x - 20, y), (x - 6, y)], fill=RED, width=3)
        d.line([(x + 6, y), (x + 20, y)], fill=RED, width=3)
        d.line([(x, y - 20), (x, y - 6)], fill=RED, width=3)
        d.line([(x, y + 6), (x, y + 20)], fill=RED, width=3)
        d.ellipse([x - 2, y - 2, x + 2, y + 2], fill=RED)
    return np.asarray(img)


def episode_video(steps, title, policy, ok, clicks=None):
    """steps: N+1 [obs | goal] frames. clicks: optional N dicts with "xy" (frame
    px of the click issued from steps[k]) and "text" (shown in the header)."""
    n = len(steps) - 1
    first = canvas(steps[0], title, "")
    out = [card(first.shape, [("EPISODE START", F_BIG), (title, F_MED), (policy, F_SMALL)], GREY)] * int(TITLE_S * FPS)
    for k, s in enumerate(steps):
        if clicks is not None and k < n:
            img = Image.fromarray(canvas(s, title, f"click {k + 1}/{n}: {clicks[k]['text']}"))
            frame = draw_clicks(img, [c["xy"] for c in clicks[:k]], clicks[k]["xy"])
        else:
            label = "start" if k == 0 else f"after click {k} / {n}"
            frame = canvas(s, title, label)
            if clicks is not None:
                frame = draw_clicks(Image.fromarray(frame), [c["xy"] for c in clicks[:k]], None)
        out += [frame] * int(STEP_S * FPS)
    verdict = "goal REACHED" if ok else "goal NOT reached"
    end = canvas(steps[-1], title, f"END: {verdict}  ({n} clicks)", border=GREEN if ok else RED)
    if clicks is not None:
        end = draw_clicks(Image.fromarray(end), [c["xy"] for c in clicks], None)
    out += [end] * int(END_S * FPS)
    return out


def mp4_writer(path):
    return imageio.get_writer(path, fps=FPS, codec="libx264", quality=8, macro_block_size=1, pixelformat="yuv420p")


def write_mp4(path, frames):
    with mp4_writer(path) as w:
        for f in frames:
            w.append_data(f)


def episode_frames(gif, policy):
    m = re.match(r"(.+)_(\d+)$", gif.stem)
    task, seed = m.group(1), m.group(2)
    steps = read_steps(gif)
    ok = matches_goal(steps[-1])
    return episode_video(steps, f"{task}   seed {seed}", policy, ok), ok, len(steps) - 1


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("roots", nargs="+", help="folders searched recursively for GIFs")
    args = p.parse_args()
    folders = sorted({g.parent for r in args.roots for g in Path(r).rglob("*.gif")})
    for folder in folders:
        policy = folder.name
        summary = []
        with mp4_writer(folder / "all_episodes.mp4") as all_w:
            for gif in sorted(folder.glob("*.gif")):
                frames, ok, n = episode_frames(gif, policy)
                write_mp4(gif.with_suffix(".mp4"), frames)
                for f in frames:
                    all_w.append_data(f)
                summary.append(f"{gif.stem}: {'match' if ok else 'no match'} in {n} clicks")
        print(f"{folder}: {len(summary)} episodes")
        for line in summary:
            print("   ", line)


if __name__ == "__main__":
    main()
