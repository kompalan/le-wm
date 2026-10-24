"""Convert OSWorld agent rollouts (xlangai/ubuntu_osworld_verified_trajs zips) to
the stable-worldmodel HDF5 layout used by train.py.

Each zip holds <...>/<domain>/<task_id>/{traj.jsonl, result.txt, step_<k>_<time>.png}.
step_<k>.png is the screen AFTER step k's action (there is no frame before step 1),
so a trajectory gives rows

    pixels = frame after step k,   action = step k+1's action   (k = 1 .. n-1)

with the last row's action NaN, as in collect_miniwob.py. A missing screenshot or
an action that can't be parsed splits the trajectory into separate episodes, so
every row-to-row transition is a single known action.

Actions are parsed into a common schema and encoded as fixed vectors by
osworld_actions.py; the parsed sub-actions are also stored as JSON
("action_json") so the encoding can be redone without the zips.

Screenshots are squashed (not padded) to img_size x img_size. The real mouse
pointer is visible in the screenshots; "cursor" records where it should be
(the last pointer coordinate, NaN if unknown) and page_changed ignores a box
around it, and the clock in the top bar.

    python convert_osworld.py data/raw/osworld/*.zip --name osworld --img-size 448
"""

import argparse
import io
import json
import logging
import multiprocessing as mp
import re
import time
import zipfile
from pathlib import Path

import h5py
import hdf5plugin
import numpy as np
import stable_worldmodel as swm
from PIL import Image

import osworld_actions as oa

_ZIPS = {}
_STEP_RE = re.compile(r"step_(\d+)_")
TOP_BAR = 30 / 1080  # GNOME top bar (clock) as a fraction of screen height
CURSOR_BOX = (-6, -6, 30, 40)  # pixels around the pointer tip (x0, y0, x1, y1) at 1920x1080


def _zip(path):
    if path not in _ZIPS:
        _ZIPS[path] = zipfile.ZipFile(path)
    return _ZIPS[path]


def list_trajectories(path):
    z = _zip(path)
    return sorted(n.rsplit("/", 1)[0] for n in z.namelist()
                  if n.endswith("/traj.jsonl") and "__MACOSX" not in n and "/._" not in n)


def changed_mask(shape, cursors, screen_wh):
    """Pixels page_changed looks at: everything but the top bar and the pointer boxes."""
    H, W = shape
    m = np.ones((H, W), bool)
    m[: int(np.ceil(TOP_BAR * H))] = False
    sw, sh = screen_wh
    for c in cursors:
        if c is None:
            continue
        px, py = (c[0] + 1) / 2 * sw, (c[1] + 1) / 2 * sh
        x0, y0, x1, y1 = CURSOR_BOX
        xa, xb = int((px + x0) * W / sw), int(np.ceil((px + x1) * W / sw))
        ya, yb = int((py + y0) * H / sh), int(np.ceil((py + y1) * H / sh))
        m[max(ya, 0):max(yb, 0), max(xa, 0):max(xb, 0)] = False
    return m


def page_changed(a, b, mask, thr=30, min_px=10):
    d = (np.abs(a.astype(np.int16) - b.astype(np.int16)).max(-1) > thr) & mask
    return float(d.sum() > min_px)


def convert_trajectory(job):
    """One trajectory dir -> list of episode dicts (contiguous runs of known transitions)."""
    zpath, tdir, img_size = job
    z = _zip(zpath)
    names = set(z.namelist())
    recs = [json.loads(l) for l in z.read(f"{tdir}/traj.jsonl").decode("utf-8", "replace").splitlines() if l.strip()]
    files = {}
    for n in names:
        if n.startswith(tdir + "/") and n.endswith(".png") and "/" not in n[len(tdir) + 1:]:
            m = _STEP_RE.match(n[len(tdir) + 1:])
            if m:
                files.setdefault(int(m.group(1)), n)
    try:
        success = float(z.read(f"{tdir}/result.txt").decode().strip())
    except (KeyError, ValueError):
        success = float("nan")

    stats = dict(steps=len(recs), unparsed=0, terminal=0, no_frame=0, truncated=0)
    screen = None
    ptr = None
    steps = []  # per step k: (subs or None, frame or None, cursor after)
    cursor = None
    for r in sorted(recs, key=lambda r: r.get("step_num", 0)):
        k = r.get("step_num")
        fname = r.get("screenshot_file")
        fpath = f"{tdir}/{fname}" if fname and f"{tdir}/{fname}" in names else files.get(k)
        frame = None
        if fpath is not None:
            try:
                im = Image.open(io.BytesIO(z.read(fpath))).convert("RGB")
                screen = screen or im.size
                frame = np.asarray(im.resize((img_size, img_size), Image.LANCZOS))
            except Exception:
                frame = None
        if ptr is None:
            ptr = oa.pointer(*(screen or (1920, 1080)))
        try:
            subs = oa.parse_step(r, ptr)
            if subs is None:
                stats["terminal"] += 1
        except Exception:
            subs = None
            stats["unparsed"] += 1
        if subs is not None:
            cursor = oa.cursor_after(subs, cursor)
        if frame is None:
            stats["no_frame"] += 1
        steps.append((subs, frame, cursor))

    # rows: frame after step i, action of step i+1
    episodes, cur = [], None
    for i, (subs, frame, c) in enumerate(steps):
        if frame is None:
            cur = None
            continue
        if cur is None:
            cur = dict(frames=[], cursors=[], subs=[], step=[])
            episodes.append(cur)
        cur["frames"].append(frame)
        cur["cursors"].append(c)
        cur["step"].append(i + 1)
        nxt = steps[i + 1] if i + 1 < len(steps) else (None, None, None)
        if nxt[0] is None or nxt[1] is None:
            cur["subs"].append(None)  # episode ends at this frame
            cur = None
        else:
            cur["subs"].append(nxt[0])

    out = []
    sw = screen or (1920, 1080)
    for ep in episodes:
        T = len(ep["frames"])
        act = np.full((T, oa.ACTION_DIM), np.nan, np.float32)
        changed = np.full(T, np.nan, np.float32)
        trunc = np.zeros(T, np.int64)
        for t in range(T - 1):
            act[t] = oa.encode_step(ep["subs"][t])
            n_real = len([a for a in ep["subs"][t] if a["type"] != "noop"])
            trunc[t] = int(n_real > oa.MAX_SUB)
            mask = changed_mask(ep["frames"][t].shape[:2], (ep["cursors"][t], ep["cursors"][t + 1]), sw)
            changed[t] = page_changed(ep["frames"][t], ep["frames"][t + 1], mask)
        stats["truncated"] += int(trunc.sum())
        cur_xy = np.array([c if c is not None else (np.nan, np.nan) for c in ep["cursors"]], np.float32)
        out.append(dict(pixels=np.stack(ep["frames"]), action=act, page_changed=changed, truncated=trunc,
                        cursor=cur_xy, step_num=np.array(ep["step"], np.int64),
                        action_json=[oa.dumps(s) if s is not None else "" for s in ep["subs"]],
                        success=success))
    parts = tdir.split("/")
    meta = dict(zip=Path(zpath).stem, domain=parts[-2], task=parts[-1], traj=tdir)
    return meta, out, stats


class Writer:
    """Append-only HDF5 writer in the swm layout, with compressed pixels."""

    STR = h5py.string_dtype()

    def __init__(self, path, img_size):
        self.f = h5py.File(path, "w", libver="latest")
        blosc = hdf5plugin.Blosc(cname="lz4", clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE)
        S = (img_size, img_size, 3)
        self.f.create_dataset("pixels", (0, *S), np.uint8, maxshape=(None, *S), chunks=(1, *S), **blosc)
        self.f.create_dataset("action", (0, oa.ACTION_DIM), np.float32, maxshape=(None, oa.ACTION_DIM))
        self.f.create_dataset("cursor", (0, 2), np.float32, maxshape=(None, 2))
        for col in ("page_changed", "success"):
            self.f.create_dataset(col, (0,), np.float32, maxshape=(None,))
        for col in ("ep_idx", "step_idx", "step_num", "truncated", "agent_idx", "domain_idx", "task_idx", "traj_idx"):
            self.f.create_dataset(col, (0,), np.int64, maxshape=(None,))
        self.f.create_dataset("action_json", (0,), self.STR, maxshape=(None,))
        self.f.create_dataset("ep_len", (0,), np.int32, maxshape=(None,))
        self.f.create_dataset("ep_offset", (0,), np.int64, maxshape=(None,))
        self.f.attrs["action_space"] = (f"osworld_actions.encode_step: {oa.MAX_SUB} slots x {oa.SUB_DIM} + count; "
                                        f"types {oa.TYPES}")
        self.agents, self.domains, self.tasks, self.trajs = {}, {}, {}, {}
        self.n_rows = 0
        self.n_eps = 0

    @staticmethod
    def _idx(table, key):
        return table.setdefault(key, len(table))

    def write(self, meta, ep):
        T = len(ep["pixels"])
        cols = {
            "pixels": ep["pixels"], "action": ep["action"], "cursor": ep["cursor"],
            "page_changed": ep["page_changed"], "success": np.full(T, ep["success"], np.float32),
            "ep_idx": np.full(T, self.n_eps), "step_idx": np.arange(T), "step_num": ep["step_num"],
            "truncated": ep["truncated"],
            "agent_idx": np.full(T, self._idx(self.agents, meta["zip"])),
            "domain_idx": np.full(T, self._idx(self.domains, meta["domain"])),
            "task_idx": np.full(T, self._idx(self.tasks, meta["task"])),
            "traj_idx": np.full(T, self._idx(self.trajs, meta["traj"])),
            "action_json": np.array(ep["action_json"], dtype=object),
        }
        for k, v in cols.items():
            self.f[k].resize(self.n_rows + T, axis=0)
            self.f[k][self.n_rows:] = v
        for k, v in (("ep_len", T), ("ep_offset", self.n_rows)):
            self.f[k].resize(self.n_eps + 1, axis=0)
            self.f[k][self.n_eps] = v
        self.n_rows += T
        self.n_eps += 1

    def close(self):
        for name, table in (("agents", self.agents), ("domains", self.domains), ("tasks", self.tasks),
                            ("trajectories", self.trajs)):
            self.f.attrs[name] = list(table)
        self.f.close()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("zips", nargs="+")
    p.add_argument("--name", default="osworld", help="written to $STABLEWM_HOME/datasets/<name>.h5")
    p.add_argument("--img-size", type=int, default=448)
    p.add_argument("--min-len", type=int, default=4, help="history_size + num_preds")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--limit", type=int, default=None, help="trajectories per zip (smoke tests)")
    args = p.parse_args()

    jobs = []
    for zp in args.zips:
        trajs = list_trajectories(zp)[: args.limit]
        jobs += [(zp, t, args.img_size) for t in trajs]
    out = Path(swm.data.utils.get_cache_dir(sub_folder="datasets"), f"{args.name}.h5")
    writer = Writer(out, args.img_size)
    totals = dict(trajectories=0, short=0)
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for i, (meta, episodes, stats) in enumerate(pool.imap_unordered(convert_trajectory, jobs, chunksize=2)):
            totals["trajectories"] += 1
            for k, v in stats.items():
                totals[k] = totals.get(k, 0) + v
            for ep in episodes:
                if len(ep["pixels"]) >= args.min_len:
                    writer.write(meta, ep)
                else:
                    totals["short"] += 1
            if (i + 1) % 50 == 0:
                logging.warning(f"[{i + 1}/{len(jobs)}] {writer.n_eps} episodes, {writer.n_rows} frames, "
                                f"{time.time() - t0:.0f}s")
    writer.close()
    print(f"wrote {writer.n_eps} episodes / {writer.n_rows} frames to {out} ({out.stat().st_size / 1e6:.0f} MB)")
    print("totals:", json.dumps(totals))


if __name__ == "__main__":
    main()
