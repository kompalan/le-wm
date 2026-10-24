"""Collect a pixels + click-action dataset on MiniWoB++ for LeWM training.

There is no public MiniWoB++ dataset with screenshots (the Stanford human
demos are DOM/event logs that can't be re-rendered), so we roll out a mixed
policy in the live environment:

  - expert (p_expert): scripted click towards the task goal
  - element (p_element): click a random interactive element (checkbox, tab, ...)
  - uniform (rest): click anywhere, mostly no-ops, so the model learns those too

Clicks that would finish the task (Submit / the target link) are swapped for
an element click, so episodes run to `max_steps` and never contain the blank
post-task page. Output follows the stable-worldmodel HDF5 layout (one row per
step, ep_len / ep_offset index), with the action of the last step set to NaN.

    python collect_miniwob.py --episodes-per-task 750 --workers 8
"""

import argparse
import logging
import multiprocessing as mp
import time
from pathlib import Path

import h5py
import hdf5plugin
import numpy as np
import stable_worldmodel as swm

from miniwob_env import TASKS, MiniWoBClickEnv, page_changed

TRAIN_SEED_OFFSET = 0
EVAL_SEED_OFFSET = 1_000_000  # demo_miniwob.py evaluates on seeds from here


def rollout(env, seed, rng, max_steps, p_expert, p_element):
    frame, _ = env.reset(seed=seed)
    frames, actions, sources, changed = [frame], [], [], []
    clean = env.clean_frame
    for _ in range(max_steps):
        u = rng.random()
        action, src = None, "uniform"
        if u < p_expert:
            action, terminal = env.expert_action()
            src = "expert"
            if terminal:
                action, src = None, "element"
        if action is None and u < p_expert + p_element:
            action, src = env.element_action(), "element"
        if action is None:
            action = rng.uniform(-1, 1, size=2).astype(np.float32)

        frame, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break  # drop the terminal click: the next frame is the blank page
        frames.append(frame)
        actions.append(action)
        sources.append(src)
        # did the page itself change (ignoring the drawn pointer)?
        changed.append(float(page_changed(clean, env.clean_frame)))
        clean = env.clean_frame

    actions.append(np.full(2, np.nan, dtype=np.float32))
    sources.append("none")
    changed.append(np.nan)
    return np.stack(frames), np.stack(actions).astype(np.float32), sources, np.array(changed, np.float32)


def worker(args):
    task, seeds, cfg = args
    env = MiniWoBClickEnv(task, img_size=cfg["img_size"], hide_text=cfg["hide_text"], draw_cursor=cfg["draw_cursor"])
    rng = np.random.default_rng(seeds[0])
    episodes = []
    try:
        for seed in seeds:
            pixels, action, src, changed = rollout(env, int(seed), rng, cfg["max_steps"], cfg["p_expert"], cfg["p_element"])
            if len(pixels) >= cfg["min_len"]:
                episodes.append(dict(task=task, seed=int(seed), pixels=pixels, action=action, src=src, page_changed=changed))
    finally:
        env.close()
    return episodes


class Writer:
    """Append-only HDF5 writer in the swm layout, with compressed pixels."""

    def __init__(self, path, img_size):
        self.f = h5py.File(path, "w", libver="latest")
        blosc = hdf5plugin.Blosc(cname="lz4", clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE)
        S = (img_size, img_size, 3)
        self.f.create_dataset("pixels", (0, *S), np.uint8, maxshape=(None, *S), chunks=(1, *S), **blosc)
        self.f.create_dataset("action", (0, 2), np.float32, maxshape=(None, 2))
        # 1 if the page (not the drawn pointer) changed between step t and t+1
        self.f.create_dataset("page_changed", (0,), np.float32, maxshape=(None,))
        for col in ("ep_idx", "step_idx", "task_idx", "seed", "is_expert"):
            self.f.create_dataset(col, (0,), np.int64, maxshape=(None,))
        self.f.create_dataset("ep_len", (0,), np.int32, maxshape=(None,))
        self.f.create_dataset("ep_offset", (0,), np.int64, maxshape=(None,))
        self.f.attrs["tasks"] = list(TASKS)
        self.f.attrs["action_space"] = "click (x, y) in [-1, 1]^2 over the 160x210 task area"
        self.n_rows = 0
        self.n_eps = 0

    def write(self, ep):
        T = len(ep["pixels"])
        cols = {
            "pixels": ep["pixels"],
            "action": ep["action"],
            "page_changed": ep["page_changed"],
            "ep_idx": np.full(T, self.n_eps),
            "step_idx": np.arange(T),
            "task_idx": np.full(T, TASKS.index(ep["task"])),
            "seed": np.full(T, ep["seed"]),
            "is_expert": np.array([s == "expert" for s in ep["src"]], dtype=np.int64),
        }
        for k, v in cols.items():
            self.f[k].resize(self.n_rows + T, axis=0)
            self.f[k][self.n_rows :] = v
        for k, v in (("ep_len", T), ("ep_offset", self.n_rows)):
            self.f[k].resize(self.n_eps + 1, axis=0)
            self.f[k][self.n_eps] = v
        self.n_rows += T
        self.n_eps += 1

    def close(self):
        self.f.close()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks", nargs="+", default=list(TASKS), choices=TASKS)
    p.add_argument("--episodes-per-task", type=int, default=750)
    p.add_argument("--max-steps", type=int, default=12)
    p.add_argument("--min-len", type=int, default=4, help="history_size + num_preds")
    p.add_argument("--p-expert", type=float, default=0.5)
    p.add_argument("--p-element", type=float, default=0.3)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--chunk", type=int, default=50, help="episodes per worker job")
    p.add_argument("--name", default="miniwob_click", help="written to $STABLEWM_HOME/datasets/<name>.h5")
    p.add_argument("--hide-text", action="store_true", help="render all page text transparent (diagnostic)")
    p.add_argument("--draw-cursor", action="store_true", help="draw a mouse pointer at the last click in observations")
    args = p.parse_args()

    cfg = dict(img_size=args.img_size, max_steps=args.max_steps, min_len=args.min_len,
               p_expert=args.p_expert, p_element=args.p_element, hide_text=args.hide_text,
               draw_cursor=args.draw_cursor)
    jobs = []
    for task in args.tasks:
        seeds = TRAIN_SEED_OFFSET + np.arange(args.episodes_per_task)
        jobs += [(task, seeds[i : i + args.chunk], cfg) for i in range(0, len(seeds), args.chunk)]

    out = Path(swm.data.utils.get_cache_dir(sub_folder="datasets"), f"{args.name}.h5")
    writer = Writer(out, args.img_size)
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for i, episodes in enumerate(pool.imap_unordered(worker, jobs)):
            for ep in episodes:
                writer.write(ep)
            logging.warning(f"[{i + 1}/{len(jobs)}] {writer.n_eps} episodes, {writer.n_rows} frames, {time.time() - t0:.0f}s")
    writer.close()
    print(f"wrote {writer.n_eps} episodes / {writer.n_rows} frames to {out} ({out.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
