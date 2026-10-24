"""Planner-free probe of a MiniWoB LeWM checkpoint.

For each held-out seed, roll the expert towards the goal. At every non-terminal
expert step, score the expert's click and N uniform random clicks by the
model's one-step cost (predicted next embedding vs goal embedding). Report the
expert click's percentile (0 = ranked best). A useful planning model puts the
expert click near 0; chance is ~50.

    python probe_miniwob.py --policy lewm_miniwob/weights_epoch_29.pt
"""

import argparse

import numpy as np
import torch

from collect_miniwob import EVAL_SEED_OFFSET
from demo_miniwob import LeWMPlanner, expert_goal, pick_device
from miniwob_env import TASKS, MiniWoBClickEnv


@torch.inference_mode()
def expert_percentiles(env, planner, seed, n_random, rng):
    goal_frame, _, n_expert = expert_goal(env, seed)
    if n_expert == 0:
        return []
    goal = planner._img(goal_frame)
    frame, _ = env.reset(seed=seed)
    model = planner.solver.model
    out = []
    for _ in range(n_expert):
        expert, terminal = env.expert_action()
        if terminal:
            break
        cands = np.concatenate([expert[None], rng.uniform(-1, 1, (n_random, 2))]).astype(np.float32)
        cands = (cands - planner.act_mean) / planner.act_std  # model plans in z-scored action space
        S = len(cands)
        # get_cost expects (B, S, T, ...) like the CEM solver feeds it
        info = {"pixels": planner._img(frame)[:, None].expand(1, S, 1, -1, -1, -1),
                "goal": goal[:, None].expand(1, S, 1, -1, -1, -1),
                "action": torch.zeros(1, S, 1, 2)}
        cost = model.get_cost(info, torch.from_numpy(cands)[None, :, None].float().to(planner.device)).squeeze(0).cpu().numpy()
        out.append(100.0 * (cost[1:] < cost[0]).mean())
        frame, *_ = env.step(expert)
    return out


@torch.inference_mode()
def state_distances(planner, dataset_path, n_episodes=200, seed=0):
    """Embedding sq-distance between consecutive frames whose page changed (per
    task), vs. between first frames of different episodes. A model that tracks
    page state keeps the first number well above 0."""
    import h5py
    import hdf5plugin  # noqa: F401

    model = planner.solver.model.model
    with h5py.File(dataset_path, "r") as f:
        L, O, T = f["ep_len"][:], f["ep_offset"][:], f["task_idx"][:]
        eps = np.random.default_rng(seed).choice(len(L), n_episodes, replace=False)
        pc = f["page_changed"] if "page_changed" in f else None  # pointer-free change mask
        episodes = [(int(T[O[e]]), f["pixels"][O[e] : O[e] + L[e]],
                     None if pc is None else pc[O[e] : O[e] + L[e] - 1] > 0.5) for e in eps]
    changed, firsts, cursor_only = {}, [], []
    for task, px, page in episodes:
        x = torch.cat([planner._img(p)[0] for p in px]).to(planner.device)
        e = model.encode({"pixels": x[None]})["emb"][0]
        if page is None:
            page = (np.abs(px[1:].astype(np.int16) - px[:-1].astype(np.int16)).max(-1) > 30).sum((1, 2)) > 10
        for i in np.nonzero(page)[0]:  # >10 pixels of the page visibly changed
            changed.setdefault(TASKS[task], []).append((e[i + 1] - e[i]).pow(2).sum().item())
        if pc is not None:  # with a drawn pointer: clicks that only moved the pointer
            moved = (np.abs(px[1:].astype(np.int16) - px[:-1].astype(np.int16)).max(-1) > 30).sum((1, 2)) > 10
            cursor_only += [(e[i + 1] - e[i]).pow(2).sum().item() for i in np.nonzero(moved & ~page)[0]]
        firsts.append(e[0])
    F = torch.stack(firsts)
    across = torch.cdist(F, F).pow(2)
    out = {t: float(np.median(v)) for t, v in sorted(changed.items())}
    if cursor_only:
        out["cursor-only"] = float(np.median(cursor_only))
    return out, float(across[across > 0].median())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", nargs="+", required=True)
    p.add_argument("--dataset", default="miniwob_click.h5")
    p.add_argument("--tasks", nargs="+", default=list(TASKS), choices=TASKS)
    p.add_argument("--num-eval", type=int, default=20)
    p.add_argument("--n-random", type=int, default=255)
    p.add_argument("--device", default="auto")
    p.add_argument("--state", action="store_true", help="report within-episode state distances instead")
    p.add_argument("--hide-text", action="store_true", help="render page text transparent (match a --hide-text dataset)")
    p.add_argument("--draw-cursor", action="store_true", help="draw a pointer at the last click (match a --draw-cursor dataset)")
    args = p.parse_args()

    device = pick_device(args.device)
    if args.state:
        import stable_worldmodel as swm
        from pathlib import Path

        path = Path(swm.data.utils.get_cache_dir(sub_folder="datasets"), args.dataset)
        for policy in args.policy:
            planner = LeWMPlanner(policy, args.dataset, horizon=1, cem_kwargs=dict(num_samples=1, n_steps=1, topk=1), device=device)
            changed, across = state_distances(planner, path)
            print(f"{policy:45s} changed-frame dist: " + "  ".join(f"{t}={d:.3f}" for t, d in changed.items()) + f"  | across episodes={across:.0f}", flush=True)
        return

    envs = {t: MiniWoBClickEnv(t, hide_text=args.hide_text, draw_cursor=args.draw_cursor) for t in args.tasks}
    print(f"{'policy':40s} " + " ".join(f"{t:>18s}" for t in args.tasks))
    for policy in args.policy:
        planner = LeWMPlanner(policy, args.dataset, horizon=1, cem_kwargs=dict(num_samples=1, n_steps=1, topk=1), device=device)
        row = []
        for task, env in envs.items():
            rng = np.random.default_rng(0)
            pct = [x for i in range(args.num_eval) for x in expert_percentiles(env, planner, EVAL_SEED_OFFSET + i, args.n_random, rng)]
            row.append(f"{np.mean(pct):8.1f} (n={len(pct):3d})")
        print(f"{policy:40s} " + " ".join(f"{r:>18s}" for r in row), flush=True)
    for env in envs.values():
        env.close()


if __name__ == "__main__":
    main()
