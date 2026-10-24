"""Goal-conditioned planning with LeWM on MiniWoB++.

For each held-out seed:
  1. Roll the scripted expert forward until its next click would finish the
     task (Submit / target link). That frame is the goal image, and its DOM
     state is the success criterion.
  2. Reset to the same seed and run MPC: CEM over click sequences, scored by
     LeWM's latent distance between the predicted final embedding and the
     goal embedding. Execute the first click, observe, replan.
  3. Success = page state matches the goal state (or the task was solved).

A uniform random-click baseline with the same budget is run on the same seeds.

The first --videos episodes per task are saved as annotated MP4s (see
make_videos.py): each click is drawn as a crosshair with the model's action,
the task-area pixel it maps to, and the planner cost. results.json also lists
every episode's clicks and outcome.

    python demo_miniwob.py --policy lewm_miniwob/weights_epoch_15.pt
    python demo_miniwob.py --policy random
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from gymnasium.spaces import Box
from torchvision.transforms import v2 as transforms

from collect_miniwob import EVAL_SEED_OFFSET
from make_videos import episode_video, mp4_writer
from miniwob_env import TASKS, MiniWoBClickEnv, frame_coords, raw_coords


def pick_device(name):
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def expert_goal(env, seed, max_steps=10, cursor=False):
    """Run the expert up to (not including) its terminal click. With cursor=True
    (and a draw_cursor env) the goal shows the pointer at the expert's last click."""
    env.reset(seed=seed)
    for n in range(max_steps):
        action, terminal = env.expert_action()
        if terminal:
            return (env.render() if cursor else env.clean_frame).copy(), env.page_state(), n
        env.step(action)
    raise RuntimeError(f"expert did not reach a terminal click in {max_steps} steps")


class ClampedCost(torch.nn.Module):
    """Clamps candidate actions in place before scoring, so CEM's elites (and
    hence its refitted mean) stay inside the valid action box.

    cost="last" is JEPA's criterion (goal distance after the final click).
    cost="sum" adds the goal distance after every planned click: with "last",
    "no-op now, useful click later" ties with "useful click now", and receding
    horizon replanning then keeps postponing the useful click."""

    def __init__(self, model, lo, hi, cost="sum"):
        super().__init__()
        self.model, self.lo, self.hi, self.cost = model, lo, hi, cost

    def get_cost(self, info, candidates):
        candidates.clamp_(self.lo, self.hi)
        cost = self.model.get_cost(info, candidates)
        if self.cost == "last":
            return cost
        pred = info["predicted_emb"][:, :, 1:]  # (B, S, T, D): state after each click
        goal = info["goal_emb"][..., -1:, :]
        return (pred - goal).pow(2).sum(-1).sum(-1)


class LeWMPlanner:
    def __init__(self, run_name, dataset, horizon, cem_kwargs, device, cost="sum", warm_start=True):
        self.device = device
        model = swm.wm.utils.load_pretrained(run_name)
        model = model.to(device).eval().requires_grad_(False)

        # actions are z-scored in training (utils.get_column_normalizer); plan in that space
        act = swm.data.load_dataset(dataset, keys_to_load=["action"], keys_to_cache=["action"]).get_col_data("action")
        act = act[~np.isnan(act).any(axis=1)]
        self.act_mean, self.act_std = act.mean(0), act.std(0, ddof=1)

        self.transform = transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
        ])
        # CEM samples are unbounded; clamp them to valid clicks so the planner
        # can't exploit the model on out-of-range actions
        lo = torch.as_tensor((-1 - self.act_mean) / self.act_std, dtype=torch.float32, device=device)
        hi = torch.as_tensor((1 - self.act_mean) / self.act_std, dtype=torch.float32, device=device)
        self.solver = swm.solver.CEMSolver(model=ClampedCost(model, lo, hi, cost), device=device, **cem_kwargs)
        self.solver.configure(
            action_space=Box(-np.inf, np.inf, shape=(1, 2)), n_envs=1,
            config=swm.PlanConfig(horizon=horizon, receding_horizon=1),
        )
        self.horizon = horizon
        self.warm_start = warm_start

    def _img(self, frame):
        return self.transform(frame)[None, None]  # (1, 1, C, H, W)

    def reset(self, goal_frame):
        self.goal = self._img(goal_frame)
        self.plan = None

    @torch.inference_mode()
    def act(self, frame):
        info = {"pixels": self._img(frame), "goal": self.goal, "action": torch.zeros(1, 1, 2)}
        out = self.solver(info, init_action=self.plan)
        plan = out["actions"]  # (1, horizon, 2), normalized
        self.plan = plan[:, 1:] if self.warm_start else None  # warm start the next solve from the shifted plan
        a = plan[0, 0].numpy() * self.act_std + self.act_mean
        return np.clip(a, -1, 1).astype(np.float32), float(out["costs"][0])


def run_episode(env, seed, goal_frame, goal_state, budget, policy, rng):
    frame, _ = env.reset(seed=seed)
    if policy is not None:
        policy.reset(goal_frame)
    frames, clicks, solved = [np.concatenate([frame, goal_frame], axis=1)], [], False
    for t in range(budget):
        if env.page_state() == goal_state:
            solved = True
            break
        if policy is None:
            action, cost = rng.uniform(-1, 1, 2).astype(np.float32), None
        else:
            action, cost = policy.act(frame)
        clicks.append(dict(action=action.tolist(), task_px=raw_coords(action).tolist(),
                           xy=frame_coords(action, frame.shape[0]).tolist(), cost=cost))
        frame, reward, terminated, truncated, _ = env.step(action)
        frames.append(np.concatenate([frame, goal_frame], axis=1))
        if terminated or truncated:
            solved = reward > 0  # e.g. clicked the target link directly
            break
    else:
        solved = env.page_state() == goal_state
    return solved, len(frames) - 1, frames, clicks


def click_text(c):
    text = f"action ({c['action'][0]:+.2f}, {c['action'][1]:+.2f}) -> px ({c['task_px'][0]:.0f}, {c['task_px'][1]:.0f})"
    return text if c["cost"] is None else f"{text}, cost {c['cost']:.2f}"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", default="random", help="checkpoint relative to $STABLEWM_HOME/checkpoints, or 'random'")
    p.add_argument("--dataset", default="miniwob_click.h5", help="training dataset (for action statistics)")
    p.add_argument("--tasks", nargs="+", default=list(TASKS), choices=TASKS)
    p.add_argument("--num-eval", type=int, default=20, help="episodes per task")
    p.add_argument("--horizon", type=int, default=4, help="clicks planned per solve")
    p.add_argument("--extra-budget", type=int, default=3, help="clicks allowed beyond the expert's count")
    p.add_argument("--num-samples", type=int, default=300)
    p.add_argument("--n-steps", type=int, default=30)
    p.add_argument("--topk", type=int, default=30)
    p.add_argument("--cost", default="sum", choices=["sum", "last"], help="goal distance summed over planned clicks, or after the last one only")
    p.add_argument("--device", default="auto")
    p.add_argument("--hide-text", action="store_true", help="render page text transparent (match a --hide-text dataset)")
    p.add_argument("--draw-cursor", action="store_true", help="draw a pointer at the last click (match a --draw-cursor dataset)")
    p.add_argument("--no-warm-start", action="store_true", help="start every solve from the initial distribution instead of the previous plan")
    p.add_argument("--goal-cursor", action="store_true", help="with --draw-cursor, the goal image also shows the pointer (at the expert's last click)")
    p.add_argument("--videos", type=int, default=3, help="annotated MP4s saved per task")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="outputs/miniwob_demo")
    args = p.parse_args()

    device = pick_device(args.device)
    rng = np.random.default_rng(args.seed)
    planner = None
    if args.policy != "random":
        cem = dict(num_samples=args.num_samples, n_steps=args.n_steps, topk=args.topk, seed=args.seed)
        planner = LeWMPlanner(args.policy, args.dataset, args.horizon, cem, device, args.cost, not args.no_warm_start)

    tag = "random" if planner is None else args.policy.replace("/", "_").removesuffix(".pt")
    out_dir = Path(args.out, tag)
    out_dir.mkdir(parents=True, exist_ok=True)

    results, episodes = {}, {}
    all_videos = mp4_writer(out_dir / "all_episodes.mp4") if args.videos > 0 else None
    for task in args.tasks:
        env = MiniWoBClickEnv(task, hide_text=args.hide_text, draw_cursor=args.draw_cursor)
        succ, steps, skipped, episodes[task] = [], [], 0, []
        for i in range(args.num_eval):
            seed = EVAL_SEED_OFFSET + i
            goal_frame, goal_state, n_expert = expert_goal(env, seed, cursor=args.goal_cursor)
            if n_expert == 0:  # e.g. "select nothing and click Submit": start == goal
                skipped += 1
                continue
            budget = n_expert + args.extra_budget
            ok, n, frames, clicks = run_episode(env, seed, goal_frame, goal_state, budget, planner, rng)
            succ.append(ok)
            steps.append(n)
            episodes[task].append(dict(seed=seed, success=bool(ok), expert_clicks=n_expert, budget=budget,
                                       clicks=[{k: v for k, v in c.items() if k != "xy"} for c in clicks]))
            logging.warning(f"{task} seed={seed} expert={n_expert} budget={budget} -> {'OK' if ok else 'fail'} in {n}")
            if i < args.videos:
                for c in clicks:
                    c["text"] = click_text(c)
                video = episode_video(frames[: len(clicks) + 1], f"{task}   seed {seed}", tag, ok, clicks)
                with mp4_writer(out_dir / f"{task}_{seed}.mp4") as w:
                    for f in video:
                        w.append_data(f)
                        all_videos.append_data(f)
        env.close()
        results[task] = dict(success=float(np.mean(succ)), n=len(succ), skipped=skipped,
                             mean_clicks=float(np.mean(steps)))
    if all_videos is not None:
        all_videos.close()

    print(f"\npolicy={args.policy}")
    print(f"{'task':20s} {'success':>8s} {'n':>4s} {'clicks':>7s}")
    for task, r in results.items():
        print(f"{task:20s} {r['success']:8.2f} {r['n']:4d} {r['mean_clicks']:7.1f}")
    (out_dir / "results.json").write_text(json.dumps(dict(args=vars(args), results=results, episodes=episodes), indent=2))
    print(f"results and videos (left: observation, right: goal) in {out_dir}")


if __name__ == "__main__":
    main()
