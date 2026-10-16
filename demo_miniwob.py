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

    python demo_miniwob.py --policy lewm_miniwob/weights_epoch_15.pt
    python demo_miniwob.py --policy random
"""

import argparse
import json
import logging
from pathlib import Path

import imageio
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from gymnasium.spaces import Box
from torchvision.transforms import v2 as transforms

from collect_miniwob import EVAL_SEED_OFFSET
from miniwob_env import TASKS, MiniWoBClickEnv


def pick_device(name):
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def expert_goal(env, seed, max_steps=10):
    """Run the expert up to (not including) its terminal click."""
    env.reset(seed=seed)
    for n in range(max_steps):
        action, terminal = env.expert_action()
        if terminal:
            return env.render().copy(), env.page_state(), n
        env.step(action)
    raise RuntimeError(f"expert did not reach a terminal click in {max_steps} steps")


class LeWMPlanner:
    def __init__(self, run_name, dataset, horizon, cem_kwargs, device):
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
        self.solver = swm.solver.CEMSolver(model=model, device=device, **cem_kwargs)
        self.solver.configure(
            action_space=Box(-np.inf, np.inf, shape=(1, 2)), n_envs=1,
            config=swm.PlanConfig(horizon=horizon, receding_horizon=1),
        )
        self.horizon = horizon

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
        self.plan = plan[:, 1:]  # warm start the next solve from the shifted plan
        a = plan[0, 0].numpy() * self.act_std + self.act_mean
        return np.clip(a, -1, 1).astype(np.float32), float(out["costs"][0])


def run_episode(env, seed, goal_frame, goal_state, budget, policy, rng):
    frame, _ = env.reset(seed=seed)
    if policy is not None:
        policy.reset(goal_frame)
    frames, solved = [np.concatenate([frame, goal_frame], axis=1)], False
    for t in range(budget):
        if env.page_state() == goal_state:
            solved = True
            break
        action = rng.uniform(-1, 1, 2).astype(np.float32) if policy is None else policy.act(frame)[0]
        frame, reward, terminated, truncated, _ = env.step(action)
        frames.append(np.concatenate([frame, goal_frame], axis=1))
        if terminated or truncated:
            solved = reward > 0  # e.g. clicked the target link directly
            break
    else:
        solved = env.page_state() == goal_state
    return solved, len(frames) - 1, frames


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
    p.add_argument("--device", default="auto")
    p.add_argument("--videos", type=int, default=3, help="GIFs saved per task")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="outputs/miniwob_demo")
    args = p.parse_args()

    device = pick_device(args.device)
    rng = np.random.default_rng(args.seed)
    planner = None
    if args.policy != "random":
        cem = dict(num_samples=args.num_samples, n_steps=args.n_steps, topk=args.topk, seed=args.seed)
        planner = LeWMPlanner(args.policy, args.dataset, args.horizon, cem, device)

    tag = "random" if planner is None else args.policy.replace("/", "_").removesuffix(".pt")
    out_dir = Path(args.out, tag)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for task in args.tasks:
        env = MiniWoBClickEnv(task)
        succ, steps, skipped = [], [], 0
        for i in range(args.num_eval):
            seed = EVAL_SEED_OFFSET + i
            goal_frame, goal_state, n_expert = expert_goal(env, seed)
            if n_expert == 0:  # e.g. "select nothing and click Submit": start == goal
                skipped += 1
                continue
            budget = n_expert + args.extra_budget
            ok, n, frames = run_episode(env, seed, goal_frame, goal_state, budget, planner, rng)
            succ.append(ok)
            steps.append(n)
            logging.warning(f"{task} seed={seed} expert={n_expert} budget={budget} -> {'OK' if ok else 'fail'} in {n}")
            if i < args.videos:
                imageio.mimsave(out_dir / f"{task}_{seed}.gif", frames, duration=700, loop=0)
        env.close()
        results[task] = dict(success=float(np.mean(succ)), n=len(succ), skipped=skipped,
                             mean_clicks=float(np.mean(steps)))

    print(f"\npolicy={args.policy}")
    print(f"{'task':20s} {'success':>8s} {'n':>4s} {'clicks':>7s}")
    for task, r in results.items():
        print(f"{task:20s} {r['success']:8.2f} {r['n']:4d} {r['mean_clicks']:7.1f}")
    (out_dir / "results.json").write_text(json.dumps(dict(args=vars(args), results=results), indent=2))
    print(f"results and GIFs (left: observation, right: goal) in {out_dir}")


if __name__ == "__main__":
    main()
