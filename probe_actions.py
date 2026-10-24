"""Does LeWM represent click coordinates sensibly?

Feeds a G x G grid of clicks covering the task area through the model and
reduces the results to 2-D with PCA, at two levels:

  action encoder   act_emb(a) on its own. It sees only (x, y), so if it has
                   learned a smooth map the PCA should look like a (possibly
                   warped) grid.
  predictor        z_hat(s, a) for a fixed start frame s per task. This should
                   NOT look like a uniform grid: clicks on blank page should
                   collapse together, and clicks on an element should separate
                   from them. Each panel also shows a heatmap over the page of
                   ||z_hat(s, a) - z_hat(s, a_blank)||: which click positions
                   the model thinks change the page.

For each level it reports the Spearman correlation between pairwise action
distances and pairwise embedding distances (1 = distances preserved in rank),
and the variance in the top two PCs.

    python probe_actions.py --policy lewm_text_dvic/weights_epoch_8.pt
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

from collect_miniwob import EVAL_SEED_OFFSET
from demo_miniwob import LeWMPlanner
from miniwob_env import TASKS, MiniWoBClickEnv, frame_coords


def pca_stats(z, acts):
    pca = PCA(n_components=2).fit(z)
    p = pca.transform(z)
    da = np.linalg.norm(acts[:, None] - acts[None], axis=-1)
    dz = np.linalg.norm(z[:, None] - z[None], axis=-1)
    iu = np.triu_indices(len(z), 1)
    rho = spearmanr(da[iu], dz[iu]).correlation
    return p, pca.explained_variance_ratio_.sum(), rho


def grid_plot(ax, p, G, title):
    """PCA points with grid lines: rows (constant y) blue->, columns (constant x) red."""
    P = p.reshape(G, G, 2)  # [iy, ix]
    for i in range(G):
        ax.plot(P[i, :, 0], P[i, :, 1], color=plt.cm.Blues(0.3 + 0.7 * i / G), lw=0.8)
        ax.plot(P[:, i, 0], P[:, i, 1], color=plt.cm.Reds(0.3 + 0.7 * i / G), lw=0.8)
    ax.scatter(p[:, 0], p[:, 1], c=np.arange(G * G) % G, cmap="viridis", s=6, zorder=3)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--dataset", default="miniwob_click.h5")
    ap.add_argument("--tasks", nargs="+", default=list(TASKS), choices=TASKS)
    ap.add_argument("--grid", type=int, default=41)
    ap.add_argument("--seed", type=int, default=EVAL_SEED_OFFSET)
    ap.add_argument("--draw-cursor", action="store_true")
    ap.add_argument("--hide-text", action="store_true")
    ap.add_argument("--out", default="outputs/action_probe")
    args = ap.parse_args()

    pl = LeWMPlanner(args.policy, args.dataset, 1, dict(num_samples=1, n_steps=1, topk=1), "cpu")
    model = pl.solver.model.model
    G = args.grid
    lin = np.linspace(-1, 1, G, dtype=np.float32)
    acts = np.stack(np.meshgrid(lin, lin), -1).reshape(-1, 2)  # row-major: [iy, ix], (x, y)
    a_norm = torch.as_tensor((acts - pl.act_mean) / pl.act_std, dtype=torch.float32)

    tag = args.policy.replace("/", "_").removesuffix(".pt")
    out = Path(args.out, tag); out.mkdir(parents=True, exist_ok=True)
    n = len(args.tasks)
    fig, axes = plt.subplots(3, n + 1, figsize=(3.6 * (n + 1), 10.5))

    with torch.inference_mode():
        act_emb = model.action_encoder(a_norm[:, None])[:, 0].numpy()
    p, var, rho = pca_stats(act_emb, acts)
    grid_plot(axes[0, 0], p, G, f"action encoder\nPC1+2 var {var:.0%}, spearman {rho:.2f}")
    print(f"{'action encoder':20s} PC1+2 var {var:.2f}  spearman(d_action, d_emb) {rho:.2f}")
    axes[1, 0].axis("off"); axes[2, 0].axis("off")
    axes[1, 0].text(0, 0.5, "rows of the click grid: blue\n(darker = lower on page)\n"
                    "columns: red\n(darker = further right)\n\n"
                    "row 3: ||z_hat(s,a) - z_hat(s,a_blank)||\nover click position\n"
                    "(a_blank = least-change click)", fontsize=9)

    for j, task in enumerate(args.tasks, 1):
        env = MiniWoBClickEnv(task, hide_text=args.hide_text, draw_cursor=args.draw_cursor)
        frame, _ = env.reset(seed=args.seed)
        with torch.inference_mode():
            emb = model.encode({"pixels": pl._img(frame)})["emb"]  # (1, 1, D)
            ae = model.action_encoder(a_norm[:, None])  # (N, 1, A)
            z = model.predict(emb.expand(len(acts), -1, -1), ae)[:, -1].numpy()
        p, var, rho = pca_stats(z, acts)
        grid_plot(axes[0, j], p, G, f"{task}: predictor\nPC1+2 var {var:.0%}, spearman {rho:.2f}")
        z0 = emb[0, 0].numpy()
        move = np.linalg.norm(z - z0, axis=1)
        blank = np.argmin(move)
        d = np.linalg.norm(z - z[blank], axis=1).reshape(G, G)
        # PCA again, coloured by change magnitude: separates "does something" clicks
        sc = axes[1, j].scatter(p[:, 0], p[:, 1], c=d.ravel(), cmap="magma", s=8)
        axes[1, j].set_title("predictor PCA, colour = change", fontsize=9)
        axes[1, j].set_xticks([]); axes[1, j].set_yticks([])
        fig.colorbar(sc, ax=axes[1, j], fraction=0.046)
        xy = np.stack([frame_coords(a, frame.shape[0]) for a in acts])
        ax = axes[2, j]
        ax.imshow(frame)
        ax.imshow(d, extent=[xy[:, 0].min(), xy[:, 0].max(), xy[:, 1].max(), xy[:, 1].min()],
                  cmap="magma", alpha=0.55, interpolation="nearest")
        ax.set_title("predicted change by click position", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        print(f"{task:20s} PC1+2 var {var:.2f}  spearman {rho:.2f}  "
              f"change min/median/max {d.min():.0f}/{np.median(d):.0f}/{d.max():.0f}")

    fig.suptitle(f"{args.policy}  (start frame of eval seed {args.seed})")
    fig.tight_layout()
    name = f"action_probe_seed{args.seed}.png" if args.seed != EVAL_SEED_OFFSET or len(args.tasks) < len(TASKS) else "action_probe.png"
    if len(args.tasks) < len(TASKS):
        name = f"action_probe_{'_'.join(args.tasks)}_seed{args.seed}.png"
    fig.savefig(out / name, dpi=110)
    print(f"saved {out / name}")


if __name__ == "__main__":
    main()
