import json
import os
from functools import partial
from pathlib import Path

import hdf5plugin
import h5py

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from module import SIGReg
from utils import get_column_normalizer, get_img_preprocessor, group_split, EpisodeBatchSampler, SaveCkptCallback


def changed_transitions(batch, thr=0.5, min_px=10):
    """(B, T-1) mask of consecutive frames whose page visibly changed.

    Uses the dataset's page_changed column when present (recorded from
    pointer-free frames, so a moved cursor alone doesn't count). Otherwise
    compares pixels: >min_px pixels differing by >thr in normalized units
    (~30/255). Identical frames encode identically, so no-op clicks can't be
    spread out and are left alone.
    """
    with torch.no_grad():
        if "page_changed" in batch:
            return torch.nan_to_num(batch["page_changed"][:, :-1], 0.0) > 0.5
        pixels = batch["pixels"]
        diff = (pixels[:, 1:] - pixels[:, :-1]).abs().amax(2)  # (B, T-1, H, W)
        return (diff > thr).flatten(2).sum(-1) > min_px


def delta_vicreg(delta, gamma):
    """VICReg variance/covariance terms on within-episode latent changes (N, D)."""
    if len(delta) < 2:
        zero = delta.sum() * 0
        return zero, zero
    delta = delta - delta.mean(0)
    std = (delta.var(0) + 1e-4).sqrt()
    var_loss = torch.relu(gamma - std).mean()
    cov = delta.T @ delta / (len(delta) - 1)
    cov_loss = (cov - torch.diag(cov.diag())).pow(2).sum() / delta.size(1)
    return var_loss, cov_loss


def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""

    ctx_len = cfg.history_size
    n_preds = cfg.num_preds
    lambd = cfg.loss.sigreg.weight

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)

    emb = output["emb"]  # (B, T, D)
    act_emb = output["act_emb"]

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, : ctx_len]

    tgt_emb = emb[:, n_preds:] # label
    pred_emb = self.model.predict(ctx_emb, ctx_act) # pred

    # LeWM loss
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"]= self.sigreg(emb.transpose(0, 1))
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]

    dv, ds = cfg.loss.get("delta_vicreg"), cfg.loss.get("delta_sigreg")
    use_dv = dv and (dv.var_weight or dv.cov_weight)
    use_ds = ds and ds.weight
    if use_dv or use_ds:
        changed = changed_transitions(batch)
        delta = (emb[:, 1:] - emb[:, :-1])[changed].float()  # (N, D)
        self.log(f"{stage}/changed_frac", changed.float().mean(), on_step=True, sync_dist=True)
    if use_dv:
        var_loss, cov_loss = delta_vicreg(delta, dv.gamma)
        output["dvar_loss"], output["dcov_loss"] = var_loss, cov_loss
        output["loss"] = output["loss"] + dv.var_weight * var_loss + dv.cov_weight * cov_loss
    if use_ds:
        # SIGReg on within-episode changes: an isotropic Gaussian, not just a variance floor
        output["dsigreg_loss"] = self.sigreg(delta[None]) if len(delta) > 1 else emb.sum() * 0
        output["loss"] = output["loss"] + ds.weight * output["dsigreg_loss"]

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output

@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop("name")
    cache_dir = os.environ.get("LOCAL_DATASET_DIR", None)
    dataset = swm.data.load_dataset(
        dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg
    )
    transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]
    
    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels") or col == "page_changed":  # 0/1 mask, not a model input
                continue
            if col in cfg.data.get("no_normalize", []):  # already scaled (e.g. OSWorld action vectors)
                continue
            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

        cfg.model.action_encoder.input_dim = cfg.data.dataset.frameskip * dataset.get_dim("action")

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    holdout = cfg.data.get("holdout")
    held_groups = None
    if holdout and holdout.frac:
        # validate on whole groups (e.g. tasks) that training never sees
        train_set, val_set, held_groups = group_split(dataset, holdout.col, holdout.frac, cfg.seed)
    else:
        train_set, val_set = spt.data.random_split(
            dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
        )

    if cfg.get("windows_per_episode"):
        loader_cfg = {k: v for k, v in cfg.loader.items() if k != "batch_size"}
        sampler = EpisodeBatchSampler(train_set, cfg.loader.batch_size, cfg.windows_per_episode, generator=rnd_gen)
        train = torch.utils.data.DataLoader(train_set, batch_sampler=sampler, **loader_cfg)
    else:
        train = torch.utils.data.DataLoader(train_set, **cfg.loader,shuffle=True, drop_last=True, generator=rnd_gen)
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)
    
    ##############################
    ##       model / optim      ##
    ##############################

    world_model = hydra.utils.instantiate(cfg.model)

    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model = world_model,
        sigreg = SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(sub_folder='checkpoints'), run_id)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)
    if held_groups is not None:
        with open(run_dir / "holdout.json", "w") as f:
            json.dump({"col": holdout.col, "values": [int(v) for v in held_groups]}, f)

    object_dump_callback = SaveCkptCallback(
        run_name=cfg.output_model_name, cfg=cfg.model, epoch_interval=1,
        hf_repo_id=cfg.hf_hub.repo_id, hf_private=cfg.hf_hub.private,
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    ckpt_path = run_dir / f"{cfg.output_model_name}_weights.ckpt"
    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=ckpt_path if ckpt_path.exists() else None,
    )

    manager()
    return


if __name__ == "__main__":
    run()
