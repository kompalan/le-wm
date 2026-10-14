import logging

import numpy as np
import torch
from stable_pretraining import data as dt
from lightning.pytorch.callbacks import Callback

def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(**imagenet_stats, source=source, target=target)
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


class ZScoreNormalizer:
    """Picklable z-score normalizer — uses a class instead of a closure so it
    survives pickle when DataLoader workers are spawned (required by LanceDataset)."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return ((x - self.mean) / self.std).float()


def get_column_normalizer(dataset, source: str, target: str):
    """Get normalizer for a specific column in the dataset."""
    col_data = dataset.get_col_data(source)
    data = torch.from_numpy(np.array(col_data))
    data = data[~torch.isnan(data).any(dim=1)]
    mean = data.mean(0, keepdim=True).clone()
    std = data.std(0, keepdim=True).clone()
    return dt.transforms.WrapTorchTransform(ZScoreNormalizer(mean, std), source=source, target=target)

class SaveCkptCallback(Callback):
    """Callback to save model checkpoint after each epoch using save_pretrained.

    If `hf_repo_id` is set, each saved checkpoint (and its config.json) is also
    pushed to that HuggingFace Hub repo in a background thread.
    """

    def __init__(self, run_name, cfg, epoch_interval: int = 1, hf_repo_id: str | None = None, hf_private: bool = True):
        super().__init__()
        self.run_name = run_name
        self.cfg = cfg
        self.epoch_interval = epoch_interval
        self.hf_repo_id = hf_repo_id
        self.hf_private = hf_private
        self._hf_api = None
        self._uploads = []

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)

        if trainer.is_global_zero:
            epoch = trainer.current_epoch + 1
            if epoch % self.epoch_interval == 0 or epoch == trainer.max_epochs:
                self._save(pl_module.model, epoch)

    def on_train_end(self, trainer, pl_module):
        # wait for pending uploads so the process doesn't exit mid-push
        for future in self._uploads:
            try:
                future.result()
            except Exception as e:
                logging.error(f"HuggingFace upload failed: {e}")
        self._uploads.clear()

    def _save(self, model, epoch):
        from stable_worldmodel.wm.utils import save_pretrained
        filename = f'weights_epoch_{epoch}.pt'
        save_pretrained(
            model,
            run_name=self.run_name,
            config=self.cfg,
            filename=filename,
        )
        if self.hf_repo_id:
            self._push_to_hub(filename, epoch)

    def _push_to_hub(self, filename, epoch):
        from huggingface_hub import HfApi
        from stable_worldmodel.data.utils import get_cache_dir

        ckpt_dir = get_cache_dir(sub_folder='checkpoints') / self.run_name
        try:
            if self._hf_api is None:
                self._hf_api = HfApi()
                self._hf_api.create_repo(self.hf_repo_id, private=self.hf_private, exist_ok=True)
            future = self._hf_api.upload_folder(
                repo_id=self.hf_repo_id,
                folder_path=ckpt_dir,
                allow_patterns=[filename, 'config.json'],
                commit_message=f'{self.run_name}: epoch {epoch}',
                run_as_future=True,
            )
            self._uploads.append(future)
        except Exception as e:
            # never crash training because of a failed upload
            logging.error(f"HuggingFace upload of {filename} failed: {e}")
