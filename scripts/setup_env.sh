#!/bin/bash

sudo apt-get install -y swig

uv sync
uv tool install -y huggingface-hub[cli]

sudo apt-get remove git
sudo apt-get install -y git-lfs
curl --proto '=https' --tlsv1.2 -sSf https://raw.githubusercontent.com/huggingface/xet-core/refs/heads/main/git_xet/install.sh | sh


mkdir -p ~/.stable_worldmodel/datasets/
git clone https://huggingface.co/datasets/quentinll/lewm-pusht ~/.stable_worldmodel/datasets/pusht
unzstd ~/.stable_worldmodel/datasets/pusht/pusht_expert_train.h5.zst
