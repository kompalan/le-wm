#!/bin/bash

git checkout feature/auto_goal_set_via_vit
apt-get install -y swig

uv sync
uv tool install huggingface-hub[cli]

apt-get remove git
apt-get install git-lfs
curl --proto '=https' --tlsv1.2 -sSf https://raw.githubusercontent.com/huggingface/xet-core/refs/heads/main/git_xet/install.sh | sh


mkdir -p ~/.stable_worldmodel/datasets/
git clone https://huggingface.co/datasets/quentinll/lewm-pusht ~/.stable_worldmodel/datasets/pusht